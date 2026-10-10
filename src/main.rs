//! Cockatiel TTS module (Rust).
//!
//! An in-process module that renders chat messages into speech using a local
//! Piper VITS model via `sherpa-onnx`. It is the DEFAULT, zero-setup TTS path:
//! a single native binary (no Python, no torch, no pip). The model is a
//! self-contained `.tar.bz2` bundle (`.onnx` + `tokens.txt` + `espeak-ng-data`)
//! downloaded ONCE at startup into a module-local `voices/` directory — never on
//! the first message, so a slow first download can't stall a live chat render.
//!
//! For model families sherpa-onnx does not support, the optional Python module
//! (`tts-experimental-py`) remains the opt-in escape hatch.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use cockatiel_client::proto::container_for_engine::Payload as EnginePayload;
use cockatiel_client::proto::container_for_module::Payload as ModulePayload;
use cockatiel_client::proto::*;
use cockatiel_client::CockatielClient;
use futures_util::{SinkExt, StreamExt};
use prost::Message;
use serde::{Deserialize, Serialize};
use sherpa_onnx::{GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsVitsModelConfig};
use tokio::sync::mpsc;
use tokio::sync::Mutex as AsyncMutex;
use tokio_tungstenite::tungstenite::Message as WsMessage;
use tracing::{error, info, warn};
use tracing_subscriber::FmtSubscriber;

type WsWriteHalf = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    WsMessage,
>;

// ── config ────────────────────────────────────────────────────────────────

/// The module's editable settings, persisted under config.json's flat
/// `module_specific` object (the convention every module uses, so the TUI
/// renders them as editable fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct Config {
    /// The sherpa-onnx VITS model bundle name (a `vits-piper-*` tarball in the
    /// sherpa-onnx `tts-models` release), a full `.tar.bz2` URL, or a local
    /// unpacked directory path.
    model: String,
    /// Max chars of text rendered to speech per message (longer is truncated).
    max_chars: usize,
    /// Max audio bytes attached to a reply (larger output is dropped).
    max_audio_bytes: usize,
    /// Whether to also play the clip on this machine (for no-display setups).
    play_locally: bool,
    /// The default voice/speaker id, keyed by the `v` flag alias (per the
    /// per-flag-defaults pattern). Blank/absent falls back to the module
    /// default (0). There is deliberately no separate `speaker_id` key — `v`
    /// IS the setting, so there is one place to change the default voice.
    #[serde(default, deserialize_with = "de_opt_i32")]
    v: Option<i32>,
    /// Reconnect backoff bounds (seconds), same shape as the Python module.
    reconnect_base_secs: u32,
    reconnect_max_secs: u32,
}

/// A JSON value (number or string — the config editor can write either) as an
/// i32, or `None` when unusable.
fn json_i32(v: &serde_json::Value) -> Option<i32> {
    match v {
        serde_json::Value::Number(n) => n.as_i64().map(|n| n as i32),
        serde_json::Value::String(s) => s.trim().parse::<i32>().ok(),
        _ => None,
    }
}

/// Deserialize a voice id that may be a JSON number OR a string, yielding `None`
/// for anything unusable so a stray value never fails the whole config parse.
fn de_opt_i32<'de, D>(d: D) -> Result<Option<i32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(v.as_ref().and_then(json_i32))
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "vits-piper-en_US-lessac-medium".into(),
            max_chars: 1000,
            max_audio_bytes: 5 * 1024 * 1024,
            play_locally: false,
            v: None,
            reconnect_base_secs: 1,
            reconnect_max_secs: 30,
        }
    }
}

/// The client's connection fields (ip/port/module_name/position/priority) live
/// at the TOP LEVEL of config.json, while the module's own settings live under
/// `module_specific` — see `CockatielConfig::load_or_create`. Read both.
fn load_config(path: &Path) -> (Config, serde_json::Value) {
    let data = std::fs::read_to_string(path).unwrap_or_default();
    let root: serde_json::Value =
        serde_json::from_str(&data).unwrap_or_else(|_| serde_json::json!({}));
    let specific = root
        .get("module_specific")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    let mut config: Config = serde_json::from_value(specific.clone()).unwrap_or_default();
    // `v` is the ONLY default-voice key. On first load, migrate the legacy
    // `speaker_id` value (if present) into `v`; otherwise start at the module
    // default (0). Either way the legacy key is dropped from the file, so there
    // is exactly one place to set the default voice.
    let mut changed = false;
    if config.v.is_none() {
        config.v = Some(specific.get("speaker_id").and_then(json_i32).unwrap_or(0));
        changed = true;
    }
    if specific.get("speaker_id").is_some() {
        changed = true;
    }
    if changed {
        write_config(path, config.v.unwrap_or(0));
    }
    (config, serde_json::Value::Null)
}

/// Write `module_specific.v` and drop the legacy `speaker_id`, preserving every
/// other key in the file. Best-effort: a read/write failure is ignored (the
/// in-memory `config.v` is still correct).
fn write_config(path: &Path, v: i32) {
    let Ok(data) = std::fs::read_to_string(path) else { return };
    let Ok(mut root) = serde_json::from_str::<serde_json::Value>(&data) else { return };
    let Some(obj) = root.as_object_mut() else { return };
    let ms = obj
        .entry("module_specific".to_string())
        .or_insert_with(|| serde_json::json!({}));
    let Some(ms) = ms.as_object_mut() else { return };
    ms.insert("v".to_string(), serde_json::json!(v));
    ms.remove("speaker_id");
    if let Ok(pretty) = serde_json::to_string_pretty(&root) {
        let _ = std::fs::write(path, pretty);
    }
}

/// The speaker id to render with. A per-message `-v` wins; otherwise the
/// CURRENT `v` is read from `config.json` on disk (so a saved config change is
/// honoured on the next message with no restart), falling back to the in-memory
/// value then the module default (0). Clamped to the flag's declared range.
fn resolve_voice(override_id: Option<i32>, in_mem_v: Option<i32>, disk_v: Option<i32>) -> i32 {
    override_id
        .or(disk_v)
        .or(in_mem_v)
        .unwrap_or(0)
        .clamp(0, 903)
}

/// Read `module_specific.v` from `config.json` on disk (best-effort), so a
/// config edit takes effect on the next message instead of requiring a restart.
/// Called only when a message supplies no `-v` override.
fn current_config_v() -> Option<i32> {
    let data = std::fs::read_to_string("config.json").ok()?;
    let root: serde_json::Value = serde_json::from_str(&data).ok()?;
    root.get("module_specific")
        .and_then(|ms| ms.get("v"))
        .and_then(json_i32)
}

// ── model / voice management ──────────────────────────────────────────────

const DEFAULT_MODEL_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models";

/// Where the model bundle is unpacked (module-local, never committed).
fn voices_dir() -> PathBuf {
    PathBuf::from("voices")
}

/// The bundle stem for a model string: `vits-piper-x` or a URL `.../x.tar.bz2`
/// both become `x`.
fn resolve_bundle(model: &str) -> String {
    model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .trim_end_matches(".tar.bz2")
        .to_string()
}

/// Download + unpack the model bundle into `voices/`, if not already present.
/// Called ONCE at startup, BEFORE connecting, so a slow first fetch never
/// stalls a live chat message.
fn ensure_model(config: &Config) -> Result<PathBuf, String> {
    // A local, already-unpacked directory is used as-is.
    if Path::new(&config.model).is_dir() {
        return Ok(PathBuf::from(&config.model));
    }

    let stem = resolve_bundle(&config.model);
    let dir = voices_dir().join(&stem);
    // The bundle's .onnx is named after the VOICE (e.g. en_US-lessac-medium),
    // not the bundle dir, so scan for any .onnx rather than a fixed name.
    let has_onnx = std::fs::read_dir(&dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                e.path()
                    .extension()
                    .map(|x| x == "onnx")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if has_onnx {
        info!("TTS model {stem} already present at {}", dir.display());
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;

    let url = if config.model.starts_with("http") {
        config.model.clone()
    } else {
        format!("{DEFAULT_MODEL_URL}/{}.tar.bz2", config.model)
    };
    let archive = dir.join(format!("{stem}.tar.bz2"));
    info!("fetching TTS model {stem} from {url}...");
    download_file(&url, &archive)?;
    unpack_bz2(&archive, &dir)?;
    std::fs::remove_file(&archive).ok();

    let has_onnx = std::fs::read_dir(&dir)
        .map(|entries| {
            entries.flatten().any(|e| {
                e.path()
                    .extension()
                    .map(|x| x == "onnx")
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false);
    if !has_onnx {
        return Err(format!(
            "model bundle did not contain any .onnx (unexpected archive layout in {})",
            dir.display()
        ));
    }
    info!("TTS model {stem} ready at {}", dir.display());
    Ok(dir)
}

/// Minimal HTTPS GET to a file. Uses `ureq` (small, std-TLS) — no heavyweight
/// HTTP client, and it inherits the OS trust store so public model downloads
/// just work.
fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    let resp = ureq::get(url)
        .timeout(Duration::from_secs(600))
        .call()
        .map_err(|e| format!("GET {url}: {e}"))?;
    let mut body = Vec::new();
    let mut reader = resp.into_reader();
    std::io::copy(&mut reader, &mut body).map_err(|e| format!("read {url}: {e}"))?;
    std::fs::write(dest, &body).map_err(|e| format!("write {}: {e}", dest.display()))?;
    Ok(())
}

fn unpack_bz2(archive: &Path, dest: &Path) -> Result<(), String> {
    let data = std::fs::read(archive).map_err(|e| format!("read {}: {e}", archive.display()))?;
    let mut decoder = bzip2::read::BzDecoder::new(data.as_slice());
    let mut decoded = Vec::new();
    std::io::copy(&mut decoder, &mut decoded).map_err(|e| format!("bzip2: {e}"))?;

    // The archives carry a top-level directory named after the bundle (e.g.
    // `vits-piper-en_US-lessac-medium/...`). We already created `dest` as that
    // same name, so strip the first path component to avoid `dest/dest/...`.
    let mut ar = tar::Archive::new(decoded.as_slice());
    let mut entries = ar
        .entries()
        .map_err(|e| format!("tar entries: {e}"))?;
    while let Some(Ok(mut entry)) = entries.next() {
        let path = entry
            .path()
            .map_err(|e| format!("tar path: {e}"))?
            .into_owned();
        // Drop the first component (the bundle dir name).
        let stripped: PathBuf = path.components().skip(1).collect();
        let out_path = dest.join(stripped);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&out_path).map_err(|e| format!("mkdir: {e}"))?;
        } else if entry.header().entry_type().is_file() {
            if let Some(parent) = out_path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("mkdir: {e}"))?;
            }
            entry
                .unpack(&out_path)
                .map_err(|e| format!("unpack {}: {e}", out_path.display()))?;
        }
    }
    Ok(())
}

// ── connection / message loop ─────────────────────────────────────────────

#[derive(Clone)]
struct Session {
    auth_token: String,
    module_name: String,
    instance_uuid7: String,
}

/// A tiny rolling "recently rendered" set so a re-delivered message is acked
/// from cache instead of re-rendered (the Python module does the same).
#[derive(Default)]
struct Dedup {
    recent: VecDeque<String>,
    cap: usize,
}

impl Dedup {
    fn new(cap: usize) -> Self {
        Self { recent: VecDeque::new(), cap: cap.max(1) }
    }
    fn contains(&self, id: &str) -> bool {
        self.recent.iter().any(|x| x == id)
    }
    fn insert(&mut self, id: String) {
        self.recent.push_back(id);
        while self.recent.len() > self.cap {
            self.recent.pop_front();
        }
    }
}

/// Build the reply container that acks an IN-PROCESS message (with audio if we
/// rendered some). The processed message is echoed UNCHANGED (this module only
/// adds audio); empty audio is still a valid ack — the stage completes.
fn ack_in_process(
    s: &Session,
    process: MessageInProcess,
    audio: Vec<u8>,
    audio_type: String,
) -> ContainerForEngine {
    ContainerForEngine {
        version: 2,
        auth_token: s.auth_token.clone(),
        module_name: s.module_name.clone(),
        module_instance_uuid7: s.instance_uuid7.clone(),
        payload: Some(EnginePayload::MessageInProcess(MessageInProcess {
            message_uuid7: process.message_uuid7,
            raw_message: process.raw_message,
            processed_message: process.processed_message,
            abandon_message: false,
            audio,
            audio_type,
        })),
    }
}

async fn send_container(write: &Arc<AsyncMutex<WsWriteHalf>>, container: ContainerForEngine) {
    let mut buf = Vec::new();
    if container.encode(&mut buf).is_ok() {
        let mut w = write.lock().await;
        let _ = w.send(WsMessage::Binary(buf)).await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| format!("tracing init: {e}"))?;

    // Config first: the model is fetched BEFORE we connect, so the first
    // message is never the thing that triggers a slow download.
    let (config, _specific) = load_config(Path::new("config.json"));
    let model_dir = ensure_model(&config)?;

    // Spawn the session task: it connects, reads messages, and reconnects
    // with backoff when the socket drops. Main then just keeps the process
    // alive (same shape as every other Rust module).
    let cfg = config.clone();
    let dir = model_dir.clone();
    tokio::spawn(async move {
        session_loop(&cfg, &dir).await;
    });

    loop {
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }
}

async fn session_loop(config: &Config, model_dir: &Path) {
    let mut backoff = config.reconnect_base_secs;
    loop {
        match run_session(config, model_dir).await {
            Ok(()) => {}
            Err(e) => error!("session error: {e}"),
        }
        warn!("engine disconnected — reconnecting in {backoff}s");
        tokio::time::sleep(Duration::from_secs(backoff as u64)).await;
        backoff = (backoff * 2).min(config.reconnect_max_secs.max(1));
    }
}

async fn run_session(
    config: &Config,
    model_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = CockatielClient::connect("config.json").await?;
    let (write, read) = client.stream.split();
    let write_shared: Arc<AsyncMutex<WsWriteHalf>> = Arc::new(AsyncMutex::new(write));
    let session = Session {
        auth_token: client.auth_token.clone(),
        module_name: client.config.module_name.clone(),
        instance_uuid7: client.instance_uuid7.clone(),
    };

    let _ = mpsc::unbounded_channel::<PromptResponse>();

    // Register the `!tts` command so the engine routes ONLY `!tts` messages to
    // this module (command scoping). The engine sends a module that registered
    // specific commands nothing else — so this module no longer receives (and
    // has to discard) every chat message.
    let commands = ContainerForEngine {
        version: 2,
        auth_token: session.auth_token.clone(),
        module_name: session.module_name.clone(),
        module_instance_uuid7: session.instance_uuid7.clone(),
        payload: Some(EnginePayload::Commands(Commands {
            commands: vec![Command {
                command_name: "tts".to_string(),
                command_flag: "!".to_string(),
                command_description: "read a message out loud via TTS (e.g. !tts hello chat)".to_string(),
                command_flags: vec![Flag {
                    flag_name: "v".to_string(),
                    flag_description: "voice/speaker id for the current model (overrides the config default)".to_string(),
                    limiting_type: 3, // FLAG_LIMIT_TYPE_RANGE
                    min_val: 0.0,
                    max_val: 903.0,
                    options: vec![],
                    value: String::new(),
                }],
            }],
            alert_on_unknown_command: false,
        })),
    };
    send_container(&write_shared, commands).await;

    let mut read = read;

    // Load the TTS engine once per session.
    let tts = build_tts(model_dir)?;

    let mut dedup = Dedup::new(1000);
    let max_chars = config.max_chars.max(1);
    let max_audio_bytes = config.max_audio_bytes;

    // Run the read loop inline: it returns when the socket drops, so the
    // caller (session_loop) can reconnect with backoff.
    loop {
        let Some(msg) = read.next().await else { break };
            let data = match msg {
                Ok(WsMessage::Binary(d)) => d,
                Ok(WsMessage::Close(_)) => break,
                Ok(_) => continue,
                Err(e) => {
                    warn!("ws error: {e}");
                    break;
                }
            };
            let Ok(container) = ContainerForModule::decode(data.as_ref()) else {
                continue;
            };

            match container.payload {
                Some(ModulePayload::AuthVerify(_)) => {
                    let reply = ContainerForEngine {
                        version: 2,
                        auth_token: session.auth_token.clone(),
                        module_name: session.module_name.clone(),
                        module_instance_uuid7: session.instance_uuid7.clone(),
                        payload: Some(EnginePayload::AuthVerify(AuthVerify {
                            cur_auth: session.auth_token.clone(),
                        })),
                    };
                    send_container(&write_shared, reply).await;
                }
                Some(ModulePayload::MessagePreProcess(pre)) => {
                    if !pre.message_uuid7.is_empty() {
                        let receipt = ContainerForEngine {
                            version: 2,
                            auth_token: session.auth_token.clone(),
                            module_name: session.module_name.clone(),
                            module_instance_uuid7: session.instance_uuid7.clone(),
                            payload: Some(EnginePayload::MessageAck(MessageAck {
                                message_uuid7: pre.message_uuid7.clone(),
                            })),
                        };
                        send_container(&write_shared, receipt).await;
                    }
                    // Pass-through ack so a stray pre-process frame never
                    // stalls the chain (this module renders in-process).
                    let ack = ContainerForEngine {
                        version: 2,
                        auth_token: session.auth_token.clone(),
                        module_name: session.module_name.clone(),
                        module_instance_uuid7: session.instance_uuid7.clone(),
                        payload: Some(EnginePayload::MessagePreProcess(MessagePreProcess {
                            message_uuid7: pre.message_uuid7,
                            raw_message: pre.raw_message,
                            audio: pre.audio,
                            audio_type: pre.audio_type,
                        })),
                    };
                    send_container(&write_shared, ack).await;
                }
                Some(ModulePayload::MessageInProcess(process)) => {
                    if !process.message_uuid7.is_empty() {
                        let receipt = ContainerForEngine {
                            version: 2,
                            auth_token: session.auth_token.clone(),
                            module_name: session.module_name.clone(),
                            module_instance_uuid7: session.instance_uuid7.clone(),
                            payload: Some(EnginePayload::MessageAck(MessageAck {
                                message_uuid7: process.message_uuid7.clone(),
                            })),
                        };
                        send_container(&write_shared, receipt).await;
                    }
                    let uuid = process.message_uuid7.clone();
                    // Which text to consider: the processed message, else the raw.
                    let mut raw = process.processed_message.clone();
                    if raw.trim().is_empty() {
                        if let Some(cm) = &process.raw_message {
                            if !cm.raw_message.trim().is_empty() {
                                raw = cm.raw_message.clone();
                            }
                        }
                    }
                    // COMMAND-SCOPED: this module only renders `!tts ...`
                    // messages. Anything else is acknowledged empty (the stage
                    // completes with no audio) so the module never speaks chat
                    // the user didn't ask for.
                    let Some(args) = parse_tts(&raw) else {
                        send_container(
                            &write_shared,
                            ack_in_process(&session, process, Vec::new(), String::new()),
                        )
                        .await;
                        continue;
                    };
                    let mut text = args.prompt;
                    if text.trim().is_empty() {
                        // A bare `!tts` with nothing to say still acks.
                        send_container(
                            &write_shared,
                            ack_in_process(&session, process, Vec::new(), String::new()),
                        )
                        .await;
                        continue;
                    }
                    if text.chars().count() > max_chars {
                        text = text.chars().take(max_chars).collect();
                    }

                    // `-v <id>` overrides the default voice for this message
                    // only; otherwise the configured `v` is read FRESH from
                    // disk so a saved config change applies without a restart.
                    let sid = resolve_voice(args.voice, config.v, current_config_v());

                    let safe = get_safe_filename(&text, 50);
                    let cached = PathBuf::from("clips")
                        .join(format!("{safe}_{}.wav", &uuid[..uuid.len().min(8)]));

                    if dedup.contains(&uuid) {
                        info!("re-delivered message {uuid}; re-sending cached clip");
                        let audio = std::fs::read(&cached).unwrap_or_default();
                        send_container(
                            &write_shared,
                            ack_in_process(&session, process, audio, "audio/wav".into()),
                        )
                        .await;
                        continue;
                    }

                    info!("rendering speech for {uuid}: {text:?} (voice {sid})");
                    let rendered = render(&tts, &text, &cached, sid);
                    let (audio, audio_type) = match rendered {
                        Ok(bytes) => {
                            if bytes.len() > max_audio_bytes {
                                warn!(
                                    "clip for {uuid} is {} bytes — over {max_audio_bytes} cap; dropping audio",
                                    bytes.len()
                                );
                                (Vec::new(), String::new())
                            } else {
                                (bytes, "audio/wav".to_string())
                            }
                        }
                        Err(e) => {
                            error!("render failed for {uuid}: {e}");
                            (Vec::new(), String::new())
                        }
                    };
                    dedup.insert(uuid.clone());
                    send_container(
                        &write_shared,
                        ack_in_process(&session, process, audio, audio_type),
                    )
                    .await;
                }
                Some(ModulePayload::MessagePostProcess(post)) => {
                    // This module renders in-process now, so it is not normally
                    // sent post-process frames. Pass-through ack (echoing every
                    // incoming field) so a stray frame can never stall the
                    // chain — mirroring the pre-process arm.
                    if !post.message_uuid7.is_empty() {
                        let receipt = ContainerForEngine {
                            version: 2,
                            auth_token: session.auth_token.clone(),
                            module_name: session.module_name.clone(),
                            module_instance_uuid7: session.instance_uuid7.clone(),
                            payload: Some(EnginePayload::MessageAck(MessageAck {
                                message_uuid7: post.message_uuid7.clone(),
                            })),
                        };
                        send_container(&write_shared, receipt).await;
                    }
                    let ack = ContainerForEngine {
                        version: 2,
                        auth_token: session.auth_token.clone(),
                        module_name: session.module_name.clone(),
                        module_instance_uuid7: session.instance_uuid7.clone(),
                        payload: Some(EnginePayload::MessagePostProcess(MessagePostProcess {
                            message_uuid7: post.message_uuid7,
                            raw_message: post.raw_message,
                            processed_message: post.processed_message,
                            audio: post.audio,
                            audio_type: post.audio_type,
                        })),
                    };
                    send_container(&write_shared, ack).await;
                }
                _ => {}
        }
    }

    Ok(())
}

/// Build the sherpa-onnx TTS engine from an unpacked model directory.
fn build_tts(model_dir: &Path) -> Result<OfflineTts, String> {
    let onnx = std::fs::read_dir(model_dir)
        .map_err(|e| format!("list {}: {e}", model_dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().map(|x| x == "onnx").unwrap_or(false))
        .ok_or_else(|| format!("no .onnx found in {}", model_dir.display()))?;

    let tokens = model_dir.join("tokens.txt");
    if !tokens.exists() {
        return Err(format!("no tokens.txt in {}", model_dir.display()));
    }
    let data_dir = model_dir.join("espeak-ng-data");
    let data_dir = if data_dir.is_dir() {
        data_dir
    } else {
        model_dir.to_path_buf()
    };

    let config = OfflineTtsConfig {
        model: sherpa_onnx::OfflineTtsModelConfig {
            vits: OfflineTtsVitsModelConfig {
                model: Some(onnx.to_string_lossy().into_owned()),
                tokens: Some(tokens.to_string_lossy().into_owned()),
                noise_scale: 0.667,
                noise_scale_w: 0.8,
                length_scale: 1.0,
                data_dir: Some(data_dir.to_string_lossy().into_owned()),
                ..Default::default()
            },
            num_threads: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    OfflineTts::create(&config).ok_or_else(|| "sherpa-onnx OfflineTts::create failed".to_string())
}

fn render(
    tts: &OfflineTts,
    text: &str,
    out: &Path,
    sid: i32,
) -> Result<Vec<u8>, String> {
    let gen = GenerationConfig {
        sid,
        ..Default::default()
    };
    let audio = tts
        .generate_with_config(text, &gen, None::<fn(&[f32], f32) -> bool>)
        .ok_or_else(|| "synthesis returned no audio".to_string())?;
    std::fs::create_dir_all(out.parent().unwrap_or(Path::new(".")))
        .map_err(|e| format!("mkdir: {e}"))?;
    if !audio.save(out.to_string_lossy().as_ref()) {
        return Err(format!("could not write {}", out.display()));
    }
    std::fs::read(out).map_err(|e| format!("read {}: {e}", out.display()))
}

fn get_safe_filename(text: &str, max: usize) -> String {
    let mut out = String::with_capacity(text.len());
    let mut prev_underscore = false;
    for c in text.chars() {
        if c.is_alphanumeric() || c == '-' {
            out.push(c);
            prev_underscore = false;
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    let trimmed = out.trim_matches('_');
    trimmed.chars().take(max).collect()
}

/// A parsed `!tts [flags] <prompt>` invocation.
struct TtsArgs {
    /// The `-v <id>` voice/speaker override, when the caller supplied one.
    voice: Option<i32>,
    /// The text to speak (the message with the command token and flags removed).
    prompt: String,
}

/// Parse a `!tts [flags] <prompt>` command. Returns `None` when the message is
/// not a `!tts` command (this module is COMMAND-SCOPED — it must never render
/// chat the user didn't ask to be spoken).
///
/// Flag syntax mirrors the engine's parser: `-v 250`, `-v:250`, or a bare `-v`
/// (value "true"). The only flag interpreted here is `v` (voice/speaker id); any
/// other flag is consumed (so it isn't spoken) but ignored.
fn parse_tts(message: &str) -> Option<TtsArgs> {
    let trimmed = message.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("!tts") {
        return None;
    }
    let after = &trimmed[4..];
    // Require a separator after the command token so `!ttssomething` isn't one.
    if !after.is_empty() && !after.starts_with(char::is_whitespace) {
        return None;
    }

    let tokens: Vec<&str> = after.split_whitespace().collect();
    let mut voice: Option<i32> = None;
    let mut words: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        let tok = tokens[i];
        if tok.starts_with('-') && tok.len() > 1 {
            let body = &tok[1..];
            let (name, inline) = match body.split_once(':') {
                Some((n, v)) => (n, Some(v)),
                None => (body, None),
            };
            let value = if let Some(v) = inline {
                v.to_string()
            } else if i + 1 < tokens.len() && !tokens[i + 1].starts_with('-') {
                i += 1;
                tokens[i].to_string()
            } else {
                "true".to_string()
            };
            if name.eq_ignore_ascii_case("v") {
                voice = value.trim().parse::<i32>().ok().filter(|n| *n >= 0);
            }
        } else {
            words.push(tok);
        }
        i += 1;
    }
    Some(TtsArgs {
        voice,
        prompt: words.join(" "),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_evicts_oldest() {
        let mut d = Dedup::new(3);
        d.insert("a".into());
        d.insert("b".into());
        d.insert("c".into());
        assert!(d.contains("a"));
        d.insert("d".into());
        assert!(!d.contains("a"), "oldest evicted");
        assert!(d.contains("d"));
    }

    #[test]
    fn safe_filename_sanitizes() {
        assert_eq!(get_safe_filename("Hello, world! #1", 50), "Hello_world_1");
        assert_eq!(get_safe_filename("short", 5), "short");
    }

    #[test]
    fn resolve_bundle_stems_urls_and_names() {
        assert_eq!(resolve_bundle("vits-piper-en_US-lessac-medium"), "vits-piper-en_US-lessac-medium");
        assert_eq!(resolve_bundle("https://host/x/vits-piper-foo.tar.bz2"), "vits-piper-foo");
    }

    fn sample_process(processed: &str) -> MessageInProcess {
        MessageInProcess {
            message_uuid7: "msg-1".into(),
            raw_message: None,
            processed_message: processed.into(),
            abandon_message: false,
            audio: Vec::new(),
            audio_type: String::new(),
        }
    }

    #[test]
    fn ack_in_process_carries_audio_and_acks_the_stage() {
        let s = Session {
            auth_token: "tok".into(),
            module_name: "tts-rs".into(),
            instance_uuid7: "uuid-1".into(),
        };
        let c = ack_in_process(
            &s,
            sample_process("hello"),
            vec![1, 2, 3],
            "audio/wav".into(),
        );
        assert_eq!(c.module_name, "tts-rs");
        assert_eq!(c.auth_token, "tok");
        let ip = match c.payload {
            Some(EnginePayload::MessageInProcess(p)) => p,
            _ => panic!("expected MessageInProcess"),
        };
        assert_eq!(ip.message_uuid7, "msg-1");
        assert_eq!(ip.processed_message, "hello", "processed message echoed unchanged");
        assert!(!ip.abandon_message);
        assert_eq!(ip.audio, vec![1, 2, 3]);
        assert_eq!(ip.audio_type, "audio/wav");
    }

    #[test]
    fn empty_audio_is_a_valid_ack() {
        let s = Session {
            auth_token: "t".into(),
            module_name: "tts-rs".into(),
            instance_uuid7: "u".into(),
        };
        let c = ack_in_process(&s, sample_process(""), Vec::new(), String::new());
        let ip = match c.payload {
            Some(EnginePayload::MessageInProcess(p)) => p,
            _ => panic!("expected MessageInProcess"),
        };
        assert!(ip.audio.is_empty());
        assert_eq!(ip.audio_type, "");
    }

    #[test]
    fn non_tts_in_process_acks_empty_audio() {
        let s = Session {
            auth_token: "t".into(),
            module_name: "tts-rs".into(),
            instance_uuid7: "u".into(),
        };
        let process = sample_process("just chatting");
        assert!(
            parse_tts(&process.processed_message).is_none(),
            "plain chat is never spoken"
        );
        let c = ack_in_process(&s, process, Vec::new(), String::new());
        let ip = match c.payload {
            Some(EnginePayload::MessageInProcess(p)) => p,
            _ => panic!("expected MessageInProcess"),
        };
        assert!(ip.audio.is_empty(), "non-!tts message acks with no audio");
        assert_eq!(ip.audio_type, "");
        assert_eq!(ip.processed_message, "just chatting");
    }

    #[test]
    fn parse_tts_only_accepts_tts_commands() {
        assert_eq!(parse_tts("!tts hello world").unwrap().prompt, "hello world");
        assert_eq!(parse_tts("  !tts   hello  ").unwrap().prompt, "hello");
        assert_eq!(parse_tts("!tts").unwrap().prompt, "");
        assert!(parse_tts("hello world").is_none(), "plain chat is never spoken");
        assert!(parse_tts("!ttssomething").is_none(), "no separator after the token");
        assert_eq!(parse_tts("!tts hello").unwrap().prompt, "hello");
        assert_eq!(parse_tts("!TTS HELLO").unwrap().prompt, "HELLO", "case-insensitive command");
    }

    #[test]
    fn parse_tts_reads_the_voice_flag_and_strips_it_from_the_prompt() {
        // `-v <id>` (space form) is read and removed from the spoken prompt.
        let a = parse_tts("!tts -v 250 hello chat").unwrap();
        assert_eq!(a.voice, Some(250));
        assert_eq!(a.prompt, "hello chat");
        // `-v:<id>` (inline form).
        let b = parse_tts("!tts -v:12 hi").unwrap();
        assert_eq!(b.voice, Some(12));
        assert_eq!(b.prompt, "hi");
        // No flag -> no override (the config default is used).
        assert_eq!(parse_tts("!tts hi there").unwrap().voice, None);
        // Non-numeric value -> ignored.
        assert_eq!(parse_tts("!tts -v loud hi").unwrap().voice, None);
        // Unknown flags are consumed but ignored; the prompt is what remains.
        let c = parse_tts("!tts -r 1.5 -v 7 hello").unwrap();
        assert_eq!(c.voice, Some(7));
        assert_eq!(c.prompt, "hello");
    }

    #[test]
    fn resolve_voice_prefers_override_then_disk_then_memory() {
        // -v wins over everything.
        assert_eq!(resolve_voice(Some(250), Some(100), Some(200)), 250);
        // No override -> the FRESH disk value wins over the stale in-memory one.
        assert_eq!(resolve_voice(None, Some(100), Some(200)), 200);
        // Disk read failed -> fall back to the in-memory value.
        assert_eq!(resolve_voice(None, Some(100), None), 100);
    }

    #[test]
    fn resolve_voice_defaults_to_zero_and_clamps() {
        assert_eq!(resolve_voice(None, None, None), 0, "no value -> module default 0");
        assert_eq!(resolve_voice(Some(-7), None, None), 0, "negative clamps to 0");
        assert_eq!(resolve_voice(Some(9999), None, None), 903, "over clamps to the range");
    }

    #[test]
    fn config_v_accepts_a_number_or_a_string() {
        let n: Config = serde_json::from_str(r#"{"v": 250}"#).unwrap();
        assert_eq!(n.v, Some(250));
        let s: Config = serde_json::from_str(r#"{"v": "250"}"#).unwrap();
        assert_eq!(s.v, Some(250), "a string id from the editor still parses");
        let blank: Config = serde_json::from_str(r#"{"v": ""}"#).unwrap();
        assert_eq!(blank.v, None, "blank falls back to the module default");
    }

    #[test]
    fn load_config_migrates_the_legacy_speaker_id_into_v_and_drops_it() {
        let dir = std::env::temp_dir().join(format!("ckt-tts-load-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{"module_specific":{"speaker_id":42}}"#).unwrap();
        let (config, _) = load_config(&path);
        assert_eq!(config.v, Some(42), "legacy speaker_id migrated into v");
        let root: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(root["module_specific"]["v"], 42, "v persisted");
        assert!(
            root["module_specific"].get("speaker_id").is_none(),
            "legacy speaker_id dropped"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_config_sets_v_and_drops_the_legacy_speaker_id() {
        let dir = std::env::temp_dir().join(format!("ckt-tts-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{"module_name":"tts-rs","module_specific":{"model":"m","speaker_id":42}}"#,
        )
        .unwrap();
        write_config(&path, 43);
        let root: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(root["module_specific"]["v"], 43, "v written");
        assert_eq!(root["module_specific"]["model"], "m", "other keys preserved");
        assert_eq!(root["module_name"], "tts-rs", "top level preserved");
        assert!(
            root["module_specific"].get("speaker_id").is_none(),
            "legacy speaker_id removed"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
