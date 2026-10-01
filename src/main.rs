//! Cockatiel TTS module (Rust).
//!
//! A post-process module that renders chat messages into speech using a local
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
    /// Optional voice/speaker id for multi-speaker models (0 = default).
    speaker_id: i64,
    /// Reconnect backoff bounds (seconds), same shape as the Python module.
    reconnect_base_secs: u64,
    reconnect_max_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "vits-piper-en_US-lessac-medium".into(),
            max_chars: 1000,
            max_audio_bytes: 5 * 1024 * 1024,
            play_locally: false,
            speaker_id: 0,
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
    let config: Config = serde_json::from_value(specific).unwrap_or_default();
    (config, serde_json::Value::Null)
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

/// Build the reply container that acks a post-process message (with audio if
/// we rendered some). Empty audio is still a valid ack — the stage completes.
fn ack_post_process(
    s: &Session,
    uuid: String,
    processed_message: String,
    audio: Vec<u8>,
    audio_type: String,
) -> ContainerForEngine {
    ContainerForEngine {
        version: 2,
        auth_token: s.auth_token.clone(),
        module_name: s.module_name.clone(),
        module_instance_uuid7: s.instance_uuid7.clone(),
        payload: Some(EnginePayload::MessagePostProcess(MessagePostProcess {
            message_uuid7: uuid,
            raw_message: None,
            processed_message,
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
        tokio::time::sleep(Duration::from_secs(backoff)).await;
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
                command_flags: vec![],
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
                    // stalls the chain (this module only declares postprocess).
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
                    let ack = ContainerForEngine {
                        version: 2,
                        auth_token: session.auth_token.clone(),
                        module_name: session.module_name.clone(),
                        module_instance_uuid7: session.instance_uuid7.clone(),
                        payload: Some(EnginePayload::MessageInProcess(MessageInProcess {
                            message_uuid7: process.message_uuid7,
                            raw_message: process.raw_message,
                            processed_message: process.processed_message,
                            abandon_message: process.abandon_message,
                            audio: process.audio,
                            audio_type: process.audio_type,
                        })),
                    };
                    send_container(&write_shared, ack).await;
                }
                Some(ModulePayload::MessagePostProcess(post)) => {
                    let uuid = post.message_uuid7;
                    if !uuid.is_empty() {
                        let receipt = ContainerForEngine {
                            version: 2,
                            auth_token: session.auth_token.clone(),
                            module_name: session.module_name.clone(),
                            module_instance_uuid7: session.instance_uuid7.clone(),
                            payload: Some(EnginePayload::MessageAck(MessageAck {
                                message_uuid7: uuid.clone(),
                            })),
                        };
                        send_container(&write_shared, receipt).await;
                    }
                    // Which text to consider: the processed message, else the raw.
                    let mut raw = post.processed_message.clone();
                    if raw.trim().is_empty() {
                        if let Some(cm) = &post.raw_message {
                            if !cm.raw_message.trim().is_empty() {
                                raw = cm.raw_message.clone();
                            }
                        }
                    }
                    // COMMAND-SCOPED: this module only renders `!tts ...`
                    // messages. Anything else is acknowledged empty (the stage
                    // completes with no audio) so the module never speaks chat
                    // the user didn't ask for.
                    let Some(mut text) = extract_tts_prompt(&raw) else {
                        send_container(
                            &write_shared,
                            ack_post_process(
                                &session,
                                uuid,
                                String::new(),
                                Vec::new(),
                                String::new(),
                            ),
                        )
                        .await;
                        continue;
                    };
                    if text.trim().is_empty() {
                        // A bare `!tts` with nothing to say still acks.
                        send_container(
                            &write_shared,
                            ack_post_process(
                                &session,
                                uuid,
                                String::new(),
                                Vec::new(),
                                String::new(),
                            ),
                        )
                        .await;
                        continue;
                    }
                    if text.chars().count() > max_chars {
                        text = text.chars().take(max_chars).collect();
                    }

                    let safe = get_safe_filename(&text, 50);
                    let cached = PathBuf::from("clips")
                        .join(format!("{safe}_{}.wav", &uuid[..uuid.len().min(8)]));

                    if dedup.contains(&uuid) {
                        info!("re-delivered message {uuid}; re-sending cached clip");
                        let audio = std::fs::read(&cached).unwrap_or_default();
                        send_container(
                            &write_shared,
                            ack_post_process(
                                &session,
                                uuid,
                                text.clone(),
                                audio,
                                "audio/wav".into(),
                            ),
                        )
                        .await;
                        continue;
                    }

                    info!("rendering speech for {uuid}: {text:?}");
                    let rendered = render(&tts, &text, &cached, config);
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
                        ack_post_process(&session, uuid, text, audio, audio_type),
                    )
                    .await;
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
    config: &Config,
) -> Result<Vec<u8>, String> {
    let gen = GenerationConfig {
        sid: config.speaker_id as i32,
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

/// Extract the text to speak from a message, IF it is a `!tts` command.
///
/// Returns `Some(prompt)` when the message is a `!tts ...` command (the prompt
/// is everything after the `!tts` token, leading whitespace trimmed), and
/// `None` for anything else. This module is COMMAND-SCOPED: it must not render
/// every chat message — only ones the user explicitly asked to be spoken.
///
/// The command shape is `!tts [flags] <prompt>`; flags are not yet interpreted
/// (voice selection lives in config), so only the prompt is returned today.
fn extract_tts_prompt(message: &str) -> Option<String> {
    let trimmed = message.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !lower.starts_with("!tts") {
        return None;
    }
    let rest = &trimmed[4..];
    // Require a separator (space) after the command token so `!ttssomething`
    // is not treated as a command.
    let rest = rest.trim_start();
    if rest.is_empty() {
        return Some(String::new());
    }
    if !trimmed[4..].starts_with(char::is_whitespace) && !trimmed[4..].is_empty() {
        return None;
    }
    Some(rest.to_string())
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

    #[test]
    fn ack_post_process_carries_audio_and_acks_the_stage() {
        let s = Session {
            auth_token: "tok".into(),
            module_name: "tts-rs".into(),
            instance_uuid7: "uuid-1".into(),
        };
        let c = ack_post_process(
            &s,
            "msg-1".into(),
            "hello".into(),
            vec![1, 2, 3],
            "audio/wav".into(),
        );
        assert_eq!(c.module_name, "tts-rs");
        assert_eq!(c.auth_token, "tok");
        let post = match c.payload {
            Some(EnginePayload::MessagePostProcess(p)) => p,
            _ => panic!("expected MessagePostProcess"),
        };
        assert_eq!(post.message_uuid7, "msg-1");
        assert_eq!(post.processed_message, "hello");
        assert_eq!(post.audio, vec![1, 2, 3]);
        assert_eq!(post.audio_type, "audio/wav");
    }

    #[test]
    fn empty_audio_is_a_valid_ack() {
        let s = Session {
            auth_token: "t".into(),
            module_name: "tts-rs".into(),
            instance_uuid7: "u".into(),
        };
        let c = ack_post_process(&s, "m".into(), String::new(), Vec::new(), String::new());
        let post = match c.payload {
            Some(EnginePayload::MessagePostProcess(p)) => p,
            _ => panic!("expected MessagePostProcess"),
        };
        assert!(post.audio.is_empty());
        assert_eq!(post.audio_type, "");
    }

    #[test]
    fn extract_tts_prompt_only_accepts_tts_commands() {
        assert_eq!(extract_tts_prompt("!tts hello world"), Some("hello world".into()));
        assert_eq!(extract_tts_prompt("  !tts   hello  "), Some("hello".into()));
        assert_eq!(extract_tts_prompt("!tts"), Some(String::new()));
        assert_eq!(extract_tts_prompt("hello world"), None, "plain chat is never spoken");
        assert_eq!(extract_tts_prompt("!ttssomething"), None, "no separator after the token");
        assert_eq!(extract_tts_prompt("!tts hello"), Some("hello".into()));
        assert_eq!(extract_tts_prompt("!TTS HELLO"), Some("HELLO".into()), "case-insensitive command");
    }
}
