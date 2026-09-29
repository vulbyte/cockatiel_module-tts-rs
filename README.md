# tts-rs (Rust)

Cockatiel's **default** text-to-speech module. A `postprocess` module that
renders chat messages into speech using a Piper VITS model via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — a single native Rust
binary with **no Python, no pip, no torch**. This is the zero-setup path the
majority of users should use.

## How it works

- Connects to the engine as a postprocess module (same protocol as every other
  Rust module, via [cockatiel-client](https://github.com/vulbyte/cockatiel_client-rs)).
- On **startup** (never on the first message), downloads a self-contained model
  bundle into `voices/` — a `.tar.bz2` of `.onnx` + `tokens.txt` +
  `espeak-ng-data` from the sherpa-onnx `tts-models` release. Default:
  `vits-piper-en_US-lessac-medium` (~67 MB).
- For each `MessagePostProcess`, renders the message text to speech, attaches
  the audio (`audio/wav`) to the reply, and the engine persists it to the
  timeline. If rendering fails, the module still acks (empty audio) so the
  message completes instead of hanging.
- Re-delivered messages are acked from a cached clip rather than re-rendered.

## Config (`config.json` → `module_specific`)

| key | default | meaning |
| --- | --- | --- |
| `model` | `vits-piper-en_US-lessac-medium` | model bundle name, a `.tar.bz2` URL, or a local unpacked dir |
| `max_chars` | `1000` | truncate longer messages before rendering |
| `max_audio_bytes` | `5242880` | drop audio larger than this |
| `play_locally` | `false` | also play on this machine |
| `speaker_id` | `0` | multi-speaker voice id |
| `reconnect_base_secs` / `reconnect_max_secs` | `1` / `30` | engine reconnect backoff |

## Dependencies

- [cockatiel-client](https://github.com/vulbyte/cockatiel_client-rs) (pinned by git rev)
- [sherpa-onnx](https://crates.io/crates/sherpa-onnx) (statically links ONNX Runtime)
- Tokio, tokio-tungstenite, prost, serde, tracing, uuid, ureq, bzip2, tar

## Build

```
cargo build --release
```

The manifest ships per-OS/arch `binary` routes, so the supervisor runs the
prebuilt binary directly — no interpreter, no environment setup.

## The optional escape hatch

For model families sherpa-onnx doesn't support (XTTS, F5, MeloTTS, etc.), the
opt-in Python module **`tts-experimental-py`** remains available for users who
want to go deeper. `tts-rs` is the recommended, stable default.