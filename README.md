# tts-rs (Rust)

Cockatiel's **default** text-to-speech module. An `inprocess` module that
renders chat messages into speech using a Piper VITS model via
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — a single native Rust
binary with **no Python, no pip, no torch**. This is the zero-setup path the
majority of users should use.

## How it works

- Connects to the engine as an **in-process** module (same protocol as every
  other Rust module, via [cockatiel-client](https://github.com/vulbyte/cockatiel_client-rs)).
- On **startup** (never on the first message), downloads a self-contained model
  bundle into `voices/` — a `.tar.bz2` of `.onnx` + `tokens.txt` +
  `espeak-ng-data` from the sherpa-onnx `tts-models` release. Default:
  `vits-piper-en_US-libritts_r-medium` (~100 MB, 904 speakers).
- It is **command-scoped**: it only renders `!tts <text>` messages (everything
  else is acked with empty audio). For each such message it attaches the audio
  (`audio/wav`) to its `MessageInProcess` ack; the engine carries the audio
  forward and hands it to the post-process displays (term-chat) **inline**, so
  the clip plays with no module-to-module fetch. If rendering fails, the module
  still acks (empty audio) so the chain completes instead of hanging.
- Re-delivered messages are acked from a cached clip rather than re-rendered.

## Command

```
!tts [-v <id>] <prompt>
```

- `<prompt>` is spoken aloud.
- `-v <id>` (or `-v:<id>`) picks the voice/speaker id **for that message only**,
  overriding the configured default. If no `-v` is given, the default (`v`) is
  used. Valid ids depend on the model (0–903 for the default `libritts_r`
  bundle). The engine parses the flag and passes it to the module; `!help tts-rs`
  lists it.

## Config (`config.json` → `module_specific`)

| key | default | meaning |
| --- | --- | --- |
| `model` | `vits-piper-en_US-libritts_r-medium` | model bundle name, a `.tar.bz2` URL, or a local unpacked dir |
| `max_chars` | `1000` | truncate longer messages before rendering |
| `max_audio_bytes` | `5242880` | drop audio larger than this |
| `play_locally` | `false` | also play on this machine |
| `v` | `0` | **default voice/speaker id** — the `-v` flag alias; auto-filled on first start |
| `reconnect_base_secs` / `reconnect_max_secs` | `1` / `30` | engine reconnect backoff |

### Changing the voice

Set `module_specific.v` to the voice/speaker id you want as the default (the
module auto-fills this key on first start; the config editor under the module's
edit screen exposes it). `-v <id>` on a single `!tts` message overrides it for
that message. The `model` field selects the voice/bundle. `v` is the only
default-voice key — there is no separate `speaker_id` setting.

The default voice is the **US English "libritts_r medium"** Piper model
(`vits-piper-en_US-libritts_r-medium`) — a 904-speaker audiobook corpus, so set
`v` to choose a voice (see below). Other ready-made bundles (all
downloadable on first start, no Python needed):

| `model` | voice |
| --- | --- |
| `vits-piper-en_US-libritts_r-medium` | US English, "libritts_r" medium, 904 speakers (default) |
| `vits-piper-en_GB-vctk-medium` | British English, VCTK, 109 speakers |
| `vits-piper-en_US-lessac-medium` | US English, "lessac" medium |
| `vits-piper-en_US-lessac-low` | US English, "lessac" low |
| `vits-piper-en_US-lessac-high` | US English, "lessac" high |
| `vits-piper-en_US-amy-medium` | US English, "amy" medium |
| `vits-piper-en_GB-alan-medium` | British English, "alan" medium |
| `vits-piper-en_GB-jenny_dioco-medium` | British English, "jenny" medium |
| `vits-piper-fr_FR-upmc-medium` | French, "upmc" medium |
| `vits-piper-de_DE-thorsten-medium` | German, "thorsten" medium |
| `vits-piper-es_ES-davefx-medium` | Spanish, "davefx" medium |

To switch voices, set `model` to one of the above (or any `vits-piper-*`
bundle in the
[sherpa-onnx `tts-models` release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)),
then restart the module (press `x` then `s` on it in the modules window). The
new bundle downloads into `voices/` on the next start. A full `.tar.bz2` URL
or a local unpacked directory also works.

`v` only matters for multi-speaker bundles (check `num_speakers` in
the model card); the single-speaker Piper voices ignore it.

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