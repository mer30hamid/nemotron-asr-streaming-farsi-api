# nemotron-asr-streaming-farsi-api

An OpenAI-compatible Persian speech-to-text API server for the Nemotron ASR streaming model. It runs locally on your CPU. No cloud, no GPU, no API key.

The server loads the Persian Nemotron model and exposes one endpoint, `POST /v1/audio/transcriptions`. Any dictation client that can talk to that endpoint can use it. VocaLinux on Linux is the one I have tested. OpenWhispr should work the same way, but I have not tested it yet.

The name comes from the original model, [mehdi-hf/nemotron-asr-streaming-farsi](https://huggingface.co/mehdi-hf/nemotron-asr-streaming-farsi), plus `-api`. The training code for that model lives at [mallahyari/nemotron-asr-streaming-farsi](https://github.com/mallahyari/nemotron-asr-streaming-farsi). It has no client name in it, because the client can change. It has no `onnx` in it either. ONNX through sherpa-onnx is the backend I use now, but a different export of the same model could replace it later without a rename.

```text
You speak (Persian)
        |
        v
+----------------------+   WAV, 16 kHz mono    +----------------------------------+
|  Dictation client    | --------------------> | nemotron-asr-streaming-farsi-api |
|  VocaLinux           |                       | Nemotron Farsi, int8             |
|  OpenWhispr          | <-------------------- | sherpa-onnx, CPU                 |
|  TapWisper, curl     |     {"text": "..."}   +----------------------------------+
+----------------------+
        |
        v
Text is typed into the focused app (dictation clients)
```

Everything runs on your own machine. Audio never leaves `127.0.0.1` unless you point a client at a LAN address yourself.

## Tested clients

| Client | Platform | Status |
| --- | --- | --- |
| VocaLinux (Remote API engine) | Linux | ✅ Tested |
| OpenWhispr (BYOK mode) | Windows, macOS, Linux | Not tested |
| TapWisper (self-hosted endpoint) | Windows, macOS | Not tested |
| Any other OpenAI-compatible client | - | Not tested |

If you test another client, please open an issue and say what worked or what did not.

## Features

- Persian dictation in any app, through VocaLinux, OpenWhispr, TapWisper, or plain `curl`
- Local only. The server binds to `127.0.0.1`, and there is no account, no telemetry, and no cloud API
- CPU only. It uses the quantized `int8` ONNX model through sherpa-onnx, so no GPU is needed
- One OpenAI-compatible endpoint: `POST /v1/audio/transcriptions`, returning `{"text": "..."}`
- Hardened container: read-only filesystem, all Linux capabilities dropped, `no-new-privileges`, and the model mounted read-only
- Small. One Python file, two dependencies (`sherpa-onnx`, `numpy`), and a `python:3.11-slim` base image

## Requirements

- Linux, with Docker Compose or Podman Compose (recommended), or Python 3.11+ to run the script directly
- The Persian Nemotron ONNX model files (see below)
- A dictation client to speak into, for example VocaLinux on Linux or OpenWhispr on Windows, macOS, and Linux. Or just `curl` to test the API
- A microphone. The client handles capture, voice activity detection, and text injection

## Get the model

The model files are not included in this project. Download them yourself before you start the server. Without them, the container will not start.

Download them from Hugging Face: **[Nimaone/nemotron-asr-streaming-farsi-onnx](https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx)**

That is the ONNX (int8) export of the Persian Nemotron model [mehdi-hf/nemotron-asr-streaming-farsi](https://huggingface.co/mehdi-hf/nemotron-asr-streaming-farsi), a Persian fine-tune of NVIDIA's Nemotron streaming ASR model, set to the `fa-IR` language slot.

### Files you must download

All four files are required. The server loads each one by name:

| File | Size | Role |
| --- | ---: | --- |
| `encoder.int8.onnx` | 628 MB | Encoder |
| `decoder.int8.onnx` | 7.0 MB | Decoder |
| `joiner.int8.onnx` | 1.7 MB | Joiner |
| `tokens.txt` | 16 KB | Tokenizer vocabulary |

Arrange the downloaded files like this:

```text
<your-model-directory>/
└── nemotron-asr-streaming-farsi-onnx/
    ├── encoder.int8.onnx
    ├── decoder.int8.onnx
    ├── joiner.int8.onnx
    └── tokens.txt
```

The server expects exactly this folder name, `nemotron-asr-streaming-farsi-onnx`, with those four files inside. Check your download with:

```bash
ls -lh /path/to/your/model-directory/nemotron-asr-streaming-farsi-onnx/
```

## Quick start

### 1. Point Compose at your model folder

Copy the example environment file and set `MODEL_DIR` to the model folder itself. That is the folder containing `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, and `tokens.txt`. Not its parent, and not a symlink.

```bash
cp .env.example .env
# Edit .env and set, for example:
# MODEL_DIR=/home/yourname/models/nemotron-asr-streaming-farsi-onnx
```

`.env` is git-ignored. It holds a path from your own machine, so never commit it. `compose.yaml` mounts it read-only exactly where the server looks for it:

```yaml
volumes:
  - ${MODEL_DIR}:/models/nemotron-asr-streaming-farsi-onnx:ro
```

With that mount, the server finds the files here:

```text
/models/nemotron-asr-streaming-farsi-onnx/
├── encoder.int8.onnx
├── decoder.int8.onnx
├── joiner.int8.onnx
└── tokens.txt
```

Those paths are fixed inside the container (`MODEL_PATH`, default `/models/nemotron-asr-streaming-farsi-onnx`). If any of the four files is missing, misnamed, or one level too deep or too shallow, model loading will fail at startup. Do not use a symlink inside a mounted volume. It is resolved inside the container, not on the host, and will break if its target is outside the mount.

All other settings (port, threads, sample rate, language, decoding method, max upload, log level) are also in `.env`. See `.env.example` for the full list with defaults. You do not need to edit `compose.yaml` or the Python file to change them.

### 2. start the server

To start the API server, you can use Docker or Python:

#### 1. Using Docker

```bash
docker compose up -d --build
```

The first start takes a little longer while the model loads. The container is named `nemotron-asr-streaming-farsi-api` and restarts automatically (`unless-stopped`).

#### 2. Using python

```bash
python3 -m venv venv # in windows: python -m venv venv
source venv/bin/activate # in windows: .\venv\Scripts\activate
pip install -r requirements.txt
MODEL_PATH=/path/to/nemotron-asr-streaming-farsi-onnx # in windows: $env:MODEL_PATH = "c:\path\to\nemotron-asr-streaming-farsi-onnx"
python nemotron_asr_streaming_farsi_api.py
```

The script reads the model location from the `MODEL_PATH` environment variable (default `/models/nemotron-asr-streaming-farsi-onnx`). Outside the container, point `MODEL_PATH` at your real model folder. All other settings are environment variables too. See **Configuration** section.

### 3. Check that it is up

```bash
curl http://127.0.0.1:8765/
```

Expected response:

```json
{"status": "ok", "service": "Nemotron Farsi ASR", "model": "nemotron-farsi"}
```

## Connect a client to it

### VocaLinux (Linux) ✅ Tested

1. Start the server (above) and leave it running.
2. Open VocaLinux **Settings** and select the **Remote API** recognition engine. Or start it from the terminal:

   ```bash
   vocalinux --engine remote_api
   ```

3. In the Remote API settings, enter:

   | Setting | Value |
   | --- | --- |
   | Server URL | `http://127.0.0.1:8765` |
   | Endpoint | `/v1/audio/transcriptions` |
   | Model | `nemotron-farsi` |

   The model name is only a label for VocaLinux. This server always transcribes with the Persian Nemotron model and ignores the `model` field in requests.

4. Dictate as usual: hold your VocaLinux dictation key, speak Persian, and release. The transcript is typed into the focused window.

> [!NOTE]
> The Remote API engine is off by default in VocaLinux and only sends audio to a server you configure. Here that server is on your own machine, so dictation stays local end to end.

### OpenWhispr (Windows, macOS, Linux) - Not tested

1. Start the server (above) and leave it running.
2. In OpenWhispr, open **Settings → Speech-to-Text / Transcription**.
3. Set the provider to **OpenAI (BYOK / Cloud)**. Not Local Whisper, not NVIDIA.
4. Set **Base URL / Endpoint** to `http://127.0.0.1:8765/v1`, **Model** to `nemotron-farsi`, and **API Key** to any dummy value such as `sk-local`. The server does not check keys, and the model field is ignored.
5. Save, then press your dictation shortcut and speak Persian.

If the client runs on a different machine from the server, `127.0.0.1` on that machine is not the server. Set `ASR_HOST=0.0.0.0` in `.env`, open port `8765` on the server's firewall, and point the client at the server's LAN address, for example `http://192.168.1.10:8765/v1`. Only do this on a trusted LAN. The server has no authentication.

Other OpenAI-compatible dictation clients work the same way: point them at `http://127.0.0.1:8765/v1` with model `nemotron-farsi` and a dummy API key. They are untested, so please report back if you try one.

## Test it without a client

Send any suitable WAV file with `curl`:

```bash
curl -X POST http://127.0.0.1:8765/v1/audio/transcriptions \
  -F "file=@test.wav"
```

```json
{"text": "سلام، این یک آزمایش است"}
```

Audio requirements are strict. This is exactly what dictation clients send:

| Property | Required value |
| --- | --- |
| Format | WAV, 16-bit PCM |
| Channels | Mono (1) |
| Sample rate | 16,000 Hz |
| Max upload size | 50 MB |

If the input audio does not meet these requirements, the API server will attempt to automatically convert it to the required format.

## Configuration

Settings come from environment variables, with the defaults below. For Docker/Podman, copy `.env.example` to `.env` and change only what you need. Do not edit `compose.yaml` or `nemotron_asr_streaming_farsi_api.py` to change them:

| Setting | Default | Meaning |
| --- | --- | --- |
| `MODEL_DIR` | *(required, no default)* | Host path to the model folder itself, set in `.env`. Mounted at `MODEL_PATH` inside the container |
| `MODEL_PATH` | `/models/nemotron-asr-streaming-farsi-onnx` | Model directory inside the container |
| `ASR_HOST` | `127.0.0.1` | Bind address (localhost only) |
| `ASR_PORT` | `8765` | HTTP port |
| `ASR_THREADS` | `8` | sherpa-onnx decoding threads. Also sets `OMP_NUM_THREADS` in the container, so the two stay in sync |
| `ASR_SAMPLE_RATE` | `16000` | Expected input sample rate |
| `ASR_LANGUAGE` | `fa-IR` | Language passed to the recognizer |
| `ASR_DECODING_METHOD` | `greedy_search` | sherpa-onnx decoding method |
| `ASR_MAX_UPLOAD_MB` | `50` | Largest audio upload the server accepts, in MB |
| `LOG_LEVEL` | `INFO` | Server log level |

If your CPU has fewer cores, lower `ASR_THREADS` in `.env`.

## API reference

### `GET /`

Health check, also used by the Compose healthcheck.

```json
{"status": "ok", "service": "Nemotron Farsi ASR", "model": "nemotron-farsi"}
```

### `POST /v1/audio/transcriptions`

OpenAI-compatible transcription endpoint. Send the audio as a multipart form field named `file`.

- Success: `200` with `{"text": "<transcript>"}`
- Failure: `400` with `{"error": "<reason>"}`, for example a wrong sample rate, stereo audio, or a missing `file` field
- Unknown paths return `404`

Under the hood, each request is decoded with a sherpa-onnx `OnlineRecognizer` (transducer, `greedy_search`, 128-dim features, language `fa-IR`). A short silence is fed before and after the audio so the streaming encoder does not clip the first or last words. The server logs the audio duration, decoding time, and real-time factor (RTF) for every request.

## Project structure

```text
nemotron-asr-streaming-farsi-api/
├── nemotron_asr_streaming_farsi_api.py   # The whole server: HTTP + sherpa-onnx decoding
├── Containerfile                         # python:3.11-slim image
├── compose.yaml                          # Hardened container setup, host networking
├── .env.example                          # All settings, with explanations and defaults
├── .env                                  # Your local copy (git-ignored, never commit)
├── .gitignore                            # Ignores .env and local model files
└── requirements.txt                      # sherpa-onnx, numpy
```

## Troubleshooting

| Symptom | Likely cause and fix |
| --- | --- |
| Container exits right away | One of `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, or `tokens.txt` is missing, or `MODEL_DIR` in `.env` is wrong. It must be the model folder itself, not its parent and not a symlink. Download all four files from [Nimaone/nemotron-asr-streaming-farsi-onnx](https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx), check the folder layout under **Get the model**, and check the logs: `docker compose logs` |
| `Expected 16000 Hz, received …` | The client sent audio at another sample rate. VocaLinux sends 16 kHz. Files tested with `curl` must be converted first |
| `Expected mono audio, received 2 channels` | The WAV is stereo. Convert it to mono before sending |
| VocaLinux / OpenWhispr reports a connection error | The server is not running, or the server URL is wrong. Test with `curl http://127.0.0.1:8765/` first |
| First or last word is cut off | Update to the current script. It already pads the audio with silence at both ends to prevent this |
| Transcription is slow | Lower `num_threads` / `OMP_NUM_THREADS` to match your CPU, and keep dictation phrases reasonably short |

## Privacy and security

- The server listens on `127.0.0.1` only and has no authentication. It is designed for your own machine. Do not expose port `8765` to a network you do not trust. If you ever run it on another host, put it on a trusted LAN or behind HTTPS.
- The container runs with a read-only filesystem, all capabilities dropped, `no-new-privileges`, and the model mounted read-only.
- Nothing is logged except the transcript line the server prints for each request, which stays in your local container logs.

## Limitations

- Persian only. The model's tokenizer and `fa-IR` tuning are Persian-specific. Other languages are not supported.
- One request at a time. Decoding is serialized behind a lock, which suits single-user dictation but not multi-user serving.
- WAV only. No MP3, WebM, resampling, or channel conversion. Clients must send 16 kHz mono WAV as-is.
- No streaming response. Despite the streaming recognizer inside, each request transcribes one complete utterance and returns the final text.
- OpenAI-compatible core only. The endpoint returns `{"text": "..."}` and ignores the `model` field. It does not return durations, segments, or other full OpenAI response fields, and it has no authentication.

## Acknowledgements

- [VocaLinux](https://github.com/VocaHQ/vocalinux) by VocaHQ, one of the dictation apps that can use this API (AGPL-3.0)
- OpenWhispr, the dictation app for Windows, macOS, and Linux that can point its BYOK mode at this API
- sherpa-onnx by the k2-fsa team, the ONNX inference runtime used for decoding
- [mehdi-hf/nemotron-asr-streaming-farsi](https://huggingface.co/mehdi-hf/nemotron-asr-streaming-farsi), the Persian Nemotron model, and its ONNX export [Nimaone/nemotron-asr-streaming-farsi-onnx](https://huggingface.co/Nimaone/nemotron-asr-streaming-farsi-onnx)
- [mallahyari/nemotron-asr-streaming-farsi](https://github.com/mallahyari/nemotron-asr-streaming-farsi), the training and evaluation code for that model
- NVIDIA Nemotron, the base streaming ASR architecture the Persian model was fine-tuned from

Model weights are distributed under their own licenses. Check the model pages above before redistributing them.
