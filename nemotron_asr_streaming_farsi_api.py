#!/usr/bin/env python3

import tempfile
import subprocess
import json
import os
import time
import wave
import logging
import threading

import numpy as np
import sherpa_onnx

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from email import policy
from email.parser import BytesParser


def _int_env(name, default):
    """Read an int from the environment, falling back to default on bad input."""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# All settings come from the environment, with the previous hard-coded
# values as defaults. Copy .env.example to .env to change them without
# editing this file.
# MODEL_PATH is the path *inside* the container (see compose.yaml).
# The host path is MODEL_DIR in .env / compose.yaml, not here.
MODEL = os.environ.get(
    "MODEL_PATH", "/models/nemotron-asr-streaming-farsi-onnx"
)

HOST = os.environ.get("ASR_HOST", "127.0.0.1")
PORT = _int_env("ASR_PORT", 8765)
SAMPLE_RATE = _int_env("ASR_SAMPLE_RATE", 16000)
NUM_THREADS = _int_env("ASR_THREADS", 8)
LANGUAGE = os.environ.get("ASR_LANGUAGE", "fa-IR")
DECODING_METHOD = os.environ.get("ASR_DECODING_METHOD", "greedy_search")
MAX_UPLOAD_BYTES = _int_env("ASR_MAX_UPLOAD_MB", 50) * 1024 * 1024
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s: %(message)s"
)

print("Loading Nemotron Farsi ONNX model...")

recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    tokens=f"{MODEL}/tokens.txt",
    encoder=f"{MODEL}/encoder.int8.onnx",
    decoder=f"{MODEL}/decoder.int8.onnx",
    joiner=f"{MODEL}/joiner.int8.onnx",
    num_threads=NUM_THREADS,
    sample_rate=SAMPLE_RATE,
    feature_dim=128,
    decoding_method=DECODING_METHOD,
)

print("Nemotron model loaded successfully.")

# Prevent simultaneous requests from decoding through the same recognizer.
recognizer_lock = threading.Lock()


def extract_audio(content_type, body):
    headers = (
        f"Content-Type: {content_type}\r\n"
        "MIME-Version: 1.0\r\n\r\n"
    ).encode("utf-8")

    message = BytesParser(policy=policy.default).parsebytes(
        headers + body
    )

    for part in message.iter_parts():
        if (
            part.get_content_disposition() == "form-data"
            and part.get_param("name", header="content-disposition") == "file"
        ):
            return part.get_payload(decode=True)

    raise ValueError("Multipart request does not contain a file field.")




def transcribe(audio_bytes):

    import io

    # ---------------------------------------------------------
    # 1. Check whether the uploaded file is already a valid
    #    WAV / PCM 16-bit / mono / 16 kHz.
    # ---------------------------------------------------------

    frames = None
    input_format = "unknown"

    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav:

            input_format = "wav"

            is_compatible = (
                wav.getcomptype() == "NONE"
                and wav.getnchannels() == 1
                and wav.getframerate() == SAMPLE_RATE
                and wav.getsampwidth() == 2
            )

            if is_compatible:
                frames = wav.readframes(wav.getnframes())
                logging.info(
                    "Input is already compatible WAV. Skipping FFmpeg."
                )

    except (wave.Error, EOFError, OSError):
        # Not a readable WAV file. FFmpeg will identify its format.
        pass

    # ---------------------------------------------------------
    # 2. Convert only when necessary.
    #    FFmpeg automatically detects the input format.
    # ---------------------------------------------------------

    if frames is None:

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False
            ) as temp_file:
                temp_path = temp_file.name

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-i", "pipe:0",
                    "-map", "0:a:0",
                    "-vn",
                    "-sn",
                    "-dn",
                    "-ac", "1",
                    "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le",
                    "-f", "wav",
                    temp_path,
                ],
                input=audio_bytes,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=120,
            )

            # Read the converted WAV from disk.
            with wave.open(temp_path, "rb") as wav:

                if (
                    wav.getcomptype() != "NONE"
                    or wav.getnchannels() != 1
                    or wav.getframerate() != SAMPLE_RATE
                    or wav.getsampwidth() != 2
                ):
                    raise ValueError(
                        "FFmpeg produced an unexpected audio format."
                    )

                frames = wav.readframes(wav.getnframes())

            logging.info(
                "Audio converted successfully: %s -> WAV PCM16 mono %d Hz",
                input_format,
                SAMPLE_RATE,
            )

        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Install FFmpeg and add it to PATH."
            )

        except subprocess.TimeoutExpired:
            raise ValueError("Audio conversion timed out.")

        except subprocess.CalledProcessError as exc:
            error = exc.stderr.decode("utf-8", errors="replace")
            raise ValueError(
                f"Unsupported or invalid audio format: {error[:1000]}"
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # ---------------------------------------------------------
    # 3. Convert PCM16 samples to float32 for Sherpa-ONNX.
    # ---------------------------------------------------------

    audio = np.frombuffer(
        frames, dtype="<i2"
    ).astype(np.float32) / 32768.0

    if len(audio) == 0:
        return ""

    # ---------------------------------------------------------
    # 4. Nemotron inference (unchanged).
    # ---------------------------------------------------------

    with recognizer_lock:

        stream = recognizer.create_stream()

        stream.set_option("language", LANGUAGE)

        # Initial silence
        stream.accept_waveform(
            SAMPLE_RATE,
            np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
        )

        stream.accept_waveform(SAMPLE_RATE, audio)

        # Tail padding
        tail_padding = np.zeros(
            int(1.0 * SAMPLE_RATE),
            dtype=np.float32
        )

        stream.accept_waveform(SAMPLE_RATE, tail_padding)

        stream.input_finished()

        start = time.perf_counter()

        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        elapsed = time.perf_counter() - start

        result = recognizer.get_result(stream)

        duration = len(audio) / SAMPLE_RATE

        logging.info(
            "Transcribed %.2fs audio in %.2fs (RTF %.3f): %s",
            duration,
            elapsed,
            elapsed / duration,
            result,
        )

        return result


class Handler(BaseHTTPRequestHandler):

    def send_json(self, status, data):
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


    def do_GET(self):
        path = self.path.split("?", 1)[0]

        # Health check
        if path == "/":
            self.send_json(200, {
                "status": "ok",
                "service": "Nemotron Farsi ASR",
                "model": "nemotron-farsi",
            })
            return

        # OpenAI-compatible model information
        if path == "/v1/models":
            self.send_json(200, {
                "object": "list",
                "data": [
                    {
                        "id": "nemotron-farsi",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }
                ],
            })
            return
        # OpenAI-compatible individual model information
        if path == "/v1/models/nemotron-farsi":
            self.send_json(200, {
                "id": "nemotron-farsi",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            })
            return

        self.send_json(404, {
            "error": {
                "message": "Not found",
                "type": "invalid_request_error",
                "code": "not_found",
            }
        })

    def do_POST(self):
        if self.path != "/v1/audio/transcriptions":
            self.send_json(404, {"error": "Unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            
            # Allow up to 1 MB for multipart headers and boundaries.
            if length <= 0 or length > MAX_UPLOAD_BYTES + 1024 * 1024:
                raise ValueError("Request size exceeds the allowed limit.")
            
            body = self.rfile.read(length)
            
            audio_bytes = extract_audio(
                self.headers.get("Content-Type", ""),
                body,
            )
            
            # Enforce the actual audio file size limit.
            if len(audio_bytes) > MAX_UPLOAD_BYTES:
                raise ValueError("Audio file exceeds the 50 MB upload limit.")
            
            text = transcribe(audio_bytes)

            self.send_json(200, {"text": text})

        except Exception as exc:
            logging.exception("Transcription request failed")
            self.send_json(400, {"error": str(exc)})

    def log_message(self, fmt, *args):
        logging.info("HTTP: " + fmt, *args)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)

    print(f"\nNemotron API listening on http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.shutdown()
