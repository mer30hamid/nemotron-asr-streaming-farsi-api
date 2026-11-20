FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install FFmpeg for audio decoding and normalization.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY nemotron_asr_streaming_farsi_api.py .

EXPOSE 8765

CMD ["python", "-u", "nemotron_asr_streaming_farsi_api.py"]
