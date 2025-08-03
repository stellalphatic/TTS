
FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV TTS_HOME=/app/.tts_models

RUN mkdir -p ${TTS_HOME}

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY download_model.py .
RUN python download_model.py

COPY voice_service.py .

EXPOSE 8765

CMD ["python", "voice_service.py"]
