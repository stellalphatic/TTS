
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

# --- CRITICAL CHANGE: Pre-download the XTTS-v2 model non-interactively ---
# Use huggingface_hub to download the model directly into the TTS_HOME directory.
# This bypasses the interactive prompt.
RUN python -c "from huggingface_hub import hf_hub_download; \
import os; \
model_repo = 'coqui/XTTS-v2'; \
model_files = [ \
    'model.json', 'vocab.json', 'config.json', \
    'dvae.pth', 'gpt_weights.pth', 'speaker_encoder.pth' \
]; \
for file_name in model_files: \
    hf_hub_download(repo_id=model_repo, filename=file_name, cache_dir=os.environ.get('TTS_HOME')); \
print('XTTS-v2 model pre-downloaded successfully!');"

COPY voice_service.py .

EXPOSE 8765

CMD ["python", "voice_service.py"]
