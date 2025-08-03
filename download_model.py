
import os
from huggingface_hub import hf_hub_download

# Get the TTS_HOME environment variable set in the Dockerfile
tts_home = os.environ.get("TTS_HOME")
if not tts_home:
    print("Error: TTS_HOME environment variable not set. Cannot download model.")
    exit(1)

model_repo = "coqui/XTTS-v2"
model_files = [
    "model.json",
    "vocab.json",
    "config.json",
    "dvae.pth",
    "gpt_weights.pth",
    "speaker_encoder.pth"
]

print(f"Starting XTTS-v2 model pre-download to {tts_home}...")

for file_name in model_files:
    try:
        hf_hub_download(repo_id=model_repo, filename=file_name, cache_dir=tts_home)
        print(f"Downloaded: {file_name}")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        exit(1) # Exit if any download fails

print("XTTS-v2 model pre-downloaded successfully!")