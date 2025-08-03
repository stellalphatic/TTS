
import os
from huggingface_hub import hf_hub_download

tts_home = os.environ.get("TTS_HOME")
if not tts_home:
    print("Error: TTS_HOME environment variable not set. Cannot download model.")
    exit(1)

model_repo = "coqui/XTTS-v2"
model_files = [
    "config.json",
    "dvae.pth",
    "model.pth",
    "speakers_xtts.pth",
    "vocab.json",
    "mel_stats.pth"
]

target_model_dir_suffix = "tts_models/multilingual/multi-dataset/xtts_v2"
full_cache_dir = os.path.join(tts_home, target_model_dir_suffix)

os.makedirs(full_cache_dir, exist_ok=True)

print(f"Starting XTTS-v2 model pre-download to {full_cache_dir}...")

for file_name in model_files:
    try:
        hf_hub_download(repo_id=model_repo, filename=file_name, cache_dir=full_cache_dir)
        print(f"Downloaded: {file_name} to {full_cache_dir}")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        exit(1) 

print("XTTS-v2 model pre-downloaded successfully!")

