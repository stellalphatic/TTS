
import os
from huggingface_hub import hf_hub_download

# Get the TTS_HOME environment variable set in the Dockerfile
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

base_tts_cache_dir_for_xtts = os.path.join(tts_home, "tts_models", "multilingual", "multi-dataset", "xtts_v2")

os.makedirs(base_tts_cache_dir_for_xtts, exist_ok=True)

print(f"Starting XTTS-v2 model pre-download to {base_tts_cache_dir_for_xtts}...")

for file_name in model_files:
    try:
        hf_hub_download(repo_id=model_repo, filename=file_name, cache_dir=base_tts_cache_dir_for_xtts)
        print(f"Downloaded: {file_name} to {base_tts_cache_dir_for_xtts}")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        exit(1) 

print("XTTS-v2 model pre-downloaded successfully!")

