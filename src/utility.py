from huggingface_hub import snapshot_download
import os

current_dir = os.getcwd()

model_names = ["google/gemma-4-E4B"]

for model_name in model_names:
    print(f"Downloading {model_name}...")
    snapshot_download(
        repo_id=model_name,
        local_dir="/home/ai-admin/Desktop/Vinit/Omnilens/models",
        local_dir_use_symlinks=False,  # actual files, no symlinks
        resume_download=True,          # resume if interrupted
    )
    print(f"✅ Successfully downloaded {model_name}")