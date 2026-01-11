import os
from huggingface_hub import snapshot_download

# This is your D: drive path in Linux
target_path = "/mnt/d/ModelManagers/HuggingFace/Models"

print(f"🚀 Starting full download to: {target_path}")

try:
    # This grabs all 5.5GB of the model files
    snapshot_download(
        repo_id="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        local_dir=target_path,
        local_dir_use_symlinks=False, # Keeps files as real files, not links
        token=True                    # Uses your saved login
    )
    print("\n✅ SUCCESS: Llama 3.1 8B is safely on your D: Drive.")
except Exception as e:
    print(f"\n❌ ERROR: {e}")