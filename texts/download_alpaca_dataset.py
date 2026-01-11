from datasets import load_dataset
import os

# 1. Define where you want things to live on your D: drive
DATASET_PATH = "/mnt/d/ModelManagers/Datasets/alpaca_cleaned"

print("Checking for internet and downloading dataset...")

# 2. Download from Hub
dataset = load_dataset("yahma/alpaca-cleaned", split="train")

# 3. Save to your permanent local folder
dataset.save_to_disk(DATASET_PATH)

print(f"Success! Dataset is now stored at: {DATASET_PATH}")
print("You can now run your training script 100% offline.")