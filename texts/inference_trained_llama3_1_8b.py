import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
import glob

# 1. Configuration & Paths
results_path = "sft_llama3_1_8b_results"

# 2. Find and choose training runs
# This looks for all folders starting with 'sft_llama_' and sorts by newest first
all_runs = sorted(glob.glob(os.path.join(results_path, "sft_llama_*")), key=os.path.getmtime, reverse=True)

if not all_runs:
    print(f"\n❌ No training results found in: {results_path}")
    exit()

print("\n📂 Available Training Runs:")
for i, run in enumerate(all_runs):
    is_latest = "(LATEST)" if i == 0 else ""
    print(f"[{i}] {os.path.basename(run)} {is_latest}")

choice = input(f"\nSelect a run number [Default 0]: ")
selected_index = int(choice) if choice.strip().isdigit() and int(choice) < len(all_runs) else 0
save_folder = all_runs[selected_index]

# 3. Load Model & Tokenizer
# We point to the save_folder; Unsloth automatically joins it with the base model
print(f"\n⏳ Loading trained model from: {os.path.basename(save_folder)} ........\n")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = save_folder,
    max_seq_length = 2048, # Must match training
    load_in_4bit = True,   # default = True
)

# 4. Define the Alpaca Prompt Template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 5. Switch to Inference Mode
print("\n🔥 Model Loaded. Switching to Inference Mode...")
FastLanguageModel.for_inference(model) # This prepares the 'warm' model for talking

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

# UI Colors for the Chat
C_USER = "\033[96m"  # Cyan
C_AI = "\033[92m"    # Green
C_RESET = "\033[0m"
C_BOLD = "\033[1m"

print(f"\n🚀 Chat with your NEW model! (Type 'exit' to finish)")
while True:
    user_prompt = input(f"\n{C_BOLD}{C_USER}👨‍💻 User:{C_RESET} ")
    if user_prompt.lower() in ["exit", "quit"]:
        break

    # Format using your Alpaca template
    # We leave 'input' empty here just like in your training immediate inference
    full_prompt = alpaca_prompt.format(user_prompt, "", "")
    inputs = tokenizer([full_prompt], return_tensors = "pt").to("cuda")

    print(f"{C_BOLD}{C_AI}🤖 AI:{C_RESET} ", end="")
    _ = model.generate(
        **inputs, 
        streamer = text_streamer, 
        max_new_tokens = 256,
        use_cache = True # Makes generation faster
    )