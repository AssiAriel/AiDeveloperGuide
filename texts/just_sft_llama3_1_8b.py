import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

# 1. Configuration & Paths
model_path = "/mnt/d/ModelManagers/HuggingFace/Models"
dataset_path = "/mnt/d/ModelManagers/Datasets/alpaca_cleaned"

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048, # default = 2048 (higher uses more VRAM)
    load_in_4bit = True,   # default = True
)

# 3. Add LoRA Adapters - The "Brain Surgery" variables
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,               # default = 16 (Higher = more capacity, but bigger files)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,      # default = 16 (Usually matches 'r')
    lora_dropout = 0,     # default = 0 (Good for stability)
    bias = "none",        # default = "none"
    use_gradient_checkpointing = "unsloth", # default = "unsloth"
)

# 4. Define the Alpaca Prompt Template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# 5. Load and Map the Dataset
dataset = load_from_disk(dataset_path)
dataset = dataset.map(formatting_prompts_func, batched = True)

# 6. Initialize Trainer - The "Exercise" variables
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048, # default = 2048
    args = TrainingArguments(
        per_device_train_batch_size = 2,  # default = 2 (Higher = more VRAM used)
        gradient_accumulation_steps = 4,  # default = 4 (Total Batch = size * steps)
        warmup_steps = 5,                 # default = 5 (Initial 'slow start' steps)
        max_steps = 60,                   # default = 60 (Total training duration)
        learning_rate = 2e-4,             # default = 2e-4 (How fast it learns)
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,                # default = 1 (How often to print Loss)
        optim = "adamw_8bit",             # default = "adamw_8bit"
        weight_decay = 0.01,              # default = 0.01 (Prevents over-learning)
        lr_scheduler_type = "linear",     # default = "linear"
        seed = 3407,                      # default = 3407 (For repeatable results)
        output_dir = "outputs",           # default = "outputs"
    ),
)

# 7. Train!
print("\n🚀 Starting Training on RTX 5070 Ti...")
trainer.train()

# 8. Save for later play
# model.save_pretrained_lora("lora_model") # default = commented out