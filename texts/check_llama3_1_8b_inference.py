import os
# Force offline mode to stop internet checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
import torch

# Path to your local files
model_path = "/mnt/d/ModelManagers/HuggingFace/Models"

# This is might take 20 second
print("\n📦 Loading model - it takes several seconds ........")
print("\n")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 2048,
    load_in_4bit = True,
)

print("\n🛠️ Patching model...")

# Manually set the Llama 3.1 template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

FastLanguageModel.for_inference(model)

print("\n🧠 Engine ready. Thinking...")

# Ask a simple question
messages = [{"role": "user", "content": "Tell me a 1-sentence fun fact about space."}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

# STREAMING SETUP
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

print("\n✨ Generating response...\n")

_ = model.generate(
    input_ids = inputs, 
    streamer = text_streamer, 
    max_new_tokens = 64,
    use_cache = True
)

print("\n✅ Generation complete.")
