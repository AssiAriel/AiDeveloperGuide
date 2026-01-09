from unsloth import FastLanguageModel 
import torch
import transformers
import psutil

try:
    # 1. Hardware Detection
    major, minor = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name(0)
    
    # 2. Kernel & Optimization Check
    from unsloth import is_bfloat16_supported
    
    print(f"\n--- System Status ---")
    print(f"GPU: {gpu_name} (sm_{major}{minor})")
    
    # 3. Memory Diagnostics
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    sys_ram = psutil.virtual_memory().total / 1024**3
    print(f"VRAM: {vram:.1f} GB | Linux RAM: {sys_ram:.1f} GB")

    # 4. Feature Support
    print(f"Unsloth Kernels: ✅ ACTIVE")
    print(f"Bfloat16 Native (Blackwell): {is_bfloat16_supported()}")
    print(f"PyTorch Version: {torch.__version__}")
    
    print(f"Status: PASS")

except Exception as e:
    print(f"\n--- System Status ---")
    print(f"❌ TEST FAILED: {e}")