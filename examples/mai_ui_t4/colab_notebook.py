#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MAI-UI on T4 - Complete Google Colab Notebook

This file can be run as a Python script or converted to a Colab notebook.
Copy the cells between the `# %%` markers into Colab.

GPU Requirements: NVIDIA T4 (16GB VRAM) - Available in free Colab tier

Usage:
    1. Open Google Colab
    2. Runtime -> Change runtime type -> GPU (T4)
    3. Copy cells from this file
    4. Run sequentially

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    MAI-UI + vLLM + T4 ARCHITECTURE                       │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Screenshot (1920×1080)                                                  │
    │       │                                                                  │
    │       ▼                                                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  IMAGE PREPROCESSING (--mm-processor-kwargs max_pixels)         │    │
    │  │  • Resize to fit max_pixels budget (512K → ~720×720)           │    │
    │  │  • Patchify into 14×14 pixel patches                            │    │
    │  │  • Result: ~2600 patches → ~650 vision tokens (after merge)    │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │       │                                                                  │
    │       ▼                                                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  QWEN2-VL VISION ENCODER                                         │    │
    │  │  • 32 transformer blocks                                         │    │
    │  │  • Attention: TORCH_SDPA (T4 doesn't support FlashAttn2)        │    │
    │  │  • Spatial merge: 2×2 patches → 1 token                         │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │       │                                                                  │
    │       ▼                                                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  LANGUAGE MODEL (2B or 8B parameters)                            │    │
    │  │  • FP16 precision (--dtype half)                                 │    │
    │  │  • PagedAttention for KV cache                                   │    │
    │  │  • Continuous batching                                           │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    │       │                                                                  │
    │       ▼                                                                  │
    │  Output: "pyautogui.click(500, 300)"                                    │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘
"""

# %% [markdown]
"""
# 🖥️ MAI-UI on T4: Optimized GUI Agent with vLLM

This notebook demonstrates running MAI-UI (a vision-language model for GUI automation)
on Google Colab's free T4 GPU using vLLM's optimized inference engine.

## What You'll Learn
1. T4 GPU architecture and its limitations
2. vLLM's memory optimization techniques (PagedAttention)
3. How to configure vLLM for optimal T4 performance
4. Running GUI grounding inference

## Requirements
- Google Colab with T4 GPU (free tier works!)
- ~15 minutes for first-time setup
"""

# %% [markdown]
"""
## 📋 Cell 1: Check GPU and Install Dependencies
"""

# %%
# Cell 1: Setup and GPU Check
import subprocess
import sys

def run_cmd(cmd):
    """Run shell command and print output."""
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode == 0

# Check GPU
print("=" * 60)
print("🔍 GPU DETECTION")
print("=" * 60)
run_cmd("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv")

# Check if we have a T4
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    compute_cap = torch.cuda.get_device_capability()
    
    print(f"\n✅ GPU: {gpu_name}")
    print(f"✅ Memory: {gpu_memory:.1f} GB")
    print(f"✅ Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    
    if "T4" in gpu_name:
        print("\n🎯 Perfect! You have a T4 GPU. This notebook is optimized for T4.")
    elif compute_cap[0] >= 8:
        print("\n✨ Great! You have an Ampere+ GPU. Even better performance expected!")
    else:
        print("\n⚠️  Note: This notebook is optimized for T4. Performance may vary.")
else:
    print("❌ No GPU detected! Please enable GPU in Runtime -> Change runtime type")
    sys.exit(1)

# Install vLLM
print("\n" + "=" * 60)
print("📦 INSTALLING DEPENDENCIES")
print("=" * 60)
run_cmd("pip install -q vllm>=0.6.0 pillow requests")
print("\n✅ Dependencies installed!")

# %% [markdown]
"""
## 🧠 Cell 2: Understanding T4 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA T4 GPU - TURING ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MEMORY: 16 GB GDDR6 @ 320 GB/s                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ████████████████████████████████████████████████████████████████████   │   │
│  │  ←────────────────────── 16 GB Total ──────────────────────────────→   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  COMPUTE: 40 SMs × 64 CUDA Cores = 2,560 CUDA Cores                            │
│           40 SMs × 8 Tensor Cores = 320 Tensor Cores (1st Gen)                  │
│                                                                                  │
│  PERFORMANCE:                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │  FP32: 8.1 TFLOPS  │  FP16: 65 TFLOPS  │  INT8: 130 TOPS             │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  T4 SUPPORTS:           T4 DOES NOT SUPPORT:                                    │
│  ✅ FP16 Tensor Cores   ❌ BF16 (requires Ampere+)                              │
│  ✅ INT8/INT4 Quant     ❌ FP8 (requires Hopper)                                │
│  ✅ PagedAttention      ❌ FlashAttention 2 (requires SM 8.0+)                  │
│  ✅ CUDA Graphs         ❌ Transformer Engine                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```
"""

# %% [markdown]
"""
## 🚀 Cell 3: Define T4-Optimized Configuration
"""

# %%
# Cell 3: T4-Optimized Configuration

# T4 Memory Budget Analysis:
# 
# MAI-UI-2B (FP16):
#   Model Weights:    ~4.0 GB (25%)
#   KV Cache:         ~3.0 GB (19%)
#   Vision Encoder:   ~1.5 GB (9%)
#   Overhead:         ~1.5 GB (9%)
#   FREE:             ~6.0 GB (38%) ✅
#
# MAI-UI-8B (4-bit BitsAndBytes):
#   Model Weights:    ~5.0 GB (31%)
#   KV Cache:         ~4.0 GB (25%)
#   Vision Encoder:   ~2.0 GB (13%)
#   Overhead:         ~2.5 GB (16%)
#   FREE:             ~2.5 GB (15%) ⚠️ Tight

# Configuration for MAI-UI-2B (Recommended for T4)
T4_CONFIG_2B = {
    "model": "osunlp/MAI-UI-2B",
    "dtype": "half",                    # FP16 for Tensor Core acceleration
    "max_model_len": 4096,              # Limit context to reduce KV cache
    "gpu_memory_utilization": 0.90,     # Reserve 10% for safety
    "enforce_eager": True,              # Disable CUDA graphs to save memory
    "max_num_seqs": 4,                  # Max concurrent requests
    "limit_mm_per_prompt": {"image": 1},  # One image per request
    "mm_processor_kwargs": {
        "min_pixels": 28 * 28,          # Minimum 28×28
        "max_pixels": 512000,           # ~720×720 max (saves memory)
    },
}

# Configuration for MAI-UI-8B with 4-bit quantization
T4_CONFIG_8B = {
    "model": "osunlp/MAI-UI-8B",
    "dtype": "half",
    "max_model_len": 2048,              # Reduced for memory
    "gpu_memory_utilization": 0.95,     # More aggressive
    "enforce_eager": True,
    "max_num_seqs": 2,                  # Fewer concurrent requests
    "quantization": "bitsandbytes",     # 4-bit quantization
    "load_format": "bitsandbytes",
    "limit_mm_per_prompt": {"image": 1},
    "mm_processor_kwargs": {
        "min_pixels": 28 * 28,
        "max_pixels": 256000,           # Smaller for memory
    },
}

# Choose configuration (default: 2B for stability)
USE_8B_MODEL = False  # Set to True for better quality (more memory)
CONFIG = T4_CONFIG_8B if USE_8B_MODEL else T4_CONFIG_2B

print("=" * 60)
print("📋 T4-OPTIMIZED CONFIGURATION")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"  {key}: {value}")
print("=" * 60)

# %% [markdown]
"""
## 🔧 Cell 4: Initialize vLLM Engine
"""

# %%
# Cell 4: Initialize vLLM Engine

from vllm import LLM, SamplingParams
import time

print("=" * 60)
print("🚀 INITIALIZING vLLM ENGINE")
print("=" * 60)
print(f"\nModel: {CONFIG['model']}")
print("This may take a few minutes on first run (downloading model)...\n")

init_start = time.time()

# Initialize LLM with T4-optimized config
llm = LLM(**CONFIG)

init_time = time.time() - init_start
print(f"\n✅ Engine initialized in {init_time:.1f} seconds")

# Print memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"\n📊 GPU Memory Usage:")
    print(f"   Allocated: {allocated:.2f} GB")
    print(f"   Reserved:  {reserved:.2f} GB")

# %% [markdown]
"""
## 📸 Cell 5: Prepare Test Image
"""

# %%
# Cell 5: Prepare Test Image

from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

def create_test_screenshot():
    """Create a simple test screenshot with UI elements."""
    # Create a window-like image
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color='#f0f0f0')
    draw = ImageDraw.Draw(img)
    
    # Draw title bar
    draw.rectangle([0, 0, width, 40], fill='#4a90d9')
    draw.text((20, 10), "Test Application", fill='white')
    
    # Draw content area
    draw.rectangle([50, 80, 750, 150], fill='white', outline='#ccc')
    draw.text((70, 105), "Username:", fill='#333')
    draw.rectangle([180, 95, 400, 130], fill='white', outline='#999')
    
    draw.rectangle([50, 170, 750, 240], fill='white', outline='#ccc')
    draw.text((70, 195), "Password:", fill='#333')
    draw.rectangle([180, 185, 400, 220], fill='white', outline='#999')
    
    # Draw login button (this is what we want the model to click)
    button_x1, button_y1 = 180, 280
    button_x2, button_y2 = 320, 330
    draw.rectangle([button_x1, button_y1, button_x2, button_y2], 
                   fill='#4CAF50', outline='#45a049')
    draw.text((220, 295), "Login", fill='white')
    
    # Draw cancel button
    draw.rectangle([340, 280, 480, 330], fill='#f44336', outline='#da190b')
    draw.text((380, 295), "Cancel", fill='white')
    
    return img

# Create test image
test_image = create_test_screenshot()
test_image.save("test_screenshot.png")

# Display the image
print("=" * 60)
print("📸 TEST SCREENSHOT")
print("=" * 60)
print("\nCreated test_screenshot.png with:")
print("  - Login form with username/password fields")
print("  - Green 'Login' button at approximately (250, 305)")
print("  - Red 'Cancel' button")
print("\nImage size:", test_image.size)

# If in Colab, display the image
try:
    from IPython.display import display
    display(test_image)
except ImportError:
    print("(Image saved to test_screenshot.png)")

# %% [markdown]
"""
## 🤖 Cell 6: Run Inference
"""

# %%
# Cell 6: Run Inference

def get_mai_ui_prompt(instruction: str) -> str:
    """Format instruction as MAI-UI prompt (Qwen2-VL ChatML format)."""
    system_prompt = (
        "You are a GUI agent. Based on the screenshot and instruction, "
        "output the action to perform. Format: pyautogui.ACTION(parameters)"
    )
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# Test instructions
test_cases = [
    "Click the Login button",
    "Click the Cancel button",
    "Click on the username input field",
]

# Sampling parameters (deterministic for reproducibility)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,
    stop=["<|im_end|>", "<|endoftext|>"],
)

print("=" * 60)
print("🤖 RUNNING INFERENCE")
print("=" * 60)

for i, instruction in enumerate(test_cases, 1):
    print(f"\n[{i}] Instruction: {instruction}")
    
    # Prepare input
    prompt = get_mai_ui_prompt(instruction)
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": test_image},
    }
    
    # Run inference
    start_time = time.time()
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    latency = time.time() - start_time
    
    # Extract result
    result = outputs[0].outputs[0].text.strip()
    tokens = len(outputs[0].outputs[0].token_ids)
    
    print(f"    → Action: {result}")
    print(f"    → Latency: {latency*1000:.0f}ms | Tokens: {tokens}")

print("\n" + "=" * 60)

# %% [markdown]
"""
## 📊 Cell 7: Benchmark Performance
"""

# %%
# Cell 7: Benchmark Performance

import statistics

def benchmark_inference(llm, image, instruction, num_runs=5):
    """Benchmark inference performance."""
    prompt = get_mai_ui_prompt(instruction)
    inputs = {"prompt": prompt, "multi_modal_data": {"image": image}}
    
    latencies = []
    for i in range(num_runs):
        start = time.time()
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        latencies.append(time.time() - start)
    
    return {
        "mean_ms": statistics.mean(latencies) * 1000,
        "std_ms": statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
        "min_ms": min(latencies) * 1000,
        "max_ms": max(latencies) * 1000,
        "tokens": len(outputs[0].outputs[0].token_ids),
    }

print("=" * 60)
print("📊 PERFORMANCE BENCHMARK")
print("=" * 60)
print("\nRunning 5 iterations for warm benchmark...\n")

results = benchmark_inference(llm, test_image, "Click the Login button", num_runs=5)

print(f"┌─────────────────────────────────────────────────────────┐")
print(f"│  Metric              │  Value                          │")
print(f"├─────────────────────────────────────────────────────────┤")
print(f"│  Mean Latency        │  {results['mean_ms']:>6.1f} ms                      │")
print(f"│  Std Dev             │  {results['std_ms']:>6.1f} ms                      │")
print(f"│  Min Latency         │  {results['min_ms']:>6.1f} ms                      │")
print(f"│  Max Latency         │  {results['max_ms']:>6.1f} ms                      │")
print(f"│  Tokens Generated    │  {results['tokens']:>6}                          │")
print(f"│  Tokens/sec (decode) │  {results['tokens'] / (results['mean_ms']/1000):>6.1f}                          │")
print(f"└─────────────────────────────────────────────────────────┘")

# Memory stats
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"\n📊 GPU Memory Status:")
    print(f"   Allocated: {allocated:.2f} GB / {total:.1f} GB ({allocated/total*100:.1f}%)")
    print(f"   Reserved:  {reserved:.2f} GB / {total:.1f} GB ({reserved/total*100:.1f}%)")

# %% [markdown]
"""
## 🎯 Cell 8: Batch Inference Example
"""

# %%
# Cell 8: Batch Inference Example

# Create multiple test images with different content
def create_varied_screenshots():
    screenshots = []
    
    # Screenshot 1: Login form
    screenshots.append({
        "image": test_image,
        "instruction": "Click the Login button",
    })
    
    # Screenshot 2: Different layout
    img2 = Image.new('RGB', (800, 600), color='#ffffff')
    draw2 = ImageDraw.Draw(img2)
    draw2.rectangle([100, 200, 300, 250], fill='#2196F3')
    draw2.text((160, 215), "Submit", fill='white')
    draw2.rectangle([350, 200, 550, 250], fill='#9E9E9E')
    draw2.text((420, 215), "Reset", fill='white')
    screenshots.append({
        "image": img2,
        "instruction": "Click the Submit button",
    })
    
    # Screenshot 3: Search box
    img3 = Image.new('RGB', (800, 600), color='#ffffff')
    draw3 = ImageDraw.Draw(img3)
    draw3.rectangle([100, 100, 600, 140], fill='white', outline='#999')
    draw3.text((120, 110), "Search...", fill='#999')
    draw3.rectangle([610, 100, 700, 140], fill='#4285F4')
    draw3.text((630, 110), "🔍", fill='white')
    screenshots.append({
        "image": img3,
        "instruction": "Click the search input box",
    })
    
    return screenshots

# Run batch inference
batch = create_varied_screenshots()

print("=" * 60)
print("🎯 BATCH INFERENCE (vLLM Continuous Batching)")
print("=" * 60)

# Prepare batch inputs
batch_inputs = []
for item in batch:
    prompt = get_mai_ui_prompt(item["instruction"])
    batch_inputs.append({
        "prompt": prompt,
        "multi_modal_data": {"image": item["image"]},
    })

# Run batch inference
batch_start = time.time()
batch_outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
batch_time = time.time() - batch_start

print(f"\nProcessed {len(batch)} requests in {batch_time:.2f}s")
print(f"Throughput: {len(batch) / batch_time:.2f} requests/second\n")

for i, (item, output) in enumerate(zip(batch, batch_outputs), 1):
    result = output.outputs[0].text.strip()
    print(f"[{i}] {item['instruction']}")
    print(f"    → {result}\n")

# %% [markdown]
"""
## 📝 Summary

### T4 Optimization Checklist

| Optimization | Flag/Setting | Impact |
|-------------|--------------|--------|
| FP16 Precision | `--dtype half` | 50% memory reduction, Tensor Core acceleration |
| Limit Context | `--max-model-len 4096` | Reduces KV cache size |
| Eager Mode | `--enforce-eager` | Saves ~500MB (disables CUDA graphs) |
| Image Resize | `max_pixels: 512000` | Fewer vision tokens, faster processing |
| Memory Utilization | `gpu_memory_utilization: 0.90` | Safety margin for OOM prevention |
| Quantization (8B) | `bitsandbytes` | 75% weight compression |

### Expected Performance on T4

| Configuration | TTFT | Decode Speed | E2E Latency |
|--------------|------|--------------|-------------|
| MAI-UI-2B FP16 | ~500ms | ~50-65 tok/s | ~1.0s |
| MAI-UI-8B 4-bit | ~800ms | ~30-40 tok/s | ~1.5s |

### T4 Limitations
- ❌ No FlashAttention 2 (uses TORCH_SDPA)
- ❌ No BF16 support (FP16 only)
- ❌ No FP8 quantization
- ⚠️ 320 GB/s bandwidth (decode is memory-bound)
"""

# %%
print("\n" + "=" * 60)
print("✅ NOTEBOOK COMPLETE")
print("=" * 60)
print("""
🎉 You have successfully run MAI-UI on T4 with vLLM!

Next Steps:
1. Try with your own screenshots
2. Integrate with pyautogui for actual GUI automation
3. Deploy as an API server using vllm.entrypoints.openai.api_server

For more examples, see:
  https://github.com/vllm-project/vllm/tree/main/examples/mai_ui_t4
""")

