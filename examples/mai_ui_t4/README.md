# MAI-UI on T4: Optimized vLLM Configuration

This directory contains production-ready code for running MAI-UI (Qwen2-VL based GUI agent) on NVIDIA T4 GPUs with vLLM.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MAI-UI + vLLM + T4 OPTIMIZATION STACK                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   Screenshot    │    │   vLLM Server   │    │      GUI Agent Client       │  │
│  │   Input         │ -> │   (Optimized)   │ -> │   (Action Execution)        │  │
│  │   1920×1080     │    │   T4 @ 16GB     │    │   pyautogui.click(x,y)      │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐  │
│  │                         T4 OPTIMIZATIONS APPLIED                           │  │
│  │  ✅ FP16 (--dtype half)           ✅ PagedAttention (automatic)           │  │
│  │  ✅ Reduced max_pixels (512K)     ✅ Continuous Batching                   │  │
│  │  ✅ Limited context (4096)        ✅ Eager mode (memory save)              │  │
│  │  ✅ 4-bit quantization (8B)       ✅ SDPA attention backend                │  │
│  └───────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Description |
|------|-------------|
| `config.py` | T4-optimized engine configurations |
| `offline_inference.py` | Batch inference script |
| `server.py` | OpenAI-compatible API server |
| `client.py` | GUI agent client library |
| `colab_notebook.py` | Complete Colab-ready code |

## Quick Start

### Option 1: Offline Inference (Batch Processing)

```bash
cd /Users/chen/Projects/vllm
python examples/mai_ui_t4/offline_inference.py \
    --model-variant 2b \
    --image screenshot.png \
    --instruction "Click the submit button"
```

### Option 2: Online Server

```bash
cd /Users/chen/Projects/vllm
python examples/mai_ui_t4/server.py --model-variant 2b --port 8000
```

### Option 3: Python API

```python
from examples.mai_ui_t4.client import MAIUIClient

client = MAIUIClient("http://localhost:8000")
action = client.get_action("screenshot.png", "Click the login button")
print(action)  # pyautogui.click(500, 300)
```

## Memory Budget (T4 - 16GB)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MAI-UI-2B (FP16):                                                          │
│  ├── Model Weights:     ~4.0 GB (25%)                                       │
│  ├── KV Cache:          ~3.0 GB (19%)                                       │
│  ├── Vision Encoder:    ~1.5 GB (9%)                                        │
│  ├── Overhead:          ~1.5 GB (9%)                                        │
│  └── FREE:              ~6.0 GB (38%) ✅ Comfortable                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  MAI-UI-8B (4-bit BitsAndBytes):                                            │
│  ├── Model Weights:     ~5.0 GB (31%)                                       │
│  ├── KV Cache:          ~4.0 GB (25%)                                       │
│  ├── Vision Encoder:    ~2.0 GB (13%)                                       │
│  ├── Overhead:          ~2.5 GB (16%)                                       │
│  └── FREE:              ~2.5 GB (15%) ⚠️ Tight                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Performance Expectations

| Configuration | TTFT | Decode Speed | E2E Latency (30 tokens) |
|--------------|------|--------------|-------------------------|
| MAI-UI-2B FP16 | ~500ms | ~50-65 tok/s | ~1.0s |
| MAI-UI-8B 4-bit | ~800ms | ~30-40 tok/s | ~1.5s |

## T4 Limitations

- ❌ No FlashAttention 2 (uses TORCH_SDPA)
- ❌ No BF16 (FP16 only)
- ❌ No FP8 quantization
- ⚠️ 320 GB/s bandwidth (memory-bound decode)

