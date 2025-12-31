# Qwen VL on vLLM: GPU-Optimized Configurations

Production-ready code for running Qwen2-VL and Qwen3-VL models on NVIDIA GPUs (T4, A100, H100, B200) with vLLM.

> 📖 **For detailed documentation, see [QWEN_VL_COMPLETE_GUIDE.md](QWEN_VL_COMPLETE_GUIDE.md)**

---

## Quick Start

### GPU Selection Matrix

| GPU | VRAM | Best Model | Config |
|-----|------|------------|--------|
| **T4** | 16 GB | Qwen3-VL-4B | 4-bit, FP16, 4K context |
| **A100** | 80 GB | Qwen3-VL-8B | BF16, 32K context |
| **H100** | 80 GB | Qwen3-VL-8B | FP8, 32K context, high throughput |
| **B200** | 192 GB | Qwen3-VL-30B | BF16, 128K context |

### Installation

```bash
pip install vllm transformers
```

### Run Inference

```bash
# T4 (Google Colab Free)
python examples/mai_ui_t4/offline_inference.py \
    --model-variant 4b-4bit \
    --image screenshot.png \
    --instruction "Click the submit button"

# A100/H100
python examples/mai_ui_t4/offline_inference.py \
    --model-variant 8b \
    --image screenshot.png \
    --instruction "Describe this image"
```

### Start Server

```bash
python examples/mai_ui_t4/server.py --model-variant 4b-4bit --port 8000
```

---

## Files

| File | Description |
|------|-------------|
| `QWEN_VL_COMPLETE_GUIDE.md` | **Complete documentation** - Architecture, parameters, optimization |
| `config.py` | GPU-optimized engine configurations |
| `gpu_configs.py` | Detailed configs for T4/A100/H100/B200 |
| `offline_inference.py` | Batch inference script |
| `server.py` | OpenAI-compatible API server |
| `client.py` | GUI agent client library |
| `colab_notebook.py` | Complete Colab-ready code |

### Notebooks

| Notebook | GPU | Description |
|----------|-----|-------------|
| `mai_ui_t4_colab.ipynb` | T4 | Free Colab, 4-bit quantization |
| `h100_colab.ipynb` | H100 | High throughput, FP8 |
| `b200_colab.ipynb` | B200 | Maximum context, 4K images |

---

## Performance

```
GPU         Model            Latency     Throughput    Max Context
════════════════════════════════════════════════════════════════════
T4          Qwen3-VL-4B      ~1000ms     ~1 req/s      4K
A100-80GB   Qwen3-VL-8B      ~300ms      ~4 req/s      32K
H100        Qwen3-VL-8B+FP8  ~200ms      ~6 req/s      32K
B200        Qwen3-VL-30B     ~100ms      ~15 req/s     128K
```

---

## Architecture Comparison

| Feature | Qwen2-VL | Qwen3-VL |
|---------|----------|----------|
| Multi-Scale Features | ❌ | ✅ DeepStack |
| Video Token Pruning | ❌ | ✅ EVS |
| MoE Variants | ❌ | ✅ 30B-A3B |
| Best for GUI Tasks | Good | **Better** (DeepStack detects small UI elements) |

See [QWEN_VL_COMPLETE_GUIDE.md](QWEN_VL_COMPLETE_GUIDE.md) for detailed architecture comparison.

---

## MAI-UI GUI Agent

This codebase supports running [MAI-UI](https://arxiv.org/abs/2512.22047), a state-of-the-art GUI agent that uses Qwen3-VL as its backbone:

- **76.7%** on AndroidWorld (SOTA)
- **73.5%** on ScreenSpot-Pro
- Trained with GRPO reinforcement learning

See the complete guide for MAI-UI integration details.
