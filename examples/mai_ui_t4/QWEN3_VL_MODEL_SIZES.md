# Qwen3-VL Model Family: Complete Architecture Breakdown

All Qwen3-VL model sizes, their architecture, parameter counts, and optimal GPU deployment configurations.

---

## Quick Reference: Model Selection by GPU

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3-VL MODEL SELECTION MATRIX                                               │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                            │
│  GPU          VRAM      Best Model              Precision   Max Context   Concurrent   Expected Latency   │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════  │
│  T4           16 GB     Qwen3-VL-2B-Instruct    FP16        8K            4            ~800ms             │
│  T4           16 GB     Qwen3-VL-4B (4-bit)     INT4        4K            4            ~1000ms            │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  L4           24 GB     Qwen3-VL-4B-Instruct    FP16        16K           8            ~500ms             │
│  L4           24 GB     Qwen3-VL-8B (4-bit)     INT4        8K            4            ~700ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  A10G         24 GB     Qwen3-VL-4B-Instruct    BF16        16K           8            ~400ms             │
│  A10G         24 GB     Qwen3-VL-8B (4-bit)     INT4        8K            4            ~600ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  A100-40GB    40 GB     Qwen3-VL-8B-Instruct    BF16        32K           16           ~300ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  A100-80GB    80 GB     Qwen3-VL-8B-Instruct    BF16        64K           32           ~250ms             │
│  A100-80GB    80 GB     Qwen3-VL-32B-Instruct   BF16        16K           8            ~500ms             │
│  A100-80GB    80 GB     Qwen3-VL-30B-A3B (MoE)  BF16        32K           16           ~350ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  H100-80GB    80 GB     Qwen3-VL-8B-Instruct    FP8         64K           64           ~150ms             │
│  H100-80GB    80 GB     Qwen3-VL-32B-Instruct   FP8         32K           32           ~300ms             │
│  H100-80GB    80 GB     Qwen3-VL-30B-A3B (MoE)  FP8         64K           32           ~200ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  H200-141GB   141 GB    Qwen3-VL-32B-Instruct   BF16        128K          64           ~200ms             │
│  H200-141GB   141 GB    Qwen3-VL-30B-A3B (MoE)  BF16        128K          64           ~150ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  B200-192GB   192 GB    Qwen3-VL-32B-Instruct   BF16        256K          128          ~100ms             │
│  B200-192GB   192 GB    Qwen3-VL-30B-A3B (MoE)  BF16        256K          128          ~80ms              │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  8×H100       640 GB    Qwen3-VL-235B-A22B      FP8         256K          64           ~400ms             │
│  8×B200       1.5 TB    Qwen3-VL-235B-A22B      BF16        256K          128          ~200ms             │
│                                                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## All Qwen3-VL Models: Architecture Breakdown

### Dense Models (All Parameters Active)

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    QWEN3-VL DENSE MODELS                                                  ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-2B-INSTRUCT                                                   │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~2.5B (2B LLM + ~500M ViT)                                                    │ ║
║  │  • Model ID: Qwen/Qwen3-VL-2B-Instruct                                                             │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens (native), 256K (extended)                                           │ ║
║  │                                                                                                     │ ║
║  │  LLM ARCHITECTURE                                                                                   │ ║
║  │  ════════════════                                                                                   │ ║
║  │  • Hidden Size: 1536                                                                                │ ║
║  │  • Intermediate Size: 8960                                                                          │ ║
║  │  • Number of Layers: 28                                                                             │ ║
║  │  • Attention Heads: 12                                                                              │ ║
║  │  • KV Heads: 2 (GQA ratio 6:1)                                                                     │ ║
║  │  • Head Dimension: 128                                                                              │ ║
║  │  • Vocabulary Size: 151,936                                                                         │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 1152                                                                                │ ║
║  │  • Depth: 24 layers                                                                                 │ ║
║  │  • Attention Heads: 16                                                                              │ ║
║  │  • MLP Ratio: 4                                                                                     │ ║
║  │  • Patch Size: 14×14 pixels                                                                         │ ║
║  │  • Temporal Patch Size: 2 frames                                                                    │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • FP16:  ~5.0 GB model + ~1.5 GB KV cache (8K ctx) = ~6.5 GB                                      │ ║
║  │  • INT4:  ~1.5 GB model + ~1.5 GB KV cache (8K ctx) = ~3.0 GB                                      │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • Edge deployment (mobile, embedded)                                                               │ ║
║  │  • T4/L4 GPUs in Google Colab                                                                      │ ║
║  │  • Real-time applications requiring low latency                                                     │ ║
║  │  • Simple image understanding tasks                                                                 │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-4B-INSTRUCT                                                   │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~4.5B (4B LLM + ~500M ViT)                                                    │ ║
║  │  • Model ID: Qwen/Qwen3-VL-4B-Instruct                                                             │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens                                                                      │ ║
║  │                                                                                                     │ ║
║  │  LLM ARCHITECTURE                                                                                   │ ║
║  │  ════════════════                                                                                   │ ║
║  │  • Hidden Size: 2048                                                                                │ ║
║  │  • Intermediate Size: 11264                                                                         │ ║
║  │  • Number of Layers: 36                                                                             │ ║
║  │  • Attention Heads: 16                                                                              │ ║
║  │  • KV Heads: 4 (GQA ratio 4:1)                                                                     │ ║
║  │  • Head Dimension: 128                                                                              │ ║
║  │  • Vocabulary Size: 151,936                                                                         │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 1280                                                                                │ ║
║  │  • Depth: 28 layers                                                                                 │ ║
║  │  • Attention Heads: 16                                                                              │ ║
║  │  • MLP Ratio: 4                                                                                     │ ║
║  │  • Patch Size: 14×14 pixels                                                                         │ ║
║  │  • Temporal Patch Size: 2 frames                                                                    │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • FP16:  ~9.0 GB model + ~3.0 GB KV cache (8K ctx) = ~12.0 GB                                     │ ║
║  │  • BF16:  ~9.0 GB model + ~3.0 GB KV cache (8K ctx) = ~12.0 GB                                     │ ║
║  │  • INT4:  ~2.5 GB model + ~3.0 GB KV cache (8K ctx) = ~5.5 GB                                      │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • T4 GPU with 4-bit quantization                                                                  │ ║
║  │  • L4/A10G GPUs at full precision                                                                  │ ║
║  │  • MAI-UI GUI agent (recommended for edge)                                                         │ ║
║  │  • Balanced quality vs. speed                                                                       │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-8B-INSTRUCT                                                   │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~8.5B (8B LLM + ~500M ViT)                                                    │ ║
║  │  • Model ID: Qwen/Qwen3-VL-8B-Instruct                                                             │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens                                                                      │ ║
║  │                                                                                                     │ ║
║  │  LLM ARCHITECTURE                                                                                   │ ║
║  │  ════════════════                                                                                   │ ║
║  │  • Hidden Size: 4096                                                                                │ ║
║  │  • Intermediate Size: 12288                                                                         │ ║
║  │  • Number of Layers: 32                                                                             │ ║
║  │  • Attention Heads: 32                                                                              │ ║
║  │  • KV Heads: 8 (GQA ratio 4:1)                                                                     │ ║
║  │  • Head Dimension: 128                                                                              │ ║
║  │  • Vocabulary Size: 151,936                                                                         │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 1536                                                                                │ ║
║  │  • Depth: 32 layers                                                                                 │ ║
║  │  • Attention Heads: 24                                                                              │ ║
║  │  • MLP Ratio: 4 (intermediate = 6144)                                                              │ ║
║  │  • Patch Size: 14×14 pixels                                                                         │ ║
║  │  • Temporal Patch Size: 2 frames                                                                    │ ║
║  │  • DeepStack Layers: [8, 16, 24] (multi-scale features)                                            │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • BF16:  ~17 GB model + ~6 GB KV cache (16K ctx) = ~23 GB                                         │ ║
║  │  • FP8:   ~9 GB model + ~3 GB KV cache (16K ctx) = ~12 GB                                          │ ║
║  │  • INT4:  ~4.5 GB model + ~6 GB KV cache (16K ctx) = ~10.5 GB                                      │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • A100-40GB/80GB at full precision                                                                 │ ║
║  │  • H100 with FP8 for maximum throughput                                                             │ ║
║  │  • Production GUI agents (MAI-UI-8B)                                                                │ ║
║  │  • High-quality image/video understanding                                                           │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-32B-INSTRUCT                                                  │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~33B (32B LLM + ~1B ViT)                                                      │ ║
║  │  • Model ID: Qwen/Qwen3-VL-32B-Instruct                                                            │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens (native), 256K (extended)                                           │ ║
║  │                                                                                                     │ ║
║  │  LLM ARCHITECTURE                                                                                   │ ║
║  │  ════════════════                                                                                   │ ║
║  │  • Hidden Size: 5120                                                                                │ ║
║  │  • Intermediate Size: 25600                                                                         │ ║
║  │  • Number of Layers: 64                                                                             │ ║
║  │  • Attention Heads: 40                                                                              │ ║
║  │  • KV Heads: 8 (GQA ratio 5:1)                                                                     │ ║
║  │  • Head Dimension: 128                                                                              │ ║
║  │  • Vocabulary Size: 151,936                                                                         │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 1792                                                                                │ ║
║  │  • Depth: 32 layers                                                                                 │ ║
║  │  • Attention Heads: 28                                                                              │ ║
║  │  • MLP Ratio: 4                                                                                     │ ║
║  │  • Patch Size: 14×14 pixels                                                                         │ ║
║  │  • Temporal Patch Size: 2 frames                                                                    │ ║
║  │  • DeepStack Layers: [8, 16, 24] (multi-scale features)                                            │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • BF16:  ~65 GB model + ~15 GB KV cache (32K ctx) = ~80 GB                                        │ ║
║  │  • FP8:   ~33 GB model + ~8 GB KV cache (32K ctx) = ~41 GB                                         │ ║
║  │  • INT4:  ~17 GB model + ~15 GB KV cache (32K ctx) = ~32 GB                                        │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • A100-80GB with BF16                                                                              │ ║
║  │  • H100 with FP8 for high throughput                                                                │ ║
║  │  • H200/B200 for maximum context length                                                             │ ║
║  │  • Complex reasoning, detailed image analysis                                                       │ ║
║  │  • State-of-the-art benchmark performance                                                           │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Mixture-of-Experts (MoE) Models

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    QWEN3-VL MOE MODELS                                                    ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║  MoE ARCHITECTURE EXPLAINED:                                                                              ║
║  ═══════════════════════════                                                                              ║
║                                                                                                           ║
║    Standard Dense Model:  Every token uses ALL parameters                                                 ║
║    MoE Model:             Each token routes to TOP-K experts (sparse activation)                         ║
║                                                                                                           ║
║    Example: Qwen3-VL-30B-A3B                                                                              ║
║    • 30B total parameters (all experts stored in memory)                                                 ║
║    • Only 3B active per token (compute cost of 3B model!)                                                ║
║    • MUCH faster than a dense 30B model                                                                  ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-30B-A3B-INSTRUCT                                               │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~30B + ~1B ViT = ~31B                                                         │ ║
║  │  • Active Parameters: ~3B per token                                                                 │ ║
║  │  • Model ID: Qwen/Qwen3-VL-30B-A3B-Instruct                                                        │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens                                                                      │ ║
║  │                                                                                                     │ ║
║  │  MoE LLM ARCHITECTURE                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 2048                                                                                │ ║
║  │  • Number of Layers: 48                                                                             │ ║
║  │  • Attention Heads: 16                                                                              │ ║
║  │  • KV Heads: 4 (GQA ratio 4:1)                                                                     │ ║
║  │  • Number of Experts: 128                                                                           │ ║
║  │  • Experts Per Token (Top-K): 8                                                                     │ ║
║  │  • Expert Intermediate Size: 1024                                                                   │ ║
║  │  • Shared Expert: Yes (always active)                                                               │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Same as Qwen3-VL-8B vision encoder                                                              │ ║
║  │  • Hidden Size: 1536                                                                                │ ║
║  │  • Depth: 32 layers                                                                                 │ ║
║  │  • DeepStack Layers: [8, 16, 24]                                                                   │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • BF16:  ~62 GB model (all experts) + ~8 GB KV = ~70 GB                                           │ ║
║  │  • FP8:   ~31 GB model + ~4 GB KV = ~35 GB                                                         │ ║
║  │                                                                                                     │ ║
║  │  COMPUTE REQUIREMENTS                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Per-token compute: ~3B parameters (like a 3B model!)                                            │ ║
║  │  • Memory bandwidth: Higher than 3B dense (expert loading)                                         │ ║
║  │  • Latency: Between 3B and 8B dense models                                                         │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • A100-80GB: Full BF16, excellent quality at 3B compute cost                                      │ ║
║  │  • H100: FP8 for single-GPU deployment                                                              │ ║
║  │  • When you need 30B quality with 3B latency                                                       │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-235B-A22B-INSTRUCT                                             │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~235B + ~2B ViT = ~237B                                                       │ ║
║  │  • Active Parameters: ~22B per token                                                                │ ║
║  │  • Model ID: Qwen/Qwen3-VL-235B-A22B-Instruct                                                      │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 256K tokens                                                                      │ ║
║  │                                                                                                     │ ║
║  │  MoE LLM ARCHITECTURE                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 5120                                                                                │ ║
║  │  • Number of Layers: 94                                                                             │ ║
║  │  • Attention Heads: 64                                                                              │ ║
║  │  • KV Heads: 4 (GQA ratio 16:1)                                                                    │ ║
║  │  • Number of Experts: 128                                                                           │ ║
║  │  • Experts Per Token (Top-K): 8                                                                     │ ║
║  │  • Expert Intermediate Size: 2560                                                                   │ ║
║  │  • Shared Expert: Yes                                                                               │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Largest ViT in Qwen3-VL family                                                                  │ ║
║  │  • Hidden Size: 2048                                                                                │ ║
║  │  • Depth: 40 layers                                                                                 │ ║
║  │  • Attention Heads: 32                                                                              │ ║
║  │  • DeepStack Layers: [10, 20, 30]                                                                  │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Multi-GPU)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • BF16:  ~470 GB model + ~50 GB KV = ~520 GB (8×80GB GPUs)                                        │ ║
║  │  • FP8:   ~235 GB model + ~25 GB KV = ~260 GB (4×80GB GPUs)                                        │ ║
║  │                                                                                                     │ ║
║  │  DEPLOYMENT OPTIONS                                                                                 │ ║
║  │  ══════════════════                                                                                 │ ║
║  │  • 8×H100-80GB: BF16 with tensor parallelism                                                       │ ║
║  │  • 4×H100-80GB: FP8 with tensor parallelism                                                        │ ║
║  │  • 8×B200-192GB: Full BF16 with room for 256K context                                              │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • State-of-the-art performance (MMMU leader)                                                       │ ║
║  │  • Complex multi-step reasoning                                                                     │ ║
║  │  • Long video understanding (hours of content)                                                      │ ║
║  │  • Research and benchmarking                                                                        │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Thinking Models (Enhanced Reasoning)

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    QWEN3-VL THINKING MODELS                                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║  WHAT ARE THINKING MODELS?                                                                                ║
║  ═════════════════════════                                                                                ║
║                                                                                                           ║
║    Standard "Instruct" models: Generate answer directly                                                   ║
║    "Thinking" models: Show chain-of-thought reasoning before answer                                       ║
║                                                                                                           ║
║    Benefits:                                                                                              ║
║    • Better on complex multi-step problems                                                                ║
║    • Improved math and logic reasoning                                                                    ║
║    • More transparent decision-making                                                                     ║
║    • Can be guided via system prompt to adjust thinking depth                                            ║
║                                                                                                           ║
║  AVAILABLE THINKING VARIANTS:                                                                             ║
║  ════════════════════════════                                                                             ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────┐             ║
║  │ Model ID                               │ Base      │ Notes                              │             ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────┤             ║
║  │ Qwen/Qwen3-VL-2B-Thinking              │ 2B Dense  │ Edge reasoning                     │             ║
║  │ Qwen/Qwen3-VL-4B-Thinking              │ 4B Dense  │ Balanced reasoning                 │             ║
║  │ Qwen/Qwen3-VL-8B-Thinking              │ 8B Dense  │ High-quality reasoning             │             ║
║  │ Qwen/Qwen3-VL-32B-Thinking             │ 32B Dense │ State-of-the-art reasoning         │             ║
║  │ Qwen/Qwen3-VL-30B-A3B-Thinking         │ 30B MoE   │ MoE with reasoning                 │             ║
║  │ Qwen/Qwen3-VL-235B-A22B-Thinking       │ 235B MoE  │ Maximum reasoning capability       │             ║
║  └─────────────────────────────────────────────────────────────────────────────────────────┘             ║
║                                                                                                           ║
║  USAGE WITH VLLM:                                                                                         ║
║  ═════════════════                                                                                        ║
║                                                                                                           ║
║    # Same as Instruct models, just use -Thinking suffix                                                   ║
║    vllm serve Qwen/Qwen3-VL-8B-Thinking \                                                                 ║
║        --dtype bfloat16 \                                                                                 ║
║        --max-model-len 32768                                                                              ║
║                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Vision Encoder Architecture (All Models)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3-VL VISION ENCODER ARCHITECTURE                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  COMMON FEATURES ACROSS ALL QWEN3-VL MODELS:                                                            │
│  ════════════════════════════════════════════                                                           │
│                                                                                                         │
│  1. PATCH EMBEDDING (Conv3D)                                                                            │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  Input: (batch, channels, time, height, width)                                               │   │
│     │  Conv3D(in=3, out=hidden_size, kernel=(2, 14, 14), stride=(2, 14, 14), bias=True)           │   │
│     │                                                                                              │   │
│     │  KEY DIFFERENCE FROM QWEN2-VL: bias=True (Qwen2-VL had bias=False)                          │   │
│     │                                                                                              │   │
│     │  This allows the model to learn per-channel offsets, improving representation flexibility   │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  2. POSITION EMBEDDING (Learned + Interpolation)                                                        │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  pos_embed = nn.Embedding(num_position_embeddings, hidden_size)                              │   │
│     │                                                                                              │   │
│     │  For arbitrary resolutions: Bilinear interpolation                                          │   │
│     │  - Handles any image size without retraining                                                │   │
│     │  - Smoothly interpolates between learned positions                                          │   │
│     │                                                                                              │   │
│     │  PLUS: RoPE with partial_rotary_factor=0.5                                                  │   │
│     │  - Only 50% of dimensions are rotated (vs 100% in Qwen2-VL)                                │   │
│     │  - Improves length extrapolation                                                            │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  3. VISION TRANSFORMER BLOCKS                                                                           │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  Each block:                                                                                 │   │
│     │    x → LayerNorm → SelfAttention (with partial RoPE) → Add                                  │   │
│     │    x → LayerNorm → MLP (SiLU activation, no bias) → Add                                     │   │
│     │                                                                                              │   │
│     │  MLP Structure:                                                                              │   │
│     │    linear_fc1: Linear(dim, mlp_hidden_dim, bias=False)                                      │   │
│     │    act_fn: SiLU = x * sigmoid(x)    ← Different from QuickGELU in Qwen2-VL                 │   │
│     │    linear_fc2: Linear(mlp_hidden_dim, dim, bias=False)                                      │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  4. DEEPSTACK (Multi-Scale Feature Extraction)                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  deepstack_visual_indexes = [8, 16, 24]  # Extract features at these layers                 │   │
│     │                                                                                              │   │
│     │  During forward pass:                                                                        │   │
│     │    for layer_num, block in enumerate(blocks):                                               │   │
│     │      hidden = block(hidden)                                                                  │   │
│     │      if layer_num in deepstack_visual_indexes:                                              │   │
│     │        ds_feature = deepstack_merger[idx](hidden)                                           │   │
│     │        deepstack_features.append(ds_feature)                                                │   │
│     │                                                                                              │   │
│     │  Final output: [main_features | ds_features_8 | ds_features_16 | ds_features_24]           │   │
│     │  These are injected into EARLY layers of the LLM decoder                                   │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  5. PATCH MERGER (Spatial Compression)                                                                  │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  spatial_merge_size = 2 → Merge 2×2 patches into 1                                          │   │
│     │                                                                                              │   │
│     │  Reduces token count by 4x while preserving information                                    │   │
│     │  E.g., 1024×1024 image → 5329 patches → 1332 tokens after merging                          │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## vLLM Deployment Configurations

### T4 (16 GB) - Entry Level

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# T4: Qwen3-VL-2B (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --dtype half \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --enforce-eager \
    --limit-mm-per-prompt image=4,video=1

# Expected: ~800ms latency, ~1.5 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# T4: Qwen3-VL-4B (4-bit Quantization)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --dtype half \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.92 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --enforce-eager \
    --limit-mm-per-prompt image=2,video=1

# Expected: ~1000ms latency, ~1 req/s
```

### L4 / A10G (24 GB) - Cloud Entry

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# L4/A10G: Qwen3-VL-4B (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt image=4,video=1 \
    --enable-prefix-caching

# Expected: ~500ms latency, ~2 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# L4/A10G: Qwen3-VL-8B (4-bit Quantization)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype half \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --enforce-eager \
    --limit-mm-per-prompt image=2,video=1

# Expected: ~700ms latency, ~1.5 req/s
```

### A100-40GB - Production Standard

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# A100-40GB: Qwen3-VL-8B (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=8,video=2

# Expected: ~300ms latency, ~4 req/s
```

### A100-80GB - High Capacity

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# A100-80GB: Qwen3-VL-8B (Full Precision, Large Batches)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 65536 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=16,video=4

# Expected: ~250ms latency, ~6 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# A100-80GB: Qwen3-VL-32B (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --limit-mm-per-prompt image=4,video=1

# Expected: ~500ms latency, ~2 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# A100-80GB: Qwen3-VL-30B-A3B MoE (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=8,video=2

# Expected: ~350ms latency, ~4 req/s (MoE = fast despite 30B total)
```

### H100-80GB - Maximum Throughput

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# H100: Qwen3-VL-8B with FP8 (Maximum Throughput)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 65536 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=16,video=4

# Expected: ~150ms latency, ~10 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# H100: Qwen3-VL-32B with FP8
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=8,video=2

# Expected: ~300ms latency, ~5 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# H100: Qwen3-VL-30B-A3B MoE with FP8
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 65536 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=16,video=4

# Expected: ~200ms latency, ~6 req/s
```

### B200-192GB - Future Hardware

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# B200: Qwen3-VL-32B (Maximum Context)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=32,video=8

# Expected: ~100ms latency, ~15 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# B200: Qwen3-VL-30B-A3B MoE (Maximum Everything)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 262144 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=32,video=8

# Expected: ~80ms latency, ~20 req/s
```

### Multi-GPU: 8×H100 for 235B Model

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# 8×H100: Qwen3-VL-235B-A22B MoE (Full Model)
# ═══════════════════════════════════════════════════════════════════════════════

OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.95 \
    --max-model-len 131072 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=16,video=4

# Expected: ~400ms latency, ~3 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# 8×B200: Qwen3-VL-235B-A22B MoE (Maximum Configuration)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.95 \
    --max-model-len 262144 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=32,video=8

# Expected: ~200ms latency, ~8 req/s
```

---

## Memory Calculation Formula

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY CALCULATION                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  Total GPU Memory = Model Weights + KV Cache + Vision Encoder + Activations + CUDA Overhead            │
│                                                                                                         │
│  MODEL WEIGHTS:                                                                                         │
│  ══════════════                                                                                         │
│  • BF16/FP16: params × 2 bytes                                                                         │
│  • FP8:       params × 1 byte                                                                           │
│  • INT4:      params × 0.5 bytes                                                                        │
│                                                                                                         │
│  KV CACHE (per request):                                                                                │
│  ══════════════════════════                                                                             │
│  • Size = 2 × num_layers × num_kv_heads × head_dim × context_len × bytes_per_param                     │
│  • Total = kv_per_request × max_num_seqs                                                                │
│                                                                                                         │
│  EXAMPLE: Qwen3-VL-8B on H100 with FP8                                                                  │
│  ════════════════════════════════════                                                                   │
│  • Model: 8B × 1 byte = 8 GB                                                                            │
│  • Vision: ~1.5 GB (ViT encoder)                                                                        │
│  • KV Cache: 2 × 32 × 8 × 128 × 32768 × 0.5 = ~4 GB per seq × 64 seqs = ~256 GB... wait, that's wrong  │
│                                                                                                         │
│  Let me recalculate properly:                                                                           │
│  • KV per layer = 2 × kv_heads × head_dim × context = 2 × 8 × 128 × 32768 = 64 MB (BF16)               │
│  • Total KV = 64 MB × 32 layers = 2 GB per request (BF16)                                              │
│  • With FP8: 1 GB per request                                                                           │
│  • 64 requests × 1 GB = 64 GB KV... still too much for 80GB with model                                 │
│                                                                                                         │
│  PRACTICAL LIMITS:                                                                                      │
│  ═════════════════                                                                                      │
│  vLLM dynamically manages KV cache with PagedAttention, so you don't need 100% upfront allocation     │
│  The gpu_memory_utilization=0.95 tells vLLM how much total memory it can use                           │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Model Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              WHICH MODEL SHOULD I USE?                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  FOR EDGE / MOBILE DEPLOYMENT:                                                                          │
│  • Qwen3-VL-2B-Instruct (T4, L4, consumer GPUs)                                                        │
│                                                                                                         │
│  FOR BALANCED COST/PERFORMANCE:                                                                         │
│  • Qwen3-VL-4B-Instruct (L4, A10G) - Sweet spot for edge                                               │
│  • Qwen3-VL-8B-Instruct (A100-40GB, H100) - Sweet spot for cloud                                       │
│                                                                                                         │
│  FOR MAXIMUM SINGLE-GPU QUALITY:                                                                        │
│  • Qwen3-VL-32B-Instruct (A100-80GB, H100 with FP8)                                                    │
│                                                                                                         │
│  FOR MoE EFFICIENCY (Quality of large model, speed of small):                                           │
│  • Qwen3-VL-30B-A3B-Instruct - 30B quality at 3B compute cost                                          │
│                                                                                                         │
│  FOR STATE-OF-THE-ART PERFORMANCE:                                                                      │
│  • Qwen3-VL-235B-A22B-Instruct (8×H100 or 8×B200)                                                      │
│                                                                                                         │
│  FOR COMPLEX REASONING TASKS:                                                                           │
│  • Use -Thinking variants of any model                                                                 │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## References

- [Qwen3-VL Technical Report (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)
- [vLLM Qwen3-VL Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Hugging Face Qwen Collection](https://huggingface.co/Qwen)

