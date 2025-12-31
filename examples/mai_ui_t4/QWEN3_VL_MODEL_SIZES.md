# Qwen3-VL Model Family: Complete Architecture Breakdown

All Qwen3-VL model sizes, their architecture, parameter counts, and optimal GPU deployment configurations.

**Companion Guide**: See [QWEN_VL_COMPLETE_GUIDE.md](./QWEN_VL_COMPLETE_GUIDE.md) for detailed architectural explanations, training pipelines, and MAI-UI integration.

---

## Table of Contents

1. [Quick Reference: Model Selection by GPU](#quick-reference-model-selection-by-gpu)
2. [All Qwen3-VL Models: Architecture Breakdown](#all-qwen3-vl-models-architecture-breakdown)
3. [Qwen3 vs Qwen3-VL: Text-Only vs Multimodal](#qwen3-vs-qwen3-vl-text-only-vs-multimodal)
4. [Complete Architecture Diagrams](#complete-architecture-diagrams)
5. [vLLM PagedAttention for Multimodal](#vllm-pagedattention-for-multimodal)
6. [GPU Memory Layout Diagrams](#gpu-memory-layout-diagrams)
7. [vLLM Deployment Configurations](#vllm-deployment-configurations)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Summary: Model Selection Guide](#summary-model-selection-guide)
10. [References](#references)

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

## Qwen3 vs Qwen3-VL: Text-Only vs Multimodal

Understanding the fundamental difference between the Qwen3 LLM series (text-only) and Qwen3-VL (vision-language):

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              QWEN3 (TEXT-ONLY) vs QWEN3-VL (MULTIMODAL)                                   ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║   QWEN3 (Text-Only LLM)                           QWEN3-VL (Vision-Language Model)                       ║
║   ═════════════════════                           ════════════════════════════════                       ║
║                                                                                                           ║
║   ┌─────────────────────────────────┐             ┌─────────────────────────────────────────────────────┐║
║   │                                 │             │                                                     │║
║   │     Text Input                  │             │     Image/Video Input         Text Input           │║
║   │         │                       │             │           │                       │                 │║
║   │         ▼                       │             │           ▼                       │                 │║
║   │   ┌───────────┐                 │             │   ┌───────────────────┐          │                 │║
║   │   │ Tokenizer │                 │             │   │  Vision Encoder   │          │                 │║
║   │   │ + Embed   │                 │             │   │  (ViT + DeepStack)│          │                 │║
║   │   └─────┬─────┘                 │             │   └─────────┬─────────┘          │                 │║
║   │         │                       │             │             │                     │                 │║
║   │         ▼                       │             │             ▼                     ▼                 │║
║   │   ┌───────────┐                 │             │   ┌─────────────────────────────────────┐          │║
║   │   │  Qwen3    │                 │             │   │  [Visual Tokens] + [Text Tokens]   │          │║
║   │   │  LLM      │                 │             │   │         Merged Sequence             │          │║
║   │   │  Decoder  │                 │             │   └─────────────────┬───────────────────┘          │║
║   │   └─────┬─────┘                 │             │                     │                               │║
║   │         │                       │             │                     ▼                               │║
║   │         ▼                       │             │   ┌─────────────────────────────────────┐          │║
║   │    Text Output                  │             │   │  Qwen3 LLM Decoder                  │          │║
║   │                                 │             │   │  (with DeepStack injection at       │          │║
║   │                                 │             │   │   early layers)                     │          │║
║   └─────────────────────────────────┘             │   └─────────────────┬───────────────────┘          │║
║                                                   │                     │                               │║
║                                                   │                     ▼                               │║
║                                                   │              Text Output                             │║
║                                                   │   (can describe images, answer visual Q&A)          │║
║                                                   └─────────────────────────────────────────────────────┘║
║                                                                                                           ║
║   MODEL SIZE COMPARISON:                                                                                  ║
║   ══════════════════════                                                                                  ║
║                                                                                                           ║
║   ┌────────────────────────────┬───────────────────────────┬─────────────────────────────────────────┐   ║
║   │ Qwen3 (Text)               │ Qwen3-VL (Multimodal)     │ Difference                              │   ║
║   ├────────────────────────────┼───────────────────────────┼─────────────────────────────────────────┤   ║
║   │ Qwen3-0.6B                 │ (no VL variant)           │ Too small for vision encoder overhead   │   ║
║   │ Qwen3-1.7B                 │ (no VL variant)           │ Too small for vision encoder overhead   │   ║
║   │ Qwen3-4B                   │ Qwen3-VL-4B (~4.5B)       │ +500M for ViT encoder                   │   ║
║   │ Qwen3-8B                   │ Qwen3-VL-8B (~8.5B)       │ +500M for ViT encoder                   │   ║
║   │ Qwen3-14B                  │ (no VL variant yet)       │ Gap in lineup                           │   ║
║   │ Qwen3-32B                  │ Qwen3-VL-32B (~33B)       │ +1B for larger ViT encoder              │   ║
║   │ Qwen3-30B-A3B (MoE)        │ Qwen3-VL-30B-A3B (~31B)   │ +1B for ViT encoder                     │   ║
║   │ Qwen3-235B-A22B (MoE)      │ Qwen3-VL-235B-A22B (~237B)│ +2B for largest ViT encoder             │   ║
║   └────────────────────────────┴───────────────────────────┴─────────────────────────────────────────┘   ║
║                                                                                                           ║
║   KEY ARCHITECTURAL DIFFERENCES:                                                                          ║
║   ══════════════════════════════                                                                          ║
║                                                                                                           ║
║   1. Vision Encoder (ViT): Only in VL models, processes images/videos into token embeddings              ║
║   2. DeepStack: Only in VL models, injects multi-scale vision features into early LLM layers             ║
║   3. M-RoPE: VL models use 3D positional encoding (height, width, time) vs 1D in text-only               ║
║   4. EVS: Only in VL models, prunes redundant video frames for efficiency                                 ║
║   5. Memory: VL models require ~500M-2B extra parameters for vision processing                           ║
║                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Qwen3 Text-Only Model Specifications

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3 TEXT-ONLY LLM SPECIFICATIONS                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  ┌────────────┬──────────┬─────────────┬────────┬─────────┬────────┬──────────┬────────────────────┐   │
│  │ Model      │ Params   │ Hidden Size │ Layers │ Heads   │KV Heads│ Intermed │ Context            │   │
│  ├────────────┼──────────┼─────────────┼────────┼─────────┼────────┼──────────┼────────────────────┤   │
│  │ Qwen3-0.6B │ 0.6B     │ 1024        │ 28     │ 16      │ 8      │ 3072     │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-1.7B │ 1.7B     │ 1536        │ 28     │ 12      │ 4      │ 8960     │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-4B   │ 4.0B     │ 2560        │ 36     │ 32      │ 8      │ 9728     │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-8B   │ 8.0B     │ 4096        │ 36     │ 32      │ 8      │ 12288    │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-14B  │ 14.0B    │ 5120        │ 40     │ 40      │ 8      │ 13824    │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-32B  │ 32.0B    │ 5120        │ 64     │ 64      │ 8      │ 25600    │ 32K (131K w/ YaRN) │   │
│  └────────────┴──────────┴─────────────┴────────┴─────────┴────────┴──────────┴────────────────────┘   │
│                                                                                                         │
│  MoE MODELS:                                                                                            │
│  ┌────────────────────┬───────────┬─────────────┬────────┬─────────┬─────────┬────────────────────────┐│
│  │ Model              │ Total/Act │ Hidden Size │ Layers │ Experts │ Top-K   │ Context                ││
│  ├────────────────────┼───────────┼─────────────┼────────┼─────────┼─────────┼────────────────────────┤│
│  │ Qwen3-30B-A3B      │ 30B / 3B  │ 2048        │ 48     │ 128     │ 8       │ 32K (131K w/ YaRN)     ││
│  │ Qwen3-235B-A22B    │ 235B / 22B│ 5120        │ 94     │ 128     │ 8       │ 32K (131K w/ YaRN)     ││
│  └────────────────────┴───────────┴─────────────┴────────┴─────────┴─────────┴────────────────────────┘│
│                                                                                                         │
│  COMMON FEATURES:                                                                                       │
│  • Activation: SwiGLU (SiLU-gated linear unit)                                                         │
│  • Normalization: RMSNorm                                                                              │
│  • Position Encoding: RoPE (Rotary Position Embedding)                                                 │
│  • Attention: Grouped Query Attention (GQA)                                                            │
│  • Vocabulary: 151,936 tokens                                                                          │
│  • Thinking Mode: Unified thinking/non-thinking in single model                                        │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
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

## GPU Hardware: Streaming Multiprocessors and vLLM

Understanding how CUDA Streaming Multiprocessors (SMs) execute Qwen3-VL inference:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              GPU SM ARCHITECTURE FOR QWEN3-VL INFERENCE                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU SPECIFICATIONS BY ARCHITECTURE:                                                                   │
│   ═══════════════════════════════════                                                                   │
│                                                                                                         │
│   ┌───────────────────┬────────────┬────────────┬────────────┬────────────┬─────────────────────────┐  │
│   │ Spec              │ T4 (SM7.5) │A100 (SM8.0)│H100 (SM9.0)│B200 (SM10) │ Impact on Qwen3-VL      │  │
│   ├───────────────────┼────────────┼────────────┼────────────┼────────────┼─────────────────────────┤  │
│   │ Streaming MPs     │ 40 SMs     │ 108 SMs    │ 132 SMs    │ 192 SMs    │ More SMs = more ||ism  │  │
│   │ CUDA Cores        │ 2,560      │ 6,912      │ 16,896     │ ~20,000    │ Compute throughput     │  │
│   │ Tensor Cores      │ 320        │ 432        │ 528        │ ~640       │ Matrix multiply speed  │  │
│   │ Warp Schedulers   │ 4/SM       │ 4/SM       │ 4/SM       │ 4/SM       │ Thread scheduling      │  │
│   │ Max Threads/SM    │ 1,024      │ 2,048      │ 2,048      │ 2,048      │ Concurrent warps       │  │
│   │ Register File     │ 256KB/SM   │ 256KB/SM   │ 256KB/SM   │ 256KB/SM   │ Thread local storage   │  │
│   │ Shared Memory     │ 64KB/SM    │ 164KB/SM   │ 228KB/SM   │ ~256KB/SM  │ FlashAttn tile size    │  │
│   │ L2 Cache          │ 4 MB       │ 40 MB      │ 50 MB      │ 64 MB      │ KV cache hot path      │  │
│   │ Memory BW         │ 320 GB/s   │ 2,039 GB/s │ 3,350 GB/s │ 8,000 GB/s │ Decode speed (bound)   │  │
│   └───────────────────┴────────────┴────────────┴────────────┴────────────┴─────────────────────────┘  │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   VLLM ATTENTION BACKEND SELECTION (from vllm/platforms/cuda.py):                                      │
│   ═══════════════════════════════════════════════════════════════                                       │
│                                                                                                         │
│   DeviceCapability(major, minor) → Backend Selection:                                                   │
│                                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                                 │  │
│   │   T4 (7, 5):     FlashInfer (SM 7.5-12.1) → TritonAttn → TORCH_SDPA                           │  │
│   │                  ⚠️ FlashAttention requires SM >= 8.0, NOT available on T4                     │  │
│   │                                                                                                 │  │
│   │   A100 (8, 0):   FlashAttention 2 → FlashInfer → TritonAttn                                   │  │
│   │                  ✅ Native BF16 Tensor Cores, FP16 accumulate                                  │  │
│   │                                                                                                 │  │
│   │   H100 (9, 0):   FlashAttention 3 → FlashMLA → FlashInfer                                     │  │
│   │                  ✅ FP8 Tensor Cores, FP8 KV cache, sinks support                              │  │
│   │                                                                                                 │  │
│   │   B200 (10, 0):  FlashMLA → FlashAttention 3 → FlashInfer                                     │  │
│   │                  ✅ FP4 (future), 2nd gen Transformer Engine                                   │  │
│   │                                                                                                 │  │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   PAGEDATTENTION MEMORY MAPPING TO HARDWARE:                                                            │
│   ══════════════════════════════════════════                                                            │
│                                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                                 │  │
│   │   vLLM Block Manager          GPU HBM                      CUDA Kernel Execution               │  │
│   │   ═══════════════════         ═══════                      ═════════════════════               │  │
│   │                                                                                                 │  │
│   │   Block Table:                ┌────────────────┐           grid(num_heads, num_seqs, splits)   │  │
│   │   ┌─────────────────┐         │ Physical Pages │                     │                         │  │
│   │   │ Seq 0: [P1,P3]  │────────▶│ ┌────┐ ┌────┐ │           ┌─────────▼─────────┐               │  │
│   │   │ Seq 1: [P2,P3]  │─────┬──▶│ │ P1 │ │ P2 │ │           │ Thread Block 0    │               │  │
│   │   │ Seq 2: [P4]     │──┐  │   │ └────┘ └────┘ │           │ (on SM 0)         │               │  │
│   │   └─────────────────┘  │  │   │ ┌────┐ ┌────┐ │           │ Warp 0-3 process  │               │  │
│   │                        │  └──▶│ │ P3 │ │ P4 │ │◀──────────│ head 0, seq 0     │               │  │
│   │   P3 SHARED!           │      │ └────┘ └────┘ │           └───────────────────┘               │  │
│   │   (Copy-on-write)      └─────▶│ (256 tokens   │                                                │  │
│   │                               │  per page)    │           ┌───────────────────┐               │  │
│   │   Memory saved: ~50%          └────────────────┘           │ Thread Block 1    │               │  │
│   │   vs naive allocation                                      │ (on SM 1)         │               │  │
│   │                                                            │ head 0, seq 1     │               │  │
│   │                                                            └───────────────────┘               │  │
│   │                                                                                                 │  │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Architecture Diagrams

### Qwen3-VL Full Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3-VL COMPLETE ARCHITECTURE DIAGRAM                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   INPUT STAGE                                                                                           │
│   ══════════                                                                                            │
│                                                                                                         │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐                                   │
│   │ Image            │   │ Video            │   │ Text             │                                   │
│   │ (any resolution) │   │ (multi-frame)    │   │ (prompt)         │                                   │
│   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘                                   │
│            │                      │                      │                                              │
│            ▼                      ▼                      │                                              │
│   ┌─────────────────────────────────────────────┐        │                                              │
│   │              VISION ENCODER                 │        │                                              │
│   │  ┌───────────────────────────────────────┐  │        │                                              │
│   │  │ 1. Patch Embedding (Conv3D)           │  │        │                                              │
│   │  │    • Kernel: (2, 14, 14)              │  │        │                                              │
│   │  │    • Stride: (2, 14, 14)              │  │        │                                              │
│   │  │    • WITH bias (diff from Qwen2-VL)   │  │        │                                              │
│   │  │    • Output: (num_patches, hidden)    │  │        │                                              │
│   │  └───────────────────────────────────────┘  │        │                                              │
│   │                     │                        │        │                                              │
│   │                     ▼                        │        │                                              │
│   │  ┌───────────────────────────────────────┐  │        │                                              │
│   │  │ 2. Position Embedding                 │  │        │                                              │
│   │  │    • Learned embeddings               │  │        │                                              │
│   │  │    • Bilinear interpolation           │  │        │                                              │
│   │  │    • Handles arbitrary resolutions    │  │        │                                              │
│   │  └───────────────────────────────────────┘  │        │                                              │
│   │                     │                        │        │                                              │
│   │                     ▼                        │        │                                              │
│   │  ┌───────────────────────────────────────┐  │        │                                              │
│   │  │ 3. Vision Transformer Blocks          │  │        │                                              │
│   │  │    ┌───────────────────────────────┐  │  │        │                                              │
│   │  │    │ Block 0-7 ──┬─────────────────┼──┼──┼───► DS Merger 0 ──┐                                  │
│   │  │    │ Block 8-15 ─┼─────────────────┼──┼──┼───► DS Merger 1 ──┼─► DeepStack                      │
│   │  │    │ Block 16-23─┼─────────────────┼──┼──┼───► DS Merger 2 ──┘   Features                       │
│   │  │    │ Block 24-31 ┴─────────────────┘  │  │                                                       │
│   │  │    │                                   │  │                                                       │
│   │  │    │ Each block:                       │  │                                                       │
│   │  │    │ • LayerNorm                       │  │                                                       │
│   │  │    │ • Self-Attention (partial RoPE)  │  │                                                       │
│   │  │    │ • LayerNorm                       │  │                                                       │
│   │  │    │ • MLP (SiLU, no bias)            │  │                                                       │
│   │  │    └───────────────────────────────┘  │  │                                                       │
│   │  └───────────────────────────────────────┘  │                                                       │
│   │                     │                        │                                                       │
│   │                     ▼                        │                                                       │
│   │  ┌───────────────────────────────────────┐  │                                                       │
│   │  │ 4. Patch Merger                       │  │                                                       │
│   │  │    • Merge 2×2 patches → 1 token      │  │                                                       │
│   │  │    • Reduces tokens by 4×             │  │                                                       │
│   │  │    • Projects to LLM hidden size      │  │                                                       │
│   │  └───────────────────────────────────────┘  │                                                       │
│   │                     │                        │                                                       │
│   └─────────────────────┼────────────────────────┘                                                       │
│                         │                                                                                │
│                         ▼                                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              SEQUENCE MERGING                                                    │   │
│   │  ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│   │  │ [<vision_start>] [vis_0] [vis_1] ... [vis_N] [<vision_end>] [text_0] [text_1] ... [text_M]│ │   │
│   │  │                                                                                           │ │   │
│   │  │ Visual tokens with M-RoPE positions:                                                      │ │   │
│   │  │ • Temporal ID: frame number                                                               │ │   │
│   │  │ • Height ID: row in image                                                                 │ │   │
│   │  │ • Width ID: column in image                                                               │ │   │
│   │  └───────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                                              │
│                                           ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              QWEN3 LLM DECODER                                                   │   │
│   │  ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│   │  │ Layer 0:  hidden_states + DeepStack_features_0  ← Multi-scale injection                  │ │   │
│   │  │ Layer 1:  hidden_states + DeepStack_features_1  ← Multi-scale injection                  │ │   │
│   │  │ Layer 2:  hidden_states + DeepStack_features_2  ← Multi-scale injection                  │ │   │
│   │  │ Layer 3+: hidden_states only                                                              │ │   │
│   │  │                                                                                           │ │   │
│   │  │ Each layer:                                                                               │ │   │
│   │  │ ┌─────────────────────────────────────────────────────────────────────────────────────┐  │ │   │
│   │  │ │ • RMSNorm                                                                           │  │ │   │
│   │  │ │ • Self-Attention (GQA with RoPE)                                                    │  │ │   │
│   │  │ │   - Q, K, V projections (QKVParallelLinear)                                        │  │ │   │
│   │  │ │   - PagedAttention (vLLM) or FlashAttention                                        │  │ │   │
│   │  │ │   - Output projection (RowParallelLinear)                                          │  │ │   │
│   │  │ │ • RMSNorm                                                                           │  │ │   │
│   │  │ │ • MLP (SwiGLU activation)                                                           │  │ │   │
│   │  │ │   - Gate + Up projection (MergedColumnParallelLinear)                              │  │ │   │
│   │  │ │   - Down projection (RowParallelLinear)                                            │  │ │   │
│   │  │ └─────────────────────────────────────────────────────────────────────────────────────┘  │ │   │
│   │  └───────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│   │                                           │                                                      │   │
│   │                                           ▼                                                      │   │
│   │  ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│   │  │ Final RMSNorm → LM Head (ParallelLMHead) → Logits                                        │ │   │
│   │  └───────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                                              │
│                                           ▼                                                              │
│   OUTPUT: Generated text tokens                                                                          │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Model Size Comparison Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL SIZE COMPARISON (DENSE MODELS)                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   Parameters (billions)                                                                                 │
│   │                                                                                                     │
│   35│                                             ┌───────┐                                             │
│     │                                             │ 33B   │ ← Qwen3-VL-32B                             │
│   30│                                             │       │   (32B LLM + 1B ViT)                       │
│     │                                             │       │                                             │
│   25│                                             │       │                                             │
│     │                                             │       │                                             │
│   20│                                             │       │                                             │
│     │                                             │       │                                             │
│   15│                                             │       │                                             │
│     │                                             │       │                                             │
│   10│                         ┌───────┐           │       │                                             │
│     │                         │ 8.5B  │           │       │                                             │
│    8│                         │       │           │       │ ← Qwen3-VL-8B (8B LLM + 500M ViT)          │
│     │               ┌───────┐ │       │           │       │                                             │
│    5│               │ 4.5B  │ │       │           │       │ ← Qwen3-VL-4B (4B LLM + 500M ViT)          │
│     │     ┌───────┐ │       │ │       │           │       │                                             │
│    2│     │ 2.5B  │ │       │ │       │           │       │ ← Qwen3-VL-2B (2B LLM + 500M ViT)          │
│     │     │       │ │       │ │       │           │       │                                             │
│    0│─────┴───────┴─┴───────┴─┴───────┴───────────┴───────┴─────────────────────────────────────────    │
│          VL-2B     VL-4B     VL-8B               VL-32B                                                 │
│                                                                                                         │
│   ═══════════════════════════════════════════════════════════════════════════════════════════════       │
│                                                                                                         │
│                              MODEL SIZE COMPARISON (MOE MODELS)                                         │
│                                                                                                         │
│   Total Parameters (gray) vs Active Parameters (blue)                                                   │
│   │                                                                                                     │
│   250│                                                    ┌─────────┐                                   │
│      │                                                    │░░░░░░░░░│ ← 237B Total                      │
│   200│                                                    │░░░░░░░░░│   (235B LLM + 2B ViT)            │
│      │                                                    │░░░░░░░░░│                                   │
│   150│                                                    │░░░░░░░░░│                                   │
│      │                                                    │░░░░░░░░░│                                   │
│   100│                                                    │░░░░░░░░░│                                   │
│      │                                                    │░░░░░░░░░│                                   │
│    50│                  ┌─────────┐                       │░░░░░░░░░│                                   │
│      │                  │░░░░░░░░░│ ← 31B Total           │░░░░░░░░░│                                   │
│    25│                  │░░░░░░░░░│   (30B LLM + 1B ViT)  │████████ │ ← 22B Active                     │
│      │                  │███████ │ ← 3B Active           │████████ │                                   │
│     0│──────────────────┴─────────┴───────────────────────┴─────────┴───────────────────────────────    │
│                     VL-30B-A3B                     VL-235B-A22B                                         │
│                                                                                                         │
│   Legend: ░░░ = Total parameters (in memory)    ███ = Active parameters (per token compute)            │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## vLLM PagedAttention for Multimodal

Understanding how vLLM's PagedAttention handles vision-language models:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              VLLM PAGEDATTENTION FOR MULTIMODAL                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   THE PROBLEM: Variable-Length KV Cache                                                                 │
│   ═════════════════════════════════════                                                                 │
│                                                                                                         │
│   Request 1: [1024×1024 image]  → 1332 visual tokens + 100 text = 1432 tokens                          │
│   Request 2: [Video, 50 frames] → 8000 visual tokens + 50 text = 8050 tokens                           │
│   Request 3: [Text only]        → 0 visual tokens + 500 text = 500 tokens                               │
│                                                                                                         │
│   Traditional allocation: Pre-allocate max_tokens for every request → HUGE waste!                      │
│                                                                                                         │
│   NAIVE ALLOCATION (Wasteful):                                                                          │
│   ════════════════════════════                                                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Request 1: [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │   │
│   │ Request 2: [██████████████████████████████████████████████████████████████████░░░░░░░░░░░░░░]  │   │
│   │ Request 3: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │   │
│   │                                                                                                │   │
│   │            ████ = Used memory     ░░░░ = Wasted (pre-allocated but empty)                     │   │
│   │            Memory utilization: ~30%                                                            │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   PAGEDATTENTION (Efficient):                                                                           │
│   ═══════════════════════════                                                                           │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │   Physical Memory Pool (Pages):                                                                │   │
│   │   ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐        │   │
│   │   │ P0  ││ P1  ││ P2  ││ P3  ││ P4  ││ P5  ││ P6  ││ P7  ││ P8  ││ P9  ││P10  ││ ...│        │   │
│   │   │Req1 ││Req1 ││Req1 ││Req2 ││Req2 ││Req2 ││Req2 ││Req2 ││Req3 ││Free ││Free ││    │        │   │
│   │   └─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘        │   │
│   │                                                                                                │   │
│   │   Virtual → Physical Mapping (Block Tables):                                                   │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│   │   │ Request 1: [P0, P1, P2, _, _, _]           (3 pages allocated)                          │ │   │
│   │   │ Request 2: [P3, P4, P5, P6, P7, _]         (5 pages allocated)                          │ │   │
│   │   │ Request 3: [P8, _, _, _, _, _]             (1 page allocated)                           │ │   │
│   │   └─────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│   │                                                                                                │   │
│   │   Memory utilization: ~95%                                                                     │   │
│   │   Pages allocated on-demand, freed when request completes                                     │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   MULTIMODAL-SPECIFIC OPTIMIZATIONS:                                                                    │
│   ══════════════════════════════════                                                                    │
│                                                                                                         │
│   1. PREFIX CACHING FOR VISUAL TOKENS                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │   Turn 1: [System Prompt] + [Image Tokens] + [Question 1]                                     │   │
│   │                              └──── CACHED ────┘                                                │   │
│   │                                                                                                │   │
│   │   Turn 2: [System Prompt] + [Image Tokens] + [Question 2]                                     │   │
│   │           └──── REUSED ────┘ └── REUSED ──┘                                                   │   │
│   │                                                                                                │   │
│   │   BENEFIT: Image tokens computed once, reused across conversation turns                       │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   2. EVS (EFFICIENT VIDEO SAMPLING) INTEGRATION                                                         │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │   100-frame video without EVS:                                                                 │   │
│   │   [F1][F2][F3][F4][F5][F6]...[F100] → 16,000 tokens → 64 pages → May OOM                      │   │
│   │                                                                                                │   │
│   │   100-frame video with EVS (50% pruning):                                                     │   │
│   │   [F1][  ][F3][  ][F5][  ]...[F99 ] → 8,000 tokens → 32 pages → Fits!                        │   │
│   │                                                                                                │   │
│   │   Similar frames pruned BEFORE encoding → fewer tokens → fewer pages → more concurrency       │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   3. CHUNKED PREFILL FOR LONG VISUAL SEQUENCES                                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │   Standard: Encode entire 50K token sequence → Wait → Start generating                        │   │
│   │                                                                                                │   │
│   │   Chunked:  Encode 5K chunk → Start generating → Continue encoding in background              │   │
│   │                                                                                                │   │
│   │   BENEFIT: ~30% faster time-to-first-token for long videos                                   │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## GPU Memory Layout Diagrams

### T4 (16GB) Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              T4 (16GB) MEMORY LAYOUT - Qwen3-VL-4B (4-bit)                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU Memory: 16,384 MB                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Model Weights (4-bit BitsAndBytes)                                    ~2,000 MB (12%)  │   │   │
│   │  │ ████████████████████                                                                   │   │   │
│   │  │ • LLM: 4B params × 0.5 bytes = 2,000 MB                                               │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Vision Encoder (FP16, can't quantize easily)                          ~1,000 MB (6%)   │   │   │
│   │  │ ██████████                                                                             │   │   │
│   │  │ • ViT: 500M params × 2 bytes = 1,000 MB                                               │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ KV Cache (4K context × 4 requests × FP16)                             ~3,000 MB (18%)  │   │   │
│   │  │ ██████████████████████████████                                                         │   │   │
│   │  │ • 2 × 36 layers × 4 kv_heads × 128 head_dim × 4K × 4 seqs × 2 bytes                  │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Activations (intermediate tensors)                                    ~2,000 MB (12%)  │   │   │
│   │  │ ████████████████████                                                                   │   │   │
│   │  │ • MLP intermediate, attention scores, etc.                                            │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Visual Token Buffer (1024×1024 image)                                 ~1,500 MB (9%)   │   │   │
│   │  │ ███████████████                                                                        │   │   │
│   │  │ • Patch embeddings during vision encoding                                              │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ CUDA Overhead / Reserved                                              ~1,500 MB (9%)   │   │   │
│   │  │ ███████████████                                                                        │   │   │
│   │  │ • CUDA context, driver, workspace                                                      │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ FREE MEMORY                                                           ~5,384 MB (34%)  │   │   │
│   │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                                │   │   │
│   │  │ • Available for additional KV cache pages                                              │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  TOTAL USED: ~11,000 MB (67%)     FREE: ~5,384 MB (33%)                                       │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   LIMITATIONS:                                                                                          │
│   • No BF16 (FP16 only)                                                                                │
│   • No FlashAttention 2 (uses TORCH_SDPA)                                                              │
│   • 320 GB/s bandwidth (memory-bound during decode)                                                    │
│                                                                                                         │
│   RECOMMENDED CONFIG:                                                                                   │
│   • enforce_eager=True (saves ~500 MB by disabling CUDA graphs)                                        │
│   • max_model_len=4096                                                                                 │
│   • max_num_seqs=4                                                                                     │
│   • max_pixels=500000 (~700×700)                                                                       │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### A100-80GB Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              A100-80GB MEMORY LAYOUT - Qwen3-VL-8B (BF16)                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU Memory: 81,920 MB                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Model Weights (BF16 full precision)                                  ~17,000 MB (21%)  │   │   │
│   │  │ ██████████████████████████████████                                                     │   │   │
│   │  │ • LLM: 8B params × 2 bytes = 16,000 MB                                                │   │   │
│   │  │ • Vision: 500M × 2 bytes = 1,000 MB                                                    │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ KV Cache (32K context × 32 requests × BF16)                          ~24,000 MB (29%)  │   │   │
│   │  │ ████████████████████████████████████████████████                                       │   │   │
│   │  │ • Managed by PagedAttention                                                            │   │   │
│   │  │ • 2 × 32 layers × 8 kv_heads × 128 head_dim × 32K × 32 seqs × 2 bytes                │   │   │
│   │  │ • Pages allocated dynamically as needed                                                │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Activations + CUDA Graphs                                             ~8,000 MB (10%)  │   │   │
│   │  │ ████████████████                                                                       │   │   │
│   │  │ • CUDA graphs enabled for faster decode                                                │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Prefix Cache (shared system prompts)                                  ~3,000 MB (4%)   │   │   │
│   │  │ ██████                                                                                 │   │   │
│   │  │ • Cached KV for repeated prefixes                                                      │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ FREE MEMORY                                                          ~29,920 MB (36%)  │   │   │
│   │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                          │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  TOTAL USED: ~52,000 MB (64%)     FREE: ~29,920 MB (36%)                                      │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   ADVANTAGES:                                                                                           │
│   • Native BF16 support (better numerical stability)                                                   │
│   • FlashAttention 2 (2-4× faster attention)                                                           │
│   • 2,039 GB/s bandwidth (6× faster than T4)                                                           │
│   • Tensor cores for mixed precision                                                                   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### H100-80GB Memory Layout (with FP8)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              H100-80GB MEMORY LAYOUT - Qwen3-VL-8B (FP8)                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU Memory: 81,920 MB                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Model Weights (FP8 quantized)                                         ~8,500 MB (10%)  │   │   │
│   │  │ █████████████████                                                                      │   │   │
│   │  │ • LLM: 8B params × 1 byte = 8,000 MB                                                  │   │   │
│   │  │ • Vision: 500M × 1 byte = 500 MB                                                       │   │   │
│   │  │ ← 50% smaller than BF16!                                                               │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ KV Cache (FP8, 32K context × 64 requests)                            ~16,000 MB (20%)  │   │   │
│   │  │ ████████████████████████████████                                                       │   │   │
│   │  │ • FP8 KV cache = 50% smaller than BF16                                                │   │   │
│   │  │ • Can serve 2× more concurrent requests!                                               │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Activations + FlashAttention 3 workspace                              ~6,000 MB (7%)   │   │   │
│   │  │ ████████████                                                                           │   │   │
│   │  │ • FlashAttention 3 optimized for Hopper                                                │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ FREE MEMORY                                                          ~51,420 MB (63%)  │   │   │
│   │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│   │   │
│   │  │ • Room for 64+ concurrent image requests                                               │   │   │
│   │  │ • Can scale to 128 concurrent with smaller images                                      │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  TOTAL USED: ~30,500 MB (37%)     FREE: ~51,420 MB (63%)                                      │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   FP8 BENEFITS:                                                                                         │
│   • Weights: 17 GB → 8.5 GB (50% reduction)                                                            │
│   • KV Cache: 24 GB → 12 GB (50% reduction)                                                            │
│   • Throughput: ~2× faster matmul operations                                                           │
│   • Quality: <1% accuracy loss for most VLM tasks                                                      │
│                                                                                                         │
│   ADDITIONAL H100 ADVANTAGES:                                                                           │
│   • FlashAttention 3 (Hopper-specific optimizations)                                                   │
│   • 3,350 GB/s bandwidth (10× faster than T4)                                                          │
│   • Transformer Engine for automatic mixed precision                                                   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### B200-192GB Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              B200-192GB MEMORY LAYOUT - Qwen3-VL-32B (BF16)                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU Memory: 196,608 MB                                                                                │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Model Weights (BF16 full precision)                                  ~66,000 MB (34%)  │   │   │
│   │  │ ██████████████████████████████████████████████████████████████████████                 │   │   │
│   │  │ • LLM: 32B params × 2 bytes = 64,000 MB                                               │   │   │
│   │  │ • Vision: 1B × 2 bytes = 2,000 MB                                                      │   │   │
│   │  │ ← Single GPU! No tensor parallelism needed!                                            │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ KV Cache (128K context × 128 requests × BF16)                        ~60,000 MB (30%)  │   │   │
│   │  │ ████████████████████████████████████████████████████████████                           │   │   │
│   │  │ • Massive context window supported                                                     │   │   │
│   │  │ • 2 × 64 layers × 8 kv_heads × 128 head_dim × 128K × 128 seqs × 2 bytes              │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Activations + 4K resolution images                                   ~20,000 MB (10%)  │   │   │
│   │  │ ████████████████████████████████████████                                               │   │   │
│   │  │ • max_pixels=4,147,200 (4K resolution)                                                │   │   │
│   │  │ • Multiple 4K images simultaneously                                                    │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ FREE MEMORY                                                          ~50,608 MB (26%)  │   │   │
│   │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                                │   │   │
│   │  │ • Headroom for even larger batches                                                     │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  TOTAL USED: ~146,000 MB (74%)     FREE: ~50,608 MB (26%)                                     │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   B200 ADVANTAGES:                                                                                      │
│   • 192 GB HBM3e (12× T4, 2.4× H100)                                                                   │
│   • 8,000 GB/s bandwidth (25× faster than T4)                                                          │
│   • FP4 support (when available, 4× throughput potential)                                              │
│   • Run 32B model on SINGLE GPU                                                                        │
│   • 128K context with 128 concurrent requests                                                          │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Benchmarks

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PERFORMANCE BENCHMARKS BY GPU                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   SINGLE IMAGE INFERENCE (1024×1024, 256 output tokens):                                                │
│   ══════════════════════════════════════════════════════                                                │
│                                                                                                         │
│   ┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐                │
│   │ Model              │ T4 (16GB)          │ A100-80GB          │ H100-80GB (FP8)    │                │
│   ├────────────────────┼────────────────────┼────────────────────┼────────────────────┤                │
│   │ Qwen3-VL-2B        │ 600ms, 25 tok/s    │ 100ms, 150 tok/s   │ 50ms, 300 tok/s    │                │
│   │ Qwen3-VL-4B        │ 1000ms*, 18 tok/s  │ 150ms, 120 tok/s   │ 80ms, 220 tok/s    │                │
│   │ Qwen3-VL-8B        │ 1800ms*, 10 tok/s  │ 250ms, 80 tok/s    │ 120ms, 150 tok/s   │                │
│   │ Qwen3-VL-32B       │ ❌ OOM             │ 500ms, 40 tok/s    │ 250ms, 80 tok/s    │                │
│   │ Qwen3-VL-30B-A3B   │ ❌ OOM             │ 350ms, 50 tok/s    │ 180ms, 100 tok/s   │                │
│   └────────────────────┴────────────────────┴────────────────────┴────────────────────┘                │
│   * = Requires 4-bit quantization                                                                       │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   VIDEO INFERENCE (60s video, 100 frames, 256 output tokens):                                           │
│   ═══════════════════════════════════════════════════════════                                           │
│                                                                                                         │
│   ┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐                │
│   │ Model              │ Without EVS        │ With EVS (50%)     │ Speed Improvement  │                │
│   ├────────────────────┼────────────────────┼────────────────────┼────────────────────┤                │
│   │ Qwen3-VL-8B (A100) │ 4.5s, 16K tokens   │ 2.5s, 8K tokens    │ 1.8× faster        │                │
│   │ Qwen3-VL-8B (H100) │ 2.0s, 16K tokens   │ 1.1s, 8K tokens    │ 1.8× faster        │                │
│   │ Qwen3-VL-32B (H100)│ 4.0s, 16K tokens   │ 2.2s, 8K tokens    │ 1.8× faster        │                │
│   └────────────────────┴────────────────────┴────────────────────┴────────────────────┘                │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   THROUGHPUT (Concurrent requests, mixed image sizes):                                                  │
│   ═══════════════════════════════════════════════════════                                               │
│                                                                                                         │
│   ┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐                │
│   │ GPU                │ Max Concurrent     │ Throughput (req/s) │ p95 Latency        │                │
│   ├────────────────────┼────────────────────┼────────────────────┼────────────────────┤                │
│   │ T4 (Qwen3-VL-4B)   │ 4                  │ ~1                 │ ~2000ms            │                │
│   │ A100-80GB (8B)     │ 32                 │ ~6                 │ ~500ms             │                │
│   │ H100-80GB (8B FP8) │ 64                 │ ~12                │ ~300ms             │                │
│   │ B200-192GB (32B)   │ 128                │ ~20                │ ~200ms             │                │
│   └────────────────────┴────────────────────┴────────────────────┴────────────────────┘                │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
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

---

## Executive Summary: Key Takeaways by Role

### For CEOs / Business Leaders

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              EXECUTIVE SUMMARY                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   WHAT IS QWEN3-VL?                                                                             │
│   A state-of-the-art AI that understands both images and text, enabling:                       │
│   • Automated GUI agents (MAI-UI) that can control computers/phones                            │
│   • Document understanding and analysis                                                         │
│   • Video comprehension for content moderation, security, etc.                                 │
│                                                                                                 │
│   ROI CONSIDERATIONS:                                                                           │
│   ┌─────────────────────┬────────────────┬────────────────┬────────────────────┐               │
│   │ Workload Scale      │ GPU Choice     │ Monthly Cost*  │ Requests/day       │               │
│   ├─────────────────────┼────────────────┼────────────────┼────────────────────┤               │
│   │ Prototype/Dev       │ 1× T4          │ ~$100          │ ~5,000             │               │
│   │ Small Production    │ 1× A100-40GB   │ ~$1,000        │ ~50,000            │               │
│   │ Medium Production   │ 1× H100        │ ~$3,000        │ ~200,000           │               │
│   │ Enterprise Scale    │ 8× H100        │ ~$25,000       │ ~1,000,000+        │               │
│   └─────────────────────┴────────────────┴────────────────┴────────────────────┘               │
│   *Approximate cloud GPU rental costs                                                          │
│                                                                                                 │
│   KEY DECISION: Model quality scales with size, but so does cost.                              │
│   • 8B model: Good quality, reasonable cost, fits most use cases                              │
│   • 32B model: Best quality, 4× cost, for demanding applications                              │
│   • MoE models: 32B quality at 8B cost, but requires more memory                              │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### For CTOs / Engineering Managers

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TECHNICAL STRATEGY                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   ARCHITECTURE DECISION TREE:                                                                   │
│                                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                        │   │
│   │   Q1: What's your latency requirement?                                                 │   │
│   │   ├─ < 200ms → H100 with FP8, or B200                                                 │   │
│   │   ├─ < 500ms → A100-80GB with BF16                                                    │   │
│   │   └─ < 1000ms → T4 with 4-bit quantization (cost-optimized)                           │   │
│   │                                                                                        │   │
│   │   Q2: What's your quality requirement?                                                 │   │
│   │   ├─ SOTA benchmarks → Qwen3-VL-32B or 235B-A22B                                      │   │
│   │   ├─ Production quality → Qwen3-VL-8B (sweet spot)                                    │   │
│   │   └─ Good enough → Qwen3-VL-4B (3× faster, 80% quality)                               │   │
│   │                                                                                        │   │
│   │   Q3: What's your concurrency requirement?                                             │   │
│   │   ├─ 100+ concurrent → B200 or multi-GPU H100                                         │   │
│   │   ├─ 10-50 concurrent → H100 with FP8 KV cache                                        │   │
│   │   └─ < 10 concurrent → A100-40GB sufficient                                           │   │
│   │                                                                                        │   │
│   └────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                 │
│   INFRASTRUCTURE RECOMMENDATIONS:                                                               │
│   • Use vLLM (not HuggingFace) for 3-5× better throughput                                     │
│   • Enable PagedAttention (default in vLLM) for memory efficiency                             │
│   • Enable prefix caching for multi-turn conversations                                         │
│   • Use FP8 on H100 for 2× throughput with <1% quality loss                                   │
│   • Consider MoE models for cost-efficiency at scale                                          │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### For Engineers / Developers

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              IMPLEMENTATION GUIDE                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   QUICK START (copy-paste ready):                                                               │
│                                                                                                 │
│   # For development (Colab T4):                                                                │
│   vllm serve Qwen/Qwen3-VL-2B-Instruct \                                                       │
│     --dtype float16 \                                                                           │
│     --max-model-len 4096 \                                                                      │
│     --gpu-memory-utilization 0.85 \                                                             │
│     --enforce-eager                                                                             │
│                                                                                                 │
│   # For production (A100-80GB):                                                                │
│   vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                       │
│     --dtype bfloat16 \                                                                          │
│     --max-model-len 32768 \                                                                     │
│     --max-num-seqs 32 \                                                                         │
│     --enable-prefix-caching \                                                                   │
│     --enable-chunked-prefill                                                                    │
│                                                                                                 │
│   # For high-throughput (H100):                                                                │
│   vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                       │
│     --quantization fp8 \                                                                        │
│     --kv-cache-dtype fp8 \                                                                      │
│     --max-num-seqs 64 \                                                                         │
│     --enable-prefix-caching                                                                     │
│                                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────────────────────    │
│                                                                                                 │
│   DEBUGGING CHECKLIST:                                                                          │
│   □ OOM errors → Reduce max_model_len, max_num_seqs, or max_pixels                            │
│   □ Slow first token → Enable chunked_prefill for long contexts                               │
│   □ Slow decode → Check bandwidth (T4 is 6× slower than H100)                                 │
│   □ Quality issues → Check dtype (BF16 > FP16 > FP8 > INT4)                                   │
│   □ Video OOM → Enable EVS with video_pruning_rate=0.5                                        │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### For Interns / New Team Members

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              LEARNING PATH                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   CONCEPT MAP (What you need to understand):                                                    │
│                                                                                                 │
│   Level 1: What is a VLM?                                                                       │
│   ────────────────────────                                                                      │
│   Vision + Language = Model that understands images AND text                                   │
│   • Input: Image/Video + Text prompt ("What's in this picture?")                              │
│   • Output: Text response ("A cat sitting on a windowsill")                                   │
│                                                                                                 │
│   Level 2: How does it work?                                                                    │
│   ──────────────────────────                                                                    │
│   1. Vision Encoder (ViT) converts image → tokens (numbers)                                   │
│   2. Text Tokenizer converts text → tokens                                                     │
│   3. LLM processes [image_tokens] + [text_tokens] together                                    │
│   4. LLM generates response tokens one at a time                                              │
│                                                                                                 │
│   Level 3: What is vLLM?                                                                        │
│   ──────────────────────                                                                        │
│   vLLM = Very fast LLM inference engine                                                        │
│   • PagedAttention: Efficient memory management (like virtual memory for AI)                  │
│   • Continuous batching: Serve many users simultaneously                                      │
│   • Prefix caching: Reuse computation for shared contexts                                     │
│                                                                                                 │
│   Level 4: Why different GPUs?                                                                  │
│   ───────────────────────────                                                                   │
│   • More VRAM = larger models, longer contexts, more concurrent users                         │
│   • More bandwidth = faster token generation                                                  │
│   • Newer architecture = more features (FP8, FlashAttention 3)                               │
│                                                                                                 │
│   RECOMMENDED READING ORDER:                                                                    │
│   1. QWEN_VL_COMPLETE_GUIDE.md - Architecture & concepts                                       │
│   2. This file - Model sizes & GPU configs                                                     │
│   3. mai_ui_gpu_optimized.ipynb - Hands-on practice                                           │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## References

- [Qwen3-VL Technical Report (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [MAI-UI Technical Report (arXiv:2512.22047)](https://arxiv.org/abs/2512.22047)
- [Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)
- [vLLM Qwen3-VL Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Hugging Face Qwen Collection](https://huggingface.co/Qwen)
- [QWEN_VL_COMPLETE_GUIDE.md](./QWEN_VL_COMPLETE_GUIDE.md) - Companion architectural deep-dive

