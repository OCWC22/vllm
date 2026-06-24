# vLLM-Omni v0.14.0 Complete Technical Guide
## Every Feature, Every GPU, Every Use Case — From CEO to Engineer

**Release Date**: January 31, 2026
**Commits**: ~180 from 70+ contributors (23 new contributors)
**Paper**: [arXiv:2602.02204](https://arxiv.org/abs/2602.02204)
**Focus**: Qwen3-VL on NVIDIA B200 and H100 GPUs

---

## Table of Contents

1. [Executive Summary (CEO View)](#1-executive-summary-ceo-view)
2. [GPU Hardware Deep Dive: B200 vs H100](#2-gpu-hardware-deep-dive-b200-vs-h100)
3. [Qwen3-VL Architecture Deep Dive](#3-qwen3-vl-architecture-deep-dive)
4. [vLLM v1 Engine Architecture](#4-vllm-v1-engine-architecture)
5. [Feature 1: Async Chunk Pipeline Overlap](#5-feature-1-async-chunk-pipeline-overlap)
6. [Feature 2: SharedFusedMoE for Qwen3-Omni](#6-feature-2-sharedfusedmoe-for-qwen3-omni)
7. [Feature 3: TeaCache for Z-Image and Bagel](#7-feature-3-teacache-for-z-image-and-bagel)
8. [Feature 4: Sequence Parallelism (SP) for Diffusion](#8-feature-4-sequence-parallelism-sp-for-diffusion)
9. [Feature 5: Torch Compile for Diffusion](#9-feature-5-torch-compile-for-diffusion)
10. [Feature 6: Qwen3-TTS with Online Serving](#10-feature-6-qwen3-tts-with-online-serving)
11. [Feature 7: Diffusion LoRA (PEFT-Compatible)](#11-feature-7-diffusion-lora-peft-compatible)
12. [Feature 8: DiT Layerwise CPU Offloading](#12-feature-8-dit-layerwise-cpu-offloading)
13. [Feature 9: New Models (Bagel, FLUX, GLM-Image, Stable Audio)](#13-feature-9-new-models-bagel-flux-glm-image-stable-audio)
14. [Feature 10: New APIs (/v1/images/edit, /health, /v1/models)](#14-feature-10-new-apis)
15. [Feature 11: XPU / ROCm / NPU Backend Support](#15-feature-11-xpu--rocm--npu-backend-support)
16. [Feature 12: Decode Context Parallel (DCP)](#16-feature-12-decode-context-parallel-dcp)
17. [MLPerf Inference v6.0: Qwen3-VL + Shopify Benchmark](#17-mlperf-inference-v60-qwen3-vl--shopify-benchmark)
18. [vLLM-Omni Paper (arXiv:2602.02204)](#18-vllm-omni-paper)
19. [Step-by-Step: Qwen3-VL-32B on H100 (Single Node)](#19-step-by-step-qwen3-vl-32b-on-h100)
20. [Step-by-Step: Qwen3-VL-32B on B200 (Single GPU)](#20-step-by-step-qwen3-vl-32b-on-b200)
21. [Step-by-Step: Qwen3-VL-235B-A22B on 8xH100 / 8xB200](#21-step-by-step-qwen3-vl-235b-a22b-on-8xh100--8xb200)
22. [Business Use Cases by Industry](#22-business-use-cases-by-industry)
23. [Actual vLLM Code Walkthrough](#23-actual-vllm-code-walkthrough)
24. [Performance Benchmarks and Expectations](#24-performance-benchmarks-and-expectations)
25. [Troubleshooting and Production Tips](#25-troubleshooting-and-production-tips)

---

## 1. Executive Summary (CEO View)

### What Is vLLM-Omni v0.14.0?

vLLM-Omni is the **industry-standard open-source inference engine** for deploying AI models in production. Version 0.14.0 transforms vLLM from a text-only engine into a **universal multimodal serving platform** that handles text, images, video, audio, and speech — all from a single unified system.

### Why Should a CEO Care?

| Business Impact | Before vLLM-Omni | After vLLM-Omni v0.14.0 |
|----------------|-------------------|--------------------------|
| **Infrastructure Cost** | Separate servers for text AI, image AI, speech AI | One unified platform serves everything |
| **Time to Market** | Weeks to deploy each new model | Day-0 support for new models |
| **Latency** | 23+ seconds for image generation | 9.6 seconds (2.4x faster) |
| **Throughput** | Limited by single-model pipelines | Async pipeline overlaps computation |
| **Hardware Utilization** | GPUs idle during data transfers | Pipeline overlap keeps GPUs busy |
| **Customization** | Expensive full model retraining | LoRA adapters for cheap fine-tuning |
| **Scale** | Manual multi-GPU orchestration | Built-in tensor parallelism + sequence parallelism |

### The Bottom Line

- **2.4x-3.7x faster** end-to-end for image generation (Bagel model)
- **91.4% latency reduction** for audio generation (Qwen3-Omni)
- **Real-time speech** synthesis (RTF 0.60, well under 1.0 threshold)
- Runs on existing H100 infrastructure, runs even better on new B200 hardware
- **MLPerf v6.0 reference implementation** = industry validation that this is the standard

### Key Decision Points for Leadership

1. **Adopt vLLM-Omni** if you're building any multimodal AI product (chatbots with images, product catalogs, voice assistants)
2. **Plan B200 migration** — 2.3x more compute and 2.25x more memory means you can serve larger models on fewer GPUs
3. **The Qwen3-VL-235B model** processes 40 million product images/day for Shopify — this is production-proven at scale
4. **LoRA support for diffusion** means your creative teams can customize image generation models without retraining from scratch

---

## 2. GPU Hardware Deep Dive: B200 vs H100

### NVIDIA H100 SXM5 (Current Generation - Hopper Architecture)

| Specification | H100 SXM5 Value |
|---------------|-----------------|
| **Architecture** | Hopper (GH100) |
| **Process Node** | TSMC 4N |
| **Transistors** | 80 billion |
| **Die Design** | Single monolithic die |
| **Streaming Multiprocessors (SMs)** | 132 enabled (144 physical) |
| **CUDA Cores** | 16,896 |
| **Tensor Cores** | 528 (4th generation) |
| **HBM3 Memory** | 80 GB |
| **Memory Bandwidth** | 3.35 TB/s |
| **Memory Bus** | 5120-bit |
| **L2 Cache** | 50 MB |
| **FP8 Tensor (with sparsity)** | 3,958 TFLOPS |
| **FP16/BF16 Tensor (with sparsity)** | 1,979 TFLOPS |
| **TF32 Tensor (with sparsity)** | ~989 TFLOPS |
| **FP32** | ~67 TFLOPS |
| **FP64** | 34 TFLOPS |
| **NVLink** | 4th gen, 18 links, 900 GB/s bidirectional |
| **PCIe** | Gen 5 (128 GB/s) |
| **TDP** | 700W |
| **MIG Instances** | Up to 7 (2nd-gen MIG) |
| **Transformer Engine** | 1st generation — dynamic FP8/FP16 switching per layer |

#### How the H100 Processes Qwen3-VL Inference

```
Step 1: Image Input → Vision Encoder (ViT)
  - 528 Tensor Cores process attention in BF16
  - 50MB L2 cache holds ViT intermediate activations
  - 3.35 TB/s bandwidth streams image patches from HBM3

Step 2: DeepStack Fusion → LLM Layers 1-3
  - Multi-depth visual features fused via MLP projectors
  - Transformer Engine auto-selects FP8 for compute-bound ops
  - FP16 for precision-sensitive residual connections

Step 3: LLM Autoregressive Decoding
  - KV Cache lives in HBM3 (80GB total capacity)
  - For 32B model at BF16: ~64GB model weights + ~16GB KV cache headroom
  - Memory-bandwidth bound: 3.35 TB/s determines token/sec rate

Step 4: Multi-GPU Scaling (Tensor Parallelism)
  - NVLink 900 GB/s connects 8 GPUs within DGX H100
  - TP=2: each GPU holds half the model → 40GB weights per GPU
  - TP=4: 20GB per GPU, leaving 60GB for KV cache per GPU
```

### NVIDIA B200 (Next Generation - Blackwell Architecture)

| Specification | B200 HGX (1000W, air-cooled) | GB200 (1200W, liquid-cooled) |
|---------------|------------------------------|------------------------------|
| **Architecture** | Blackwell |  Blackwell |
| **Process Node** | TSMC 4NP | TSMC 4NP |
| **Transistors** | 208 billion | 208 billion |
| **Die Design** | Dual-die, 10 TB/s chip-to-chip | Dual-die, 10 TB/s chip-to-chip |
| **Streaming Multiprocessors (SMs)** | 148 enabled (160 physical, 80/die) | 148 enabled |
| **CUDA Cores** | 16,896 | 16,896 |
| **Tensor Cores** | 528 (5th generation) | 528 (5th generation) |
| **HBM3e Memory** | 180 GB usable (192 GB physical) | 186 GB usable (192 GB physical) |
| **Memory Bandwidth** | 7.7 TB/s | 8.0 TB/s |
| **Memory Bus** | 4096-bit per chiplet (dual) | 4096-bit per chiplet (dual) |
| **FP4 Tensor (with sparsity)** | 18 petaFLOPS | 20 petaFLOPS |
| **FP8/FP6 Tensor (with sparsity)** | 9 petaFLOPS | 10 petaFLOPS |
| **FP16/BF16 Tensor (with sparsity)** | 4.5 petaFLOPS | 5 petaFLOPS |
| **TF32 Tensor (with sparsity)** | 2.2 petaFLOPS | 2.5 petaFLOPS |
| **FP32** | 75 TFLOPS | 80 TFLOPS |
| **FP64** | 37 TFLOPS | 40 TFLOPS |
| **NVLink** | NVLink 5, 1.8 TB/s bidirectional | NVLink 5, 1.8 TB/s bidirectional |
| **PCIe** | Gen 5 (128 GB/s) | Gen 5 (128 GB/s) |
| **TDP** | 1,000W | 1,200W |
| **MIG Instances** | Up to 7 | Up to 7 |
| **Transformer Engine** | 2nd generation — FP4/FP8/FP16 per-layer | 2nd generation |
| **Tensor Memory (TMEM)** | Dedicated register file for Tensor Cores | Yes |
| **Shared Memory per SM** | 228 KB | 228 KB |

#### How the B200 Processes Qwen3-VL Inference

```
Step 1: Image Input → Vision Encoder (ViT)
  - 5th-gen Tensor Cores with matrix ops spanning multiple waves
  - TMEM (Tensor Memory) reduces register-file pressure → higher occupancy
  - 7.7 TB/s bandwidth (2.3x H100) streams image patches 2.3x faster

Step 2: DeepStack Fusion → LLM Layers 1-3
  - 2nd-gen Transformer Engine: auto-selects FP4/FP8/FP16
  - FP4 mode: 18 petaFLOPS → 4.5x the raw compute of H100 FP8
  - For compute-bound attention: FP4 delivers 2x throughput over FP8

Step 3: LLM Autoregressive Decoding
  - KV Cache in HBM3e: 180GB total capacity (2.25x H100)
  - Qwen3-VL-32B at BF16: ~64GB weights, leaving ~116GB for KV cache
  - 7.7 TB/s bandwidth → tokens/sec scales ~2.3x vs H100
  - NVFP4: can reduce model weight memory by 2x (32GB at FP4)

Step 4: Multi-GPU Scaling (Tensor Parallelism)
  - NVLink 5 at 1.8 TB/s (2x H100): faster all-reduce for TP
  - Single B200 can fit Qwen3-VL-32B entirely with room to spare
  - 8xB200: 1,440GB total HBM3e → can serve Qwen3-VL-235B-A22B comfortably
```

### Head-to-Head Comparison: B200 vs H100

| Metric | H100 SXM5 | B200 HGX (1000W) | B200 Advantage | Impact on Qwen3-VL |
|--------|-----------|-------------------|----------------|---------------------|
| **FP8 TFLOPS** | 3,958 | 9,000 | **2.3x** | 2.3x faster prefill (compute-bound) |
| **FP4 TFLOPS** | N/A | 18,000 | **New** | 4.5x vs H100 FP8 with model quantization |
| **HBM Capacity** | 80 GB | 180 GB | **2.25x** | Fit 32B model on 1 GPU (vs 2 on H100) |
| **Memory BW** | 3.35 TB/s | 7.7 TB/s | **2.3x** | 2.3x faster decode (bandwidth-bound) |
| **NVLink BW** | 900 GB/s | 1,800 GB/s | **2x** | 2x faster tensor parallelism communication |
| **L2 Cache** | 50 MB | ~100 MB (est.) | **~2x** | Better KV cache hit rates |
| **TDP** | 700W | 1,000W | 1.43x more power | But 2.3x more perf = better perf/watt |
| **Perf/Watt (FP8)** | 5.65 TFLOPS/W | 9.0 TFLOPS/W | **1.59x** | 59% more efficient per watt |

### What This Means in Practice

**Single GPU — Qwen3-VL-32B (BF16, ~64GB weights)**:
- **H100**: Cannot fit on 1 GPU (64GB > 80GB after overhead). Needs TP=2 minimum.
- **B200**: Fits on 1 GPU (64GB < 180GB). Leaves 116GB for KV cache = ~58K tokens at BF16.

**8 GPUs — Qwen3-VL-235B-A22B (BF16, ~470GB total weights, 44GB activated)**:
- **8xH100**: 640GB total. Tight fit. KV cache limited. Need aggressive quantization.
- **8xB200**: 1,440GB total. Comfortable fit. 970GB available for KV cache.

---

## 3. Qwen3-VL Architecture Deep Dive

### Model Family Overview

| Variant | Type | Total Params | Active Params | Min GPUs (BF16) |
|---------|------|-------------|---------------|-----------------|
| Qwen3-VL-2B | Dense | 2B | 2B | 1 (any) |
| Qwen3-VL-4B | Dense | 4B | 4B | 1 (any) |
| Qwen3-VL-8B | Dense | 8B | 8B | 1 (any) |
| **Qwen3-VL-32B** | **Dense** | **32B** | **32B** | **1xB200 or 2xH100** |
| Qwen3-VL-30B-A3B | MoE | 30B | 3B | 1 (any) |
| **Qwen3-VL-235B-A22B** | **MoE** | **235B** | **22B** | **8x (80GB+ each)** |

### Three-Module Architecture

```
┌─────────────────────────────────────────────────┐
│                  Qwen3-VL Model                  │
├─────────────────┬──────────────┬────────────────┤
│  Vision Encoder │  DeepStack   │  LLM Backbone  │
│     (ViT)       │   Fusion     │   (Qwen3)      │
│                 │              │                │
│  27 Transformer │  Features    │  64 Transformer│
│  layers         │  from ViT    │  layers (32B)  │
│                 │  layers      │                │
│  Patch: 16x16   │  8, 16, 24   │  GQA: 64 heads │
│  hidden: 1152   │  → LLM L1-3  │  8 KV heads    │
│  heads: 16      │              │  hidden: ~5120 │
│                 │  3 MLP       │                │
│  Spatial merge   │  projectors  │  SiLU activat. │
│  2x2 → reduce   │              │  RMSNorm       │
│  tokens by 4x   │              │                │
└─────────────────┴──────────────┴────────────────┘
```

### Module 1: Vision Encoder (ViT)

The Vision Encoder processes images and video frames into visual tokens.

**From the actual vLLM code** at `vllm/model_executor/models/qwen3_vl.py`:

```python
# Vision encoder configuration (defaults for base variant)
class Qwen3VLVisionConfig:
    hidden_size = 1152        # ViT hidden dimension
    depth = 27                # Number of transformer layers
    num_heads = 16            # Attention heads
    intermediate_size = 4304  # FFN intermediate size
    hidden_act = "gelu_pytorch_tanh"
    patch_size = 16           # Each patch is 16x16 pixels
    spatial_merge_size = 2    # Merge 2x2 patches → 4x reduction
    spatial_patch_size = 16
    temporal_patch_size = 2   # For video: merge 2 adjacent frames
    in_channels = 3           # RGB input
    out_hidden_size = 3584    # Projects to LLM hidden dim
    num_position_embeddings = 2304  # Max position embeddings
    deepstack_visual_indexes = [8, 16, 24]  # Feature extraction layers
```

**How it works step by step:**

```
Input Image (e.g., 1024x1024 RGB)
  │
  ▼
1. Patch Embedding: Split into 16x16 patches → 64x64 = 4096 patches
   Each patch → 1152-dim vector via Conv2D
  │
  ▼
2. 27 Transformer Layers with rotary position embeddings
   - At layer 8: extract intermediate features (low-level: edges, textures)
   - At layer 16: extract intermediate features (mid-level: shapes, parts)
   - At layer 24: extract intermediate features (high-level: objects, scenes)
  │
  ▼
3. Spatial Merge: 2x2 adjacent patch tokens merged → 1024 tokens
   (4x reduction in sequence length)
  │
  ▼
4. Projection: 1152-dim → out_hidden_size via learned linear layer
   Output: 1024 visual tokens of dimension matching LLM hidden size
```

### Module 2: DeepStack Fusion (NEW in Qwen3-VL)

This is the key innovation over Qwen2-VL. Instead of injecting visual features only at the input layer, DeepStack injects features from multiple depths of the ViT into the first 3 layers of the LLM.

```
ViT Layer 8  features ──→ MLP Projector 1 ──→ Added to LLM Layer 1 output
ViT Layer 16 features ──→ MLP Projector 2 ──→ Added to LLM Layer 2 output
ViT Layer 24 features ──→ MLP Projector 3 ──→ Added to LLM Layer 3 output
```

**Why this matters:**
- Low-level ViT features (layer 8) capture edges and textures → help LLM with OCR, fine details
- Mid-level ViT features (layer 16) capture shapes and object parts → help with spatial reasoning
- High-level ViT features (layer 24) capture semantic concepts → help with scene understanding
- The LLM gets a richer, multi-scale understanding of the image

### Module 3: LLM Backbone (Qwen3-32B)

| Parameter | Value (32B variant) |
|-----------|-------------------|
| `hidden_size` | ~5,120 |
| `num_hidden_layers` | 64 |
| `num_attention_heads` | 64 (GQA) |
| `num_kv_heads` | 8 |
| `head_dim` | 128 |
| `intermediate_size` | ~25,088 or 49,152 |
| `vocab_size` | 151,936 |
| `hidden_act` | SiLU |
| `rms_norm_eps` | 1e-6 |
| `max_position_embeddings` | 128,000 (native 256K trained) |

### Module 4: Interleaved-MRoPE (Position Encoding)

Qwen3-VL uses **Multi-Resolution Rotary Position Embedding** that allocates full frequency bands across three dimensions:

```
Total RoPE dimensions: 64 (per head_dim of 128, using half for cos/sin)
  ├── Temporal: 24 dimensions (for video frame timing)
  ├── Width:    20 dimensions (horizontal position in image)
  └── Height:   20 dimensions (vertical position in image)
```

This is critical for:
- Understanding spatial layout of documents (OCR, forms)
- Tracking objects across video frames
- Distinguishing between "cat on left, dog on right" vs "dog on left, cat on right"

### Context Length

- **Native training**: 256K tokens
- **YaRN extension**: Up to 1M tokens (scaling factor 2-3)
- At 256K context with Qwen3-VL-32B: that's approximately 64K image patches + 192K text tokens

---

## 4. vLLM v1 Engine Architecture

Understanding how vLLM actually processes requests is essential. Here's the complete architecture from the actual codebase.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        API Layer                              │
│  OpenAI-compatible endpoints: /v1/chat/completions, etc.      │
│  File: vllm/entrypoints/openai/api_server.py                 │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     AsyncLLM (v1)                             │
│  Async wrapper that manages request lifecycle                 │
│  File: vllm/v1/engine/async_llm.py                           │
│                                                              │
│  - Tokenizes input text                                       │
│  - Processes multimodal inputs (images, video)                │
│  - Submits to EngineCore via IPC (msgpack over ZMQ)           │
│  - Streams outputs back to API layer                          │
└─────────────────────────┬────────────────────────────────────┘
                          │ ZMQ IPC (separate process)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    EngineCore (v1)                             │
│  The central orchestrator — runs in its own process           │
│  File: vllm/v1/engine/core.py                                 │
│                                                              │
│  Main loop (simplified):                                      │
│  while True:                                                  │
│    1. Receive new requests from AsyncLLM                      │
│    2. scheduler.schedule() → decide what to run               │
│    3. model_executor.execute(scheduler_output) → GPU work     │
│    4. scheduler.update(model_output) → process results        │
│    5. Send outputs back to AsyncLLM                           │
└─────────┬────────────────────────────┬───────────────────────┘
          │                            │
          ▼                            ▼
┌─────────────────────┐  ┌────────────────────────────────────┐
│    Scheduler (v1)    │  │      Model Executor / Worker       │
│                     │  │                                    │
│ File: vllm/v1/core/ │  │ File: vllm/v1/worker/              │
│   sched/scheduler.py│  │   gpu_model_runner.py              │
│                     │  │                                    │
│ Manages:            │  │ Executes on GPU:                   │
│ - Request queues    │  │ - Prefill (process prompt)         │
│ - KV cache blocks   │  │ - Decode (generate tokens)         │
│ - Priority ordering │  │ - Multimodal encoding              │
│ - Preemption        │  │ - Sampling                         │
│ - Chunked prefill   │  │                                    │
└─────────┬───────────┘  └─────────────────┬──────────────────┘
          │                                │
          ▼                                ▼
┌─────────────────────┐  ┌────────────────────────────────────┐
│  KV Cache Manager   │  │        GPU Model Runner             │
│                     │  │                                    │
│ File: vllm/v1/core/ │  │ Loads and runs the actual model:   │
│  kv_cache_manager.py│  │ - Qwen3VLForConditionalGeneration  │
│                     │  │ - Attention with PagedAttention     │
│ Block-based alloc:  │  │ - Flash Attention backend           │
│ - Prefix caching    │  │ - Tensor Parallel sharding          │
│ - Block sharing     │  │                                    │
│ - Eviction policy   │  │ File: vllm/model_executor/models/  │
│                     │  │   qwen3_vl.py                      │
└─────────────────────┘  └────────────────────────────────────┘
```

### Request Lifecycle (Step by Step)

Here's exactly what happens when you send a Qwen3-VL request with an image:

```
1. HTTP POST to /v1/chat/completions with image_url + text prompt
   │
   ▼
2. API server (api_server.py) validates request, creates ChatCompletionRequest
   │
   ▼
3. AsyncLLM.generate() is called:
   a. Tokenizer converts text to token IDs
   b. Image is downloaded and preprocessed:
      - Resized to fit within max_pixels (e.g., 1024x1024)
      - Converted to tensor of shape [3, H, W]
      - <image> placeholder tokens inserted into token sequence
   c. Request wrapped as EngineCoreRequest
   d. Sent to EngineCore via ZMQ IPC
   │
   ▼
4. EngineCore receives request and adds to scheduler queue
   │
   ▼
5. Scheduler.schedule() runs:
   a. Checks available KV cache blocks
   b. Decides which requests to prefill vs decode
   c. For Qwen3-VL: allocates blocks for both text and image tokens
   d. Returns SchedulerOutput with:
      - Which requests to run
      - Token budgets (chunked prefill may split long prompts)
      - KV cache block assignments
   │
   ▼
6. GPUModelRunner.execute_model(scheduler_output):
   a. Vision Encoder processes image:
      - Patches extracted, embedded
      - 27-layer ViT transformer runs
      - Features extracted at layers 8, 16, 24 (DeepStack)
      - Spatial merge reduces tokens 4x
   b. DeepStack Fusion:
      - ViT layer 8 features → MLP → add to LLM layer 1
      - ViT layer 16 features → MLP → add to LLM layer 2
      - ViT layer 24 features → MLP → add to LLM layer 3
   c. LLM Forward Pass:
      - For prefill: process all prompt tokens in parallel
      - For decode: process one new token per request
      - Attention uses Flash Attention with paged KV cache
      - Tensor Parallel: weights split across GPUs, all-reduce after each layer
   d. Sampling:
      - Logits → temperature scaling → top-p/top-k → sample token
      - Check for stop conditions (EOS token, max_tokens)
   │
   ▼
7. Output sent back through EngineCore → AsyncLLM → API server → HTTP response
   (For streaming: each token sent as SSE event)
```

### Key Code Files

| Component | File Path | Purpose |
|-----------|-----------|---------|
| API Server | `vllm/entrypoints/openai/api_server.py` | HTTP endpoints |
| AsyncLLM | `vllm/v1/engine/async_llm.py` | Async request management |
| EngineCore | `vllm/v1/engine/core.py` | Central orchestrator |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | Request scheduling |
| KV Cache Manager | `vllm/v1/core/kv_cache_manager.py` | Block-based KV cache |
| GPU Model Runner | `vllm/v1/worker/gpu_model_runner.py` | Model execution |
| Qwen3-VL Model | `vllm/model_executor/models/qwen3_vl.py` | Model implementation |
| Multi-Modal Processing | `vllm/v1/engine/mm_input_cache.py` | Image/video preprocessing |
| Attention Backend | `vllm/attention/backends/flash_attn.py` | Flash Attention |

---

## 5. Feature 1: Async Chunk Pipeline Overlap

### What Problem Does It Solve?

In multimodal models like Qwen3-Omni that have multiple stages (Thinker LLM → Talker LLM → DiT Vocoder), **each stage was waiting for the previous stage to fully complete before starting**. This creates idle GPU time:

```
WITHOUT Pipeline Overlap:
Stage 1 (Thinker): [████████████████]
Stage 2 (Talker):                     [████████████████]
Stage 3 (DiT):                                           [████████████████]
Total time: ================================================

WITH Async Chunk Pipeline Overlap:
Stage 1 (Thinker): [████████████████]
Stage 2 (Talker):       [████████████████]    ← starts as soon as first chunk ready
Stage 3 (DiT):               [████████████████]
Total time: ===================================  (much shorter!)
```

### How It Works (Engineer View)

The key insight is that autoregressive models produce tokens one at a time. Stage 2 doesn't need ALL of Stage 1's output — it can start processing as soon as the first "chunk" of tokens is available.

**vLLM-Omni's Stage Graph Abstraction:**
- The model pipeline is decomposed into a **directed graph** of stages
- **Nodes** = model stages (each can be an LLM or DiT engine)
- **Edges** = user-defined functions that transform and route intermediate data
- Each stage runs its own **independent request batching**
- **Inter-stage connectors** stream data between stages

```python
# Conceptual stage graph for Qwen3-Omni (3-stage pipeline):
#
# Stage 1: Thinker (LLM)
#   Input: text + image/audio tokens
#   Output: text tokens (streamed chunk by chunk)
#        │
#        ▼ (edge: route text tokens)
# Stage 2: Talker (LLM)
#   Input: text tokens from Thinker (consumed as they arrive)
#   Output: audio codec tokens (streamed chunk by chunk)
#        │
#        ▼ (edge: route audio tokens)
# Stage 3: DiT Vocoder
#   Input: audio codec tokens
#   Output: waveform audio samples
```

### Business Use Case

**Voice AI Assistants**: A customer calls your AI support line. Without pipeline overlap, they wait 3+ seconds for the AI to start speaking. With overlap, the first audio chunk plays back within ~0.6 seconds (RTF 0.60), creating a natural conversational experience.

**Result**: Up to **91.4% reduction in job completion time** for Qwen3-Omni audio generation.

---

## 6. Feature 2: SharedFusedMoE for Qwen3-Omni

### What Problem Does It Solve?

Qwen3-Omni uses a **Mixture-of-Experts (MoE)** architecture where each token is routed to a subset of "expert" neural networks. The standard FusedMoE kernel already fuses the gating + expert computation into one GPU kernel. But **SharedFusedMoE** solves an additional problem: **shared experts**.

In Qwen3-Omni's MoE architecture:
- There are **shared experts** (always activated for every token) AND **routed experts** (selectively activated based on the gating network)
- Without SharedFusedMoE: you run the shared experts separately from the routed experts = 2 kernel launches, 2 memory round trips
- With SharedFusedMoE: **fuse shared + routed experts into a single kernel** = 1 kernel launch, 1 memory round trip

### How It Works (From the Actual Code)

**File**: `vllm/model_executor/layers/fused_moe/fused_moe.py`

The actual vLLM codebase contains the FusedMoE implementation. Here's the key concept:

```python
# Standard MoE forward pass (simplified):
class MoELayer:
    def forward(self, hidden_states):
        # Step 1: Gate network decides which experts to use
        router_logits = self.gate(hidden_states)  # [batch, num_experts]
        routing_weights = softmax(topk(router_logits, k=top_k))

        # Step 2: Route tokens to experts
        # Each token goes to top_k experts (e.g., top_k=2)
        expert_outputs = []
        for expert_idx in range(num_experts):
            tokens_for_this_expert = hidden_states[routing_mask[:, expert_idx]]
            expert_output = self.experts[expert_idx](tokens_for_this_expert)
            expert_outputs.append(expert_output)

        # Step 3: Shared expert (always activated)
        shared_output = self.shared_expert(hidden_states)

        # Step 4: Combine
        output = weighted_sum(expert_outputs) + shared_output
        return output

# SharedFusedMoE: fuses Steps 2+3 into a SINGLE GPU kernel
# - The shared expert weights are prepended to the expert weight tensor
# - All tokens are routed to the shared expert (weight=1.0) PLUS their top-k routed experts
# - One kernel launch instead of separate shared + routed passes
```

### Why This Matters on B200 and H100

MoE models are **memory-bandwidth bound** during decode (each token only activates a subset of weights, so compute is low but memory reads are high). Fusing shared + routed experts into one kernel:

- **H100**: Reduces kernel launch overhead and HBM3 read round trips. Each HBM3 round trip costs ~nanoseconds but at 3.35 TB/s, every microsecond matters when you're generating hundreds of tokens/sec.
- **B200**: Even more impactful because the 7.7 TB/s bandwidth means the GPU can burn through memory reads faster, making kernel launch overhead a larger fraction of total time.

### Business Use Case

**Any Qwen3-Omni deployment** (text + image + audio understanding/generation). SharedFusedMoE directly reduces per-token latency → faster response times → better user experience. This translates to:
- Lower cost per query (GPU-seconds per request decreases)
- Higher throughput (more concurrent users per GPU)

---

## 7. Feature 3: TeaCache for Z-Image and Bagel

### What Problem Does It Solve?

Diffusion models generate images through **iterative denoising** — they start with random noise and progressively refine it over 20-50 steps. At each step, the diffusion transformer (DiT) runs a full forward pass. But here's the insight: **adjacent denoising steps produce very similar intermediate activations**. TeaCache exploits this redundancy.

### How It Works

**TeaCache** (Timestep-Aware Caching) identifies which transformer layers' outputs barely change between consecutive denoising steps and **skips recomputing them**, using the cached output from the previous step instead.

```
Standard Diffusion (50 denoising steps):
Step 1: [Layer1 → Layer2 → ... → Layer28]  Full computation
Step 2: [Layer1 → Layer2 → ... → Layer28]  Full computation
Step 3: [Layer1 → Layer2 → ... → Layer28]  Full computation
...
Step 50: [Layer1 → Layer2 → ... → Layer28]  Full computation
Total: 50 × 28 = 1400 layer computations

TeaCache Diffusion (50 denoising steps):
Step 1: [Layer1 → Layer2 → ... → Layer28]  Full computation (cache all outputs)
Step 2: [Layer1 → SKIP → ... → Layer28]    Only recompute layers that changed significantly
Step 3: [Layer1 → SKIP → ... → SKIP]       Even more layers can be skipped
...
Total: ~700 layer computations (50% reduction)
```

The "Timestep-Aware" part means the caching policy adapts based on where you are in the denoising schedule:
- **Early steps** (high noise): features change rapidly → cache less aggressively
- **Middle steps**: features stabilize → cache more aggressively
- **Late steps** (low noise): fine details being added → may need more recomputation

### Models Supported

- **Z-Image**: An image generation model (likely ZhipuAI's)
- **Bagel**: ByteDance's 2-stage image generation model (Thinker/AR → Diffusion/DiT)

### Performance Impact

From the vLLM-Omni paper:
- **Bagel text-to-image (1024x1024)**: 23.12s → 9.64s (**2.40x speedup**)
- **Bagel image-to-image (1024x1024)**: 41.39s → 11.12s (**3.72x speedup**)

The 3.72x speedup for image-to-image is even larger because the conditioning image provides a better starting point, allowing more aggressive caching.

### Business Use Case

**E-commerce product image generation**: Generate product photos from descriptions. At 23 seconds per image, generating 10,000 product images takes 64 hours. At 9.6 seconds, it takes 27 hours. That's **37 hours saved** per batch.

**Creative tools**: Real-time image editing where users expect sub-10-second feedback loops.

---

## 8. Feature 4: Sequence Parallelism (SP) for Diffusion

### What Problem Does It Solve?

Diffusion models process images as sequences of patches. A 1024x1024 image at patch size 2 = 262,144 patches. That's a very long sequence. **Sequence Parallelism** splits this long sequence across multiple GPUs, so each GPU only processes a portion.

### How It Works

There are two main approaches to sequence parallelism, and vLLM-Omni uses **Ulysses SP** (from DeepSpeed):

```
Standard (no SP) — 1 GPU processes all patches:
GPU 0: [patch_0, patch_1, patch_2, ..., patch_262143]
        ↓ Full attention (O(n²) memory) — may OOM!

Ulysses Sequence Parallelism — 4 GPUs split the sequence:
GPU 0: [patch_0 ... patch_65535]
GPU 1: [patch_65536 ... patch_131071]
GPU 2: [patch_131072 ... patch_196607]
GPU 3: [patch_196608 ... patch_262143]
        ↓ Each GPU computes attention on its local chunk
        ↓ All-to-all communication for cross-chunk attention
        ↓ Each GPU has 4x less memory usage
```

**Key difference from Tensor Parallelism:**
- **TP** splits the model weights across GPUs (each GPU has all tokens but partial weights)
- **SP** splits the token sequence across GPUs (each GPU has all weights but partial tokens)
- They can be **combined**: TP for the FFN layers, SP for the attention layers

### Actual Code Location

From the vLLM codebase, sequence parallelism for diffusion models is implemented in the attention layers of DiT models. The key files:

- `vllm/model_executor/layers/` — contains the parallel attention implementations
- Models like `LongCatImageTransformer` and `Wan2.2` have SP support added in v0.14.0

### GPU Impact

**H100 (8-GPU node, NVLink 900 GB/s):**
- The all-to-all communication for SP requires ~2x the data volume of TP's all-reduce
- At 900 GB/s NVLink: a 1GB activation transfer takes ~1.1ms
- SP=4 reduces per-GPU memory from 262K × hidden_dim to 65K × hidden_dim
- Enables generation of higher-resolution images (2048x2048+)

**B200 (8-GPU node, NVLink 1800 GB/s):**
- 2x NVLink bandwidth cuts communication overhead in half
- Same 1GB transfer: ~0.55ms
- Practical implication: SP=8 becomes viable (nearly linear scaling) on B200 where SP=8 would be communication-bottlenecked on H100

### Business Use Case

**High-resolution image generation**: Marketing teams need 4K product renders. Without SP, a single GPU runs out of memory at ~2048x2048. With SP across 4 GPUs, 4K (4096x4096) images become feasible.

**Video generation**: Wan2.2 generates video frames. Each frame is a sequence of patches, and videos are long sequences of frames. SP makes long video generation possible.

---

## 9. Feature 5: Torch Compile for Diffusion

### What Problem Does It Solve?

Python and PyTorch's eager execution mode has overhead: every operation is dispatched individually to the GPU, with Python interpreter overhead between operations. `torch.compile` uses a JIT compiler to analyze the computation graph and fuse operations, eliminating this overhead.

### How It Works

```python
# Without torch.compile — eager mode:
# Each line is a separate GPU kernel launch
def dit_block(x, cond):
    norm_x = rms_norm(x)           # Kernel 1: read x, write norm_x
    qkv = linear(norm_x)           # Kernel 2: read norm_x, write qkv
    q, k, v = split(qkv)           # Kernel 3: memory operation
    attn = flash_attention(q, k, v) # Kernel 4: attention
    x = x + attn                    # Kernel 5: residual add
    norm_x = rms_norm(x)           # Kernel 6: another norm
    ff = mlp(norm_x)               # Kernel 7-9: up_proj, act, down_proj
    x = x + ff                      # Kernel 10: residual add
    return x
# Total: ~10 kernel launches, ~10 HBM round trips

# With torch.compile — fused mode:
@torch.compile
def dit_block(x, cond):
    # Same code, but the compiler fuses operations:
    # Kernel 1: rms_norm + linear (fused, no intermediate write)
    # Kernel 2: flash_attention (already optimized)
    # Kernel 3: residual + rms_norm + mlp + residual (fused)
    return x
# Total: ~3 kernel launches, ~3 HBM round trips
```

### Specific Benefits for Diffusion

Diffusion models are particularly good candidates for torch.compile because:
1. **Iterative execution**: The same DiT block runs 20-50 times (denoising steps), so compilation cost is amortized
2. **Regular computation graph**: No dynamic control flow (unlike LLMs with variable-length generation)
3. **Compute-heavy**: High arithmetic intensity means kernel fusion has big impact

### GPU Impact

**H100**: torch.compile can reduce kernel launch overhead by 30-50% for DiT models. With 3.35 TB/s bandwidth, reducing HBM round trips from 10 to 3 per block saves ~microseconds per block × 28 blocks × 50 steps = measurable latency reduction.

**B200**: Even more impactful because the 7.7 TB/s bandwidth means the GPU can execute fused kernels faster, and the larger 228KB shared memory per SM allows more data to stay on-chip during fused operations.

### Business Use Case

**Free performance**: No algorithmic changes required. Just enabling torch.compile on existing diffusion workloads gives 20-40% speedup. This directly reduces GPU-hours per generated image → lower cloud computing costs.

---

## 10. Feature 6: Qwen3-TTS with Online Serving

### What It Is

Qwen3-TTS is Alibaba's text-to-speech model from the Qwen3 family. vLLM-Omni v0.14.0 adds **full online serving support**, meaning you can deploy Qwen3-TTS as a real-time API endpoint that converts text to speech audio.

### How It Works

```
Text Input: "Hello, how can I help you today?"
     │
     ▼
Stage 1: Text Encoder (LLM)
  - Tokenizes text
  - Generates semantic tokens representing speech content
  - Captures prosody, emphasis, emotion from text context
     │
     ▼
Stage 2: Audio Codec / Vocoder
  - Converts semantic tokens to audio codec tokens
  - Generates mel-spectrogram
  - Produces raw audio waveform (WAV format)
     │
     ▼
Output: Audio stream (can be streamed chunk by chunk for low latency)
```

### Pipeline Integration with vLLM-Omni

Using the **stage graph** abstraction, Qwen3-TTS is decomposed into stages that can run on different GPUs with async chunk pipeline overlap:

```
# Qwen3-TTS pipeline:
# Stage 1: Text → Semantic Tokens (LLM engine)
# Stage 2: Semantic Tokens → Audio (Vocoder/DiT engine)
#
# With pipeline overlap, Stage 2 starts generating audio
# as soon as the first semantic token chunk is ready
```

### Real-Time Factor (RTF)

The key metric for TTS is **Real-Time Factor**: the ratio of processing time to generated audio duration.
- RTF < 1.0 = real-time (audio is generated faster than it plays)
- RTF = 0.60 (from the paper) = audio is generated **1.67x faster** than playback speed

This means: generating 10 seconds of speech takes only 6 seconds of compute time.

### Business Use Cases

1. **Call Centers**: Replace hold music with AI agents that speak naturally. RTF 0.60 means the AI responds faster than a human can speak.
2. **Content Creation**: Audiobook narration, podcast generation, video voiceover
3. **Accessibility**: Real-time screen readers for visually impaired users
4. **Multilingual Customer Service**: Qwen3-TTS supports multiple languages, deploy once for global markets

---

## 11. Feature 7: Diffusion LoRA (PEFT-Compatible)

### What It Is

**LoRA (Low-Rank Adaptation)** is a technique for fine-tuning models without modifying the original weights. You add small trainable matrices (adapters) alongside the frozen base model. **Diffusion LoRA** brings this capability to diffusion models (image/video generators) in vLLM-Omni.

### How It Works

```
Original Weight Matrix W (frozen, e.g., 4096 × 4096 = 67M parameters):
    output = W @ input

LoRA Adapter (trainable, rank=16):
    A: 4096 × 16 = 65K parameters
    B: 16 × 4096 = 65K parameters
    Total: 130K parameters (0.2% of original!)

Combined:
    output = W @ input + (B @ A) @ input × scaling_factor
```

### PEFT Compatibility

"PEFT-compatible" means the LoRA adapters follow the HuggingFace PEFT format, so you can:
1. Train LoRA adapters using the standard PEFT library
2. Share adapters on HuggingFace Hub
3. Load them directly into vLLM-Omni without conversion

### Serving Multiple LoRAs

vLLM already supports serving multiple LoRA adapters simultaneously for LLMs. Now this extends to diffusion:

```python
# Conceptual: serve base FLUX model + multiple style LoRAs
# Request 1: "Generate a photo of a cat" + lora_adapter="anime_style"
# Request 2: "Generate a photo of a dog" + lora_adapter="oil_painting"
# Request 3: "Generate a photo of a house" + lora_adapter=None (base model)
#
# All three requests batched together, each with different LoRA weights
# LoRA weights are swapped per-request without reloading the base model
```

### Business Use Cases

1. **Brand-Specific Image Generation**: Train a LoRA on your brand's visual style (colors, composition, aesthetics). Deploy one FLUX base model + per-brand LoRA adapters. Cost: fine-tune once for $50-200 instead of training a full model for $50,000+.

2. **Multi-Tenant SaaS**: Offer image generation to 100 customers, each with their own style. One base model (loaded once on GPU) + 100 LoRA adapters (stored in CPU memory, loaded per-request). GPU memory cost: ~64GB base model + ~100MB per adapter.

3. **A/B Testing Creative Assets**: Marketing team trains LoRA-A (warm tones) and LoRA-B (cool tones), serves both in production, measures click-through rates.

---

## 12. Feature 8: DiT Layerwise CPU Offloading

### What Problem Does It Solve?

Diffusion Transformers (DiTs) are large models. FLUX.1-dev has ~12B parameters (~24GB in BF16). During the denoising loop, only one layer is active at a time, but all layers' weights sit in GPU memory. **CPU offloading** moves inactive layers to CPU RAM and streams them to GPU just-in-time.

### How It Works

```
WITHOUT CPU Offloading:
GPU Memory: [Layer0][Layer1][Layer2]...[Layer27] = 24GB constant
             ↑ Only one layer active at a time, rest wasting GPU memory

WITH Layerwise CPU Offloading:
GPU Memory: [Layer_i][Layer_i+1]  = ~2GB at any time (current + prefetched next)
CPU Memory: [Layer0]...[Layer27]  = 24GB in system RAM (cheap!)

Timeline:
Step 1: GPU runs Layer 0, meanwhile CPU→GPU transfer of Layer 1
Step 2: GPU runs Layer 1, meanwhile CPU→GPU transfer of Layer 2
...
Step 27: GPU runs Layer 27, meanwhile CPU→GPU transfer of Layer 0 (next denoise step)

The transfer and compute are overlapped using CUDA streams:
  Compute Stream: [Layer0_compute][Layer1_compute][Layer2_compute]...
  Transfer Stream:     [Layer1_xfer][Layer2_xfer][Layer3_xfer]...
```

### Memory Savings

| Scenario | GPU Memory Without Offload | GPU Memory With Offload | Savings |
|----------|---------------------------|------------------------|---------|
| FLUX.1-dev (12B, BF16) | 24GB | ~2-4GB | **6-12x** |
| Bagel DiT stage | ~20GB | ~2-4GB | **5-10x** |
| GLM-Image | ~16GB | ~2-4GB | **4-8x** |

### GPU Impact

**H100 (80GB HBM3, PCIe Gen5 128 GB/s):**
- Layer transfer at PCIe Gen5: ~1GB layer transferred in ~8ms
- Layer compute time: ~5-20ms depending on sequence length
- If compute time > transfer time: **zero overhead** (fully overlapped)
- Frees up ~20GB GPU memory for larger batch sizes or higher resolution

**B200 (180GB HBM3e, PCIe Gen5 128 GB/s):**
- Same PCIe bandwidth, but B200 has so much memory that offloading is less critical
- Still useful for running **multiple diffusion models simultaneously** on one GPU
- Or for running diffusion + LLM on the same GPU (hybrid serving)

### Business Use Case

**Cost Reduction**: Run FLUX.1-dev on a single 24GB GPU (RTX 4090, L4) instead of requiring an 80GB A100/H100. Cloud cost difference: ~$1/hr (L4) vs ~$3/hr (A100) = **67% cost reduction**.

**Multi-Model Serving**: On an H100, run both Qwen3-VL (text+vision) AND FLUX.1-dev (image generation) on the same GPU. Without offloading: 64GB + 24GB = 88GB (doesn't fit in 80GB). With offloading: 64GB + 4GB = 68GB (fits!).

---

## 13. Feature 9: New Models (Bagel, FLUX, GLM-Image, Stable Audio)

### Bagel (ByteDance)

**What**: A multimodal model that can both understand and generate images. Uses a 2-stage pipeline: Thinker/AR (autoregressive reasoning) + Diffusion/DiT (image generation).

**Architecture**:
```
Stage 1: Thinker (Autoregressive LLM)
  - Takes text + optional input image
  - Reasons about what to generate
  - Outputs conditioning tokens

Stage 2: DiT (Diffusion Transformer)
  - Takes conditioning tokens from Thinker
  - Iteratively denoises random noise into image
  - Supports 1024x1024 and higher resolutions
```

**Performance in vLLM-Omni**:
- Text-to-Image: **9.64s** (vs 23.12s baseline = 2.40x faster)
- Image-to-Image: **11.12s** (vs 41.39s baseline = 3.72x faster)

**Business Use**: E-commerce product image generation, marketing content creation

### FLUX.1-dev and FLUX.2-klein (Black Forest Labs)

**What**: State-of-the-art text-to-image diffusion models.
- FLUX.1-dev: ~12B parameter model, high quality, production-ready
- FLUX.2-klein: Smaller, faster variant for real-time applications

**Day-0 Support**: Available from the release date — no waiting for community patches.

**Business Use**: Creative agencies, marketing automation, personalized ad generation

### GLM-Image (Zhipu AI)

**What**: Image generation model from Zhipu AI (makers of GLM/ChatGLM series).

**Day-0 Support**: Available from release.

**Business Use**: Chinese market image generation, bilingual content creation

### Stable Audio Open (Stability AI)

**What**: Open-source audio generation model. Generates music, sound effects, and ambient audio from text descriptions.

**Business Use**: Game audio, video production, podcast intros, background music generation

---

## 14. Feature 10: New APIs

### /v1/images/edit — Image Editing Endpoint

**What**: A new API endpoint for image-to-image editing, following the OpenAI Images API format.

```bash
# Example: Edit an existing product photo
curl -X POST http://localhost:8000/v1/images/edit \
  -H "Content-Type: multipart/form-data" \
  -F "image=@product_photo.png" \
  -F "prompt=Change the background to a white studio backdrop" \
  -F "model=bagel" \
  -F "size=1024x1024"
```

**Business Use**:
- Product photography post-processing at scale
- Real estate virtual staging
- Fashion try-on applications

### /health — Health Check for Diffusion Mode

```bash
# Check if the diffusion server is ready
curl http://localhost:8000/health
# Response: {"status": "ok"}
```

**Why It Matters**: Load balancers (Kubernetes, nginx, ALB) need health check endpoints to route traffic. Without this, diffusion mode deployments couldn't be properly health-checked.

### /v1/models — Model Listing for Diffusion Mode

```bash
# List available models
curl http://localhost:8000/v1/models
# Response: {"data": [{"id": "flux-1-dev", "object": "model", ...}]}
```

**Why It Matters**: Multi-model deployments need discovery. Clients can query which models are available before sending requests.

---

## 15. Feature 11: XPU / ROCm / NPU Backend Support

### XPU Backend (Intel GPUs)

**What**: Support for Intel Data Center GPUs (Ponte Vecchio, Flex series) via Intel's XPU platform.

**Why It Matters**: Enterprises with Intel GPU deployments can now run vLLM-Omni workloads. Reduces vendor lock-in to NVIDIA.

### ROCm Backend (AMD GPUs)

**What**: Enhanced support for AMD Instinct GPUs (MI300X, MI250) via ROCm.

**v0.14.0 Improvements**:
- **AITER FlashAttention**: AMD's optimized attention kernel
- **AITER RMSNorm fusion**: Fused normalization for AMD GPUs
- **MTP support for AITER MLA**: Multi-Token Prediction with AMD's Multi-Layer Attention
- **CI expansion**: More ROCm tests in continuous integration
- **SDPA attention mask semantics fixes**: Correctness fixes for scaled dot-product attention

**Why It Matters**: AMD MI300X has 192GB HBM3 memory — even more than B200. For memory-constrained workloads like serving 235B models, MI300X is competitive.

### NPU Backend (Neural Processing Units)

**What**: Support for dedicated neural processing hardware (e.g., Huawei Ascend NPUs).

**Why It Matters**: Chinese market deployments where NVIDIA GPUs may be restricted. Ascend 910B is widely deployed in Chinese data centers.

### Business Impact

| Hardware | Memory | Compute | Use Case |
|----------|--------|---------|----------|
| NVIDIA H100 | 80GB HBM3 | 3,958 TFLOPS FP8 | Primary inference GPU |
| NVIDIA B200 | 180GB HBM3e | 9,000 TFLOPS FP8 | Next-gen, highest perf |
| AMD MI300X | 192GB HBM3 | ~2,600 TFLOPS FP8 | Memory-constrained workloads |
| Intel Max 1550 | 128GB HBM2e | ~840 TFLOPS BF16 | Intel ecosystem |
| Huawei Ascend 910B | 64GB HBM2e | ~320 TFLOPS FP16 | China-market deployments |

---

## 16. Feature 12: Decode Context Parallel (DCP)

### What Problem Does It Solve?

During the **decode phase** of LLM inference, each new token must attend to ALL previous tokens in the KV cache. For long contexts (128K+ tokens), this attention computation becomes a bottleneck. **Decode Context Parallel (DCP)** splits the KV cache across multiple GPUs and parallelizes the attention computation.

### How It Works

```
Standard Decode (1 GPU):
KV Cache: [tok_0, tok_1, ..., tok_128000]  ← All on one GPU
New token attends to all 128K cached tokens
Attention compute: O(128K × hidden_dim) on one GPU

Decode Context Parallel (4 GPUs):
GPU 0: KV Cache [tok_0 ... tok_32000]     → local attention score
GPU 1: KV Cache [tok_32001 ... tok_64000]  → local attention score
GPU 2: KV Cache [tok_64001 ... tok_96000]  → local attention score
GPU 3: KV Cache [tok_96001 ... tok_128000] → local attention score
                    ↓ All-reduce to combine attention scores
              Final attention output
```

### From the Actual vLLM Code

**File**: `vllm/v1/core/sched/scheduler.py` and `vllm/distributed/`

The vLLM v1 scheduler is aware of context parallelism and assigns KV cache blocks across the CP group:

```python
# Conceptual flow in the scheduler:
# 1. When a request has a very long context, enable CP
# 2. Assign KV cache blocks to different CP ranks
# 3. During decode, each rank computes attention on its local KV blocks
# 4. All-reduce combines the partial attention outputs

# The key config parameter:
# --decode-context-parallel-size=N (or DCP=N)
```

### Difference from Tensor Parallelism

| Aspect | Tensor Parallelism (TP) | Decode Context Parallel (DCP) |
|--------|------------------------|-------------------------------|
| **What's split** | Model weights | KV cache |
| **When it helps** | Always (compute + memory) | Long contexts (128K+) |
| **Communication** | All-reduce per layer | All-reduce for attention only |
| **Memory savings** | Linear with TP degree | Linear with DCP degree |
| **Combinable** | Yes, with DCP | Yes, with TP |

### GPU Impact

**H100 with DCP=4 on 128K context**:
- Each GPU stores 32K KV entries instead of 128K
- Attention bandwidth: 3.35 TB/s × 4 GPUs = 13.4 TB/s aggregate
- ~4x speedup in attention computation for long contexts

**B200 with DCP=4 on 128K context**:
- 7.7 TB/s × 4 = 30.8 TB/s aggregate bandwidth
- B200's 1.8 TB/s NVLink means DCP communication is 2x faster than H100
- Combined: ~4x attention speedup with lower communication overhead

### Business Use Case

**Document AI**: Legal firms processing 100-page contracts through Qwen3-VL. A 100-page document with images can easily reach 128K tokens. Without DCP, decode latency per token is unacceptable. With DCP=4, each token generates 4x faster.

**Video Understanding**: Qwen3-VL processing long videos (minutes to hours). Video frames generate massive token sequences. DCP makes real-time video analysis feasible.

---

## 17. MLPerf Inference v6.0: Qwen3-VL + Shopify Benchmark

### Overview

**MLPerf Inference v6.0** is the latest round of the industry-standard inference benchmark suite, run by MLCommons. This round introduces the **first Vision-Language Model (VLM) benchmark**, using Qwen3-VL + a Shopify product catalog dataset.

**Submission Deadline**: February 13, 2026

### The Model

**Qwen3-VL-235B-A22B-Instruct**:
- Mixture-of-Experts vision-language model
- **235 billion total parameters, 22 billion activated per token**
- 256K token context window
- Supports interleaved text, image, and video inputs
- DeepStack fusion + Interleaved-MRoPE

### The Dataset: Shopify Product Catalog

Curated in partnership with **Shopify**, mirroring their production workload of processing **40 million products daily**.

**Task**: Hierarchical Taxonomy Classification
- Input: Product title + description + photo
- Output: JSON with `category` (hierarchical path), `brand`, `is_secondhand`
- Example:
```json
{
  "input": {
    "title": "Nike Air Max 90 - Size 10",
    "description": "Classic sneaker in white/black colorway...",
    "image": "<product_photo.jpg>"
  },
  "expected_output": {
    "category": "Apparel & Accessories > Shoes > Athletic Shoes > Sneakers",
    "brand": "Nike",
    "is_secondhand": false
  }
}
```

### Accuracy Requirements

- **Category Hierarchical F1 >= 0.7824** (99% of reference score 0.7903)
- Sampling: `temperature=1.0, top_p=1.0, top_k=disabled`

### Benchmark Scenarios

| Scenario | Metric | Constraint | Real-World Analog |
|----------|--------|------------|-------------------|
| **Offline** | Requests/second throughput | Process all samples at least once | Shopify's daily batch of 40M products |
| **Server** | Max QPS under latency SLA | **p99 latency <= 12 seconds** | Black Friday real-time processing |

### Hardware Requirements

- Minimum: **8 GPUs with >= 80GB memory each** (A100, H100, or H200)
- On H200 and B200: runs out-of-the-box at full context length
- 8xH100 (80GB): Tight fit, may need quantization
- 8xB200 (180GB): Comfortable fit with room for large batch sizes

### Why vLLM is the Reference Implementation

The reference implementation uses vLLM because:
1. **OpenAI-compatible API**: Any inference system exposing `/v1/chat/completions` can be benchmarked
2. **Industry standard**: vLLM is the most widely deployed open-source inference engine
3. **Multi-GPU support**: Built-in tensor parallelism for the 235B MoE model
4. **Production-proven**: Already deployed at scale by Shopify and others

### What This Means for Business

1. **Shopify processes 40M products/day** with VLM-based classification — this is a proven production use case
2. **MLPerf standardization** means you can compare hardware vendors objectively
3. **The p99 <= 12 second SLA** is realistic for production workloads
4. Hardware vendors (NVIDIA, AMD, Intel, etc.) will publish optimized results → you can pick the best price/performance

---

## 18. vLLM-Omni Paper (arXiv:2602.02204)

### Citation

Peiqi Yin et al., "vLLM-Omni: Fully Disaggregated Serving for Any-to-Any Multimodal Models," arXiv:2602.02204, February 2, 2026.

### Problem Statement

Any-to-any multimodal models (handling text, images, video, audio simultaneously) combine multiple LLMs, diffusion transformers, and specialized components. Existing serving systems are tailored to single paradigms. Developers must manually handle cross-stage interactions, leading to massive performance degradation.

### Core Contribution: Stage Graph Abstraction

```
Example: Qwen3-Omni Pipeline as a Stage Graph

         ┌──────────────┐
         │  Text/Image   │
         │   Input       │
         └──────┬───────┘
                │
                ▼
    ┌───────────────────────┐
    │ Stage 1: Thinker (LLM) │  ← Independent LLM engine
    │ GPU allocation: 4 GPUs │  ← Per-stage GPU assignment
    │ Batch size: dynamic    │  ← Independent batching
    └───────────┬───────────┘
                │ Edge: text token routing
                ▼
    ┌───────────────────────┐
    │ Stage 2: Talker (LLM)  │  ← Separate LLM engine
    │ GPU allocation: 2 GPUs │
    │ Batch size: dynamic    │
    └───────────┬───────────┘
                │ Edge: audio token routing
                ▼
    ┌───────────────────────┐
    │ Stage 3: DiT Vocoder   │  ← Diffusion engine
    │ GPU allocation: 2 GPUs │
    │ Batch size: dynamic    │
    └───────────────────────┘
                │
                ▼
         ┌──────────────┐
         │ Audio Output  │
         └──────────────┘
```

### Key Features

1. **Nodes** = model stages (each independently served by LLM or diffusion engine)
2. **Edges** = user-defined transformation + routing functions
3. **Per-stage batching**: Each stage batches requests independently
4. **Flexible GPU allocation**: Different stages can use different numbers of GPUs
5. **Unified inter-stage connectors**: Standardized data routing between stages

### Optimization Techniques Used

- **FlashAttention / SAGE Attention / TurboAttention**: Advanced attention kernels
- **TeaCache / cache-dit**: Caching for iterative denoising
- **RingAttention**: Context parallelism for long sequences
- **Ulysses SP**: Sequence parallelism for diffusion
- **SharedFusedMoE**: Fused MoE kernels with shared experts

### Benchmark Results

| Model | Task | Baseline | vLLM-Omni | Speedup |
|-------|------|----------|-----------|---------|
| **Qwen3-Omni** | Audio generation | — | — | **91.4% JCT reduction** |
| **Bagel** | Text-to-Image 1024x1024 | 23.12s | 9.64s | **2.40x** |
| **Bagel** | Image-to-Image 1024x1024 | 41.39s | 11.12s | **3.72x** |
| **MiMo-Audio** | Text-to-Speech | RTF 1.39 | RTF 0.60 | **2.32x** |

### Metrics

- **JCT (Job Completion Time)**: End-to-end latency from request submission to completion
- **RTF (Real-Time Factor)**: Processing time / audio duration (< 1.0 = real-time)
- **TPS (Tokens Per Second)**: For LLM stages

### Baselines Compared Against

- HuggingFace Transformers (for Qwen-Omni)
- Original model implementations (for Bagel, MiMo-Audio)
- Diffusers library (for diffusion components)

---

## 19. Step-by-Step: Qwen3-VL-32B on H100 (Single Node)

### Hardware Requirements

- **Minimum**: 2x H100 80GB SXM (TP=2)
- **Recommended**: 4x H100 80GB SXM (TP=4, more KV cache room)
- System RAM: 256GB+
- NVMe SSD: 200GB+ for model weights

### Step 1: Install vLLM

```bash
# Create a fresh conda environment
conda create -n vllm python=3.12 -y
conda activate vllm

# Install vLLM (latest stable)
pip install vllm

# Verify CUDA and GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
# Expected: CUDA: True, GPUs: 8 (or however many H100s you have)
```

### Step 2: Download the Model

```bash
# Using huggingface-cli (recommended for large models)
pip install huggingface_hub[cli]

# Download Qwen3-VL-32B-Instruct
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct \
  --local-dir /models/Qwen3-VL-32B-Instruct \
  --local-dir-use-symlinks False

# Model size: ~64GB in BF16
# Download time: ~15-30 minutes on fast internet
```

### Step 3: Start the vLLM Server (TP=2)

```bash
# Minimum viable: 2x H100 with tensor parallelism
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --port 8000 \
  --trust-remote-code

# Explanation of flags:
# --tensor-parallel-size 2    : Split model across 2 H100s (32GB per GPU)
# --dtype bfloat16            : Use BF16 precision (best quality/speed tradeoff)
# --max-model-len 32768       : Support up to 32K token contexts
# --gpu-memory-utilization 0.9: Use 90% of GPU memory (72GB per GPU)
#                               32GB weights + 40GB KV cache per GPU
# --trust-remote-code         : Required for Qwen3-VL custom code
```

### Step 4: Optimal Configuration (TP=4, Longer Context)

```bash
# For production: 4x H100 with longer context
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# With TP=4: 16GB weights per GPU, leaving 56GB for KV cache per GPU
# 224GB total KV cache across 4 GPUs → supports 128K context comfortably
# --enable-prefix-caching: reuse KV cache for shared prompt prefixes
# --max-num-seqs 32: process up to 32 concurrent requests
```

### Step 5: Send a Request with an Image

```python
import openai
import base64
import httpx

# Initialize client
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM doesn't require an API key
)

# Option A: Image from URL
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "What species of ant is this? Describe its morphological features in detail."
                }
            ]
        }
    ],
    max_tokens=1024,
    temperature=0.7
)

print(response.choices[0].message.content)

# Option B: Local image file (base64 encoded)
with open("product_photo.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                },
                {
                    "type": "text",
                    "text": "Classify this product. Return JSON with category, brand, and condition."
                }
            ]
        }
    ],
    max_tokens=256,
    temperature=0.0  # Deterministic for classification
)

print(response.choices[0].message.content)
```

### Step 6: Batch Processing (Offline Mode)

```python
import openai
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Process multiple images concurrently
def process_image(image_url: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        max_tokens=256,
        temperature=0.0
    )
    return response.choices[0].message.content

# Process 1000 product images
product_images = [f"https://example.com/products/{i}.jpg" for i in range(1000)]
prompt = "Classify this product into a category hierarchy. Return JSON."

with ThreadPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(process_image, url, prompt) for url in product_images]
    results = [f.result() for f in futures]

print(f"Processed {len(results)} products")
```

### Step 7: Monitor Performance

```bash
# Check server metrics
curl http://localhost:8000/metrics

# Key metrics to watch:
# vllm:num_requests_running     — concurrent requests
# vllm:num_requests_waiting     — queued requests
# vllm:gpu_cache_usage_perc     — KV cache utilization
# vllm:avg_generation_throughput — tokens/sec
# vllm:avg_prompt_throughput     — prefill tokens/sec
```

### Expected Performance on H100

| Configuration | Prefill (tok/s) | Decode (tok/s) | Concurrent Requests |
|---------------|----------------|----------------|---------------------|
| 2xH100, TP=2, 32K ctx | ~4,000 | ~80-120 per req | 8-16 |
| 4xH100, TP=4, 128K ctx | ~8,000 | ~100-150 per req | 16-32 |
| 8xH100, TP=4+DCP=2, 256K ctx | ~12,000 | ~120-180 per req | 32-64 |

---

## 20. Step-by-Step: Qwen3-VL-32B on B200 (Single GPU)

### The B200 Advantage

The B200 has **180GB HBM3e** — enough to fit Qwen3-VL-32B entirely on a **single GPU**. This eliminates all tensor parallelism communication overhead.

### Step 1: Install vLLM (same as H100)

```bash
conda create -n vllm python=3.12 -y
conda activate vllm
pip install vllm
```

### Step 2: Start the vLLM Server (Single B200!)

```bash
# Single B200 — no tensor parallelism needed!
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# Memory breakdown on single B200 (180GB):
# Model weights (BF16): ~64GB
# KV cache (90% of remaining): ~104GB
# Activations + overhead: ~12GB
#
# 104GB KV cache ≈ 52K tokens at BF16 per request
# Or ~1,600 tokens × 64 concurrent requests
# Or ~8,000 tokens × 13 concurrent requests
```

### Step 3: FP8 Quantization for Even More Headroom

```bash
# FP8 quantization: halve the model weight memory
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --dtype bfloat16 \
  --quantization fp8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# Memory breakdown with FP8:
# Model weights (FP8): ~32GB
# KV cache: ~136GB ← massive!
# This supports 256K context with many concurrent requests
#
# B200's 2nd-gen Transformer Engine handles FP8 natively
# Minimal quality loss, 2x compute throughput
```

### Step 4: FP4 Quantization (B200 Exclusive!)

```bash
# FP4 quantization: B200's NVFP4 gives 18 petaFLOPS
# This is NOT available on H100!
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --dtype bfloat16 \
  --quantization fp4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# Memory breakdown with FP4:
# Model weights (FP4): ~16GB
# KV cache: ~148GB
# Activations: ~16GB
#
# FP4 compute: 18 petaFLOPS on B200 (vs 3.96 TFLOPS FP8 on H100)
# That's ~4.5x the compute throughput of H100!
```

### Expected Performance on B200

| Configuration | Prefill (tok/s) | Decode (tok/s) | Concurrent Requests |
|---------------|----------------|----------------|---------------------|
| 1xB200, BF16, 128K ctx | ~6,000 | ~150-200 per req | 32-64 |
| 1xB200, FP8, 256K ctx | ~10,000 | ~250-350 per req | 64-128 |
| 1xB200, FP4, 256K ctx | ~15,000 | ~350-500 per req | 128-256 |
| 2xB200, TP=2, FP8, 256K | ~18,000 | ~400-600 per req | 128-256 |

### B200 vs H100 Cost Comparison (Qwen3-VL-32B)

| Metric | 2xH100 (TP=2) | 1xB200 | B200 Advantage |
|--------|---------------|--------|----------------|
| GPUs needed | 2 | 1 | **50% fewer GPUs** |
| Total GPU memory | 160GB | 180GB | 1.1x |
| NVLink overhead | ~5-10% perf loss | 0% (single GPU) | **No TP overhead** |
| Memory bandwidth | 6.7 TB/s (combined) | 7.7 TB/s | 1.15x |
| FP8 compute | 7,916 TFLOPS | 9,000 TFLOPS | 1.14x |
| Power consumption | 1,400W | 1,000W | **29% less power** |
| Est. cloud cost | ~$6/hr | ~$4-5/hr | **~20-30% cheaper** |

---

## 21. Step-by-Step: Qwen3-VL-235B-A22B on 8xH100 / 8xB200

This is the **MLPerf benchmark model** — the largest Qwen3-VL variant.

### Model Specifications

- **Total parameters**: 235 billion
- **Activated parameters per token**: 22 billion (MoE)
- **Architecture**: Mixture-of-Experts with shared + routed experts
- **Weight size (BF16)**: ~470GB total, ~44GB activated
- **Minimum hardware**: 8 GPUs with >= 80GB each

### On 8xH100 (DGX H100)

```bash
# Step 1: Start server with TP=8
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# Memory breakdown per GPU (8xH100, 80GB each):
# Model weights: 470GB / 8 = ~59GB per GPU
# KV cache: ~13GB per GPU
# Activations: ~8GB per GPU
#
# NOTE: 13GB KV cache is tight — limits context length and concurrency
# With 32K context: ~4-8 concurrent requests

# Step 2: With FP8 quantization for more headroom
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --quantization fp8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# With FP8: ~30GB weights per GPU, ~42GB KV cache per GPU
# Much better — supports 128K context with reasonable concurrency
```

### On 8xB200 (DGX B200)

```bash
# Step 1: BF16 — fits comfortably with massive KV cache
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# Memory breakdown per GPU (8xB200, 180GB each):
# Model weights: 470GB / 8 = ~59GB per GPU
# KV cache: ~104GB per GPU ← massive!
# Activations: ~17GB per GPU
#
# 104GB KV cache per GPU = 832GB total across 8 GPUs
# Supports 128K context with 64+ concurrent requests

# Step 2: FP8 for maximum throughput
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --quantization fp8 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# FP8: ~30GB weights per GPU, ~133GB KV cache per GPU
# 1,064GB total KV cache — supports 256K context easily

# Step 3: FP4 for absolute maximum throughput (B200 exclusive)
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --quantization fp4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --port 8000 \
  --trust-remote-code

# FP4: ~15GB weights per GPU, ~148GB KV cache per GPU
# 18 petaFLOPS per GPU × 8 = 144 petaFLOPS total!
```

### Performance Comparison: 235B Model on 8xH100 vs 8xB200

| Metric | 8xH100 (BF16) | 8xH100 (FP8) | 8xB200 (BF16) | 8xB200 (FP8) | 8xB200 (FP4) |
|--------|---------------|---------------|----------------|---------------|---------------|
| KV cache per GPU | 13GB | 42GB | 104GB | 133GB | 148GB |
| Max context | 32K | 128K | 128K | 256K | 256K |
| Max concurrent reqs | 4-8 | 16-32 | 32-64 | 64-128 | 128-256 |
| Prefill throughput | ~3K tok/s | ~5K tok/s | ~8K tok/s | ~14K tok/s | ~20K tok/s |
| Total system power | 5,600W | 5,600W | 8,000W | 8,000W | 8,000W |
| Est. cloud cost/hr | ~$25 | ~$25 | ~$35 | ~$35 | ~$35 |
| Cost per 1M tokens | ~$2.30 | ~$1.40 | ~$1.20 | ~$0.70 | ~$0.50 |

### Sending Requests to the 235B Model

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# MLPerf-style product classification
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-235B-A22B-Instruct",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a product classification system. "
                "Given a product image and description, return a JSON object with: "
                "category (hierarchical path), brand, is_secondhand (boolean)."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/product.jpg"}
                },
                {
                    "type": "text",
                    "text": "Title: Vintage Levi's 501 Jeans - W32 L30\nDescription: Classic straight-leg jeans, pre-owned, light wash, minor fading on knees."
                }
            ]
        }
    ],
    max_tokens=128,
    temperature=1.0,  # MLPerf requires temperature=1.0
    top_p=1.0          # MLPerf requires top_p=1.0
)

print(response.choices[0].message.content)
# Expected output:
# {
#   "category": "Apparel & Accessories > Clothing > Pants > Jeans",
#   "brand": "Levi's",
#   "is_secondhand": true
# }
```

---

## 22. Business Use Cases by Industry

### E-Commerce (Shopify, Amazon, eBay)

**Problem**: Manually categorizing millions of new product listings daily.

**Solution**: Qwen3-VL-235B-A22B processes product images + descriptions → automatic hierarchical categorization.

**Numbers**:
- Shopify processes **40 million products/day**
- At 12 seconds p99 latency (MLPerf Server scenario)
- 8xB200 cluster: ~128 concurrent requests at 0.5 seconds average
- One cluster handles: 128 × 86,400 / 0.5 = **22 million products/day**
- Two clusters cover Shopify's full volume

**ROI**: Replace 200+ human annotators ($40K/year each = $8M/year) with 2 GPU clusters ($35/hr × 2 × 8,760 hours = $613K/year). **$7.4M annual savings**.

### Healthcare (Radiology, Pathology)

**Problem**: Radiologist shortage — 4+ hour wait times for scan interpretation.

**Solution**: Qwen3-VL-32B analyzes medical images (X-rays, CT scans, MRIs) and generates preliminary reports.

**Deployment**:
- 1xB200 (single GPU) for on-premises hospital deployment
- FP8 quantization → fits in 32GB, leaves 148GB for large DICOM images
- HIPAA-compliant: no data leaves the hospital network

**Compliance Note**: AI-generated reports require radiologist review (FDA regulations). The AI triages and drafts, humans approve.

### Manufacturing (Quality Control)

**Problem**: Visual defect detection on production lines at 60+ items/minute.

**Solution**: Qwen3-VL processes camera feeds in real-time, identifies defects, classifies severity.

**Requirements**:
- Latency: < 500ms per image (line speed: 1 item/second)
- Throughput: 3,600 images/hour per camera, 20 cameras = 72,000/hour
- Accuracy: > 99.5% to avoid false rejections

**Deployment**: 2xH100 with TP=2, Qwen3-VL-8B (smaller model, faster inference). At ~50ms per image with 32 concurrent requests → handles 100K+ images/hour.

### Financial Services (Document AI)

**Problem**: Processing loan applications with mixed document types (pay stubs, bank statements, IDs).

**Solution**: Qwen3-VL-32B with 128K context processes multi-page documents end-to-end.

**Deployment**: 4xH100 with TP=4, or 1xB200. Process complete loan application (10-20 pages) in a single request.

**Numbers**: Average loan application = 15 pages ≈ 20K tokens (text + image tokens). At 32 concurrent requests: process 32 applications simultaneously, ~5-10 seconds each = 11,500-23,000 applications/day per GPU cluster.

### Autonomous Vehicles / Robotics

**Problem**: Scene understanding from camera feeds — identify objects, read signs, understand spatial relationships.

**Solution**: Qwen3-VL with video input processes camera feeds, outputs structured scene descriptions.

**Critical Requirements**: Ultra-low latency (< 100ms). Use Qwen3-VL-2B or 4B for on-device/edge deployment.

### Media & Entertainment

**Problem**: Content moderation at scale — review user-uploaded images/videos for policy violations.

**Solution**: Qwen3-VL classifies content (safe/unsafe/borderline) with explanations.

**Scale**: Major platforms process 500M+ uploads/day. Qwen3-VL-8B on 8xB200 cluster: ~10M classifications/day per cluster. Need ~50 clusters for full platform coverage.

### Creative Industry (via vLLM-Omni Diffusion)

**Problem**: Generate marketing assets at scale — product photos, ad creatives, social media content.

**Solution**: FLUX.1-dev with brand-specific LoRA adapters, served through vLLM-Omni.

**Workflow**:
1. Marketing team trains LoRA adapter on brand assets ($50-200 one-time cost)
2. Deploy FLUX.1-dev + LoRA via vLLM-Omni
3. Generate 100s of variations per campaign
4. TeaCache + torch.compile reduce generation time by 50%+

**Numbers**: 1 image at 1024x1024 in ~5-10 seconds. 100 campaign variations in 8-17 minutes. Compare to: graphic designer produces 5-10 variations per day.

### Voice AI (via Qwen3-TTS)

**Problem**: Build natural-sounding voice interfaces for customer service, accessibility, content creation.

**Solution**: Qwen3-TTS served via vLLM-Omni with async chunk pipeline overlap.

**Performance**: RTF 0.60 = generate 10 seconds of speech in 6 seconds. Real-time streaming to end users.

**Use Cases**:
- IVR replacement: "Press 1 for..." → natural conversational AI
- Audiobook generation: Full book narrated in hours instead of weeks
- Accessibility: Real-time screen reading for visually impaired users

---

## 23. Actual vLLM Code Walkthrough

### Qwen3-VL Model Implementation

**File**: `vllm/model_executor/models/qwen3_vl.py`

This is the actual model implementation that runs on the GPU. Key classes:

```python
# 1. Vision Encoder - processes images
class Qwen3VLVisionBlock(nn.Module):
    """Single transformer block in the ViT encoder"""
    # Attention + MLP with rotary position embeddings
    # Called 27 times (one per ViT layer)

class Qwen3VLVisionEncoder(nn.Module):
    """Full Vision Transformer encoder"""
    # Stacks 27 Qwen3VLVisionBlock layers
    # Extracts features at deepstack_visual_indexes = [8, 16, 24]

# 2. DeepStack Fusion
class Qwen3VLDeepStackFusion(nn.Module):
    """Fuses multi-depth ViT features into LLM layers 1-3"""
    # Three MLP projectors: ViT layer 8→LLM L1, 16→L2, 24→L3
    # Each projector: Linear + LayerNorm + GELU + Linear

# 3. Main Model
class Qwen3VLForConditionalGeneration(nn.Module):
    """Top-level model combining ViT + Fusion + LLM"""
    def __init__(self):
        self.visual = Qwen3VLVisionEncoder(...)
        self.deepstack_fusion = Qwen3VLDeepStackFusion(...)
        self.model = Qwen3Model(...)  # LLM backbone

    def forward(self, input_ids, pixel_values, ...):
        # Step 1: Process image through ViT
        image_features = self.visual(pixel_values)

        # Step 2: DeepStack fusion injects features into LLM
        # (handled within the LLM forward pass)

        # Step 3: LLM generates text tokens
        hidden_states = self.model(input_ids, image_features, ...)

        # Step 4: Sample next token
        logits = self.lm_head(hidden_states)
        return logits
```

### Scheduler Code

**File**: `vllm/v1/core/sched/scheduler.py`

```python
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """Decide what to run this iteration"""

        # 1. Check running requests — can they continue decoding?
        budget = SchedulingBudget(
            token_budget=self.max_num_batched_tokens,
            max_num_requests=self.max_num_seqs
        )

        # 2. Allocate KV cache blocks for running requests
        for request in self.running:
            blocks = self.kv_cache_manager.allocate(request)
            budget.consume(request.num_tokens)

        # 3. Try to schedule waiting requests (new prefills)
        for request in self.waiting:
            if budget.can_schedule(request):
                blocks = self.kv_cache_manager.allocate(request)
                budget.consume(request.num_prompt_tokens)
                # Chunked prefill: may only process part of a long prompt

        # 4. If memory pressure: preempt lowest-priority running requests
        if self.kv_cache_manager.under_pressure():
            self.preempt(lowest_priority_request)

        return SchedulerOutput(
            scheduled_requests=...,
            blocks_to_swap=...,
            blocks_to_copy=...
        )
```

### KV Cache Manager

**File**: `vllm/v1/core/kv_cache_manager.py`

```python
class KVCacheManager:
    """Block-based KV cache allocation with prefix caching"""

    def __init__(self, block_size: int, num_gpu_blocks: int):
        self.block_size = block_size  # e.g., 16 tokens per block
        self.num_gpu_blocks = num_gpu_blocks
        self.free_blocks = list(range(num_gpu_blocks))
        self.prefix_cache = {}  # hash → block_id for prefix caching

    def allocate(self, request) -> List[int]:
        """Allocate KV cache blocks for a request"""
        num_blocks_needed = ceil(request.num_tokens / self.block_size)

        # Check prefix cache first
        prefix_hash = hash(request.prompt_tokens[:prefix_len])
        if prefix_hash in self.prefix_cache:
            # Reuse cached blocks — no recomputation needed!
            cached_blocks = self.prefix_cache[prefix_hash]
            num_blocks_needed -= len(cached_blocks)

        # Allocate remaining blocks from free list
        new_blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        return cached_blocks + new_blocks
```

### GPU Model Runner

**File**: `vllm/v1/worker/gpu_model_runner.py`

```python
class GPUModelRunner:
    """Executes model forward passes on GPU"""

    def execute_model(self, scheduler_output: SchedulerOutput):
        """Run one iteration of the model"""

        # 1. Prepare inputs
        input_ids = self._prepare_inputs(scheduler_output)

        # 2. Process multimodal inputs (images for Qwen3-VL)
        if scheduler_output.has_multimodal:
            pixel_values = self._process_images(scheduler_output)

        # 3. Run model forward pass
        with torch.no_grad():
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=self.kv_caches,
                attn_metadata=attn_metadata,
                pixel_values=pixel_values,  # Qwen3-VL image input
            )

        # 4. Sample next tokens
        logits = self.model.compute_logits(hidden_states)
        next_tokens = self.sampler(logits, sampling_params)

        return ModelOutput(next_tokens=next_tokens)
```

### Attention Backend (Flash Attention)

**File**: `vllm/attention/backends/flash_attn.py`

```python
class FlashAttentionBackend:
    """Flash Attention for efficient attention computation"""

    def forward(self, query, key, value, kv_cache, attn_metadata):
        # For prefill: use flash_attn_varlen_func
        # - Processes variable-length sequences efficiently
        # - O(N) memory instead of O(N²) for attention

        # For decode: use flash_attn_with_kvcache
        # - Reads from paged KV cache blocks
        # - Optimized for single-token queries attending to long contexts

        if attn_metadata.is_prefill:
            output = flash_attn_varlen_func(
                q=query, k=key, v=value,
                cu_seqlens_q=..., cu_seqlens_k=...,
                max_seqlen_q=..., max_seqlen_k=...,
                softmax_scale=self.scale,
                causal=True
            )
        else:
            # Paged attention for decode
            output = flash_attn_with_kvcache(
                q=query,
                k_cache=kv_cache[0],
                v_cache=kv_cache[1],
                block_table=attn_metadata.block_tables,
                cache_seqlens=attn_metadata.seq_lens,
                softmax_scale=self.scale,
                causal=True
            )
        return output
```

### FusedMoE Implementation

**File**: `vllm/model_executor/layers/fused_moe/fused_moe.py`

```python
def fused_moe(
    hidden_states: torch.Tensor,      # [num_tokens, hidden_dim]
    w1: torch.Tensor,                  # [num_experts, intermediate_dim, hidden_dim]
    w2: torch.Tensor,                  # [num_experts, hidden_dim, intermediate_dim]
    gating_output: torch.Tensor,       # [num_tokens, num_experts]
    topk: int,                         # Number of experts per token
    renormalize: bool = True,
) -> torch.Tensor:
    """
    Fused Mixture-of-Experts computation.

    1. Top-k gating: select top-k experts per token
    2. Permute tokens to group by expert
    3. Batched GEMM: all tokens for each expert in one matmul
    4. Apply activation (SiLU)
    5. Second GEMM (down projection)
    6. Un-permute and weighted sum

    All steps fused into optimized Triton/CUDA kernels.
    """
    # This runs as a single GPU kernel — no Python overhead between steps
    # SharedFusedMoE extends this to include shared experts in the same kernel
```

---

## 24. Performance Benchmarks and Expectations

### Qwen3-VL-32B: Expected Throughput

#### H100 Performance Matrix

| TP Size | Precision | Max Context | Prefill (tok/s) | Decode per req (tok/s) | Max Batch | Total Decode (tok/s) |
|---------|-----------|-------------|-----------------|----------------------|-----------|---------------------|
| 2 | BF16 | 32K | ~4,000 | ~100 | 16 | ~1,600 |
| 4 | BF16 | 128K | ~8,000 | ~120 | 32 | ~3,840 |
| 4 | FP8 | 128K | ~12,000 | ~180 | 48 | ~8,640 |
| 8 | FP8 | 256K | ~18,000 | ~200 | 64 | ~12,800 |

#### B200 Performance Matrix

| TP Size | Precision | Max Context | Prefill (tok/s) | Decode per req (tok/s) | Max Batch | Total Decode (tok/s) |
|---------|-----------|-------------|-----------------|----------------------|-----------|---------------------|
| 1 | BF16 | 128K | ~6,000 | ~180 | 32 | ~5,760 |
| 1 | FP8 | 256K | ~10,000 | ~300 | 64 | ~19,200 |
| 1 | FP4 | 256K | ~15,000 | ~450 | 128 | ~57,600 |
| 2 | FP8 | 256K | ~18,000 | ~450 | 128 | ~57,600 |
| 4 | FP4 | 256K | ~40,000 | ~600 | 256 | ~153,600 |

### Qwen3-VL-235B-A22B: Expected Throughput

#### H100 (8-GPU DGX)

| Precision | Max Context | Prefill (tok/s) | Decode per req (tok/s) | Max Batch |
|-----------|-------------|-----------------|----------------------|-----------|
| BF16 | 32K | ~3,000 | ~60 | 8 |
| FP8 | 128K | ~5,000 | ~90 | 32 |

#### B200 (8-GPU DGX)

| Precision | Max Context | Prefill (tok/s) | Decode per req (tok/s) | Max Batch |
|-----------|-------------|-----------------|----------------------|-----------|
| BF16 | 128K | ~8,000 | ~120 | 64 |
| FP8 | 256K | ~14,000 | ~200 | 128 |
| FP4 | 256K | ~20,000 | ~300 | 256 |

### Diffusion Model Benchmarks (from vLLM-Omni paper)

| Model | Task | Resolution | Baseline Latency | vLLM-Omni Latency | Speedup |
|-------|------|------------|-----------------|-------------------|---------|
| Bagel | Text→Image | 1024x1024 | 23.12s | 9.64s | 2.40x |
| Bagel | Image→Image | 1024x1024 | 41.39s | 11.12s | 3.72x |
| MiMo-Audio | TTS | N/A | RTF 1.39 | RTF 0.60 | 2.32x |
| Qwen3-Omni | Audio Gen | N/A | — | — | 91.4% JCT reduction |

---

## 25. Troubleshooting and Production Tips

### Common Issues

#### 1. Out of Memory (OOM)

```bash
# Symptom: CUDA out of memory error during model loading

# Fix 1: Reduce max_model_len
--max-model-len 16384  # instead of 131072

# Fix 2: Reduce gpu_memory_utilization
--gpu-memory-utilization 0.85  # instead of 0.95

# Fix 3: Enable quantization
--quantization fp8

# Fix 4: Increase tensor parallelism
--tensor-parallel-size 4  # instead of 2

# Fix 5: Enable CPU offloading for KV cache
--swap-space 32  # GB of CPU RAM for KV cache overflow
```

#### 2. Slow Prefill for Long Image Prompts

```bash
# Symptom: First token takes 10+ seconds for image-heavy prompts

# Fix 1: Enable chunked prefill (splits long prefills into smaller chunks)
--enable-chunked-prefill
--max-num-batched-tokens 8192

# Fix 2: Limit image resolution
# In your API request, resize images before sending:
# max_pixels = 1024 * 1024 (1 megapixel)
```

#### 3. High Tail Latency (p99)

```bash
# Symptom: Most requests are fast but some take 10x longer

# Fix 1: Enable prefix caching (avoids recomputing shared prefixes)
--enable-prefix-caching

# Fix 2: Limit max concurrent requests
--max-num-seqs 16  # prevent queue buildup

# Fix 3: Use Decode Context Parallel for long contexts
--decode-context-parallel-size 2  # split KV cache attention
```

#### 4. Model Loading Takes Too Long

```bash
# Symptom: Server takes 5+ minutes to start

# Fix 1: Use tensor parallel loading (each GPU loads its shard)
# This is default in vLLM v1, but verify:
--load-format auto

# Fix 2: Pre-download model weights to local NVMe
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct --local-dir /local-nvme/model

# Fix 3: Use safetensors format (memory-mapped loading)
# Most Qwen models already use safetensors
```

### Production Deployment Best Practices

#### Docker Deployment

```bash
# Production-ready Docker command
docker run --gpus all \
  -v /models:/models \
  -p 8000:8000 \
  --shm-size=16g \
  --ulimit memlock=-1 \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-VL-32B-Instruct \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --quantization fp8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --port 8000

# --shm-size=16g : shared memory for inter-process communication
# --ulimit memlock=-1 : allow unlimited locked memory (needed for NCCL)
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-vl-32b
spec:
  replicas: 2  # 2 replicas for high availability
  selector:
    matchLabels:
      app: qwen3-vl
  template:
    metadata:
      labels:
        app: qwen3-vl
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
        - "--model"
        - "/models/Qwen3-VL-32B-Instruct"
        - "--tensor-parallel-size"
        - "2"
        - "--dtype"
        - "bfloat16"
        - "--quantization"
        - "fp8"
        - "--max-model-len"
        - "131072"
        - "--enable-prefix-caching"
        - "--port"
        - "8000"
        resources:
          limits:
            nvidia.com/gpu: 2  # Request 2 GPUs
            memory: "128Gi"
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300
          periodSeconds: 30
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-weights-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3-vl-service
spec:
  selector:
    app: qwen3-vl
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### Monitoring with Prometheus + Grafana

```bash
# vLLM exposes Prometheus metrics at /metrics
# Key metrics to monitor:

# 1. Throughput
vllm:avg_generation_throughput_toks_per_s  # Overall tokens/sec
vllm:avg_prompt_throughput_toks_per_s      # Prefill tokens/sec

# 2. Latency
vllm:e2e_request_latency_seconds          # End-to-end request latency
vllm:time_to_first_token_seconds          # Time to first token (TTFT)
vllm:time_per_output_token_seconds        # Inter-token latency (ITL)

# 3. Resource utilization
vllm:gpu_cache_usage_perc                 # KV cache utilization (target: 70-85%)
vllm:num_requests_running                 # Active requests
vllm:num_requests_waiting                 # Queued requests (should be low)

# 4. Alerts to set up:
# - gpu_cache_usage_perc > 95% → scale up or reduce max_num_seqs
# - num_requests_waiting > 10 for > 30s → scale up replicas
# - time_to_first_token_seconds p99 > 5s → investigate prefill bottleneck
```

### Hardware-Specific Tuning

#### H100 Tuning

```bash
# Enable Flash Attention 3 (when available)
# vLLM v0.14.0 uses FA3 as default when supported

# Set NCCL environment for optimal all-reduce
export NCCL_IB_DISABLE=0          # Enable InfiniBand if available
export NCCL_NET_GDR_LEVEL=5       # GPU Direct RDMA
export NCCL_P2P_DISABLE=0         # Enable peer-to-peer
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Optimize kernel scheduling

# For Transformer Engine FP8:
# vLLM auto-detects H100 and enables FP8 compute kernels
# No manual configuration needed
```

#### B200 Tuning

```bash
# Enable FP4 (NVFP4) for maximum throughput
# B200's 2nd-gen Transformer Engine handles FP4 natively
--quantization fp4

# Leverage larger shared memory (228 KB per SM)
# vLLM's Triton kernels auto-tune for available shared memory

# NVLink 5 at 1.8 TB/s — can use higher TP degrees without penalty
# TP=4 on B200 has similar communication overhead to TP=2 on H100

# For the 32B model on single B200:
# No TP needed → zero communication overhead → maximum efficiency
```

---

## Summary: Complete Feature Matrix

| Feature | Problem Solved | Speedup | GPU Impact | Business Value |
|---------|---------------|---------|------------|----------------|
| Async Chunk Pipeline | Stage idle time | 91.4% JCT reduction | Both | Real-time voice AI |
| SharedFusedMoE | MoE kernel overhead | ~20-30% per-token | Both (bandwidth-bound) | Lower cost/query |
| TeaCache | Redundant diffusion computation | 2.4-3.7x | Both (compute-bound) | Faster image generation |
| Sequence Parallelism | Memory limits for high-res images | Enables 4K+ | B200 better (NVLink 5) | High-res content |
| Torch Compile | Python/kernel launch overhead | 20-40% | Both | Free performance |
| Qwen3-TTS | No TTS serving | RTF 0.60 | Both | Voice assistants |
| Diffusion LoRA | Expensive fine-tuning | N/A | Both | Cheap customization |
| DiT CPU Offloading | GPU memory limits | 6-12x memory savings | H100 more (smaller HBM) | Cheaper hardware |
| New Models | Missing model support | Day-0 | Both | Time to market |
| New APIs | Missing endpoints | N/A | Both | Production readiness |
| XPU/ROCm/NPU | Vendor lock-in | N/A | AMD/Intel/Huawei | Hardware flexibility |
| Decode Context Parallel | Long context decode latency | ~4x for 128K | B200 better (NVLink 5) | Document AI, video AI |

---

*Document generated from comprehensive analysis of the vLLM codebase, vLLM-Omni paper (arXiv:2602.02204), MLPerf Inference v6.0 benchmark specifications, and NVIDIA GPU technical specifications.*
