# vLLM Office Hours #43 — Complete Technical Guide

**Date**: February 12, 2026
**Speakers**: Burkhard Ringlein (IBM Research), Michael Goin (Red Hat, vLLM Committer), Sasa Zelenovic (Red Hat)
**Audience**: New Engineers, CTOs, and CEOs

Everything covered in the session — from 30,000-foot business impact to GPU register-level
execution — in one document.

---

## Table of Contents

### For CEOs (Start Here)
- [Executive Summary: What Happened and Why It Matters](#executive-summary)
- [Business Impact: Cost, Speed, Scale Numbers](#business-impact)

### For CTOs (Architecture Decisions)
- [1. What Is vLLM?](#1-what-is-vllm)
- [2. What's New in the Past Two Weeks](#2-whats-new-in-the-past-two-weeks)
- [3. Streaming Requests and Realtime API](#3-streaming-requests-and-realtime-api)
- [4. vLLM-Omni v0.14.0 — Multi-Model Pipelines](#4-vllm-omni-v0140)
- [5. MLPerf Inference v6.0 — First VLM Benchmark](#5-mlperf-inference-v60)
- [6. GPT-OSS on NVIDIA Blackwell](#6-gpt-oss-on-nvidia-blackwell)
- [7. vLLM on GB200 — WideEP at Rack Scale](#7-vllm-on-gb200-wideep)
- [8. NVFP4 Quantization — 4-Bit Inference](#8-nvfp4-quantization)
- [9. New Model Support](#9-new-model-support)
- [10. Context Parallel Deployment](#10-context-parallel-deployment)
- [11. Hardware Decision Matrix](#11-hardware-decision-matrix)

### For Engineers (Code-Level Deep Dive)
- [12. The Triton Attention Backend — Full Kernel Analysis](#12-triton-attention-backend)
- [13. Q-Blocks: The GQA Optimization (Before vs. After)](#13-q-blocks-gqa-optimization)
- [14. Parallel Tiled Softmax — The Two-Kernel Trick](#14-parallel-tiled-softmax)
- [15. CUDA Graphs and the Static Launch Grid](#15-cuda-graphs-and-static-launch-grid)
- [16. GPU Execution Map](#16-gpu-execution-map)
- [17. All 6 Parallelism Strategies Explained](#17-parallelism-strategies)
- [18. Helion — The Next-Generation DSL](#18-helion)

### Community
- [19. Meetup Schedule and Getting Involved](#19-community)

---

## Executive Summary

**What happened**: vLLM Office Hours #43 presented the most significant collection of inference
breakthroughs since the project's founding. In a single session:

1. **Triton attention backend achieves 100.7% of FlashAttention 3 on H100** — from ~800 lines of
   portable Triton code vs. ~70,000 lines of CUDA. The same code runs on NVIDIA, AMD, and Intel.

2. **vLLM-Omni v0.14.0 ships** — the first stable release of a fully disaggregated system for
   serving any-to-any multimodal models (text + image + audio + video in, text + image + audio out).
   Up to **91.4% faster** than baseline.

3. **MLPerf v6.0 debuts the first VLM benchmark** with Shopify's product catalog — 40 million
   products/day classified by AI instead of human annotators.

4. **GPT-OSS hits 1.5 million tokens/second** on GB200 NVL72 with vLLM optimizations.

5. **NVFP4 quantization** reduces model size by 3x vs. BF16 while recovering 98-99% accuracy
   on 70B+ models, running natively on Blackwell tensor cores.

6. **Four new model families supported**: Step 3.5 Flash, GLM-OCR, Qwen3-Coder-Next, GLM-5.

**Why it matters**: These advances collectively mean that serving frontier AI models is becoming
cheaper, faster, more portable across hardware vendors, and accessible to more organizations.

---

## Business Impact

| Metric | Before (6 months ago) | After (today) | Improvement |
|--------|----------------------|---------------|-------------|
| **Cost to serve 70B model** | ~$2.30/1M tokens (H100 BF16) | ~$0.50/1M tokens (B200 FP4) | **4.6x cheaper** |
| **Decode speed** | ~100 tok/s (H100 BF16) | ~830 tok/s (B200 FP4) | **8.3x faster** |
| **Multi-model pipeline latency** | Sum of all stages | Max of any stage (async overlap) | **Up to 91% reduction** |
| **Hardware vendor lock-in** | Separate kernels per GPU | Single Triton source for all | **Eliminated** |
| **Product classification (Shopify)** | 200+ human annotators, $8M/yr | 2 GPU clusters, $613K/yr | **13x cheaper** |
| **Voice AI latency** | Not supported natively | Real-time factor < 1.0 | **Now possible** |
| **Models servable on 1 GPU** | Up to ~13B (BF16) | Up to ~32B (FP4 on B200) | **2.5x larger** |

---

## 1. What Is vLLM?

vLLM is the most widely deployed open-source LLM inference engine:

- **70,000+ GitHub stars**, 800+ PRs per month
- **500,000+ GPUs** deployed 24/7 globally
- **2,000+ contributors** from 50+ companies (NVIDIA, AMD, Red Hat, IBM, Meta, Google, Intel, etc.)
- **100+ model architectures** supported
- **10+ hardware platforms**: CUDA, ROCm, Gaudi/XPU, TPU, Neuron, CPU, Ascend, Metal, MACA, RBLN, Spyre, MLU, Kunlun

### Quick Start

```bash
# Install
uv pip install vllm --torch-backend=auto

# Serve any supported model
vllm serve deepseek-ai/DeepSeek-V3.1 -tp 8
```

This exposes an OpenAI-compatible API at `http://localhost:8000/v1/chat/completions`.

---

## 2. What's New in the Past Two Weeks

| Feature | Impact | Details |
|---------|--------|---------|
| Streaming Requests & Realtime API | Voice AI, real-time audio | WebSocket protocol, Voxtral Mini 4B |
| vLLM-Omni v0.14.0 | Multi-model pipelines | Stage graph, disaggregated serving |
| MLPerf v6.0 + Shopify | Industry benchmark | First VLM benchmark, Qwen3-VL-235B |
| GPT-OSS on Blackwell | +38% throughput | FlashInfer, torch.compile, FP8 KV cache |
| vLLM on GB200 | 1.5M tok/s | WideEP, 72-GPU NVLink domain |
| Step 3.5 Flash | New VLM model | 61 layers, 48 MoE experts, vision encoder |
| GLM-OCR | Audio understanding | Dasheng audio encoder + Qwen2 LLM |
| Qwen3-Coder-Next | Code generation | Hybrid linear+full attention, 512 MoE experts |
| GLM-5 | New model family | MoE with MTP speculative decoding |
| NVFP4 quantization | 3x model shrink | Native on B200, 98-99% accuracy at 70B+ |
| Context Parallel | Long-context serving | PCP for prefill, DCP for decode |
| **Triton attention backend** | **Performance portability** | **100.7% of FA3 on H100, SOTA on MI300X** |

---

## 3. Streaming Requests and Realtime API

### What It Is

vLLM now supports the **OpenAI Realtime API protocol** — a WebSocket-based interface for
streaming audio in and out of language models in real-time.

### Key Model: Voxtral Mini 4B Realtime

- **Model**: `mistralai/Voxtral-Mini-4B-Realtime-2602`
- **Size**: 4 billion parameters
- **Capability**: Processes streaming audio input and generates streaming audio output
- **Real-Time Factor**: < 1.0 (generates audio faster than real-time)
- **Protocol**: WebSocket with binary audio frames

### Why It Matters

Before this, serving voice AI required stitching together ASR → LLM → TTS as three separate
services. Now, a single vLLM deployment handles the entire audio conversation loop.

### For CTOs

```
Architecture BEFORE:         Architecture AFTER:
  Client → ASR service       Client ↔ vLLM (WebSocket)
         → LLM service             Single endpoint
         → TTS service             Single GPU allocation
         → Client                  Sub-second latency
  3 services, 3 GPU pools    1 service, 1 GPU pool
```

---

## 4. vLLM-Omni v0.14.0

### What It Is

A fully disaggregated serving system for **any-to-any multimodal models**. It decomposes complex
AI pipelines into independently managed stages connected by a **stage graph**.

### The Stage Graph

```yaml
# Example: Qwen2.5-Omni three-stage pipeline
Stage 0: Thinker (AR LLM)
    GPUs: 0,1 (TP=2), memory: 60%
    Output: hidden states (latent)
         │
         ▼  thinker2talker transition
Stage 1: Talker (AR LLM)
    GPUs: 1, memory: 30%
    Input: Stage 0
    Output: codec sequences (latent)
         │
         ▼
Stage 2: Code2Wav (DiT Vocoder)
    GPUs: 0, memory: 10%
    Input: Stage 1
    Output: audio waveforms (final)
```

Each stage runs in its own process with its own engine, scheduler, and GPU allocation.
Stages communicate via **OmniConnector** (shared memory for single-node, RDMA for multi-node).

### Performance

| Model | Job Completion Time Reduction | Throughput Speedup |
|-------|------------------------------|-------------------|
| Qwen2.5-Omni | 61.6% faster | 1.97x (Talker TPS) |
| Qwen3-Omni | **91.4% faster** | **12.97x (Thinker TPS)** |

### Supported Pipelines

| Category | Models |
|----------|--------|
| Omni-modality | Qwen2.5-Omni (3B, 7B), Qwen3-Omni (30B-A3B) |
| Image generation | FLUX.1-dev, FLUX.2-klein, SD3, GLM-Image, Z-Image |
| Video generation | Wan2.2 |
| Audio/TTS | Qwen3-TTS, Stable Audio Open |
| Multi-stage | Bagel-7B-MoT (Thinker/AR + Diffusion/DiT) |

### For CTOs: When to Use vLLM-Omni

Use vLLM-Omni when your product requires **multiple AI models in a pipeline** (e.g., understand
image → reason → generate speech → generate image). Use standard vLLM when you have a single
model serving text/chat.

---

## 5. MLPerf Inference v6.0

### What It Is

The first **Vision-Language Model (VLM) benchmark** in MLPerf history, created in partnership
with Shopify. It measures how fast an AI system can classify product images and descriptions
into a hierarchical taxonomy.

### The Task

```
Input:  Product title + description + photograph
Output: JSON with hierarchical category, brand, is_secondhand flag

Example:
  Input:  "Nike Air Max 90" + photo of sneakers
  Output: {"category": "Apparel > Shoes > Athletic > Sneakers",
           "brand": "Nike", "is_secondhand": false}
```

### The Numbers

| Metric | Value |
|--------|-------|
| Reference model | Qwen3-VL-235B-A22B-Instruct |
| Minimum GPUs | 8 × 80GB (H100 or better) |
| Accuracy threshold | Category F1 ≥ 0.7824 (99% of reference) |
| p99 latency SLA | ≤ 12 seconds (server scenario) |
| Shopify's daily volume | 40 million products |

### Hardware Comparison

| Config | Throughput | Cost/1M Tokens |
|--------|-----------|----------------|
| 8×H100 BF16 | ~3K tok/s | ~$2.30 |
| 8×H100 FP8 | ~5K tok/s | ~$1.40 |
| 8×B200 BF16 | ~8K tok/s | ~$1.20 |
| 8×B200 FP8 | ~14K tok/s | ~$0.70 |
| 8×B200 FP4 | ~20K tok/s | ~$0.50 |

### For CEOs: The Shopify ROI

```
Human annotation: 200+ annotators × $40K/year = $8M/year
AI classification: 2 GPU clusters × $306K/year = $613K/year
Annual savings: $7.4 million (92% reduction)
```

---

## 6. GPT-OSS on NVIDIA Blackwell

### What Is GPT-OSS?

OpenAI's open-weight Mixture-of-Experts model family:

| Model | Total Params | Active Params | Native Precision |
|-------|-------------|---------------|-----------------|
| gpt-oss-120b | 117B | 5.1B | MXFP4 (4-bit) |
| gpt-oss-20b | 21B | 3.6B | MXFP4 (4-bit) |

### Architecture Highlights

- **48 MoE experts**, top-4 routing with Sigmoid activation
- **Alternating sliding window** (128 tokens) and full attention layers
- **Attention sinks**: Learned per-head values in the softmax denominator
- **Native MXFP4**: Weights are 4-bit from the start (not post-training quantized)

### Blackwell Optimizations

| Optimization | Impact |
|-------------|--------|
| FlashInfer integration | Primary kernel backend for attention + MoE |
| torch.compile fusion | Automatic kernel fusion, fewer launches |
| AllReduce + RMSNorm fusion | Single kernel instead of two |
| FP8 KV cache | More concurrent requests per GPU |
| Async scheduling | CPU/GPU overlap (5-15% throughput gain) |

### Results

```
Max throughput:             +38%
Min latency:                +13%
GB200 NVL72 peak:           1.5 million tokens/second
```

### For CTOs: GPT-OSS vs. DeepSeek-R1

| Aspect | GPT-OSS-120B | DeepSeek-R1 |
|--------|-------------|-------------|
| Active params | 5.1B | ~37B |
| Total params | 117B | 671B |
| Native precision | MXFP4 (4-bit) | BF16 |
| Single-GPU decode | Yes (H100) | No (needs 8+ GPUs) |
| Attention sinks | Yes (streaming-friendly) | No |
| Sliding window | 128 tokens (alternating) | No |
| Best for | Low-latency, local deployment | Reasoning, long-form generation |

---

## 7. vLLM on GB200 — WideEP

### What Is WideEP?

**Wide Expert Parallelism** distributes MoE experts across up to 72 GPUs in a single NVLink
domain. Instead of each GPU holding all experts (tensor parallel), each GPU holds only a few.

### GB200 NVL72 Specifications

| Spec | Value |
|------|-------|
| GPUs per NVLink domain | 72 NVIDIA Blackwell |
| NVLink bandwidth per GPU | 1,800 GB/s bidirectional |
| Aggregate NVLink bandwidth | **130 TB/s** across the rack |
| HBM per GPU | 192 GB (HBM3e) |
| HBM bandwidth per GPU | 8 TB/s |
| Total HBM per rack | **13.8 TB** |

### Performance Results (DeepSeek-R1 on GB200)

| Metric | Value | vs. H200 |
|--------|-------|----------|
| Prefill TPGS | 26.2K tok/GPU/s | 3-5x faster |
| Decode TPGS | 10.1K tok/GPU/s | 3-5x faster |
| DeepSeek-R1 8K/1K | — | **15x faster than H200** |

### Key Technologies

| Technology | Purpose |
|-----------|---------|
| WideEP (EP=64) | Spread 256 experts across 64 GPUs (~4 experts/GPU/layer) |
| FP4 MoE weights | 4x memory reduction for expert weights |
| FP8 MLA attention | Better accuracy for attention projections |
| DeepEP all-to-all | Custom kernels for expert token dispatch |
| Dual Batch Overlap (DBO) | Overlap all-to-all communication with computation |
| EPLB | Load-balance hot/cold experts across GPUs |
| NVLink-C2C | CPU↔GPU link for weight offloading (no PCIe bottleneck) |

### For CTOs: Deployment Topology

```
Prefill:  4 instances × 2 GPUs each (DP+EP)  = 8 GPUs for prefill
Decode:   1 instance  × 8 GPUs      (DP+EP)  = 8 GPUs for decode
Total:    16 GPUs for one DeepSeek-R1 serving endpoint
```

---

## 8. NVFP4 Quantization

### What It Is

NVIDIA's proprietary 4-bit floating-point format for inference, with **hierarchical (dual-level)
scaling** that preserves dynamic range far better than simple 4-bit formats.

### How It Works

```
Layer 1: Per-16-element block scale (FP8 E4M3)
  Every 16 contiguous values share one 8-bit scale factor
  Scale = global_scale × (max_abs_in_block / 6.0)

Layer 2: Per-tensor global scale (FP32)
  One scalar per tensor (or per expert in MoE)
  Provides overall dynamic range normalization

Data values: E2M1 format (1 sign + 2 exponent + 1 mantissa = 4 bits)
  Representable: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} and negatives
  Two values packed per byte
```

### Why NVFP4 Beats MXFP4

| Property | MXFP4 (GPT-OSS native) | NVFP4 |
|----------|------------------------|-------|
| Block size | 32 elements | **16 elements** (2x finer) |
| Scale format | E8M0 (power-of-two only) | **E4M3 (mantissa precision)** |
| Global scale | None | **FP32 per-tensor** |
| Quantization error | Higher | **88% lower** |

### Memory Savings

| Precision | Bits/element | Effective (with scales) | Reduction vs BF16 |
|-----------|-------------|------------------------|-------------------|
| BF16 | 16 | 16 | 1x (baseline) |
| FP8 | 8 | ~8.5 | ~1.9x |
| **NVFP4** | 4 | **~4.5** | **~3.5x** |

### Accuracy at Scale

| Model Size | BF16 Accuracy Recovery |
|-----------|----------------------|
| 8B | ~95-97% |
| 30B MoE | ~98% |
| 70B dense | ~98-99% |
| 400B+ MoE | ~99%+ |
| MMLU-Pro 5-shot | 62.58% (NVFP4) vs 62.62% (FP8) — essentially identical |

### vLLM Integration

```bash
# Serve an NVFP4-quantized model (auto-detected from checkpoint)
vllm serve nvidia/Llama-3.3-70B-Instruct-FP4    # Dense model
vllm serve nvidia/Qwen3-30B-A3B-FP4             # MoE model

# Override GEMM backend (default: auto-select)
VLLM_NVFP4_GEMM_BACKEND=cutlass vllm serve nvidia/Llama-3.3-70B-Instruct-FP4
```

Backend selection priority:
1. **flashinfer-cutlass** (preferred on B200)
2. **cutlass** (native SM100/SM120 GEMM)
3. **marlin** (fallback for older GPUs, weight-only quantization)

### For CEOs: What FP4 Means for Your Budget

```
70B model on B200:
  BF16: 140 GB → needs 1 GPU (barely fits) or 2 GPUs
  FP8:   70 GB → fits 1 GPU with 122 GB for KV cache
  FP4:   40 GB → fits 1 GPU with 152 GB for KV cache (3x more concurrent users)

Cost reduction:
  BF16 on 2 GPUs:  $4.60/1M tokens
  FP4 on 1 GPU:    $0.50/1M tokens → 9.2x cheaper
```

---

## 9. New Model Support

### Step 3.5 Flash (StepFun)

| Aspect | Detail |
|--------|--------|
| Architecture | MoE VLM: 61 layers, 48 experts, top-3 routing |
| Attention | Multi-Query (1 KV head), head dim 256 |
| Vision | 63-layer ViT, 728px patches, sliding-window cropping for high-res |
| Shared expert | Yes (runs in parallel with routed experts) |
| Max context | 65,536 tokens |
| Serving | `vllm serve stepfun-ai/step3 --limit-mm-per-prompt '{"image": 4}'` |

### GLM-OCR (MiDashengLM)

| Aspect | Detail |
|--------|--------|
| Architecture | Audio-language model: Dasheng audio encoder + Qwen2 LLM |
| Audio encoder | 12-layer transformer, mel-spectrogram frontend (16kHz, 64 mel bins) |
| Parameters | ~7B |
| Modalities | Text + Audio (speech understanding) |
| Serving | `vllm serve mispeech/midashenglm-7b` |

### Qwen3-Coder-Next

| Aspect | Detail |
|--------|--------|
| Architecture | **Hybrid** linear + full attention MoE |
| Innovation | Gated Delta Net (GDN) linear attention on 3/4 of layers, full attention on 1/4 |
| Experts | 512 routed experts, top-10 routing (~3B active of ~80B total) |
| Tool calling | Native XML-based function calling via `Qwen3CoderToolParser` |
| MTP | Multi-Token Prediction for speculative decoding |
| Serving | `vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct` |

### GLM-5 (GLM-4.5/4.6/4.7 Family)

| Aspect | Detail |
|--------|--------|
| Architecture | SharedFusedMoE with MTP speculative decoding |
| Variants | Text-only (GLM-4.5), Vision+Text (GLM-4.1V), Vision+MoE (GLM-4.5V) |
| Vision | EVA2CLIP encoder, MRoPE position encoding |
| Serving | `vllm serve zai-org/GLM-4.5` |

---

## 10. Context Parallel Deployment

### The Problem

Large models with few KV-heads (like DeepSeek-R1 with 1 KV-head) suffer severe **KV cache
duplication** when using tensor parallelism. With TP=8 and 1 KV-head, the KV cache is
duplicated **8 times** — wasting 7/8 of the KV cache memory.

### The Solution: Two New Parallelism Dimensions

| Strategy | Flag | What It Does |
|----------|------|-------------|
| **Prefill Context Parallel (PCP)** | `-pcp N` | Splits long prefill across N GPUs → faster TTFT |
| **Decode Context Parallel (DCP)** | `-dcp N` | Shards KV cache along token dimension → no duplication |

### Example: DeepSeek-R1

```bash
# Without DCP: KV cache duplicated 8x (only 1 KV-head with TP=8)
vllm serve deepseek-ai/DeepSeek-R1 -tp 8

# With DCP: KV cache NOT duplicated (sharded across all 8 GPUs)
vllm serve deepseek-ai/DeepSeek-R1 -tp 8 -dcp 8

# Result: 8x more KV cache capacity → 8x more concurrent requests
```

### When to Use What

| Scenario | Use |
|----------|-----|
| Model fits on 1 GPU | No parallelism needed |
| Model too large for 1 GPU | `-tp N` (tensor parallel) |
| Very deep model (100+ layers) | `-pp N` (pipeline parallel) |
| MoE model with many experts | `--enable-expert-parallel` |
| TP > num_kv_heads (KV duplication) | Add `-dcp N` |
| Long-context prefill is bottleneck | Add `-pcp N` |
| Need to serve many users concurrently | `-dp N` (data parallel) |

---

## 11. Hardware Decision Matrix

### For CTOs: Which GPU for Which Workload?

| Workload | Best GPU | Why | vLLM Config |
|----------|---------|-----|-------------|
| Chat (7B-13B models) | 1× H100 or 1× B200 | Fits on 1 GPU, decode-bound | `vllm serve model -tp 1` |
| Chat (70B models) | 2× B200 FP4 or 4× H100 FP8 | FP4 halves GPU count | `vllm serve model -tp 2` |
| DeepSeek-R1 (671B) | 8× B200 + DCP | WideEP + DCP eliminates duplication | `-tp 8 -dcp 8 --enable-expert-parallel` |
| Qwen3-VL-235B (VLM) | 8× B200 TP=8 | Vision+MoE, needs high memory | `-tp 8` |
| Voice AI (real-time) | 1× B200 | Voxtral 4B fits easily | `vllm serve voxtral --enable-realtime` |
| Multi-model pipeline | vLLM-Omni | Stage graph with per-stage GPU | YAML stage config |
| MLPerf benchmark | 8× B200 FP4 | Maximum throughput | `-tp 8` with NVFP4 checkpoint |
| Cost-optimized AMD | MI300X | Triton backend is SOTA | `vllm serve model -tp 4` |

---

## 12. Triton Attention Backend

### The Special Topic of Office Hours #43

The Triton attention backend is vLLM's **performance-portable** attention implementation —
a single Triton source that runs on NVIDIA, AMD, and Intel GPUs.

### Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/attention/ops/triton_unified_attention.py` | ~1,060 | Three Triton JIT kernels |
| `vllm/v1/attention/backends/triton_attn.py` | ~500 | Backend wrapper, CUDA graph support |
| `vllm/attention/ops/triton_reshape_and_cache_flash.py` | ~160 | KV cache store kernel |

Authors (IBM Research Zurich):
```
Burkhard Ringlein, Jan van Lunteren, Chih-Chieh Yang, Thomas Parnell
```

### Three Kernels

```
kernel_unified_attention_2d   ← Prefill + large-batch decode
  Grid: (total_q_blocks, num_kv_heads)
  Used when: batch > threshold OR any prefill request

kernel_unified_attention_3d   ← Small-batch long-context decode
  Grid: (total_q_blocks, num_kv_heads, 16_segments)
  Used when: batch ≤ threshold AND pure decode

reduce_segments               ← Merges 16 partial softmax results
  Grid: (num_tokens, num_query_heads)
  Used after: 3D kernel only
```

### Benchmark Result

```
H100 SXM, Llama-3.1-8B, batch_size=1, input_length=500:
  Flash Attention 3 (70K LoC CUDA):    baseline
  Triton Unified Attention (~800 LoC): 100.7% of FA3 (FASTER)

MI300X, same workload:
  Previous SOTA:                       baseline
  Triton Unified Attention:            5.8x speedup (same source!)
```

### Feature Support

| Feature | Triton | Flash Attn 3 | FlashInfer |
|---------|--------|-------------|------------|
| Sinks (GPT-OSS) | Yes | Yes (SM ≥ 9.0) | Via TRTLLM extension |
| ALiBi | Yes | No (falls back to FA2) | Yes |
| Softcap | Yes | Yes | Yes |
| Sliding window | Yes (2-level optimization) | Yes | Yes |
| Multimodal prefix | Yes (bidirectional ranges) | No | No |
| FP8 KV cache | Yes | Yes | Yes |
| CUDA graphs | ALWAYS (highest level) | ALWAYS (FA3) / UNIFORM (FA2) | ALWAYS |
| AMD ROCm | Yes (default backend) | No | No |
| Intel XPU | Yes | No | No |

---

## 13. Q-Blocks: GQA Optimization

### The Problem

Grouped-Query Attention (GQA) groups multiple query heads per KV head. A naive implementation
loads K and V from HBM once per query head — wasting bandwidth.

### Before (Naive)

```
For each KV head:
  For each of 8 query heads sharing this KV head:
    Load K from HBM              ← 8 redundant loads!
    Load V from HBM              ← 8 redundant loads!
    Compute attention(Q, K, V)
```

### After (Q-Blocks)

```python
# triton_unified_attention.py:127-133
# Pack all 8 query heads × 2 tokens into a single BLOCK_M=16 tile:
#   Row 0: token 0, head 0
#   Row 1: token 0, head 1
#   ...
#   Row 7: token 0, head 7
#   Row 8: token 1, head 0
#   ...
#   Row 15: token 1, head 7

# K and V loaded ONCE per tile, indexed by kv_head_idx only:
k_offset = physical_block_idx * stride + kv_head_idx * stride  # ← single KV head

# tl.dot(Q, K) computes scores for ALL 16 rows × all KV positions simultaneously
S += scale * tl.dot(Q, K)  # [16, TILE_SIZE] — all heads and tokens at once
```

**Result**: K/V loaded from HBM **1x** instead of **8x** per tile. 8x bandwidth saving for GQA ratio=8.

---

## 14. Parallel Tiled Softmax

### The Problem

During decode (1 query token per request), batch_size=1 with 8 KV-heads launches only **8 thread
blocks** on a GPU with 132 SMs. **124 SMs sit idle**.

### The Solution

Split each sequence's KV into **16 segments**, launch 16× more thread blocks:

```
Before: 1 × 8 = 8 thread blocks → 8 of 132 SMs used (6%)
After:  1 × 8 × 16 = 128 thread blocks → 128 of 132 SMs used (97%)
```

Each segment computes partial `(output, max, exp_sum)`. The `reduce_segments` kernel merges
them using online softmax — mathematically exact, no approximation.

### Why Two Kernels?

The talk explains: *"Triton doesn't have a global barrier."* Without a barrier, thread blocks
in different segments cannot synchronize within a single kernel. The solution is two separate
kernel launches:

1. **3D kernel**: Each segment independently computes partial attention
2. **reduce_segments**: Reads all partial results, merges them

The trade-off is launch overhead (~10μs per kernel) vs. parallelism. For long sequences, the
parallelism wins massively.

---

## 15. CUDA Graphs and Static Launch Grid

### The Challenge

CUDA graphs "freeze" kernel launch parameters including the grid dimensions. But batch size
changes every step. How do you use a fixed grid with a variable batch?

### The Solution: Over-Provision + Early Exit + Binary Search

```python
# Step 1: Over-provisioned grid (triton_unified_attention.py:910)
total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs  # Upper bound, not exact

# Step 2: Early exit for empty blocks (line 124)
if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
    return  # Thread block exits in ~10 cycles

# Step 3: Binary search to find sequence (line 34-53)
# Each thread block discovers its own sequence by searching GPU-side metadata
seq_idx = find_seq_idx(query_start_len_ptr, target_idx, num_seqs, ...)
```

The metadata tensor's **address** is frozen in the CUDA graph. Its **contents** are updated
each step. Thread blocks read the current metadata at execution time.

### CUDA Graph Support Level

```python
# triton_attn.py:115
_cudagraph_support = AttentionCGSupport.ALWAYS  # Highest level (value=3)
# Supports mixed prefill+decode within a single captured graph
```

---

## 16. GPU Execution Map

### Where Each Operation Runs (H100)

| Operation | Hardware Unit | Memory Tier |
|-----------|-------------|-------------|
| `tl.dot(Q, K)` — attention scores | **Tensor Cores** (FP16/BF16 HMMA) | Q in registers, K in SMEM |
| `tl.dot(P, V)` — attention output | **Tensor Cores** | P in registers, V in SMEM |
| `tl.exp(S - m_j)` — softmax | **CUDA cores** (FP32 ALUs) | Registers |
| `tl.max(S)` — running maximum | **CUDA cores** (reduction) | Registers |
| K/V tile load | **HBM** → L2 → SMEM | Block table indirection through L2 |
| Block table lookup | **HBM** → L2 cache | Small metadata tensor |
| Output store | SMEM → **HBM** | Write-back after all tiles |
| `find_seq_idx` binary search | **CUDA cores** (integer ops) | `query_start_len_ptr` in HBM |
| Sliding window mask | **CUDA cores** (comparisons) | Registers |
| ALiBi bias | **CUDA cores** (FP32 multiply-add) | Slope in registers |
| FP8 dequantization | **CUDA cores** (FP32 multiply) | Scale factor in registers |
| Sink initialization | **HBM** read (per-head values) | One load per thread block |

### Memory Sizes (Llama-3.1-8B, 1 KV head group)

| Buffer | Size | Location |
|--------|------|----------|
| Q tile | BLOCK_M × 128 × 2B = 4 KB | Registers + SMEM |
| K tile | 128 × TILE_SIZE × 2B = 4 KB | SMEM (from HBM) |
| V tile | TILE_SIZE × 128 × 2B = 4 KB | SMEM (from HBM) |
| Scores | BLOCK_M × TILE_SIZE × 4B = 1 KB | Registers (FP32) |
| Accumulator | BLOCK_M × 128 × 4B = 8 KB | Registers (FP32) |
| Running max/sum | BLOCK_M × 4B = 64 B | Registers |

---

## 17. All 6 Parallelism Strategies

### Layout Order

vLLM initializes process groups in this order:
```
ExternalDP × DP × PP × PCP × TP
```

### Decision Table

| Strategy | What It Shards | When to Use | CLI |
|----------|---------------|-------------|-----|
| **Tensor Parallel (TP)** | Weights + KV heads | Model too big for 1 GPU | `-tp 4` |
| **Pipeline Parallel (PP)** | Layers | Very deep models (100+ layers) | `-pp 2` |
| **Data Parallel (DP)** | Requests | High throughput, model fits per-rank | `-dp 4` |
| **Expert Parallel (EP)** | MoE experts | Large MoE models (256+ experts) | `--enable-expert-parallel` |
| **Prefill CP (PCP)** | Prefill computation | Long-context TTFT bottleneck | `-pcp 4` |
| **Decode CP (DCP)** | KV cache (tokens) | KV duplication from TP > num_kv_heads | `-dcp 8` |

### Example: DeepSeek-R1 on 8 GPUs

```bash
# Problem: TP=8 with 1 KV-head → 8x KV cache duplication
# Solution: Add DCP=8 to shard KV cache along token dimension

vllm serve deepseek-ai/DeepSeek-R1 \
  -tp 8 \
  -dcp 8 \
  --enable-expert-parallel \
  --dtype bfloat16

# Result: KV cache NOT duplicated → 8x more concurrent requests
```

---

## 18. Helion — The Next-Generation DSL

### What It Is

Helion is a new DSL from PyTorch described as *"tiled PyTorch"* or *"higher-level Triton"*.
It lets you write attention in PyTorch-like syntax, and Helion compiles it to optimized Triton.

```
Abstraction Stack:
  Helion (PyTorch-like) → Triton (tiled) → PTX/GCN/SPIR-V (GPU ISA)
```

### Current Status

**Not yet in vLLM.** A draft paged attention kernel was submitted as PR #27293, and a blog
post at `pytorch.org/blog/portable-paged-attention-in-helion/` describes the approach. This
is experimental research — a preview of where the ecosystem is heading.

### Why It Matters

If Helion achieves performance parity with hand-written Triton, it would reduce the barrier
to writing high-performance GPU kernels from "GPU kernel expert" to "PyTorch developer."
This dramatically expands the contributor pool for projects like vLLM.

---

## 19. Community

### Q1 2026 In-Person Meetup Schedule

| Date | City | Notes |
|------|------|-------|
| Feb 24 | Munich | |
| Feb 28 | Pune | |
| Mar 3 | Wellington | |
| Mar 5 | Tokyo, Auckland | |
| Mar 7 | Hong Kong | |
| Mar 10 | New York City (llm-d), Warsaw | |
| Mar 12 | Vienna (livestreamed) | |
| Mar 15 | Beijing | |
| Mar 19 | Boston | |
| Mar 21 | Beijing | |
| Mar 25 | Stockholm | |
| Mar 28 | Mumbai | |

### How to Get Involved

| Action | Link |
|--------|------|
| Join Slack | `slack.vllm.ai` (10,000+ members) |
| Contribute | Check "good first issue" tags on GitHub |
| Review PRs | Comment on PRs that interest you |
| Join RFCs | Participate in design discussions |
| Apply to Red Hat | Red Hat is hiring vLLM engineers |

### Upcoming Office Hours

| Date | Topic |
|------|-------|
| Feb 26 | vLLM Project Update & Discussion |
| Mar 12 | Vienna vLLM Meetup Livestream |
| Mar 26 | Latest Trends in AI Agent Applications and vLLM |

---

## Quick Reference: vLLM Command Cheat Sheet

```bash
# Basic serving
vllm serve Qwen/Qwen3-32B -tp 2

# With FP8 quantization
vllm serve Qwen/Qwen3-32B -tp 2 --kv-cache-dtype fp8

# With NVFP4 (auto-detected from checkpoint)
vllm serve nvidia/Llama-3.3-70B-Instruct-FP4 -tp 2

# DeepSeek-R1 with all optimizations
vllm serve deepseek-ai/DeepSeek-R1 -tp 8 -dcp 8 --enable-expert-parallel

# GPT-OSS with async scheduling
vllm serve openai/gpt-oss-120b -tp 1

# VLM with image support
vllm serve Qwen/Qwen3-VL-32B -tp 2 --limit-mm-per-prompt '{"image": 4}'

# Force Triton attention backend (useful for AMD or debugging)
VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve model

# Context parallel for long-context
vllm serve model -tp 8 -dcp 4 -pcp 2

# Voice AI (Realtime API)
vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602

# Multi-model pipeline (requires vLLM-Omni)
pip install vllm-omni
vllm-omni serve --stage-config qwen2_5_omni.yaml
```

---

*This document covers every topic from vLLM Office Hours #43 (February 12, 2026), from CEO-level
business impact to GPU register-level kernel execution, with actual code paths, line numbers, and
deployment commands.*
