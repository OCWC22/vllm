# Hardware-Level LLM Inference on NVIDIA Blackwell B200

**A conceptual, diagram-rich guide for understanding GPU-level inference execution**

---

## Table of Contents
- [1. Executive Summary](#1-executive-summary)
- [2. Blackwell B200 GPU Architecture](#2-blackwell-b200-gpu-architecture)
- [3. Anatomy of an Inference Request](#3-anatomy-of-an-inference-request)
- [4. Hardware-Level Execution Path](#4-hardware-level-execution-path)
- [5. Stage-by-Stage Deep Dive](#5-stage-by-stage-deep-dive)
- [6. Comparison: No Engine vs vLLM vs Optimized B200](#6-comparison-no-engine-vs-vllm-vs-optimized-b200)
- [7. Trade-offs and Bottlenecks](#7-trade-offs-and-bottlenecks)
- [8. Practical Implications](#8-practical-implications)

---

# 1. Executive Summary

## Why This Document Exists

Understanding LLM inference at the hardware level is critical for:
1. **Making informed architecture decisions** - knowing what limits throughput and latency
2. **Optimizing deployment costs** - maximizing utilization of expensive GPU hardware
3. **Debugging performance issues** - understanding where bottlenecks occur
4. **Evaluating new hardware** - comparing GPUs on metrics that actually matter for LLMs

## The Blackwell B200 in Context

The NVIDIA Blackwell B200 represents a significant leap in GPU architecture designed specifically for AI workloads:

| Specification | Hopper H100 | Blackwell B200 | Improvement |
|---------------|-------------|----------------|-------------|
| **Transistors** | 80 billion | 208 billion | 2.6× |
| **HBM Capacity** | 80GB HBM3 | 192GB HBM3e | 2.4× |
| **HBM Bandwidth** | 3.35 TB/s | 8.0 TB/s | 2.4× |
| **FP8 Tensor** | ~2 PFLOPS | ~4.5 PFLOPS | 2.25× |
| **FP4 Tensor** | N/A | ~9 PFLOPS | New! |
| **NVLink** | 900 GB/s | 1.8 TB/s | 2× |

**Key Insight**: The B200's improvements are strategically targeted at LLM inference bottlenecks:
- More HBM for larger KV caches → more concurrent requests
- Higher bandwidth → faster decode (memory-bound)
- FP4/FP8 support → higher throughput with reduced precision
- More SMs → better prefill performance (compute-bound)

---

# 2. Blackwell B200 GPU Architecture

## Mental Model: The GPU as a Memory-Compute System

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NVIDIA BLACKWELL B200 GPU                              │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         HBM3e MEMORY (192 GB)                          │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │  Model Weights    │  KV Cache         │  Activations/Workspace  │   │ │
│  │  │  (~70-140 GB      │  (~20-100 GB      │  (~10-30 GB)            │   │ │
│  │  │   for 70B model)  │   dynamic)        │                         │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │                              ↕ 8.0 TB/s bandwidth                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                      ↕                                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    STREAMING MULTIPROCESSORS (132 SMs)                 │ │
│  │                                                                        │ │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐      │ │
│  │  │  SM 0   │ │  SM 1   │ │  SM 2   │ │  SM 3   │ ... │ SM 131  │      │ │
│  │  │┌───────┐│ │┌───────┐│ │┌───────┐│ │┌───────┐│     │┌───────┐│      │ │
│  │  ││Tensor ││ ││Tensor ││ ││Tensor ││ ││Tensor ││     ││Tensor ││      │ │
│  │  ││ Cores ││ ││ Cores ││ ││ Cores ││ ││ Cores ││     ││ Cores ││      │ │
│  │  ││ FP8   ││ ││ FP8   ││ ││ FP8   ││ ││ FP8   ││     ││ FP8   ││      │ │
│  │  ││ FP4   ││ ││ FP4   ││ ││ FP4   ││ ││ FP4   ││     ││ FP4   ││      │ │
│  │  │└───────┘│ │└───────┘│ │└───────┘│ │└───────┘│     │└───────┘│      │ │
│  │  │┌───────┐│ │┌───────┐│ │┌───────┐│ │┌───────┐│     │┌───────┐│      │ │
│  │  ││ SRAM  ││ ││ SRAM  ││ ││ SRAM  ││ ││ SRAM  ││     ││ SRAM  ││      │ │
│  │  ││ 256KB ││ ││ 256KB ││ ││ 256KB ││ ││ 256KB ││     ││ 256KB ││      │ │
│  │  ││(L1/SM)││ ││(L1/SM)││ ││(L1/SM)││ ││(L1/SM)││     ││(L1/SM)││      │ │
│  │  │└───────┘│ │└───────┘│ │└───────┘│ │└───────┘│     │└───────┘│      │ │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘     └─────────┘      │ │
│  │                                                                        │ │
│  │  Total Shared Memory: ~33 MB (256KB × 132 SMs)                        │ │
│  │  L2 Cache: ~60 MB                                                      │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         NVLink 5.0 Interconnect                        │ │
│  │                         1.8 TB/s bidirectional                         │ │
│  │                    (for multi-GPU tensor parallelism)                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory Hierarchy Latency/Bandwidth

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MEMORY HIERARCHY (B200)                              │
│                                                                         │
│   Registers (per thread)    ← ~0 cycles latency, ~TB/s effective       │
│         ↓                                                               │
│   Shared Memory (256KB/SM)  ← ~20 cycles, ~19 TB/s aggregate           │
│         ↓                                                               │
│   L1 Cache (unified w/SRAM) ← ~30 cycles                               │
│         ↓                                                               │
│   L2 Cache (~60 MB)         ← ~200 cycles, ~12 TB/s                    │
│         ↓                                                               │
│   HBM3e (192 GB)            ← ~400 cycles, 8 TB/s                      │
│         ↓                                                               │
│   NVLink (to other GPUs)    ← ~1000 cycles, 1.8 TB/s                   │
│         ↓                                                               │
│   PCIe 5.0 (to CPU)         ← ~5000 cycles, 128 GB/s                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

KEY INSIGHT FOR LLM INFERENCE:
┌─────────────────────────────────────────────────────────────────────────┐
│  PREFILL (process prompt):  Compute-bound → use Tensor Cores           │
│  DECODE  (generate tokens): Memory-bound → limited by HBM bandwidth    │
│                                                                         │
│  B200's 8 TB/s bandwidth is THE critical spec for decode performance   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 3. Anatomy of an Inference Request

## Example Request

```
User Prompt: "Explain quantum computing in simple terms"
              ↓
        [38 input tokens]
              ↓
        Model generates 150 output tokens
              ↓
Total: 38 (prefill) + 150 (decode steps)
```

## Request Lifecycle Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REQUEST LIFECYCLE                                    │
│                                                                             │
│  ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌─────────┐   ┌────────────┐ │
│  │ Receive │ → │ Tokenize  │ → │ Prefill │ → │ Decode  │ → │ Detokenize │ │
│  │ Request │   │           │   │ (1×)    │   │ (N×)    │   │ & Return   │ │
│  └─────────┘   └───────────┘   └─────────┘   └─────────┘   └────────────┘ │
│      ↓             ↓               ↓             ↓              ↓         │
│    CPU           CPU             GPU           GPU            CPU         │
│   ~0.1ms       ~0.5ms         ~20-100ms     ~15-30ms/tok     ~0.1ms      │
│                                                                             │
│  Total latency = TTFT + (num_tokens × per-token latency)                   │
│                = ~25ms + (150 × 20ms) = ~3.0 seconds                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 4. Hardware-Level Execution Path

## Complete Flow: Request → GPU → Response

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE INFERENCE EXECUTION PATH                         │
│                                                                              │
│ ┌─────────────────────────┐                                                 │
│ │     APPLICATION LAYER    │                                                 │
│ │  (Python/vLLM/TensorRT)  │                                                 │
│ └────────────┬─────────────┘                                                 │
│              ↓                                                               │
│ ┌─────────────────────────┐     ┌─────────────────────────────────────────┐ │
│ │     SCHEDULER LAYER      │     │ What: Decides WHICH requests run WHEN   │ │
│ │                          │ ←── │ Why: Maximize throughput, minimize TTFT │ │
│ │  • Request queuing       │     │ How: Token budget, preemption, batching │ │
│ │  • Continuous batching   │     └─────────────────────────────────────────┘ │
│ │  • KV cache allocation   │                                                 │
│ └────────────┬─────────────┘                                                 │
│              ↓                                                               │
│ ┌─────────────────────────┐     ┌─────────────────────────────────────────┐ │
│ │     MEMORY MANAGER       │     │ What: Allocates/tracks KV cache blocks  │ │
│ │                          │ ←── │ Why: Avoid fragmentation, enable reuse  │ │
│ │  • Block allocation      │     │ How: PagedAttention, prefix caching     │ │
│ │  • Reference counting    │     └─────────────────────────────────────────┘ │
│ │  • LRU eviction          │                                                 │
│ └────────────┬─────────────┘                                                 │
│              ↓                                                               │
│ ┌─────────────────────────┐     ┌─────────────────────────────────────────┐ │
│ │     MODEL EXECUTION      │     │ What: Runs transformer forward pass     │ │
│ │                          │ ←── │ Why: Convert inputs → logits            │ │
│ │  • Input preparation     │     │ How: CUDA kernels, tensor operations    │ │
│ │  • Attention computation │     └─────────────────────────────────────────┘ │
│ │  • FFN layers            │                                                 │
│ └────────────┬─────────────┘                                                 │
│              ↓                                                               │
│ ┌─────────────────────────┐     ┌─────────────────────────────────────────┐ │
│ │      SAMPLING LAYER      │     │ What: Converts logits → token IDs       │ │
│ │                          │ ←── │ Why: Generate the actual output         │ │
│ │  • Temperature           │     │ How: Softmax, top-k/p, sampling         │ │
│ │  • Top-k / Top-p         │     └─────────────────────────────────────────┘ │
│ │  • Random sampling       │                                                 │
│ └────────────┬─────────────┘                                                 │
│              ↓                                                               │
│ ┌─────────────────────────┐                                                 │
│ │    OUTPUT PROCESSING     │                                                 │
│ │  • Detokenization        │                                                 │
│ │  • Streaming response    │                                                 │
│ └──────────────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## GPU Kernel Launch Sequence

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              GPU KERNEL EXECUTION (Per Forward Pass)                         │
│                                                                              │
│  Timeline →                                                                  │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  PREFILL (38 tokens, process all at once):                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Layer 0:                                                              │ │
│  │  ┌─────────┐ ┌───────────────────────┐ ┌────────┐ ┌──────────────────┐ │ │
│  │  │ LayerNorm│→│ QKV Projection (GEMM)│→│Attn Kern│→│ FFN (2×GEMM)    │ │ │
│  │  │ ~5µs    │ │ ~100µs               │ │ ~80µs  │ │ ~200µs          │ │ │
│  │  └─────────┘ └───────────────────────┘ └────────┘ └──────────────────┘ │ │
│  │                                                                        │ │
│  │  Layer 1-79: [repeat above pattern]                                   │ │
│  │                                                                        │ │
│  │  Total per layer: ~400µs × 80 layers = ~32ms prefill                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  DECODE (1 token at a time, repeated 150×):                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Layer 0:                                                              │ │
│  │  ┌─────────┐ ┌───────────────────────┐ ┌────────┐ ┌──────────────────┐ │ │
│  │  │ LayerNorm│→│ QKV Projection (GEMM)│→│PagedAttn│→│ FFN (2×GEMM)    │ │ │
│  │  │ ~2µs    │ │ ~50µs (batch 64)     │ │ ~150µs │ │ ~100µs          │ │ │
│  │  └─────────┘ └───────────────────────┘ └────────┘ └──────────────────┘ │ │
│  │                                                                        │ │
│  │  Total per layer: ~300µs × 80 layers = ~24ms per token                │ │
│  │  MEMORY-BOUND: Reading all KV cache + weights dominates               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 5. Stage-by-Stage Deep Dive

## Stage 1: Request Ingestion

### What is Happening
```
HTTP POST /v1/completions
{"prompt": "Explain quantum computing in simple terms", "max_tokens": 150}
                                    ↓
                        [API Server Receives]
                                    ↓
                        [Tokenization: text → token IDs]
                        "Explain" → 50123
                        "quantum" → 31456
                        ... (38 tokens total)
                                    ↓
                        [Request Object Created]
                        {request_id: "abc123",
                         token_ids: [50123, 31456, ...],
                         max_tokens: 150,
                         sampling_params: {...}}
```

### What Problem It Solves
- Converts human-readable text into a format the model understands (token IDs)
- Validates request parameters before committing GPU resources
- Creates tracking state for the request lifecycle

### Why This Approach
- Tokenization must happen on CPU (string processing)
- Stateless API allows horizontal scaling of API servers
- Request objects enable tracking through the async pipeline

### Output Produced
- `Request` object with validated parameters
- List of input token IDs ready for embedding lookup

---

## Stage 2: Scheduling Decision

### What is Happening
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCHEDULER DECISION TREE                              │
│                                                                              │
│  New Request "abc123" arrives                                                │
│         ↓                                                                    │
│  ┌─────────────────────────────────────────┐                                │
│  │ Check: Is there KV cache space?         │                                │
│  │        Need: 38 input + 150 output = 188 tokens                          │
│  │        Available blocks: 50 (50 × 16 = 800 tokens)                       │
│  └─────────────────────────────────────────┘                                │
│         ↓ YES                                                                │
│  ┌─────────────────────────────────────────┐                                │
│  │ Check: Can we add to current batch?      │                                │
│  │        Current batch: 42 requests        │                                │
│  │        Token budget: 2048 total          │                                │
│  │        Used: 1800, New request: 38       │                                │
│  │        1800 + 38 = 1838 < 2048 ✓        │                                │
│  └─────────────────────────────────────────┘                                │
│         ↓ YES                                                                │
│  ┌─────────────────────────────────────────┐                                │
│  │ Check: Prefix cache hit?                 │                                │
│  │        Hash tokens [0:16] → Block X      │                                │
│  │        Hash tokens [16:32] → MISS        │                                │
│  │        Reuse 16 tokens of computation!   │                                │
│  └─────────────────────────────────────────┘                                │
│         ↓                                                                    │
│  ┌─────────────────────────────────────────┐                                │
│  │ Allocate KV blocks:                      │                                │
│  │   Physical blocks: [P47, P48, P49, ...]  │                                │
│  │   Logical → Physical mapping stored      │                                │
│  └─────────────────────────────────────────┘                                │
│         ↓                                                                    │
│  Request "abc123" scheduled for PREFILL                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Problem It Solves
- **Memory fragmentation**: PagedAttention eliminates 60-80% waste
- **Starvation**: Ensures new requests get scheduled fairly
- **Throughput**: Batches requests to amortize weight loading

### Why This Approach
- Continuous batching vs. static batching → 2-4× higher throughput
- Paged allocation enables efficient memory utilization
- Prefix caching saves redundant computation

### Output Produced
- `SchedulerOutput` containing:
  - List of requests to execute this step
  - Block tables mapping logical → physical KV blocks
  - Computed prefix lengths for each request

---

## Stage 3: Prefill (Initial Token Processing)

### What is Happening on GPU Hardware

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREFILL: GPU HARDWARE VIEW                                │
│                                                                              │
│  INPUT: 38 tokens for request "abc123"                                       │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 1: Embedding Lookup                                                ││
│  │                                                                         ││
│  │   HBM (embedding table)     SMs (compute)        HBM (output)           ││
│  │   ┌─────────────────┐       ┌───────────┐       ┌─────────────────┐     ││
│  │   │ Token 50123 → row│  →   │ Copy to   │   →   │ Embed[0] = vec  │     ││
│  │   │ Token 31456 → row│      │ output    │       │ Embed[1] = vec  │     ││
│  │   │     ...          │      │ buffer    │       │ Embed[37] = vec │     ││
│  │   └─────────────────┘       └───────────┘       └─────────────────┘     ││
│  │                                                                         ││
│  │   Memory: Read 38 × 8KB = 304KB (tiny)                                  ││
│  │   Compute: None (pure memory operation)                                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 2: Self-Attention (per layer, 80 layers)                           ││
│  │                                                                         ││
│  │   ┌─────────────────────────────────────────────────────────────────┐   ││
│  │   │ 2a. QKV Projection (GEMM - General Matrix Multiply)             │   ││
│  │   │                                                                 │   ││
│  │   │   X[38, 8192]  ×  W_qkv[8192, 24576]  =  QKV[38, 24576]        │   ││
│  │   │                                                                 │   ││
│  │   │   ┌──────────┐     ┌──────────┐     ┌──────────┐               │   ││
│  │   │   │ Hidden   │     │ Weights  │     │ Q, K, V  │               │   ││
│  │   │   │ States   │  ×  │ (FP8)    │  =  │ tensors  │               │   ││
│  │   │   └──────────┘     └──────────┘     └──────────┘               │   ││
│  │   │                                                                 │   ││
│  │   │   Compute: 38 × 8192 × 24576 × 2 = 15.4 GFLOPs                 │   ││
│  │   │   Memory:  Read weights 200MB (shared across batch)             │   ││
│  │   │   B200 Tensor Cores: ~100µs at FP8                             │   ││
│  │   └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                         ││
│  │   ┌─────────────────────────────────────────────────────────────────┐   ││
│  │   │ 2b. Attention Score Computation                                 │   ││
│  │   │                                                                 │   ││
│  │   │   Q[38, 128, 64]  ×  K^T[38, 128, 64]  =  Scores[38, 128, 38]  │   ││
│  │   │                                                                 │   ││
│  │   │   For each of 128 attention heads:                              │   ││
│  │   │   - Compute Q·K^T for all 38 query positions                    │   ││
│  │   │   - Apply causal mask (upper triangle = -∞)                     │   ││
│  │   │   - Softmax over key dimension                                  │   ││
│  │   │   - Multiply by V to get output                                 │   ││
│  │   │                                                                 │   ││
│  │   │   ┌───────────────────────────────────────────┐                 │   ││
│  │   │   │ FLASH ATTENTION on B200:                  │                 │   ││
│  │   │   │ - Tiles computation to fit in SRAM        │                 │   ││
│  │   │   │ - Fuses softmax with matmul               │                 │   ││
│  │   │   │ - Never materializes full attention matrix│                 │   ││
│  │   │   │ - Memory: O(N) instead of O(N²)           │                 │   ││
│  │   │   └───────────────────────────────────────────┘                 │   ││
│  │   └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                         ││
│  │   ┌─────────────────────────────────────────────────────────────────┐   ││
│  │   │ 2c. Store K, V to KV Cache (Paged)                              │   ││
│  │   │                                                                 │   ││
│  │   │   For token positions 0-37:                                     │   ││
│  │   │     slot = block_table[logical_block] × block_size + offset     │   ││
│  │   │     kv_cache[layer][slot] = (K[pos], V[pos])                    │   ││
│  │   │                                                                 │   ││
│  │   │   ┌─────────────────────────────────────────────────────────┐   │   ││
│  │   │   │           PHYSICAL KV CACHE BLOCKS IN HBM               │   │   ││
│  │   │   │                                                         │   │   ││
│  │   │   │   Block P47: [tok0-tok15 K/V for all 80 layers]        │   │   ││
│  │   │   │   Block P48: [tok16-tok31 K/V for all 80 layers]       │   │   ││
│  │   │   │   Block P49: [tok32-tok37 K/V, partially filled]       │   │   ││
│  │   │   │                                                         │   │   ││
│  │   │   │   Each block: 16 tokens × 80 layers × 2 (K,V) ×        │   │   ││
│  │   │   │               128 heads × 64 dims × 2 bytes = 20.5MB   │   │   ││
│  │   │   └─────────────────────────────────────────────────────────┘   │   ││
│  │   └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 3: FFN (Feed-Forward Network) - per layer                          ││
│  │                                                                         ││
│  │   X[38, 8192]  →  GEMM1  →  SiLU  →  GEMM2  →  X'[38, 8192]            ││
│  │                                                                         ││
│  │   GEMM1: X × W_up[8192, 28672] = 38 × 8192 × 28672 × 2 = 18 GFLOPs     ││
│  │   GEMM2: H × W_down[28672, 8192] = 38 × 28672 × 8192 × 2 = 18 GFLOPs   ││
│  │                                                                         ││
│  │   Total FFN: ~36 GFLOPs per layer × 80 layers = 2.9 TFLOPs              ││
│  │   B200 at FP8: 4.5 PFLOPS → ~0.6ms compute (but memory-bound in decode) ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  PREFILL SUMMARY FOR 38 TOKENS:                                              │
│  ───────────────────────────────                                             │
│  Total Compute: ~4 TFLOPs                                                    │
│  Total Memory:  ~150GB read (weights), ~60MB write (KV cache)                │
│  Time: ~25-35ms (COMPUTE-BOUND on B200)                                      │
│  Why compute-bound: Large matrix multiplies fully utilize Tensor Cores       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Problem It Solves
- Processes the entire prompt in parallel (not sequentially)
- Builds the KV cache needed for all future decode steps
- Provides the first token logits

### Why This Approach
- Parallel processing is ~N× faster than sequential for N tokens
- Flash Attention reduces memory from O(N²) to O(N)
- FP8 Tensor Cores on B200 provide 2.25× compute vs H100

### Output Produced
- Logits for the last token position (used for first generated token)
- KV cache populated with K,V tensors for all 38 input positions

---

## Stage 4: Decode Loop (Token Generation)

### What is Happening on GPU Hardware

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DECODE: GPU HARDWARE VIEW (Per Token)                     │
│                                                                              │
│  INPUT: 1 new token (position 38+i), KV cache has positions 0 to 37+i       │
│  BATCHED: 64 requests generating simultaneously                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 1: QKV Projection for NEW token only                               ││
│  │                                                                         ││
│  │   X[64×1, 8192]  ×  W_qkv[8192, 24576]  =  QKV[64, 24576]              ││
│  │   (64 requests, 1 token each)                                           ││
│  │                                                                         ││
│  │   Compute: 64 × 1 × 8192 × 24576 × 2 = 26 GFLOPs (tiny!)               ││
│  │   Memory:  Read 200MB weights → MEMORY BOUND                            ││
│  │                                                                         ││
│  │   ┌─────────────────────────────────────────────────────────────────┐   ││
│  │   │ ROOFLINE ANALYSIS:                                              │   ││
│  │   │                                                                 │   ││
│  │   │   Compute: 26 GFLOPs                                           │   ││
│  │   │   Memory:  200 MB to read                                       │   ││
│  │   │   Arithmetic Intensity: 26 GFLOP / 0.2 GB = 130 FLOP/byte      │   ││
│  │   │                                                                 │   ││
│  │   │   B200 roofline: 4.5 PFLOPS / 8 TB/s = 562 FLOP/byte           │   ││
│  │   │                                                                 │   ││
│  │   │   130 << 562 → MEMORY BOUND                                     │   ││
│  │   │   Time = 200MB / 8TB/s = 25µs (limited by bandwidth)           │   ││
│  │   └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 2: PagedAttention Kernel                                           ││
│  │                                                                         ││
│  │   ┌─────────────────────────────────────────────────────────────────┐   ││
│  │   │ For each request in batch (64 total):                           │   ││
│  │   │   1. Look up block_table for this request                       │   ││
│  │   │   2. For each block (non-contiguous in memory!):                │   ││
│  │   │      - Load K block from physical address                       │   ││
│  │   │      - Compute Q·K^T for tokens in this block                   │   ││
│  │   │      - Accumulate softmax denominator                           │   ││
│  │   │   3. For each block again:                                       │   ││
│  │   │      - Load V block                                              │   ││
│  │   │      - Multiply by attention weights                            │   ││
│  │   │      - Accumulate output                                        │   ││
│  │   │   4. Store new K,V to cache                                     │   ││
│  │   └─────────────────────────────────────────────────────────────────┘   ││
│  │                                                                         ││
│  │   ┌─────────────────────────────────────────────────────────────────┐   ││
│  │   │              PAGED ATTENTION MEMORY ACCESS                      │   ││
│  │   │                                                                 │   ││
│  │   │   Request "abc123" (context length = 100 tokens):               │   ││
│  │   │                                                                 │   ││
│  │   │   Block Table: [Logical] → [Physical]                          │   ││
│  │   │                 0 → P47                                         │   ││
│  │   │                 1 → P12                                         │   ││
│  │   │                 2 → P89                                         │   ││
│  │   │                 3 → P03                                         │   ││
│  │   │                 4 → P56                                         │   ││
│  │   │                 5 → P71                                         │   ││
│  │   │                 6 → P23 (partially filled)                      │   ││
│  │   │                                                                 │   ││
│  │   │   HBM Layout (non-contiguous but fully utilized):               │   ││
│  │   │   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐           │   ││
│  │   │   │P03 │P05 │P12 │P17 │P23 │P47 │P56 │P71 │P89 │... │           │   ││
│  │   │   │abc │REQ │abc │REQ │abc │abc │abc │abc │abc │... │           │   ││
│  │   │   │ B3 │ X  │ B1 │ Y  │ B6 │ B0 │ B4 │ B5 │ B2 │... │           │   ││
│  │   │   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘           │   ││
│  │   │                                                                 │   ││
│  │   │   Memory read per request: 7 blocks × 2.6MB = 18.2 MB          │   ││
│  │   │   For batch of 64: ~1.2 GB read                                │   ││
│  │   │   At 8 TB/s: ~150µs                                             │   ││
│  │   └─────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 3: FFN Layers (same as prefill, but for 1 token × 64 batch)        ││
│  │                                                                         ││
│  │   X[64, 8192]  ×  W_up[8192, 28672]  ×  W_down[28672, 8192]             ││
│  │                                                                         ││
│  │   Compute: 64 × 8192 × 28672 × 2 × 2 = 60 GFLOPs per layer             ││
│  │   Memory:  ~470 MB weights per layer                                    ││
│  │   Arithmetic Intensity: 60 GFLOP / 0.47 GB = 128 FLOP/byte             ││
│  │   → Still MEMORY BOUND (128 << 562)                                     ││
│  │                                                                         ││
│  │   Time: 470MB / 8TB/s = 60µs per layer                                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  DECODE STEP SUMMARY (Batch size 64, avg 100 token context):                 │
│  ──────────────────────────────────────────────────────────                  │
│  Per Layer:                                                                  │
│    QKV Projection:  ~25µs  (memory-bound, reading weights)                   │
│    PagedAttention:  ~150µs (memory-bound, reading KV cache)                  │
│    FFN:             ~120µs (memory-bound, reading weights)                   │
│    Other:           ~5µs   (layernorm, residual, etc.)                       │
│    Total:           ~300µs per layer                                         │
│                                                                              │
│  For 80 layers: 300µs × 80 = 24ms per decode step                            │
│  Throughput: 64 tokens / 24ms = 2,667 tokens/second                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ WHY DECODE IS MEMORY-BOUND (The Critical Insight)                       ││
│  │                                                                         ││
│  │   Total data read per decode step (batch 64):                           ││
│  │     Model weights:    ~140 GB (70B params × 2 bytes)                    ││
│  │     KV cache:         ~80 GB (64 reqs × avg 100 tok × 10KB/tok/layer)   ││
│  │     Total:            ~220 GB                                            ││
│  │                                                                         ││
│  │   At 8 TB/s: 220 GB / 8 TB/s = 27.5 ms                                  ││
│  │                                                                         ││
│  │   This is THE bottleneck. More compute won't help.                      ││
│  │   Only higher bandwidth (B200 vs H100) or smaller models help.          ││
│  │   Batch size helps amortize weight loading (read once, use 64×)         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Problem It Solves
- Generates output tokens autoregressively
- Paged memory access enables non-contiguous KV cache utilization
- Batching amortizes weight loading across multiple requests

### Why This Approach
- Autoregressive generation is inherent to transformer architecture
- PagedAttention custom kernel handles non-contiguous memory efficiently
- Continuous batching maximizes GPU utilization

### Output Produced
- Logits for the next token position
- Updated KV cache with new K,V vectors appended

---

## Stage 5: Sampling

### What is Happening

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAMPLING STAGE                                       │
│                                                                              │
│  INPUT: Logits tensor [batch_size=64, vocab_size=128256]                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 1: Apply Temperature                                               ││
│  │   logits = logits / temperature                                         ││
│  │   (temperature=0.7 makes distribution "sharper")                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 2: Top-K Filtering                                                 ││
│  │   Keep only top 50 tokens, set rest to -∞                               ││
│  │                                                                         ││
│  │   Before: [0.1, 0.05, 0.02, 0.001, ..., 0.0001] (128256 values)        ││
│  │   After:  [0.1, 0.05, 0.02, -∞, ..., 0.001] (50 non-masked)            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 3: Top-P (Nucleus) Filtering                                       ││
│  │   Keep smallest set of tokens with cumulative prob ≥ 0.95              ││
│  │   Typically keeps 10-30 tokens                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 4: Softmax → Probabilities                                         ││
│  │   probs = softmax(filtered_logits)                                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STEP 5: Multinomial Sampling (GPU)                                      ││
│  │   next_token = sample from categorical distribution                     ││
│  │                                                                         ││
│  │   Uses cuRAND for random number generation                              ││
│  │   Typically ~100µs for batch of 64                                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  OUTPUT: token_ids [64] - one new token per request                         │
│                                                                              │
│  Latency: ~200µs total (negligible compared to model forward pass)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What Problem It Solves
- Converts continuous logits to discrete token choices
- Enables controllable randomness (temperature, top-k, top-p)
- Supports deterministic generation with seed

### Why This Approach
- Sampling parameters allow quality/creativity trade-off
- GPU-accelerated for batch efficiency
- Fused kernels minimize memory round-trips

### Output Produced
- Token ID(s) for each request in the batch
- Ready for detokenization or next decode step

---

## Stage 6: Continuous Batching Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              CONTINUOUS BATCHING: REQUEST INTERLEAVING                       │
│                                                                              │
│  Time →  0ms   10ms   20ms   30ms   40ms   50ms   60ms   70ms   80ms        │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Req A: [PREFILL 50tok]──────────────[D][D][D][D][D][D][D][D]→ complete     │
│                                                                              │
│  Req B:        [PREFILL 30tok]───────[D][D][D][D][D]→ complete              │
│                                                                              │
│  Req C:              [P20]───────────[D][D][D][D][D][D][D][D][D][D]→...     │
│                                                                              │
│  Req D:                    [PREFILL 100tok]──────────────[D][D][D]→...      │
│                                                                              │
│  Req E:                              [P15]──[D][D][D][D][D][D]→ complete    │
│                                                                              │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  BATCH COMPOSITION AT t=40ms:                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Req A: Decode step 3 (1 token)                                         ││
│  │  Req B: Decode step 1 (1 token)                                         ││
│  │  Req C: Decode step 1 (1 token)                                         ││
│  │  Req D: Prefill (100 tokens) ← Different compute pattern!              ││
│  │  Req E: Prefill (15 tokens)                                             ││
│  │                                                                         ││
│  │  Total tokens this step: 3 decode + 115 prefill = 118 tokens            ││
│  │  Scheduler decides to run this as mixed batch or split                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  KEY INSIGHT: Without continuous batching, Req D would wait until A,B,C     │
│  all complete. With it, D starts immediately → lower TTFT, higher GPU util. │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ CHUNKED PREFILL (Advanced Feature):                                     ││
│  │                                                                         ││
│  │ If Req D's prefill is too long, chunk it:                               ││
│  │   Chunk 1: tokens [0:512]   → batch with existing decodes               ││
│  │   Chunk 2: tokens [512:1024] → next step                                ││
│  │   Chunk 3: tokens [1024:end] → ...                                      ││
│  │                                                                         ││
│  │ Why: Prevents long prefills from blocking decode progress               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 6. Comparison: No Engine vs vLLM vs Optimized B200

## Comparison Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE APPROACH COMPARISON                             │
│                                                                              │
│                      │ Naive PyTorch │   vLLM       │ Optimized B200       │
│ ─────────────────────┼───────────────┼──────────────┼───────────────────── │
│ Memory Management    │ Contiguous    │ Paged        │ Paged + FP8 KV       │
│ Fragmentation        │ 60-80%        │ ~4%          │ ~4%                  │
│ Max Concurrent Reqs  │ 4-8           │ 50-200       │ 100-400              │
│ Batching             │ Static        │ Continuous   │ Continuous + Chunked │
│ Prefix Caching       │ None          │ Hash-based   │ Hash-based           │
│ Quantization Support │ Manual        │ FP8/INT8     │ FP8/FP4 native       │
│ TTFT (50 tokens)     │ ~100ms        │ ~25ms        │ ~15ms                │
│ Decode (per token)   │ ~50ms         │ ~20ms        │ ~12ms                │
│ Throughput (tok/s)   │ 200-500       │ 2000-5000    │ 5000-10000           │
│ GPU Utilization      │ 10-30%        │ 60-80%       │ 80-95%               │
│ ─────────────────────┼───────────────┼──────────────┼───────────────────── │
│ Complexity           │ Low           │ Medium       │ High                 │
│ Setup Time           │ Minutes       │ Hours        │ Days                 │
│ Debugging Ease       │ Easy          │ Medium       │ Hard                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Comparison: Memory Utilization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY LAYOUT COMPARISON                                  │
│                                                                              │
│  NAIVE PYTORCH (Pre-allocated, contiguous):                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ┌───────────────────────────────────────────────────────────────────┐  ││
│  │  │ Req 1: Reserved 4096 tokens ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ││  ││
│  │  │        (Using only 200)      ▲ WASTED                            ││  ││
│  │  └───────────────────────────────────────────────────────────────────┘  ││
│  │  ┌───────────────────────────────────────────────────────────────────┐  ││
│  │  │ Req 2: Reserved 4096 tokens ███████████░░░░░░░░░░░░░░░░░░░░░░░░░ ││  ││
│  │  │        (Using only 350)      ▲ WASTED                            ││  ││
│  │  └───────────────────────────────────────────────────────────────────┘  ││
│  │  ┌───────────────────────────────────────────────────────────────────┐  ││
│  │  │ Req 3: OOM! Cannot allocate another 4096-token block             ││  ││
│  │  └───────────────────────────────────────────────────────────────────┘  ││
│  │                                                                         ││
│  │  Total: 8192 tokens reserved, 550 used → 93% waste                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  VLLM PAGED (On-demand, non-contiguous):                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐    ││
│  │  │ R1 │ R1 │ R2 │ R1 │ R2 │ R2 │ R3 │ R3 │ R4 │ R4 │ R5 │Free│Free│    ││
│  │  │ B0 │ B1 │ B0 │ B2 │ B1 │ B2 │ B0 │ B1 │ B0 │ B1 │ B0 │    │    │    ││
│  │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘    ││
│  │   16   16   16   16   16   16   16   16   16   16   16                  ││
│  │  tokens each block                                                       ││
│  │                                                                         ││
│  │  Total: 5 requests running, only ~4% internal fragmentation             ││
│  │  Can add more requests as blocks become available                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  B200 WITH FP8 KV CACHE:                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Same paging as vLLM, but:                                              ││
│  │  - KV values stored in FP8 (1 byte) vs FP16 (2 bytes)                   ││
│  │  - 2× more tokens fit in same memory                                    ││
│  │  - 192GB HBM3e vs 80GB HBM3 → 2.4× base capacity                       ││
│  │  - Combined: ~4.8× more concurrent requests possible                    ││
│  │                                                                         ││
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐    ││
│  │  │ R1 │ R2 │ R3 │ R4 │ R5 │ R6 │ R7 │ R8 │ R9 │R10 │R11 │R12 │... │    ││
│  │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘    ││
│  │                                                                         ││
│  │  Same memory, 4× more requests → 4× throughput potential                ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Comparison: Throughput

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THROUGHPUT ANALYSIS                                       │
│                                                                              │
│  Workload: 1000 requests, 100 input tokens, 50 output tokens each           │
│  Model: Llama-70B                                                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ NAIVE PYTORCH (Static batching, batch size 4):                          ││
│  │                                                                         ││
│  │   Batch 1: Process 4 requests                                           ││
│  │     Prefill: 400 tokens → ~200ms                                        ││
│  │     Decode:  50 steps × 4 reqs × ~50ms = ~10 seconds                    ││
│  │     Wait for ALL to complete before next batch                          ││
│  │                                                                         ││
│  │   Total: 250 batches × ~10.2s = 2,550 seconds                           ││
│  │   Throughput: 1000 × 50 / 2550 = 20 tokens/second                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ VLLM (Continuous batching, dynamic batch up to 64):                     ││
│  │                                                                         ││
│  │   No waiting - new requests join immediately                            ││
│  │   Batch size grows to 64, stays high throughout                         ││
│  │                                                                         ││
│  │   Prefill: 100 tok × 64 reqs (chunked) → distributed across steps       ││
│  │   Decode:  64 reqs × 50 steps × ~25ms/step = ~1250ms per 64 reqs        ││
│  │                                                                         ││
│  │   Total: ~1000 reqs / (50 tok × ~25ms amortized) ≈ 50 seconds          ││
│  │   Throughput: 50,000 / 50 = 1,000 tokens/second                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ B200 OPTIMIZED (FP8, larger batches, faster memory):                    ││
│  │                                                                         ││
│  │   FP8 weights + FP8 KV cache → 2× memory efficiency                    ││
│  │   8 TB/s bandwidth → 2.4× faster decode                                ││
│  │   Larger batch (up to 256) → better weight amortization                ││
│  │                                                                         ││
│  │   Decode: 256 reqs × 50 steps × ~12ms/step = ~600ms per 256 reqs       ││
│  │                                                                         ││
│  │   Total: ~1000 reqs / 256 × 0.6s ≈ 2.5 seconds                         ││
│  │   Throughput: 50,000 / 2.5 = 20,000 tokens/second                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  SUMMARY:                                                                    │
│  ───────────────────────────────────────────────────────────────────────────│
│  │ Approach         │ Tokens/sec │ Improvement vs Naive │                   │
│  ├──────────────────┼────────────┼──────────────────────┤                   │
│  │ Naive PyTorch    │ 20         │ 1× (baseline)        │                   │
│  │ vLLM             │ 1,000      │ 50×                  │                   │
│  │ B200 Optimized   │ 20,000     │ 1,000×               │                   │
│  └──────────────────────────────────────────────────────────────────────────│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Code Comparison

### Naive PyTorch (No Inference Engine)

```python
# Naive approach - what breaks at scale
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b")

def generate_naive(prompts, max_tokens=50):
    """Static batching, no KV cache optimization, no paging"""
    results = []
    
    # Problem 1: Must pad all sequences to same length
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    
    # Problem 2: Pre-allocate KV cache for max possible length
    # Each request reserves max_length × layers × heads × head_dim × 2 (K,V)
    # For 70B model: 4096 × 80 × 64 × 128 × 2 × 2 bytes ≈ 10GB per request!
    
    # Problem 3: Static batch - must wait for longest sequence
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model(**inputs)
        
        # All sequences generate one token, even if some are done
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1)
        inputs["input_ids"] = torch.cat([inputs["input_ids"], 
                                          next_tokens.unsqueeze(1)], dim=1)
    
    return tokenizer.batch_decode(inputs["input_ids"])

# Why this fails:
# 1. Memory: 4 concurrent requests × 10GB = 40GB just for KV cache
# 2. Waste: Short responses wait for long ones
# 3. Throughput: GPU sits idle during memory transfers
# 4. No prefix sharing: Repeated prompts recompute everything
```

### vLLM Approach

```python
# vLLM approach - what it fixes
from vllm import LLM, SamplingParams

# Initialize once - handles all complexity internally
llm = LLM(
    model="meta-llama/Llama-2-70b",
    tensor_parallel_size=4,              # Multi-GPU
    gpu_memory_utilization=0.90,         # Use 90% of GPU memory
    enable_prefix_caching=True,          # Reuse common prefixes
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=50,
)

# Just send requests - vLLM handles:
# 1. Paged KV cache allocation (no fragmentation)
# 2. Continuous batching (no waiting for slow requests)
# 3. Prefix caching (reuse repeated prompts)
# 4. Optimal scheduling (balance TTFT vs throughput)
outputs = llm.generate(prompts, sampling_params)

# What vLLM does internally:
# - Allocates 16-token blocks on-demand
# - Schedules requests based on available memory
# - Batches prefill and decode operations efficiently
# - Shares KV blocks for identical prefixes
```

### B200 Optimized (with vLLM)

```python
# B200-specific optimizations
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70b",
    tensor_parallel_size=1,                    # B200 can handle 70B on single GPU!
    gpu_memory_utilization=0.95,               # More aggressive with 192GB
    quantization="fp8",                        # Native FP8 on Blackwell
    kv_cache_dtype="fp8",                      # FP8 KV cache
    enable_prefix_caching=True,
    enable_chunked_prefill=True,               # Better latency distribution
    max_num_batched_tokens=8192,               # Larger batches on B200
    max_num_seqs=256,                          # More concurrent requests
)

# B200-specific features used:
# 1. FP8 Tensor Cores: 2× throughput vs FP16
# 2. 192GB HBM3e: 2.4× more KV cache capacity
# 3. 8 TB/s bandwidth: 2.4× faster decode
# 4. SM100 kernels: Optimized for Blackwell architecture
```

---

# 7. Trade-offs and Bottlenecks

## Remaining Bottlenecks on B200

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BOTTLENECKS THAT PERSIST ON B200                          │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 1. MEMORY BANDWIDTH IS STILL THE DECODE BOTTLENECK                      ││
│  │                                                                         ││
│  │    Even with 8 TB/s, decode is memory-bound:                            ││
│  │    - Reading 140GB model weights per step still takes ~17ms             ││
│  │    - KV cache reads add to this                                         ││
│  │    - Compute (4.5 PFLOPS) remains underutilized during decode           ││
│  │                                                                         ││
│  │    Solution: Higher batch sizes, speculative decoding, model sharding   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 2. LONG CONTEXT ATTENTION COMPLEXITY                                    ││
│  │                                                                         ││
│  │    Attention is O(N²) in sequence length for each layer                ││
│  │    - 100K context: ~10 billion attention operations per layer          ││
│  │    - Flash Attention helps but doesn't eliminate the scaling            ││
│  │    - Long prompts cause TTFT spikes                                     ││
│  │                                                                         ││
│  │    Solution: Chunked prefill, sliding window attention, MLA             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 3. MULTI-GPU COMMUNICATION OVERHEAD                                     ││
│  │                                                                         ││
│  │    For models larger than 192GB (need multiple B200s):                  ││
│  │    - Tensor parallel requires all-reduce after each layer               ││
│  │    - NVLink 5.0 at 1.8 TB/s is fast but not infinite                   ││
│  │    - Synchronization points limit scaling efficiency                    ││
│  │                                                                         ││
│  │    Solution: Pipeline parallelism, expert parallelism for MoE           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 4. KV CACHE STILL LIMITS CONCURRENCY                                    ││
│  │                                                                         ││
│  │    For long contexts (32K+ tokens):                                     ││
│  │    - Each request: 32K × 80 layers × 2 (K,V) × 128 × 64 × 1 byte (FP8) ││
│  │    - = 1.3 GB per request                                               ││
│  │    - 192GB / 1.3GB = ~150 concurrent long-context requests max          ││
│  │                                                                         ││
│  │    Solution: KV cache compression, attention sinks, cache eviction      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ 5. QUANTIZATION ACCURACY TRADE-OFFS                                     ││
│  │                                                                         ││
│  │    FP8 and FP4 reduce memory but may impact:                            ││
│  │    - Model accuracy (small degradation usually acceptable)              ││
│  │    - Specific task performance (varies by model/task)                   ││
│  │    - Reproducibility (numerical differences)                            ││
│  │                                                                         ││
│  │    Solution: Calibration, per-layer quantization, fallback to FP16      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Trade-off Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRADE-OFF MATRIX                                     │
│                                                                              │
│  ┌───────────────────┬────────────────────┬─────────────────────────────┐   │
│  │ Optimization      │ Benefit            │ Cost                        │   │
│  ├───────────────────┼────────────────────┼─────────────────────────────┤   │
│  │ FP8 Quantization  │ 2× throughput      │ Slight accuracy loss        │   │
│  │ FP4 Quantization  │ 4× memory savings  │ More accuracy loss          │   │
│  │ Larger batch size │ Better throughput  │ Higher latency per request  │   │
│  │ Prefix caching    │ Reduce prefill     │ Memory for cache storage    │   │
│  │ Chunked prefill   │ Lower TTFT tail    │ Slightly higher avg latency │   │
│  │ Speculative decode│ Lower token latency│ Wasted compute on rejection │   │
│  │ Tensor parallel   │ Larger models      │ Communication overhead      │   │
│  │ KV cache eviction │ More concurrency   │ Recomputation on cache miss │   │
│  └───────────────────┴────────────────────┴─────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 8. Practical Implications

## When to Use Which Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DECISION FRAMEWORK                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ USE NAIVE PYTORCH WHEN:                                                 ││
│  │   ✓ Prototyping / experimenting                                         ││
│  │   ✓ Single request, no concurrency needed                               ││
│  │   ✓ Debugging model behavior                                            ││
│  │   ✓ Academic research with small-scale runs                             ││
│  │   ✗ Production serving                                                  ││
│  │   ✗ Cost-sensitive deployments                                          ││
│  │   ✗ Latency-sensitive applications                                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ USE VLLM WHEN:                                                          ││
│  │   ✓ Production LLM serving                                              ││
│  │   ✓ Multiple concurrent users                                           ││
│  │   ✓ Variable-length requests                                            ││
│  │   ✓ Repeated prompts (prefix caching helps)                             ││
│  │   ✓ Need OpenAI-compatible API                                          ││
│  │   ✗ Extremely latency-sensitive (single-digit ms)                       ││
│  │   ✗ Custom model architectures not supported                            ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ USE B200-OPTIMIZED VLLM WHEN:                                           ││
│  │   ✓ Maximum throughput required                                         ││
│  │   ✓ Large models (70B+) on single GPU                                   ││
│  │   ✓ High concurrency (100+ requests)                                    ││
│  │   ✓ FP8/FP4 accuracy acceptable                                         ││
│  │   ✓ Budget allows B200 hardware                                         ││
│  │   ✗ Need bit-exact reproducibility                                      ││
│  │   ✗ Cost-constrained environments                                       ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Metrics to Monitor

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MONITORING CHECKLIST                                      │
│                                                                              │
│  LATENCY METRICS:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  • TTFT (Time To First Token): Target < 100ms for interactive use       ││
│  │  • ITL (Inter-Token Latency): Target < 50ms for smooth streaming        ││
│  │  • p50/p95/p99 latencies: Watch for tail latency spikes                 ││
│  │  • Queue wait time: Indicates capacity issues                           ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  THROUGHPUT METRICS:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  • Tokens per second (TPS): Primary throughput measure                  ││
│  │  • Requests per second (RPS): User-facing throughput                    ││
│  │  • Batch size distribution: Should stay high for efficiency             ││
│  │  • GPU utilization: Target 80%+ for cost efficiency                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  MEMORY METRICS:                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  • KV cache utilization: High = good, unless causing preemptions        ││
│  │  • Block fragmentation: Should stay < 10%                               ││
│  │  • Prefix cache hit rate: Higher = better (for repeated prompts)        ││
│  │  • Preemption rate: Should be near zero in normal operation             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│  DIAGNOSTIC PATTERNS:                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  High TTFT, low throughput     → Insufficient batch size                ││
│  │  High p99 latency              → Long request or preemption             ││
│  │  Low GPU util, high queue time → Memory pressure                        ││
│  │  Low prefix cache hits         → Diverse prompts or small cache         ││
│  │  OOM errors                    → Reduce max_num_seqs or context length  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Configuration Recommendations for B200

```python
# Recommended vLLM configuration for Blackwell B200
from vllm import LLM, SamplingParams

# For throughput-optimized serving
llm_throughput = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=1,           # B200 can handle 70B on single GPU
    gpu_memory_utilization=0.92,      # Leave 8% for activations
    quantization="fp8",               # Native FP8 for 2× throughput
    kv_cache_dtype="fp8",             # FP8 KV cache for 2× capacity
    enable_prefix_caching=True,       # Always enable for production
    enable_chunked_prefill=True,      # Reduce TTFT spikes
    max_num_batched_tokens=16384,     # Large batches for B200
    max_num_seqs=256,                 # High concurrency
    enforce_eager=False,              # Enable CUDA graphs
)

# For latency-optimized serving
llm_latency = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,      # More headroom
    quantization="fp8",
    kv_cache_dtype="fp8",
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    max_num_batched_tokens=4096,      # Smaller batches for lower latency
    max_num_seqs=64,                  # Lower concurrency
    max_seq_len_to_capture=2048,      # Optimize CUDA graphs for short seqs
)

# Environment variables for B200 optimization
# export VLLM_ATTENTION_BACKEND=FLASHINFER  # Or FLASH_ATTN for FA3
# export VLLM_USE_V1=1                       # Use vLLM v1 architecture
# export CUDA_VISIBLE_DEVICES=0              # Single B200
```

---

## Summary: Why Optimized Inference Matters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE BOTTOM LINE                                           │
│                                                                              │
│  LLM inference is fundamentally memory-bound during decode:                  │
│  ──────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  1. THE PROBLEM:                                                             │
│     • Each decode step reads entire model weights + KV cache                 │
│     • For 70B model: ~140GB read per decode step                            │
│     • GPU compute sits 80-90% idle waiting for memory                        │
│                                                                              │
│  2. THE SOLUTIONS:                                                           │
│     • Paged KV cache: Eliminates 60-80% memory fragmentation                │
│     • Continuous batching: Amortizes weight loading across requests          │
│     • Prefix caching: Avoids redundant prefill computation                   │
│     • FP8/FP4 quantization: 2-4× memory reduction                           │
│                                                                              │
│  3. THE RESULTS (B200 vs Naive):                                            │
│     • Throughput: 20,000 vs 20 tokens/sec (1000× improvement)               │
│     • Concurrency: 256 vs 4 requests (64× improvement)                       │
│     • Cost efficiency: ~100× better $/token                                 │
│                                                                              │
│  4. REMAINING CHALLENGES:                                                    │
│     • Memory bandwidth still limits decode speed                             │
│     • Long contexts still expensive (O(N²) attention)                        │
│     • Multi-GPU communication overhead for largest models                    │
│                                                                              │
│  5. KEY INSIGHT:                                                             │
│     The difference between naive and optimized inference is not 2-3×.       │
│     It's 100-1000×. This is why inference engines like vLLM exist,          │
│     and why hardware like B200 (with massive bandwidth) matters.             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180) - PagedAttention and efficient LLM serving
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Memory-efficient attention
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) - B200 specifications
- [vLLM Documentation](https://docs.vllm.ai/) - Configuration and API reference
- [vLLM GitHub](https://github.com/vllm-project/vllm) - Source code (see `csrc/attention/mla/` for SM100 kernels)

---

*Document generated for understanding hardware-level LLM inference on NVIDIA Blackwell B200 GPU*
