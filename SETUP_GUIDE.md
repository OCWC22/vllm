# vLLM Setup Guide: Serving Kimi K2.5, Qwen3-VL-235B, and Qwen3-32B-VL on H100/B200 GPUs

A practical, detailed guide for engineers with no prior background in GPU inference.
This guide covers what happens inside vLLM, how GPUs actually execute model inference,
and how to configure everything for three specific models on NVIDIA H100 and B200 hardware.

---

## Table of Contents

1. [What Is vLLM?](#1-what-is-vllm)
2. [GPU Fundamentals for LLM Inference](#2-gpu-fundamentals-for-llm-inference)
3. [vLLM Architecture Deep Dive](#3-vllm-architecture-deep-dive)
4. [KV Cache and PagedAttention](#4-kv-cache-and-pagedattention)
5. [Tensor Parallelism and Multi-GPU Serving](#5-tensor-parallelism-and-multi-gpu-serving)
6. [Multimodal / Vision-Language Model Pipeline](#6-multimodal--vision-language-model-pipeline)
7. [Model-Specific Guides](#7-model-specific-guides)
   - [Kimi K2.5](#kimi-k25-moonshot-ai)
   - [Qwen3-VL-235B-A22B](#qwen3-vl-235b-a22b)
   - [Qwen3-32B-VL](#qwen3-32b-vl)
8. [H100 vs B200 Hardware Comparison](#8-h100-vs-b200-hardware-comparison)
9. [Practical Deployment Commands](#9-practical-deployment-commands)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What Is vLLM?

vLLM is a high-throughput inference engine for large language models (LLMs). It takes a
model's weights (downloaded from Hugging Face), loads them onto one or more GPUs, and
serves an OpenAI-compatible HTTP API so applications can send prompts and receive responses.

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                        YOUR APPLICATION                        │
 │              (sends HTTP requests with prompts)                 │
 └──────────────────────────────┬──────────────────────────────────┘
                                │  HTTP POST /v1/chat/completions
                                ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     vLLM API SERVER                             │
 │  (FastAPI app: vllm/entrypoints/openai/api_server.py)          │
 │                                                                 │
 │  Receives requests, validates them, passes to the Engine        │
 └──────────────────────────────┬──────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     vLLM ENGINE CORE                            │
 │  (vllm/v1/engine/)                                             │
 │                                                                 │
 │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
 │  │  Scheduler   │  │ KV Cache Mgr │  │  Structured Output Mgr │ │
 │  │ (sched/)     │  │ (core/)      │  │                        │ │
 │  └──────┬───────┘  └──────┬───────┘  └────────────────────────┘ │
 └─────────┼─────────────────┼────────────────────────────────────-┘
           │                 │
           ▼                 ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │                     GPU WORKERS                                 │
 │  (vllm/v1/worker/gpu_worker.py)                                │
 │                                                                 │
 │  Each GPU gets one Worker process containing:                   │
 │  ┌────────────────────────────────────────────────────────┐     │
 │  │  GPUModelRunner (gpu_model_runner.py)                  │     │
 │  │  - Loads the model onto GPU memory                     │     │
 │  │  - Manages KV cache tensors                            │     │
 │  │  - Runs forward passes (prefill + decode)              │     │
 │  │  - Samples next tokens                                 │     │
 │  └────────────────────────────────────────────────────────┘     │
 └─────────────────────────────────────────────────────────────────┘
```

### The Two Phases of Generation

Every text generation request goes through two phases:

```
 PHASE 1: PREFILL                         PHASE 2: DECODE
 (Process all input tokens at once)       (Generate one token at a time)

 Input: "What is the capital of France?"  Already cached: all input tokens
                                          + previously generated tokens
        ┌───────────────────────┐
        │  All input tokens     │              ┌──────────────┐
        │  processed in         │              │ Generate ONE │
        │  ONE forward pass     │──────────▶   │ new token    │──┐
        │  (compute-bound)      │              │ per step     │  │
        └───────────────────────┘              │ (memory-     │  │
                                               │  bandwidth-  │  │
        Result: KV cache filled                │  bound)      │  │
        for all input tokens                   └──────┬───────┘  │
                                                      │          │
                                               Token: "The"     │
                                                      │ loop     │
                                                      ▼ back     │
                                               Token: "capital"◄─┘
                                               Token: "of"
                                               Token: "France"
                                               Token: "is"
                                               Token: "Paris"
                                               Token: "."
                                               Token: <EOS>  ──▶ DONE
```

**Prefill** is compute-bound: you have lots of tokens to process, so the GPU's
arithmetic units (tensor cores) are the bottleneck.

**Decode** is memory-bandwidth-bound: you process only 1 token per step, but you
must read the entire model's weights from GPU memory each time. The GPU spends
most of its time waiting for data to arrive from HBM (High Bandwidth Memory).

---

## 2. GPU Fundamentals for LLM Inference

### What's Inside a GPU?

A modern NVIDIA GPU (H100, B200) is organized like this:

```
 ┌──────────────────────────────── GPU DIE ────────────────────────────────┐
 │                                                                         │
 │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      ┌─────────┐    │
 │  │   SM 0  │ │   SM 1  │ │   SM 2  │ │   SM 3  │ ...  │ SM 131  │    │
 │  │         │ │         │ │         │ │         │      │ (H100)  │    │
 │  │ Tensor  │ │ Tensor  │ │ Tensor  │ │ Tensor  │      │         │    │
 │  │ Cores   │ │ Cores   │ │ Cores   │ │ Cores   │      │         │    │
 │  │ 128 FP32│ │ 128 FP32│ │ 128 FP32│ │ 128 FP32│      │         │    │
 │  │ cores   │ │ cores   │ │ cores   │ │ cores   │      │         │    │
 │  │         │ │         │ │         │ │         │      │         │    │
 │  │ 256KB   │ │ 256KB   │ │ 256KB   │ │ 256KB   │      │ 256KB   │    │
 │  │ L1/SMEM │ │ L1/SMEM │ │ L1/SMEM │ │ L1/SMEM │      │ L1/SMEM │    │
 │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      └─────────┘    │
 │                                                                         │
 │  ┌─────────────────────────────────────────────────────────────────┐    │
 │  │                         L2 CACHE (50MB on H100)                 │    │
 │  └─────────────────────────────────────────────────────────────────┘    │
 │                                                                         │
 │  ┌─────────────────────────────────────────────────────────────────┐    │
 │  │                    HBM3 (80GB on H100 SXM)                      │    │
 │  │                    Memory Bandwidth: 3.35 TB/s                  │    │
 │  │                                                                 │    │
 │  │    This is where model weights and KV cache live                │    │
 │  └─────────────────────────────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────────────────────────────┘

 SM = Streaming Multiprocessor (the basic compute unit)
```

### Key GPU Concepts

**Streaming Multiprocessor (SM):** The fundamental compute unit of an NVIDIA GPU.
Each SM contains:
- Tensor Cores: specialized matrix-multiply units (the main workhorse for LLM math)
- CUDA Cores: general-purpose floating point units
- Shared Memory / L1 Cache: fast on-chip memory (~256KB per SM on H100)
- Warp Schedulers: each SM can run multiple "warps" (groups of 32 threads)

**HBM (High Bandwidth Memory):** The main GPU memory. This is where model weights
and the KV cache are stored. Despite the name "high bandwidth," reading from HBM
is still the main bottleneck during decode. Think of it as the GPU's "hard drive" -
big but relatively slow to access compared to on-chip memory.

**Tensor Cores:** Specialized hardware circuits that can multiply small matrices
(e.g., 16x16) in a single clock cycle. This is what makes GPUs so fast at the
matrix multiplications that dominate transformer inference.

**FLOPS (Floating Point Operations Per Second):** Measures how fast the GPU can
do arithmetic. H100 SXM does ~990 TFLOPS at FP16 (half precision). But FLOPS
alone don't tell you how fast inference will be — you also need to consider
memory bandwidth.

**Memory Bandwidth:** Measures how fast data can be read from/written to HBM.
H100 SXM provides 3.35 TB/s. During decode, the GPU must read every model
weight once per token generated, so bandwidth determines decode speed.

**Arithmetic Intensity:** The ratio of compute operations to memory operations.
- High arithmetic intensity (prefill): GPU is doing lots of math per byte read
  → compute-bound → tensor cores are the bottleneck
- Low arithmetic intensity (decode): GPU reads lots of data but does little math
  per byte → memory-bandwidth-bound → HBM bandwidth is the bottleneck

```
 THE ROOFLINE MODEL (simplified)
 ─────────────────────────────────
 Performance depends on whether you're compute-bound or memory-bound:

                  ▲ Performance (FLOPS achieved)
                  │
  Compute  ───────┤━━━━━━━━━━━━━━━━━━━━━━━━━━━  ← Peak FLOPS (990 TF)
  Ceiling         │                          ╱
                  │                        ╱
                  │                      ╱
                  │                    ╱
                  │                  ╱  ← Memory bandwidth ceiling
                  │                ╱      (slope = 3.35 TB/s)
                  │              ╱
                  │            ╱
                  │          ╱
                  │        ╱
                  │      ╱
                  │    ╱
                  │  ╱
                  │╱
                  └──────────────────────────────▶
                     Arithmetic Intensity (FLOP/byte)

          DECODE lives        PREFILL lives
          here (left)         here (right)
          memory-bound        compute-bound
```

### Why This Matters for LLM Serving

During **decode** (generating tokens one at a time), the arithmetic intensity is
very low. For each output token, the GPU must read ALL model weights from HBM
but only does a small amount of computation. This means:

- A 70B parameter model in FP16 = 140 GB of weights
- To generate one token, read 140 GB from HBM
- H100 bandwidth = 3.35 TB/s → can read 140 GB in ~42ms
- That's ~24 tokens/second per request (for a single user)

vLLM's key insight: **batch multiple requests together**. If you serve 32 users
at once, you read the weights once but do 32x the computation. This pushes you
toward the compute-bound regime and dramatically improves throughput.

---

## 3. vLLM Architecture Deep Dive

### Component Overview

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │ vllm serve <model> --tensor-parallel-size 8                        │
 └───────────────┬─────────────────────────────────────────────────────┘
                 │
                 ▼
 ┌───────────────────────────────────────────────┐
 │          API Server (FastAPI)                  │
 │    vllm/entrypoints/openai/api_server.py      │
 │                                                │
 │  - /v1/chat/completions                        │
 │  - /v1/completions                             │
 │  - /v1/embeddings                              │
 │  Validates requests, streams responses         │
 └─────────────────────┬─────────────────────────┘
                       │
                       ▼
 ┌───────────────────────────────────────────────┐
 │         Engine Core Client                     │
 │    vllm/v1/engine/core_client.py              │
 │                                                │
 │  Communicates with Engine Core process         │
 │  (in-process or multi-process)                 │
 └─────────────────────┬─────────────────────────┘
                       │
                       ▼
 ┌───────────────────────────────────────────────────────────────────┐
 │                    ENGINE CORE PROCESS                            │
 │                                                                   │
 │  ┌─────────────────────────┐  ┌──────────────────────────────┐   │
 │  │       Scheduler          │  │     KV Cache Manager          │   │
 │  │  vllm/v1/core/sched/    │  │  vllm/v1/core/               │   │
 │  │  scheduler.py            │  │  kv_cache_manager.py          │   │
 │  │                          │  │                               │   │
 │  │  - Decides which reqs    │  │  - Tracks which KV cache      │   │
 │  │    to run each step      │  │    blocks are free/used       │   │
 │  │  - Enforces max batch    │  │  - Implements prefix caching  │   │
 │  │    size and token budget │  │    (reuse KV for shared       │   │
 │  │  - Handles preemption    │  │     prompt prefixes)          │   │
 │  │    (pausing requests     │  │  - Manages block allocation   │   │
 │  │     when memory is full) │  │    and eviction               │   │
 │  └────────────┬─────────────┘  └──────────────┬────────────────┘   │
 │               │                               │                   │
 │               ▼                               ▼                   │
 │  ┌───────────────────────────────────────────────────────────┐    │
 │  │                   SchedulerOutput                         │    │
 │  │  (which requests to process, token budgets, block tables) │    │
 │  └─────────────────────────┬─────────────────────────────────┘    │
 └────────────────────────────┼──────────────────────────────────────┘
                              │
                              ▼ (sent to all GPU workers)
 ┌───────────────────────────────────────────────────────────────────┐
 │                        GPU WORKERS (one per GPU)                  │
 │                                                                   │
 │  GPU 0 (driver)      GPU 1             GPU 2           GPU 7     │
 │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌────────┐ │
 │  │   Worker      │  │   Worker      │  │   Worker    │  │ Worker │ │
 │  │   ┌────────┐  │  │   ┌────────┐  │  │  ┌──────┐  │  │┌─────┐│ │
 │  │   │Model   │  │  │   │Model   │  │  │  │Model │  │  ││Model││ │
 │  │   │Runner  │  │  │   │Runner  │  │  │  │Runner│  │  ││Run. ││ │
 │  │   │        │  │  │   │        │  │  │  │      │  │  ││     ││ │
 │  │   │ Model  │  │  │   │ Model  │  │  │  │Model │  │  ││Model││ │
 │  │   │ Shard  │  │  │   │ Shard  │  │  │  │Shard │  │  ││Shard││ │
 │  │   │ 1/8th  │  │  │   │ 1/8th  │  │  │  │1/8th │  │  ││1/8th││ │
 │  │   └────────┘  │  │   └────────┘  │  │  └──────┘  │  │└─────┘│ │
 │  └──────────────┘  └──────────────┘  └────────────┘  └────────┘ │
 │                                                                   │
 │  All workers communicate via NCCL (NVIDIA Collective              │
 │  Communications Library) for tensor parallel all-reduce ops       │
 └───────────────────────────────────────────────────────────────────┘
```

### The Scheduler (vllm/v1/core/sched/scheduler.py)

The scheduler is the "brain" of vLLM. Every step (every forward pass), it decides:

1. **Which new requests to admit** (from the waiting queue)
2. **Which running requests to continue** (already being decoded)
3. **Which requests to preempt** (pause to free memory for higher-priority work)
4. **How many tokens each request gets** (prefill may process many tokens; decode processes one)

```
 Scheduler Decision Loop (every step):
 ──────────────────────────────────────

 ┌──────────────────────────────────┐
 │  Waiting Queue                   │
 │  [Req A: 2000 tokens to prefill] │
 │  [Req B: 500 tokens to prefill]  │
 │  [Req C: 100 tokens to prefill]  │
 └──────────────┬───────────────────┘
                │
                ▼
 ┌──────────────────────────────────────────────────┐
 │  Can we fit more requests?                        │
 │                                                    │
 │  Check constraints:                                │
 │  1. max_num_seqs (e.g., 256)                       │
 │     → max concurrent requests                      │
 │  2. max_num_batched_tokens (e.g., 8192)            │
 │     → max tokens processed in one forward pass     │
 │  3. KV cache blocks available?                     │
 │     → enough memory for this request's KV cache?   │
 └──────────────┬───────────────────────────────────┘
                │ Yes: schedule it
                ▼
 ┌──────────────────────────────────────────────────┐
 │  Running Set                                      │
 │  [Req X: decode (1 new token)]                    │
 │  [Req Y: decode (1 new token)]                    │
 │  [Req A: prefill (2000 new tokens)]  ← just added │
 │                                                    │
 │  Total tokens this step: 1 + 1 + 2000 = 2002      │
 │  This is under the 8192 budget → OK                │
 └──────────────────────────────────────────────────┘
```

### The Model Runner (vllm/v1/worker/gpu_model_runner.py)

The GPUModelRunner is the component that actually executes the model on the GPU.
It handles:

- **Loading the model** from Hugging Face weights into GPU memory
- **Managing input preparation** (token IDs, positions, attention metadata)
- **Running forward passes** through the transformer layers
- **Sampling** next tokens from the output logits
- **CUDA graph capture** for optimizing repeated decode operations

Key code path:
```
GPUModelRunner.__init__()
  → get_model_loader() → loads weights from HF
  → allocate KV cache tensors on GPU

GPUModelRunner.execute_model()
  → prepare inputs (token_ids, positions, block_tables)
  → model.forward(input_ids, positions, kv_caches, attn_metadata)
  → sample next tokens from logits
  → return ModelRunnerOutput
```

---

## 4. KV Cache and PagedAttention

### What Is the KV Cache?

In a transformer, each attention layer computes Key (K) and Value (V) vectors for
every token. During decode, we need the K and V vectors for ALL previous tokens
(not just the current one) to compute attention. Rather than recomputing them
every step, we **cache** them.

```
 ATTENTION MECHANISM (simplified):

 For each layer, for each token position:

   Q (Query)  = current_token × W_q     ← Only for the NEW token
   K (Key)    = current_token × W_k     ← Only for the NEW token (cached for reuse)
   V (Value)  = current_token × W_v     ← Only for the NEW token (cached for reuse)

   Attention = softmax(Q × K_all^T / √d) × V_all
                          ▲                   ▲
                          │                   │
                    All K vectors        All V vectors
                    (from cache +        (from cache +
                     new token)           new token)
```

The KV cache can be HUGE. For a model like Qwen3-VL-235B:
- 94 layers × 2 (K and V) × 64 KV heads × 128 head_dim × sequence_length × bytes
- For a 32K context: roughly ~28 GB per request at FP16
- This is why you need lots of GPU memory!

### PagedAttention: Virtual Memory for KV Cache

Traditional inference engines pre-allocate a contiguous block of memory for each
request's KV cache up to the maximum sequence length. This wastes huge amounts of
memory because most requests don't use the full context length.

vLLM's **PagedAttention** (inspired by operating system virtual memory) solves this:

```
 TRADITIONAL APPROACH (wasteful):
 ─────────────────────────────────
 GPU Memory:
 ┌──────────────────────────────────────────────────────────┐
 │ Req A: [████████░░░░░░░░░░░░░░░░░░░░░░░░░░]             │
 │         used(8)     wasted(26) ← allocated for max_len  │
 │                                                          │
 │ Req B: [██████████████░░░░░░░░░░░░░░░░░░░░]             │
 │         used(14)        wasted(20)                       │
 │                                                          │
 │ Req C: [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]             │
 │         used(2)              wasted(32)                  │
 │                                                          │
 │ ░░░░░ = wasted memory (allocated but unused)             │
 └──────────────────────────────────────────────────────────┘

 VLLM's PAGED APPROACH (efficient):
 ────────────────────────────────────
 GPU Memory divided into fixed-size BLOCKS (e.g., 16 tokens each):

 Block Pool:
 ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
 │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ B8 │ B9 │B10 │B11 │
 │ A  │ A  │ B  │ B  │ B  │ C  │free│free│free│free│free│free│
 └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

 Block Table (maps request → blocks):
 ┌──────────────────────┐
 │ Req A → [B0, B1]     │  8 tokens using 2 blocks (waste: up to 15 tokens)
 │ Req B → [B2, B3, B4] │  14 tokens using 3 blocks
 │ Req C → [B5]         │  2 tokens using 1 block
 │ Free  → [B6..B11]    │  6 blocks available for new requests
 └──────────────────────┘

 When Req A generates more tokens and fills B1, a new block is allocated:
 Req A → [B0, B1, B6]   (B6 taken from free pool)
```

### vLLM KV Cache Implementation

The KV cache management lives in:
- `vllm/v1/core/kv_cache_manager.py` — high-level allocation/deallocation
- `vllm/v1/core/block_pool.py` — the pool of free/used blocks with prefix caching
- `vllm/v1/core/kv_cache_utils.py` — block hashing and utility functions
- `vllm/attention/ops/paged_attn.py` — the actual GPU kernels that read/write paged KV

The `BlockPool` class (`block_pool.py`) maintains:
- A **free block queue** (blocks available for allocation)
- A **block hash map** (for prefix caching: reuse KV blocks for shared prefixes)

```python
# From vllm/v1/core/block_pool.py (simplified concept)
class BlockPool:
    # Maps block hashes to blocks — enables prefix caching
    cached_block_hash_to_block: BlockHashToBlockMap

    # Queue of free blocks, ordered by eviction priority
    free_block_queue: FreeKVCacheBlockQueue
```

**Prefix caching** is a powerful optimization: if two requests share the same system
prompt, the KV cache blocks for that shared prefix are computed once and reused.

---

## 5. Tensor Parallelism and Multi-GPU Serving

### Why Multi-GPU?

Large models don't fit on a single GPU:

```
 Model Size vs GPU Memory:

 Model                    Parameters    FP16 Size    FP8 Size
 ─────────────────────────────────────────────────────────────
 Qwen3-32B-VL             ~32B          ~64 GB       ~32 GB
 Qwen3-VL-235B-A22B (MoE) ~235B         ~470 GB      ~235 GB
 Kimi K2.5 (MoE)          ~1T*          ~400 GB**    ~200 GB**

 GPU Memory Available:
 ─────────────────────────────────────────────────────────────
 H100 SXM                                 80 GB
 B200                                    192 GB

 * Kimi K2.5 is built on Kimi-K2 (~1T total, ~32B active, MoE)
 ** Actual weight size depends on active + routing parameters
```

### Tensor Parallelism (TP)

Tensor parallelism **splits individual weight matrices across GPUs**. Each GPU holds
a vertical or horizontal slice of every layer's weights and processes the same input
tokens in parallel.

```
 TENSOR PARALLELISM (TP=4) — Splitting a Linear Layer
 ─────────────────────────────────────────────────────

 Original weight matrix W: [4096 × 16384]

 Split by COLUMNS across 4 GPUs (ColumnParallelLinear):
 ┌─────────────┬─────────────┬─────────────┬─────────────┐
 │   GPU 0     │   GPU 1     │   GPU 2     │   GPU 3     │
 │ W[:, 0:4096]│W[:, 4096:   │W[:, 8192:   │W[:, 12288:  │
 │             │     8192]   │    12288]   │    16384]   │
 │ [4096×4096] │ [4096×4096] │ [4096×4096] │ [4096×4096] │
 └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
   Y_partial_0   Y_partial_1  Y_partial_2   Y_partial_3
        │             │             │             │
        └─────────────┴──────┬──────┴─────────────┘
                             │
                      ALL-REDUCE (sum)
                      via NVLink/NVSwitch
                             │
                             ▼
                    Y_complete [batch × 16384]
                    (identical on all GPUs)
```

The ALL-REDUCE operation is where GPUs communicate. On H100 and B200 systems with
NVSwitch, this happens over NVLink at very high bandwidth:
- H100: 900 GB/s bidirectional NVLink
- B200: 1800 GB/s bidirectional NVLink (5th gen)

### Expert Parallelism (EP) for MoE Models

Qwen3-VL-235B and Kimi K2.5 are **Mixture of Experts (MoE)** models. Instead of
every token going through all parameters, a router selects a subset of "expert"
sub-networks for each token.

```
 MIXTURE OF EXPERTS (simplified):
 ─────────────────────────────────

 Input token embedding
        │
        ▼
 ┌──────────────┐
 │    Router     │  Scores each expert, picks top-K (e.g., top-8 of 128)
 │  (small MLP)  │
 └──┬───┬───┬───┘
    │   │   │
    ▼   ▼   ▼     Only selected experts activate
 ┌────┐┌────┐┌────┐┌────┐                    ┌────┐
 │Exp0││Exp3││Exp7││Exp │  ...  (120 idle)    │E127│
 │ ██ ││ ██ ││ ██ ││ 15 │                     │    │
 │ACT ││ACT ││ACT ││ ██ │                     │idle│
 └──┬─┘└──┬─┘└──┬─┘└──┬─┘                    └────┘
    │     │     │     │
    └─────┴──┬──┴─────┘
             │
    Weighted sum of expert outputs
             │
             ▼
    Output token embedding
```

**Expert Parallelism** distributes experts across GPUs instead of replicating them:

```
 EXPERT PARALLELISM (EP=8) with 128 experts:
 ─────────────────────────────────────────────

 GPU 0: Experts   0-15    (16 experts)
 GPU 1: Experts  16-31    (16 experts)
 GPU 2: Experts  32-47    (16 experts)
 GPU 3: Experts  48-63    (16 experts)
 GPU 4: Experts  64-79    (16 experts)
 GPU 5: Experts  80-95    (16 experts)
 GPU 6: Experts  96-111   (16 experts)
 GPU 7: Experts 112-127   (16 experts)

 When a token needs Expert 50, it gets routed to GPU 3.
 This requires ALL-TO-ALL communication between GPUs.
```

In vLLM, enable expert parallelism with `--enable-expert-parallel`. The implementation
uses `FusedMoE` (`vllm/model_executor/layers/fused_moe/`) which fuses the expert
routing, dispatching, and computation into efficient GPU kernels.

### How TP Maps to Code

The parallelism is implemented through distributed linear layers:

```python
# vllm/model_executor/layers/linear.py

ColumnParallelLinear   # Splits weight columns across GPUs
                       # Each GPU computes a partial output
                       # Used for: Q/K/V projections, FFN up-projections

RowParallelLinear      # Splits weight rows across GPUs
                       # Each GPU has partial inputs, produces partial sums
                       # Followed by all-reduce
                       # Used for: output projections, FFN down-projections

ReplicatedLinear       # Full copy on every GPU (for small layers)
                       # Used for: vision model projectors, small adapters
```

The workers coordinate via `vllm/distributed/parallel_state.py` which sets up
NCCL process groups for tensor-parallel, pipeline-parallel, and expert-parallel
communication.

---

## 6. Multimodal / Vision-Language Model Pipeline

All three target models (Kimi K2.5, Qwen3-VL-235B, Qwen3-32B-VL) are **vision-language
models** (VLMs). They can process both text and images/video.

### How Multimodal Input Flows Through vLLM

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  User Request (OpenAI Chat Completions API)                      │
 │  {                                                               │
 │    "messages": [{                                                │
 │      "role": "user",                                             │
 │      "content": [                                                │
 │        {"type": "image_url", "image_url": {"url": "..."}},      │
 │        {"type": "text", "text": "What's in this image?"}        │
 │      ]                                                           │
 │    }]                                                            │
 │  }                                                               │
 └───────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  MULTIMODAL PROCESSOR                                            │
 │  (vllm/multimodal/)                                              │
 │                                                                  │
 │  1. Download/decode the image                                    │
 │  2. Run model-specific image processor (resize, normalize, etc.) │
 │  3. Convert image to pixel tensors                               │
 │  4. Calculate how many "image tokens" this image produces        │
 │  5. Insert placeholder tokens into the text token sequence       │
 └───────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  TOKEN SEQUENCE WITH PLACEHOLDERS                                │
 │                                                                  │
 │  [BOS] [IMG_0] [IMG_1] ... [IMG_576] What's in this image? [EOS]│
 │         ▲                    ▲                                   │
 │         └────────────────────┘                                   │
 │         These placeholder tokens will be replaced                │
 │         by vision encoder embeddings                             │
 └───────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  VISION ENCODER (runs on GPU during prefill)                     │
 │                                                                  │
 │  Kimi K2.5:  MoonViT (custom ViT based on InternViT)            │
 │              vllm/model_executor/models/moonvit.py               │
 │                                                                  │
 │  Qwen3-VL:   Qwen3_VisionTransformer (custom ViT with 3D Conv)  │
 │              vllm/model_executor/models/qwen3_vl.py              │
 │                                                                  │
 │  pixel_values → [patch_embed → transformer_layers] → features    │
 │                                                                  │
 │  Output: image feature embeddings [num_patches × hidden_dim]     │
 └───────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  MULTIMODAL PROJECTOR                                            │
 │                                                                  │
 │  Maps vision encoder output dimension → LLM hidden dimension     │
 │                                                                  │
 │  Kimi K2.5:  KimiVLMultiModalProjector                           │
 │              LayerNorm → Linear → GELU → Linear                  │
 │              (kimi_vl.py:106-139)                                │
 │                                                                  │
 │  Qwen3-VL:   (integrated into vision model's merger layer)       │
 └───────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  EMBEDDING MERGE                                                 │
 │                                                                  │
 │  Replace placeholder token embeddings with vision features:      │
 │                                                                  │
 │  [BOS_emb] [vis_0] [vis_1] ... [vis_576] [What_emb] [is_emb]...│
 │             ▲                    ▲                               │
 │             └────────────────────┘                               │
 │             Replaced by projected vision features                │
 │                                                                  │
 │  Now proceed with normal LLM forward pass                       │
 └──────────────────────────────────────────────────────────────────┘
```

### The --mm-encoder-tp-mode Option

By default, the vision encoder runs on a single GPU (replicated). With
`--mm-encoder-tp-mode data`, vLLM distributes different images to different GPUs
for parallel processing:

```
 --mm-encoder-tp-mode data  (Data Parallel for Vision Encoder)
 ────────────────────────────────────────────────────────────

 Batch has 4 images to process with TP=4:

 Without data mode (default):       With data mode:
 ┌────────┐                         ┌────────┐
 │ GPU 0  │ processes ALL 4 images  │ GPU 0  │ processes image 0
 │ GPU 1  │ (idle for vision)       │ GPU 1  │ processes image 1
 │ GPU 2  │ (idle for vision)       │ GPU 2  │ processes image 2
 │ GPU 3  │ (idle for vision)       │ GPU 3  │ processes image 3
 └────────┘                         └────────┘

 Result: 10-45% throughput improvement for multimodal workloads!
```

---

## 7. Model-Specific Guides

### Kimi K2.5 (Moonshot AI)

**Architecture:** MoE vision-language model built on Kimi-K2 base (DeepSeek-V2 architecture).
Continually pretrained on ~15T mixed visual+text tokens.

**Key specs:**
- ~1T total parameters, ~32B active per token (MoE)
- Based on DeepSeek-V2 architecture (uses MLA — Multi-head Latent Attention)
- Vision encoder: MoonViT (custom, based on InternViT)
- Supports images (no video in initial release)

**vLLM code path:**
- Model: `vllm/model_executor/models/kimi_vl.py` → `KimiVLForConditionalGeneration`
- Vision: `vllm/model_executor/models/moonvit.py` → `MoonVitPretrainedModel`
- Language: Reuses `DeepseekV2Model` from `vllm/model_executor/models/deepseek_v2.py`
- Registry: `vllm/model_executor/models/registry.py:344`

**Minimum hardware:**
- 8× H100 80GB (TP=8) or 4× B200 192GB (TP=4)

**Serve command (H100 8-GPU):**
```bash
vllm serve moonshotai/Kimi-K2.5 \
    -tp 8 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code
```

**Serve command (B200 4-GPU):**
```bash
vllm serve moonshotai/Kimi-K2.5 \
    -tp 4 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code
```

**Optimization tips:**
- Use `--enable-expert-parallel` for better throughput with expert parallelism
- Use `--mm-processor-cache-type shm` for shared-memory caching of preprocessed images
  (better performance at high TP)
- If workload has mostly unique images: `--mm-processor-cache-gb 0` to skip caching
- Recommended temperature: 1.0 (thinking mode), 0.6 (instant mode), top_p=0.95

---

### Qwen3-VL-235B-A22B

**Architecture:** MoE vision-language model (Mixture of Experts variant of Qwen3-VL).

**Key specs:**
- 235B total parameters, ~22B active per token
- Uses Qwen3 MoE architecture with shared+routed experts
- Vision encoder: Qwen3_VisionTransformer with 3D convolution for video support
- Supports both images and video
- Uses M-RoPE (Multi-dimensional Rotary Position Embedding) for spatial awareness

**vLLM code path:**
- Model: `vllm/model_executor/models/qwen3_vl_moe.py` → `Qwen3VLMoeForConditionalGeneration`
- Vision: `vllm/model_executor/models/qwen3_vl.py` → `Qwen3_VisionTransformer`
- Language: Reuses `Qwen3MoeModel` from `vllm/model_executor/models/qwen3_moe.py`
- Registry: `vllm/model_executor/models/registry.py:414`

**Minimum hardware:**
- 8× H100 80GB (TP=8) or 4× B200 192GB (TP=4)

**Serve command (H100 8-GPU, FP8 quantized):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt video=0
```

**Serve command (H100 8-GPU, BF16 full precision):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt video=0
```

**Serve command (B200 4-GPU, FP8):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --tensor-parallel-size 4 \
    --mm-encoder-tp-mode data
```

**Optimization tips:**
- Set `OMP_NUM_THREADS=1` to avoid CPU thread contention during preprocessing
- Use `--limit-mm-per-prompt video=0` if only processing images (saves memory
  that would be reserved for video embedding buffers)
- FP8 quantized version (`-FP8` suffix) halves memory requirements with minimal
  quality loss — strongly recommended for 8×H100
- If you hit OOM, reduce `--max-model-len` (e.g., to 32768 or 65536)

---

### Qwen3-32B-VL

**Architecture:** Dense (non-MoE) vision-language model. The smaller, faster sibling
of the 235B variant.

**Key specs:**
- ~32B parameters (all active — not MoE)
- Same Qwen3-VL vision architecture as the 235B
- Same M-RoPE and 3D conv video support
- Fits on a single H100 in FP8, or 2× H100 in BF16

**vLLM code path:**
- Model: `vllm/model_executor/models/qwen3_vl.py` → `Qwen3VLForConditionalGeneration`
- Vision: `vllm/model_executor/models/qwen3_vl.py` → `Qwen3_VisionTransformer`
- Registry: `vllm/model_executor/models/registry.py:413`

**Minimum hardware:**
- 1× H100 80GB (FP8) or 1× B200 192GB (BF16 with room to spare)
- 2× H100 80GB for BF16 with longer contexts

**Serve command (single H100, FP8):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --quantization fp8 \
    --max-model-len 32768
```

**Serve command (2× H100, BF16):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --tensor-parallel-size 2 \
    --mm-encoder-tp-mode data
```

**Serve command (single B200, BF16):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-32B-Instruct
```

**Optimization tips:**
- The 32B model is dense (not MoE), so `--enable-expert-parallel` is NOT applicable
- Single-GPU deployment is simplest and avoids TP communication overhead
- B200's 192GB lets you run full BF16 with very long contexts comfortably

---

## 8. H100 vs B200 Hardware Comparison

```
 ┌──────────────────────┬─────────────────────┬─────────────────────┐
 │                      │      H100 SXM       │       B200          │
 ├──────────────────────┼─────────────────────┼─────────────────────┤
 │ Architecture         │ Hopper (2022)       │ Blackwell (2024)    │
 │ HBM Memory           │ 80 GB HBM3          │ 192 GB HBM3e       │
 │ Memory Bandwidth     │ 3.35 TB/s           │ 8.0 TB/s           │
 │ FP16 Tensor TFLOPS   │ 990 TFLOPS          │ 2,250 TFLOPS       │
 │ FP8 Tensor TFLOPS    │ 1,979 TFLOPS        │ 4,500 TFLOPS       │
 │ NVLink Bandwidth     │ 900 GB/s            │ 1,800 GB/s         │
 │ TDP                  │ 700W                │ 1000W              │
 │ Transistors          │ 80B                 │ 208B               │
 │ SMs                  │ 132                 │ 192                 │
 ├──────────────────────┼─────────────────────┼─────────────────────┤
 │ DECODE ADVANTAGE     │ baseline            │ ~2.4x faster       │
 │ (memory BW bound)    │                     │ (8.0/3.35)         │
 │                      │                     │                     │
 │ PREFILL ADVANTAGE    │ baseline            │ ~2.3x faster       │
 │ (compute bound)      │                     │ (2250/990)         │
 │                      │                     │                     │
 │ MEMORY ADVANTAGE     │ baseline            │ 2.4x more          │
 │                      │                     │ (192/80)           │
 └──────────────────────┴─────────────────────┴─────────────────────┘
```

### What This Means for Our Models

```
 ┌──────────────────┬──────────────────────────┬─────────────────────────┐
 │ Model            │ H100 SXM Configuration   │ B200 Configuration      │
 ├──────────────────┼──────────────────────────┼─────────────────────────┤
 │                  │                          │                         │
 │ Kimi K2.5        │ 8× H100 (TP=8)          │ 4× B200 (TP=4)         │
 │ (~400GB weights) │ 640 GB total memory      │ 768 GB total memory     │
 │                  │ Memory is tight           │ Comfortable fit         │
 │                  │                          │                         │
 │ Qwen3-VL-235B   │ 8× H100 (TP=8)          │ 4× B200 (TP=4)         │
 │ (~470GB BF16)    │ Use FP8 version!         │ FP8 or BF16 both work  │
 │ (~235GB FP8)     │ 640 GB total memory      │ 768 GB total memory     │
 │                  │                          │                         │
 │ Qwen3-32B-VL    │ 1× H100 (FP8) or        │ 1× B200 (BF16)         │
 │ (~64GB BF16)     │ 2× H100 (TP=2, BF16)    │ Tons of headroom        │
 │ (~32GB FP8)      │                          │                         │
 └──────────────────┴──────────────────────────┴─────────────────────────┘

 Memory budget per GPU (approximate):
 ────────────────────────────────────
 Total GPU Memory
   - Model weights            (largest chunk)
   - KV cache                 (grows with concurrent requests × context length)
   - Activation memory        (temporary, during forward pass)
   - CUDA kernels + overhead  (~1-2 GB)
   = Available for KV cache determines max concurrent requests
```

---

## 9. Practical Deployment Commands

### Installation

```bash
# Install vLLM (requires Python 3.9+ and CUDA 12.x)
pip install vllm

# Or install nightly for latest model support (needed for Kimi K2.5)
pip install vllm-nightly
```

### Quick-Start Commands

**Kimi K2.5 on 8× H100:**
```bash
vllm serve moonshotai/Kimi-K2.5 \
    -tp 8 \
    --mm-encoder-tp-mode data \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --trust-remote-code \
    --enable-expert-parallel
```

**Qwen3-VL-235B on 8× H100 (FP8):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --mm-encoder-tp-mode data \
    --limit-mm-per-prompt video=0
```

**Qwen3-VL-235B on 4× B200 (BF16):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --mm-encoder-tp-mode data
```

**Qwen3-32B-VL on 1× H100 (FP8):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --quantization fp8 \
    --max-model-len 32768
```

**Qwen3-32B-VL on 1× B200 (BF16):**
```bash
OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-32B-Instruct
```

### Sending a Request (with an image)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-32B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "https://example.com/photo.jpg"}
          },
          {
            "type": "text",
            "text": "Describe what you see in this image."
          }
        ]
      }
    ],
    "max_tokens": 512
  }'
```

---

## 10. Troubleshooting

### Out of Memory (OOM)

**Symptoms:** `torch.cuda.OutOfMemoryError` or process killed by OOM killer.

**Solutions:**
1. Reduce `--max-model-len` (e.g., 32768 instead of 131072)
2. Use FP8 quantization (`--quantization fp8` or use the `-FP8` model variant)
3. Add `--limit-mm-per-prompt video=0` if not using video
4. Increase tensor parallelism (more GPUs = more total memory)
5. Reduce `--max-num-seqs` to limit concurrent requests

### Slow Performance

**Symptoms:** Low tokens/second, high latency.

**Solutions:**
1. Ensure NVLink is available (`nvidia-smi topo -m` should show NV connections)
2. Use `--mm-encoder-tp-mode data` for multimodal workloads
3. Enable expert parallelism for MoE models: `--enable-expert-parallel`
4. Set `OMP_NUM_THREADS=1` to avoid CPU contention
5. Use `--mm-processor-cache-type shm` for repeated image patterns

### Model Not Found / Loading Errors

**Solutions:**
1. Ensure you have `--trust-remote-code` for models that need it (Kimi K2.5)
2. Check Hugging Face access: some models require accepting a license
3. For Kimi K2.5: may need vLLM nightly (`pip install vllm-nightly`)
4. Verify model name matches exactly (case-sensitive)

---

## Appendix: Key Source Files Reference

| Component | File Path | Key Class/Function |
|---|---|---|
| API Server | `vllm/entrypoints/openai/api_server.py` | FastAPI app |
| Engine Core | `vllm/v1/engine/` | `EngineCoreClient` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `Scheduler` |
| KV Cache Manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` |
| Block Pool | `vllm/v1/core/block_pool.py` | `BlockPool`, `FreeKVCacheBlockQueue` |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | `Worker` |
| GPU Model Runner | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner` |
| PagedAttention Ops | `vllm/attention/ops/paged_attn.py` | `PagedAttention` |
| TP Linear Layers | `vllm/model_executor/layers/linear.py` | `ColumnParallelLinear`, `RowParallelLinear` |
| FusedMoE | `vllm/model_executor/layers/fused_moe/` | `FusedMoE` |
| Model Registry | `vllm/model_executor/models/registry.py` | Architecture → implementation map |
| Kimi K2.5 Model | `vllm/model_executor/models/kimi_vl.py` | `KimiVLForConditionalGeneration` |
| MoonViT Encoder | `vllm/model_executor/models/moonvit.py` | `MoonVitPretrainedModel` |
| Qwen3 VL Model | `vllm/model_executor/models/qwen3_vl.py` | `Qwen3VLForConditionalGeneration` |
| Qwen3 VL MoE | `vllm/model_executor/models/qwen3_vl_moe.py` | `Qwen3VLMoeForConditionalGeneration` |
| Multimodal Registry | `vllm/multimodal/__init__.py` | `MULTIMODAL_REGISTRY` |
| Parallel State | `vllm/distributed/parallel_state.py` | TP/PP/EP process groups |

---

*Guide generated from vLLM source code analysis. Model specifications and GPU
benchmarks sourced from official documentation and Hugging Face model cards.*
