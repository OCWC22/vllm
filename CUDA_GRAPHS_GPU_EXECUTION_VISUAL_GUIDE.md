# CUDA Graphs and GPU Execution — Complete ASCII Visualization

Everything that happens on the GPU during vLLM inference, visualized.

---

## Table of Contents

1. [The Big Picture: What Runs Where](#1-the-big-picture)
2. [CUDA Graph Lifecycle: Capture → Replay](#2-cuda-graph-lifecycle)
3. [One Decode Step: Clock-by-Clock Timeline](#3-one-decode-step)
4. [Full vs Piecewise CUDA Graphs](#4-full-vs-piecewise)
5. [Inside the GPU: SM Architecture](#5-inside-the-gpu)
6. [Every Operation → Hardware Unit](#6-every-operation-to-hardware-unit)
7. [Triton Attention: Thread Block Layout](#7-triton-attention-thread-block-layout)
8. [NVFP4 Linear: Data Flow Through Tensor Cores](#8-nvfp4-linear)
9. [Memory Hierarchy: Where Data Lives](#9-memory-hierarchy)
10. [Multi-GPU: Tensor Parallel Communication](#10-multi-gpu)
11. [Batch Size Padding and Dispatch](#11-batch-size-padding)
12. [Prefill vs Decode: Why Everything Changes](#12-prefill-vs-decode)

---

## 1. The Big Picture

```
┌─── YOUR REQUEST: "What is the meaning of life?" ──────────────────────────────┐
│                                                                                │
│  TOKENIZER (CPU)                                                               │
│  "What is the meaning of life?" → [1, 2, 3, 4, 5, 6, 7]                      │
│                                                                                │
│  SCHEDULER (CPU)                                                               │
│  Batch this with 31 other requests → 32 tokens to process                     │
│                                                                                │
│  ┌─── GPU ──────────────────────────────────────────────────────────────────┐  │
│  │                                                                          │  │
│  │  CUDA GRAPH REPLAY ← entire forward pass in ONE CPU call                │  │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │  │
│  │  │                                                                    │  │  │
│  │  │  For each of ~60 layers:                                          │  │  │
│  │  │                                                                    │  │  │
│  │  │  ┌─RMSNorm──┐  ┌─FP4 Quant─┐  ┌─FP4 GEMM──────┐                │  │  │
│  │  │  │CUDA cores│→ │CUDA cores │→ │TENSOR CORES   │  QKV projection  │  │  │
│  │  │  │memory bw │  │memory bw  │  │compute bound  │                  │  │  │
│  │  │  └──────────┘  └───────────┘  └───────────────┘                  │  │  │
│  │  │        │                                                          │  │  │
│  │  │        ▼                                                          │  │  │
│  │  │  ┌─RoPE──────┐  ┌─Triton Attention─────────────────────┐         │  │  │
│  │  │  │CUDA cores │→ │ Q·K = TENSOR CORES                  │         │  │  │
│  │  │  │cos/sin LUT│  │ exp() = SFU                         │         │  │  │
│  │  │  └───────────┘  │ max/sum = CUDA CORES                │         │  │  │
│  │  │                  │ P·V = TENSOR CORES                  │         │  │  │
│  │  │                  │ KV cache read = MEMORY CONTROLLER   │         │  │  │
│  │  │                  └─────────────────────────────────────┘         │  │  │
│  │  │        │                                                          │  │  │
│  │  │        ▼                                                          │  │  │
│  │  │  ┌─FP4 Quant─┐  ┌─FP4 GEMM──────┐                              │  │  │
│  │  │  │CUDA cores │→ │TENSOR CORES   │  Output projection             │  │  │
│  │  │  └───────────┘  └───────────────┘                                │  │  │
│  │  │        │                                                          │  │  │
│  │  │        ▼                                                          │  │  │
│  │  │  ┌─AllReduce──────────────┐  (only with tensor parallelism)      │  │  │
│  │  │  │NVLINK: cross-GPU xfer │                                       │  │  │
│  │  │  │CUDA CORES: FP32 sum   │                                       │  │  │
│  │  │  └────────────────────────┘                                      │  │  │
│  │  │        │                                                          │  │  │
│  │  │        ▼                                                          │  │  │
│  │  │  ┌─RMSNorm──┐  ┌─Router──────┐  ┌─MoE Experts────────┐         │  │  │
│  │  │  │CUDA cores│→ │CUDA cores   │→ │TENSOR CORES        │         │  │  │
│  │  │  │          │  │sigmoid+topk │  │FP4 group GEMM      │         │  │  │
│  │  │  └──────────┘  └─────────────┘  │shared expert in || │         │  │  │
│  │  │                                  └─────────────────────┘         │  │  │
│  │  │        │                                                          │  │  │
│  │  │  ┌─Residual Add─┐                                                │  │  │
│  │  │  │CUDA cores    │  x = x + attn_out + mlp_out                   │  │  │
│  │  │  └──────────────┘                                                │  │  │
│  │  │                                                                    │  │  │
│  │  │  (repeat for all layers)                                          │  │  │
│  │  │                                                                    │  │  │
│  │  │  ┌─Final RMSNorm─┐  ┌─LM Head GEMM────┐  ┌─Sampling──────────┐ │  │  │
│  │  │  │CUDA cores     │→ │TENSOR CORES     │→ │CUDA CORES + SMEM │ │  │  │
│  │  │  └───────────────┘  │[32, vocab_size]  │  │radix histogram   │ │  │  │
│  │  │                      └─────────────────┘  │top-k select      │ │  │  │
│  │  │                                            └──────────────────┘ │  │  │
│  │  └────────────────────────────────────────────────────────────────────┘  │  │
│  │                                                                          │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  DETOKENIZER (CPU)                                                             │
│  [42] → "42"  ← streamed to client                                           │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CUDA Graph Lifecycle

### Phase 1: Startup — Allocate Persistent Buffers

```
vllm/v1/worker/gpu_model_runner.py:484-511

CPU Memory (pinned)              GPU Memory (HBM)
┌──────────────────┐             ┌──────────────────┐
│ input_ids  [512] │ ──────────→ │ input_ids  [512] │ ← FIXED ADDRESS
│ positions  [512] │ ──────────→ │ positions  [512] │ ← FIXED ADDRESS
│ seq_lens   [256] │ ──────────→ │ seq_lens   [256] │ ← FIXED ADDRESS
│ block_table[256] │ ──────────→ │ block_table[256] │ ← FIXED ADDRESS
│ slot_map   [512] │ ──────────→ │ slot_map   [512] │ ← FIXED ADDRESS
│ query_start[257] │ ──────────→ │ query_start[257] │ ← FIXED ADDRESS
└──────────────────┘             └──────────────────┘
       ↑                                ↑
  CpuGpuBuffer.cpu                CpuGpuBuffer.gpu
  (numpy view for                 (these addresses get
   fast writes)                    frozen in the graph)
```

### Phase 2: Capture — Record Kernel Launches

```
vllm/compilation/cuda_graph.py:257-275

For each batch size in [512, 496, ..., 256, 248, ..., 8, 4, 2, 1]:
                        (largest first, to reuse memory pool)

  ┌─ WARMUP (mode=NONE) ──────────────────────────────────────────┐
  │                                                                 │
  │  Run model forward EAGERLY with dummy data.                    │
  │  Purpose: trigger torch.compile / Inductor to compile kernels  │
  │  for this specific tensor shape.                                │
  │                                                                 │
  │  gpu_model_runner.py:4790 → _dummy_run(mode=NONE)             │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─ CAPTURE (mode=FULL) ─────────────────────────────────────────┐
  │                                                                 │
  │  torch.cuda.CUDAGraph()                                        │
  │  with torch.cuda.graph(cudagraph, pool=global_pool):           │
  │                                                                 │
  │    ┌──────────────────────────────────────────────────┐        │
  │    │ RECORDING: Every CUDA API call is captured:      │        │
  │    │                                                   │        │
  │    │  cudaLaunchKernel(rmsnorm, grid, block, args)    │        │
  │    │  cudaLaunchKernel(fp4_quant, grid, block, args)  │        │
  │    │  cudaLaunchKernel(cutlass_gemm, grid, block, args│        │
  │    │  cudaLaunchKernel(triton_attn, grid, block, args)│        │
  │    │  cudaLaunchKernel(fp4_quant, ...)                │        │
  │    │  cudaLaunchKernel(cutlass_gemm, ...)             │        │
  │    │  cudaLaunchKernel(rmsnorm, ...)                  │        │
  │    │  cudaLaunchKernel(moe_router, ...)               │        │
  │    │  cudaLaunchKernel(cutlass_moe_gemm, ...)         │        │
  │    │  ... (× 60 layers × ~8 kernels each)             │        │
  │    │  cudaLaunchKernel(lm_head_gemm, ...)             │        │
  │    │                                                   │        │
  │    │  Total: ~500 kernel launches recorded             │        │
  │    └──────────────────────────────────────────────────┘        │
  │                                                                 │
  │  Store: entries[BatchDescriptor(512)] = CUDAGraphEntry(        │
  │           cudagraph=<the recorded graph>,                       │
  │           output=<weak ref to output tensor>                    │
  │         )                                                       │
  └─────────────────────────────────────────────────────────────────┘

  Repeat for batch_size=496, 480, ..., 8, 4, 2, 1
  (each graph stored separately in the entries dict)
```

### Phase 3: Runtime — Pad + Lookup + Update + Replay

```
Every decode step:

  ┌─ CPU: SCHEDULER ────────────────────────────────────────────────┐
  │  "This step has 37 tokens across 37 requests"                   │
  │                                                                  │
  │  1. PAD:  37 → 40  (next captured size via O(1) array lookup)  │
  │     vllm/config/compilation.py:1115  bs_to_padded_graph_size[37]│
  │                                                                  │
  │  2. DISPATCH:  BatchDescriptor(40, 37, uniform=True, lora=None)│
  │     vllm/v1/cudagraph_dispatcher.py:143  → CUDAGraphMode.FULL  │
  │                                                                  │
  │  3. UPDATE PERSISTENT BUFFERS:                                   │
  │     Write 37 real token IDs into input_ids.cpu[0:37]            │
  │     Write 37 real positions into positions.cpu[0:37]            │
  │     Write 37 real seq_lens into seq_lens.cpu[0:37]              │
  │     Copy CPU → GPU:  input_ids.copy_to_gpu(40)  (async DMA)    │
  │     Copy CPU → GPU:  positions.copy_to_gpu(40)                  │
  │     Copy CPU → GPU:  seq_lens.copy_to_gpu(37)                  │
  │                                                                  │
  │  Positions 37-39 are PADDING (zero-filled, 3 wasted tokens).   │
  └──────────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─ GPU: REPLAY ───────────────────────────────────────────────────┐
  │                                                                  │
  │  entry = entries[BatchDescriptor(40, ...)]                      │
  │  entry.cudagraph.replay()   ← ONE CUDA API CALL                │
  │                                                                  │
  │  The ~500 recorded kernels execute in the EXACT same order,     │
  │  with the EXACT same grid/block dimensions, reading from        │
  │  the EXACT same GPU addresses — but now those addresses         │
  │  contain REAL data (written in step 3 above).                   │
  │                                                                  │
  │  Thread blocks for padded tokens (37, 38, 39):                  │
  │    - Triton attention: early exit at line 124                   │
  │      if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:     │
  │          return   ← ~10 GPU cycles, no real work               │
  │    - CUTLASS GEMM: computes on padding, result discarded        │
  │    - RMSNorm: runs on padding, result discarded                 │
  │                                                                  │
  │  Output: logits tensor at entry.output (same GPU address)       │
  │  Only logits[0:37] are real; logits[37:40] are garbage.        │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘
          │
          ▼
  ┌─ CPU: SAMPLE + SEND ───────────────────────────────────────────┐
  │  Read logits[0:37], sample 37 tokens, stream to 37 clients     │
  └─────────────────────────────────────────────────────────────────┘
```

### Why This Matters: The Numbers

```
WITHOUT CUDA GRAPHS:
  CPU issues ~500 kernel launches × ~5μs each = 2,500 μs = 2.5 ms CPU overhead
  GPU idle between launches (bubbles):          ~500 × 1μs = 0.5 ms wasted
  Total overhead per step:                      ~3.0 ms

WITH CUDA GRAPHS:
  CPU issues 1 replay() call:                   ~1 μs
  GPU executes all 500 kernels back-to-back:    0 ms bubbles
  Total overhead per step:                      ~0.001 ms

  Speedup: ~3,000x reduction in launch overhead
  (matters most for small batches where GPU compute is also fast)
```

---

## 3. One Decode Step — Clock-by-Clock Timeline

```
Time (microseconds) →
0        100       200       300       400       500       600

CPU:
├─schedule─┤─update bufs─┤─replay()─┤                    ├─sample─┤
                          │ 1 μs     │                    │        │
                          └──────────┘                    │        │
                               │                          │        │
GPU:                           ▼                          ▼        │
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Layer 0:                                                        │
│  ├─RMSNorm──┤─FP4Q──┤──FP4 GEMM(QKV)──┤─RoPE─┤                │
│  │ CUDA+SFU │ CUDA  │  TENSOR CORES    │ CUDA │                │
│  │ 3μs      │ 2μs   │  15μs            │ 2μs  │                │
│  │          │       │                  │      │                │
│  ├────────Triton Attention───────────┤                          │
│  │ Load K: MEM  │ Q·K: TC │ softmax: │ P·V: TC │ Store: MEM   │
│  │ 4μs          │ 2μs     │ CUDA+SFU │ 2μs     │ 1μs          │
│  │              │         │ 3μs      │         │              │
│  │                                                              │
│  ├─FP4Q─┤──FP4 GEMM(O)──┤─AllReduce──┤                        │
│  │ CUDA │  TENSOR CORES  │  NVLINK    │                        │
│  │ 2μs  │  10μs          │  5μs       │                        │
│  │                                                              │
│  ├─RMSNorm──┤─Router─┤───MoE (shared + routed)───┤─Residual─┤ │
│  │ CUDA+SFU │ CUDA   │  TENSOR CORES (group GEMM)│ CUDA     │ │
│  │ 3μs      │ 2μs    │  25μs                     │ 1μs      │ │
│  │                                                              │
│  Layer 1-59: (same pattern, ~80μs per layer)                    │
│  │                                                              │
│  ├─Final RMSNorm─┤──LM Head GEMM──┤                            │
│  │ 3μs           │  20μs          │                            │
│  │                                                              │
│  Total GPU time: ~60 layers × 80μs + 23μs = ~4,823 μs ≈ 4.8ms │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘

CPU:                                                    ├─sample─┤
                                                        │ 50μs   │
                                                        ▼
                                              Stream token to client
```

---

## 4. Full vs Piecewise CUDA Graphs

### FULL Mode (FA3 or Triton backend)

```
The ENTIRE forward pass is ONE graph:

  ┌─── CUDA Graph (FULL) ──────────────────────────────────────┐
  │                                                             │
  │  RMSNorm → FP4Q → GEMM → RoPE → ATTENTION → FP4Q → GEMM  │
  │  → AllReduce → RMSNorm → Router → MoE → Residual          │
  │  → (repeat × 60)                                           │
  │  → Final RMSNorm → LM Head                                │
  │                                                             │
  │  ~500 kernels, ALL inside the graph                        │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

  replay() ──→ ALL 500 kernels fire back-to-back

  Requirements:
    - Attention backend must declare ALWAYS or UNIFORM_BATCH support
    - FA3 (ALWAYS) and Triton (ALWAYS) support this
    - FA2 only supports UNIFORM_BATCH (decode-only, same query length)
```

### PIECEWISE Mode (when attention can't be graphed)

```
The graph is SPLIT at attention ops. Each piece is a separate graph:

  ┌─── Graph Piece 0 ──────────────────────────────────────────┐
  │  Embedding → RMSNorm → FP4Q → QKV GEMM → RoPE            │
  └─────────────────────────────────────────────────────────────┘
           │
           ▼  (EAGER execution — not graphed)
  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
  │  ATTENTION (runs outside graph, different every step)       │
  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
           │
           ▼
  ┌─── Graph Piece 1 ──────────────────────────────────────────┐
  │  O Projection → AllReduce → RMSNorm → Router → MoE →      │
  │  Residual → RMSNorm → FP4Q → QKV GEMM → RoPE             │
  └─────────────────────────────────────────────────────────────┘
           │
           ▼  (EAGER — not graphed)
  ┌ ─ ─ ─ ATTENTION ─ ─ ─ ┐
  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
           │
           ▼
  ┌─── Graph Piece 2 ──────────────────────────────────────────┐
  │  O Projection → AllReduce → RMSNorm → ... (next layer)    │
  └─────────────────────────────────────────────────────────────┘
           │
           ... (60 layers = ~120 graph pieces + 60 eager attention calls)
           │
           ▼
  ┌─── Graph Piece 121 ────────────────────────────────────────┐
  │  Final RMSNorm → LM Head GEMM                             │
  └─────────────────────────────────────────────────────────────┘

  This saves ~75% of launch overhead (GEMMs and norms are graphed)
  but attention still pays ~60 × 5μs = 300μs of launch overhead.
```

### Compound Modes

```
vllm/config/compilation.py:52-92

FULL_AND_PIECEWISE = (FULL, PIECEWISE):
  ┌──────────────────────────────────────────────────┐
  │ Batch type        │ Mode used                    │
  ├───────────────────┼──────────────────────────────┤
  │ Pure decode       │ FULL (one graph, fastest)    │
  │ Mixed prefill+dec │ PIECEWISE (attention eager)  │
  └──────────────────────────────────────────────────┘

  dispatch() logic (vllm/v1/cudagraph_dispatcher.py:143-183):
    1. Try FULL exact match (e.g., BatchDescriptor(40, 37, uniform=True))
    2. Try FULL relaxed match (relax num_reqs and uniform flag)
    3. Try PIECEWISE relaxed match (always succeeds if size ≤ max)
    4. Fall back to NONE (eager execution)
```

---

## 5. Inside the GPU: SM Architecture

### H100 SM (Streaming Multiprocessor)

```
┌─────────────────── One SM (×132 on H100) ────────────────────────┐
│                                                                    │
│  ┌─ Sub-partition 0 ──┐  ┌─ Sub-partition 1 ──┐                  │
│  │ 32 FP32 CUDA Cores │  │ 32 FP32 CUDA Cores │                  │
│  │ 16 FP64 CUDA Cores │  │ 16 FP64 CUDA Cores │                  │
│  │ 1 Tensor Core unit │  │ 1 Tensor Core unit │                  │
│  │ 1 SFU  (exp,rsqrt) │  │ 1 SFU  (exp,rsqrt) │                  │
│  │ Warp Scheduler      │  │ Warp Scheduler      │                  │
│  │ Register File 16KB  │  │ Register File 16KB  │                  │
│  └─────────────────────┘  └─────────────────────┘                  │
│  ┌─ Sub-partition 2 ──┐  ┌─ Sub-partition 3 ──┐                  │
│  │ 32 FP32 CUDA Cores │  │ 32 FP32 CUDA Cores │                  │
│  │ 16 FP64 CUDA Cores │  │ 16 FP64 CUDA Cores │                  │
│  │ 1 Tensor Core unit │  │ 1 Tensor Core unit │                  │
│  │ 1 SFU  (exp,rsqrt) │  │ 1 SFU  (exp,rsqrt) │                  │
│  │ Warp Scheduler      │  │ Warp Scheduler      │                  │
│  │ Register File 16KB  │  │ Register File 16KB  │                  │
│  └─────────────────────┘  └─────────────────────┘                  │
│                                                                    │
│  ┌─ Shared Memory / L1 Cache ─────────────────────────────────┐   │
│  │  228 KB configurable (SMEM up to 228KB, L1 gets remainder) │   │
│  │  Used by: Triton tile loads, CUB reductions, histograms    │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  ┌─ TMA Unit (Hopper only) ──────────────────────────────────┐   │
│  │  Asynchronous memory copy engine                           │   │
│  │  Fire-and-forget HBM → SMEM transfers                     │   │
│  │  Used by: FA3 warp-specialized producer warps              │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                    │
│  Totals per SM:                                                    │
│    128 FP32 CUDA cores                                            │
│    4 Tensor Core units (989 TFLOPS total across 132 SMs)         │
│    4 SFUs (exp, rsqrt, sin, cos — ~3.9 TFLOPS total)            │
│    64 KB register file                                            │
│    228 KB SMEM/L1                                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### B200 SM (5th Generation)

```
┌─────────────────── One SM (×192 on B200) ────────────────────────┐
│                                                                    │
│  Same 4-sub-partition layout as H100, but:                        │
│                                                                    │
│  ┌─ Key Differences ──────────────────────────────────────────┐   │
│  │                                                             │   │
│  │  5th-gen Tensor Cores:                                     │   │
│  │    - Native FP4 (E2M1) with block scaling                 │   │
│  │    - OpClassBlockScaledTensorOp (CUTLASS operator class)   │   │
│  │    - Reads E2M1 data + E4M3 scale, dequantizes in HW      │   │
│  │    - ~7,702 TFLOPS FP4 (vs 989 TFLOPS FP16 on H100)      │   │
│  │                                                             │   │
│  │  Memory:                                                    │   │
│  │    - 192 GB HBM3e at 8 TB/s (vs 80GB HBM3 at 3.35 TB/s)  │   │
│  │    - 128 MB L2 cache (vs 50 MB on H100)                    │   │
│  │                                                             │   │
│  │  Interconnect:                                              │   │
│  │    - NVLink 5.0 at 1.8 TB/s bidi (vs 900 GB/s on H100)   │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 6. Every Operation → Hardware Unit

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ OPERATION                     │ HARDWARE UNIT      │ BOUND BY     │ PTX / INSTRUCTION   │
├───────────────────────────────┼────────────────────┼──────────────┼─────────────────────┤
│                               │                    │              │                     │
│ ── ATTENTION ──               │                    │              │                     │
│ Q · K^T (score matrix)       │ TENSOR CORES       │ Compute(pf)  │ wgmma / mma.sync    │
│                               │                    │ Memory(dec)  │                     │
│ exp(S - max)                  │ SFU + CUDA CORES   │ CUDA thruput │ ex2.approx + FMA    │
│ max(S, axis=1)                │ CUDA CORES         │ CUDA thruput │ fmax + __shfl_xor   │
│ sum(P, axis=1)                │ CUDA CORES         │ CUDA thruput │ fadd + __shfl_xor   │
│ P · V (weighted output)      │ TENSOR CORES       │ Compute(pf)  │ wgmma / mma.sync    │
│                               │                    │ Memory(dec)  │                     │
│ Load K/V tiles from HBM       │ MEMORY CONTROLLER  │ HBM bw       │ cp.async / TMA      │
│ Store output to HBM           │ MEMORY CONTROLLER  │ HBM bw       │ st.global           │
│ Block table lookup             │ L2 CACHE           │ L2 latency   │ ld.global           │
│ Causal mask compare            │ CUDA CORES         │ CUDA thruput │ setp.ge             │
│ Softcap: tanh via exp         │ SFU + CUDA CORES   │ CUDA thruput │ ex2.approx + FMA    │
│ ALiBi: position × slope       │ CUDA CORES         │ CUDA thruput │ fma                 │
│ Sink values: add to scores    │ CUDA CORES         │ CUDA thruput │ fadd                │
│ FP8 dequant: × scale          │ CUDA CORES         │ CUDA thruput │ fmul                │
│                               │                    │              │                     │
│ ── LINEAR LAYERS ──           │                    │              │                     │
│ BF16 GEMM (cuBLAS)           │ TENSOR CORES       │ Compute      │ wgmma               │
│ FP8 GEMM (cuBLAS)            │ TENSOR CORES       │ Compute      │ wgmma (fp8)         │
│ FP4 GEMM (CUTLASS, B200)     │ 5th-gen TC         │ Compute      │ BlkScaledTensorOp   │
│ FP4 activation quant          │ CUDA CORES         │ Memory bw    │ cvt.rn.e2m1x2 + FMA│
│ FP4 scale compute (per-16)    │ CUDA CORES         │ Memory bw    │ fabs+fmax+frcp+cvt  │
│ FP4 byte packing (2 per u8)   │ CUDA CORES         │ CUDA thruput │ or.b32             │
│                               │                    │              │                     │
│ ── NORMALIZATION ──           │                    │              │                     │
│ RMSNorm: sum of squares       │ CUDA CORES         │ Memory bw    │ fma                 │
│ RMSNorm: block reduction      │ CUDA CORES + SMEM  │ Memory bw    │ __shfl_xor + st.s   │
│ RMSNorm: rsqrt(variance)      │ SFU                │ SFU thruput  │ rsq.approx.ftz.f32  │
│ RMSNorm: scale + write        │ CUDA CORES         │ Memory bw    │ fmul + cvt + st.g   │
│                               │                    │              │                     │
│ ── POSITION ENCODING ──       │                    │              │                     │
│ RoPE: x*cos - y*sin           │ CUDA CORES         │ Memory bw    │ fma + fsub          │
│ RoPE: y*cos + x*sin           │ CUDA CORES         │ Memory bw    │ fma + fadd          │
│ RoPE: cos/sin lookup          │ L2 CACHE           │ L2 hit rate  │ ld.global (cached)  │
│                               │                    │              │                     │
│ ── ACTIVATION ──              │                    │              │                     │
│ SiLU: exp(-x)                 │ SFU                │ SFU thruput  │ ex2.approx          │
│ SiLU: 1/(1+exp) × x          │ CUDA CORES         │ Memory bw    │ frcp + fmul         │
│ Gate multiply: silu(g) × up   │ CUDA CORES         │ Memory bw    │ fmul                │
│                               │                    │              │                     │
│ ── MoE ROUTING ──             │                    │              │                     │
│ Sigmoid router scores         │ SFU + CUDA CORES   │ CUDA thruput │ ex2.approx + frcp   │
│ Grouped top-k selection       │ CUDA CORES         │ CUDA thruput │ sort + select       │
│ Score correction bias add     │ CUDA CORES         │ CUDA thruput │ fadd                │
│ Token dispatch (permute)      │ MEMORY CONTROLLER  │ Memory bw    │ ld.global + st.g    │
│ Expert GEMM (FP4 group)      │ TENSOR CORES       │ Compute      │ BlkScaledTensorOp   │
│                               │                    │              │                     │
│ ── KV CACHE ──                │                    │              │                     │
│ reshape_and_cache: write K,V  │ MEMORY CONTROLLER  │ Memory bw    │ st.global (scatter) │
│ FP8 KV scale+convert          │ CUDA CORES         │ Memory bw    │ fmul + cvt          │
│                               │                    │              │                     │
│ ── COMMUNICATION ──           │                    │              │                     │
│ AllReduce: cross-GPU read     │ NVLINK             │ NVLink bw    │ ld.global (peer)    │
│ AllReduce: element-wise sum   │ CUDA CORES         │ NVLink bw    │ fadd (FP32)         │
│ AllReduce: fence/barrier      │ NVLINK             │ NVLink lat   │ st.release.sys      │
│                               │                    │              │                     │
│ ── SAMPLING ──                │                    │              │                     │
│ Logit load (vocab_size)       │ MEMORY CONTROLLER  │ Memory bw    │ ld.global (float4)  │
│ Radix histogram               │ CUDA CORES + SMEM  │ SMEM atomics │ atom.shared.add     │
│ Top-k selection                │ CUDA CORES + SMEM  │ SMEM bw      │ BlockRadixSort      │
│ Random sampling (multinomial)  │ CUDA CORES         │ CUDA thruput │ curand + fmul       │
│                               │                    │              │                     │
└───────────────────────────────┴────────────────────┴──────────────┴─────────────────────┘
```

---

## 7. Triton Attention: Thread Block Layout

### 2D Kernel (Prefill / Large-Batch Decode)

```
Grid: (total_q_blocks, num_kv_heads)

                    KV Head 0    KV Head 1    KV Head 2    ...    KV Head 7
                    ─────────    ─────────    ─────────           ─────────
Q Block 0           [TB 0,0]     [TB 0,1]     [TB 0,2]    ...    [TB 0,7]
Q Block 1           [TB 1,0]     [TB 1,1]     [TB 1,2]    ...    [TB 1,7]
Q Block 2           [TB 2,0]     [TB 2,1]     [TB 2,2]    ...    [TB 2,7]
...                    ...          ...          ...                 ...
Q Block 31          [TB 31,0]    [TB 31,1]    [TB 31,2]   ...    [TB 31,7]

Total: 32 × 8 = 256 thread blocks → 256 of 132 SMs occupied (some SMs get 2 TBs)
```

### Inside One Thread Block (Q-Block Layout for GQA=8)

```
BLOCK_M = 16 rows (since GQA ratio ≤ 16)
Each row is ONE (token, query_head) pair:

Row  0: Token 0, Query Head 0  ─┐
Row  1: Token 0, Query Head 1   │
Row  2: Token 0, Query Head 2   │  All 8 query heads
Row  3: Token 0, Query Head 3   │  for Token 0
Row  4: Token 0, Query Head 4   │  (share same KV head)
Row  5: Token 0, Query Head 5   │
Row  6: Token 0, Query Head 6   │
Row  7: Token 0, Query Head 7  ─┘
Row  8: Token 1, Query Head 0  ─┐
Row  9: Token 1, Query Head 1   │  All 8 query heads
Row 10: Token 1, Query Head 2   │  for Token 1
Row 11: Token 1, Query Head 3   │  (share same KV head)
Row 12: Token 1, Query Head 4   │
Row 13: Token 1, Query Head 5   │
Row 14: Token 1, Query Head 6   │
Row 15: Token 1, Query Head 7  ─┘

K and V are loaded ONCE per tile, shared across all 16 rows:

  K tile: [HEAD_SIZE × TILE_SIZE] loaded from HBM → SMEM    ← 1 load
  V tile: [TILE_SIZE × HEAD_SIZE] loaded from HBM → SMEM    ← 1 load

  S = Q × K^T → [16 × TILE_SIZE]     (tensor cores)
  P = softmax(S) → [16 × TILE_SIZE]   (CUDA cores + SFU)
  O += P × V → [16 × HEAD_SIZE]       (tensor cores)

Without Q-blocks: K and V loaded 8× (once per query head) — 8× HBM waste
With Q-blocks:    K and V loaded 1× — all heads packed in BLOCK_M dimension
```

### 3D Kernel (Small-Batch Decode with Parallel Segments)

```
Grid: (total_q_blocks, num_kv_heads, 16)

Problem: Batch=1, 8 KV heads → 8 thread blocks → 8 of 132 SMs busy (6%)

Solution: Split each sequence's KV cache into 16 segments:

Sequence (10,000 tokens in KV cache):
├── Segment 0:  tokens    0- 624  ──→ Thread Block (0, kv_h, 0)
├── Segment 1:  tokens  625-1249  ──→ Thread Block (0, kv_h, 1)
├── Segment 2:  tokens 1250-1874  ──→ Thread Block (0, kv_h, 2)
├── Segment 3:  tokens 1875-2499  ──→ Thread Block (0, kv_h, 3)
│   ...
├── Segment 14: tokens 8750-9374  ──→ Thread Block (0, kv_h, 14)
└── Segment 15: tokens 9375-9999  ──→ Thread Block (0, kv_h, 15)

Each thread block computes partial: (output_i, max_i, exp_sum_i)

Total: 1 × 8 × 16 = 128 thread blocks → 128 of 132 SMs busy (97%)


Then a second kernel merges the 16 partial results:

reduce_segments kernel:
  Grid: (num_tokens, num_query_heads)

  For each query head:
    global_max = max(max_0, max_1, ..., max_15)
    For each segment i:
      correction = exp(max_i - global_max)
      global_output += output_i × correction
      global_expsum += expsum_i × correction
    final = global_output / global_expsum

  This is EXACT (online softmax identity), not an approximation.
```

---

## 8. NVFP4 Linear: Data Flow Through Tensor Cores

```
Input: hidden_states [32 tokens × 4096 dims] in BF16

═══════════════════════════════════════════════════════════════
STEP A: Quantize Activations (CUDA Cores)
csrc/quantization/fp4/nvfp4_quant_kernels.cu
═══════════════════════════════════════════════════════════════

BF16 input:  [0.73, -1.2, 0.05, 0.91, -0.34, 2.1, -0.8, 0.12,   ← 16 values
              0.45, -0.67, 1.5, -0.23, 0.88, -1.1, 0.32, -0.56]

Step A1: Find max absolute value across 16 elements
  max_abs = 2.1

Step A2: Compute block scale (FP8 E4M3)
  scale = global_scale × (2.1 / 6.0) = global_scale × 0.35
  scale_fp8 = cast_to_E4M3(scale)     ← 8-bit scale factor

Step A3: Quantize each value to E2M1 (4-bit float)
  E2M1 representable: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}

  0.73 / scale → round to E2M1 → 1.0    (4 bits)
  -1.2 / scale → round to E2M1 → -2.0   (4 bits)
  ...

Step A4: Pack two E2M1 values into one uint8
  [1.0, -2.0] → packed as [low_nibble | high_nibble] → 0xAB  (1 byte)

Output:
  x_fp4:        [32, 2048] uint8   (4096 dims / 2 per byte)
  x_blockscale: [32, 256]  FP8     (4096 dims / 16 per scale)


═══════════════════════════════════════════════════════════════
STEP B: FP4 GEMM (5th-Gen Tensor Cores on B200)
csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu
═══════════════════════════════════════════════════════════════

                    x_fp4 [32 × 2048]              weight_fp4 [N × 2048]
                    ┌──────────────────┐            ┌──────────────────┐
                    │ E2M1 packed uint8│            │ E2M1 packed uint8│
                    │ 2 values / byte  │            │ 2 values / byte  │
                    └────────┬─────────┘            └────────┬─────────┘
                             │                               │
                    ┌────────┴─────────┐            ┌────────┴─────────┐
                    │ x_blockscale     │            │ weight_scale     │
                    │ [32 × 256] FP8   │            │ [N × 256] FP8   │
                    │ 1 scale / 16 vals│            │ 1 scale / 16 vals│
                    └────────┬─────────┘            └────────┬─────────┘
                             │                               │
                             ▼                               ▼
                    ┌────────────────────────────────────────────────┐
                    │         B200 5th-Gen Tensor Core               │
                    │                                                │
                    │  For each 256×128×256 tile:                    │
                    │    1. Load FP4 data tile from SMEM             │
                    │    2. Load FP8 block scales from SMEM          │
                    │    3. Hardware dequantize:                      │
                    │       actual_value = E2M1_value × E4M3_scale   │
                    │    4. Multiply-accumulate in FP32:              │
                    │       C_fp32 += A_dequant × B_dequant          │
                    │    5. After all K tiles:                        │
                    │       output = C_fp32 × alpha                  │
                    │       alpha = input_global_scale × weight_gs   │
                    │       cast FP32 → BF16                         │
                    │                                                │
                    └────────────────────┬───────────────────────────┘
                                         │
                                         ▼
                              output [32 × N] BF16
```

---

## 9. Memory Hierarchy: Where Data Lives

```
┌─────────────────────── REGISTER FILE ──────────────────────────────┐
│ Size: 256 KB per SM (64 KB per sub-partition × 4)                  │
│ Latency: 0 cycles (same cycle as compute)                          │
│ What lives here:                                                    │
│   - Q tile for current attention computation                       │
│   - Attention score accumulator (acc) [BLOCK_M × HEAD_SIZE] FP32  │
│   - Running max (m_j) and exp_sum (l_j) per row                   │
│   - RoPE cos/sin values after load                                  │
│   - Loop variables, indices, masks                                  │
│ CUDA graph status: addresses frozen, contents computed each replay │
└────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────── SHARED MEMORY (SMEM) ──────────────────────┐
│ Size: up to 228 KB per SM (H100), configurable                     │
│ Latency: ~20-30 cycles                                             │
│ What lives here:                                                    │
│   - K tile: [HEAD_SIZE × TILE_SIZE] × 2B = 4-8 KB                │
│   - V tile: [TILE_SIZE × HEAD_SIZE] × 2B = 4-8 KB                │
│   - CUB BlockReduce temp storage (RMSNorm, sampling)              │
│   - Sampling histogram bins [2048 × 4B = 8 KB]                    │
│   - CUTLASS GEMM pipeline buffers (double/triple buffered)        │
│ CUDA graph status: same as registers (transient)                   │
└────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────── L2 CACHE ──────────────────────────────────┐
│ Size: 50 MB (H100) / 128 MB (B200)                                │
│ Latency: ~200 cycles                                               │
│ What lives here:                                                    │
│   - RoPE cos/sin lookup table (~512 KB, fits entirely in L2)      │
│   - Block table tensor (small, frequently reused)                  │
│   - Recently-accessed KV cache pages (hot working set)             │
│   - Weight tiles that get reused across batch elements             │
│ CUDA graph status: contents may differ (cache is transparent)      │
└────────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────── HBM (High Bandwidth Memory) ───────────────┐
│ Size: 80 GB (H100) / 192 GB (B200)                                │
│ Bandwidth: 3.35 TB/s (H100) / 8 TB/s (B200)                      │
│ Latency: ~400-600 cycles                                           │
│ What lives here:                                                    │
│                                                                     │
│   PERSISTENT BUFFERS (frozen addresses, updated contents):         │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │ input_ids  [max_tokens]          ~2 KB   (int32)         │     │
│   │ positions  [max_tokens]          ~4 KB   (int64)         │     │
│   │ seq_lens   [max_reqs]            ~1 KB   (int32)         │     │
│   │ block_table[max_reqs, max_blks]  ~64 KB  (int32)        │     │
│   │ slot_mapping[max_tokens]         ~2 KB   (int32)         │     │
│   │ query_start_loc[max_reqs+1]      ~1 KB   (int32)        │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│   MODEL WEIGHTS (read-only during inference):                      │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │ BF16: each linear layer = in_dim × out_dim × 2 bytes    │     │
│   │ FP8:  each linear layer = in_dim × out_dim × 1 byte     │     │
│   │ FP4:  each linear layer = in_dim × out_dim × 0.5 bytes  │     │
│   │       + block_scales: out_dim × (in_dim/16) × 1 byte    │     │
│   │       + global_scale: 1 × 4 bytes                       │     │
│   │                                                          │     │
│   │ Example (70B FP4): ~35 GB weights + ~4 GB scales        │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│   KV CACHE (grows with context length):                            │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │ Layout: [2, num_blocks, block_size, num_kv_heads, head]  │     │
│   │ BF16: 2 × num_blocks × 16 × 8 × 128 × 2B per block    │     │
│   │ FP8:  same but 1B per element (half the memory)          │     │
│   │                                                          │     │
│   │ Budget: total HBM - weights - activation workspace       │     │
│   │ Example (70B FP4 on B200):                               │     │
│   │   192 GB - 39 GB weights - 8 GB workspace = 145 GB KV   │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                     │
│   ACTIVATION WORKSPACE (reused each layer):                        │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │ hidden_states: [batch, hidden_dim] × 2B                  │     │
│   │ residual:      [batch, hidden_dim] × 2B                  │     │
│   │ QKV output:    [batch, (num_heads+2*num_kv_heads)*head]  │     │
│   │ MoE intermediates: [batch, intermediate_dim] × experts   │     │
│   │ logits:        [batch, vocab_size] × 4B (last layer)     │     │
│   └──────────────────────────────────────────────────────────┘     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Multi-GPU: Tensor Parallel Communication

```
4-GPU Tensor Parallel: Each GPU holds 1/4 of attention heads and 1/4 of MLP

GPU 0                 GPU 1                 GPU 2                 GPU 3
┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│ Heads 0-7   │       │ Heads 8-15  │       │ Heads 16-23 │       │ Heads 24-31 │
│ (QKV GEMM)  │       │ (QKV GEMM)  │       │ (QKV GEMM)  │       │ (QKV GEMM)  │
│ (Attention)  │       │ (Attention)  │       │ (Attention)  │       │ (Attention)  │
│ (O Project) │       │ (O Project) │       │ (O Project) │       │ (O Project) │
└──────┬──────┘       └──────┬──────┘       └──────┬──────┘       └──────┬──────┘
       │                     │                     │                     │
       └─────────┬───────────┴──────────┬──────────┴──────────┬──────────┘
                 │                      │                     │
                 ▼                      ▼                     ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                    ALLREDUCE over NVLink                     │
       │                                                             │
       │  Phase 1: Each GPU reads from all other GPUs via NVLink    │
       │           GPU 0 reads GPU 1,2,3 partial sums               │
       │           GPU 1 reads GPU 0,2,3 partial sums               │
       │           GPU 2 reads GPU 0,1,3 partial sums               │
       │           GPU 3 reads GPU 0,1,2 partial sums               │
       │                                                             │
       │  Phase 2: Each GPU sums all partials (CUDA cores, FP32)   │
       │                                                             │
       │  Sync: st.release.sys / ld.acquire.sys (NVLink fences)    │
       │                                                             │
       │  H100: 4 × 900 GB/s = 3.6 TB/s aggregate NVLink          │
       │  B200: 4 × 1.8 TB/s = 7.2 TB/s aggregate NVLink          │
       │                                                             │
       │  Time for 32 tokens × 4096 dims × 2B:                     │
       │    262 KB / (900 GB/s / 4 GPUs) ≈ 1.2 μs (H100)         │
       │    262 KB / (1.8 TB/s / 4 GPUs) ≈ 0.6 μs (B200)         │
       └─────────────────────────────────────────────────────────────┘
                 │                      │                     │
                 ▼                      ▼                     ▼
       ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
       │ GPU 0: full  │       │ GPU 1: full  │       │ GPU 2: full  │  ...
       │ hidden state │       │ hidden state │       │ hidden state │
       └──────────────┘       └──────────────┘       └──────────────┘
                 │
                 ▼
       (Feed into MoE / next layer — each GPU has the same full result)
```

### Expert Parallel (MoE) Communication

```
8-GPU Expert Parallel: Each GPU holds ~32 of 256 experts

                          ROUTER (all GPUs compute in parallel)
                          ┌──────────────────────────────────┐
                          │ Token 0 → Expert 12, 45, 189     │
                          │ Token 1 → Expert 3, 77, 201      │
                          │ Token 2 → Expert 45, 99, 156     │
                          │ ...                               │
                          └──────────────┬───────────────────┘
                                         │
                                         ▼
                          ┌──── ALL-TO-ALL DISPATCH ────┐
                          │                              │
  GPU 0 (experts 0-31)   │ Token 0 needs expert 12 ──→ │ stays on GPU 0
  GPU 1 (experts 32-63)  │ Token 0 needs expert 45 ──→ │ sent to GPU 1
  GPU 5 (experts 160-191)│ Token 0 needs expert 189 ─→ │ sent to GPU 5
                          │                              │
                          │ (NVLink all-to-all)          │
                          └──────────────────────────────┘
                                         │
                                         ▼
                          Each GPU runs its local expert GEMMs
                          on the tokens routed to it
                                         │
                                         ▼
                          ┌──── ALL-TO-ALL GATHER ──────┐
                          │ Collect results back to      │
                          │ the GPU that owns each token  │
                          └──────────────────────────────┘
```

---

## 11. Batch Size Padding and Dispatch

```
vllm/config/compilation.py:1115-1128 + vllm/v1/cudagraph_dispatcher.py:143-183

Pre-captured graph sizes: [1, 2, 4, 8, 16, 24, 32, 40, ..., 248, 256, 272, ..., 512]

Actual batch → Padded → Graph selected:

  Actual: 1  ────→ Padded: 1   ────→ Graph for batch=1   (no waste)
  Actual: 3  ────→ Padded: 4   ────→ Graph for batch=4   (1 wasted token)
  Actual: 5  ────→ Padded: 8   ────→ Graph for batch=8   (3 wasted tokens)
  Actual: 37 ────→ Padded: 40  ────→ Graph for batch=40  (3 wasted tokens)
  Actual: 129────→ Padded: 136 ────→ Graph for batch=136 (7 wasted tokens)
  Actual: 255────→ Padded: 256 ────→ Graph for batch=256 (1 wasted token)
  Actual: 513────→ No graph    ────→ EAGER execution     (too large)

Padding lookup is O(1):
  bs_to_padded_graph_size = [0, 1, 2, 4, 4, 8, 8, 8, 8, 16, ...]
                             ^  ^  ^  ^     ^              ^
                             0  1  2  3     5              9

  padded = bs_to_padded_graph_size[actual_batch_size]
```

### Dispatch Priority

```
CudagraphDispatcher.dispatch(num_tokens=37, uniform_decode=True):

  Step 1: Pad 37 → 40

  Step 2: Create BatchDescriptor(num_tokens=40, num_reqs=37, uniform=True)

  Step 3: Try FULL exact match
    Key: (40, 37, True, None)
    Found in cudagraph_keys[FULL]? → YES (captured during warmup)
    Return: (CUDAGraphMode.FULL, BatchDescriptor(40, 37, True))

  If not found:
  Step 4: Try FULL relaxed match
    Key: (40, None, False, None)    ← num_reqs=None, uniform=False
    Found in cudagraph_keys[FULL]? → maybe

  If not found:
  Step 5: Try PIECEWISE relaxed match
    Key: (40, None, False, None)
    Found in cudagraph_keys[PIECEWISE]? → usually yes

  If nothing found:
  Step 6: Return CUDAGraphMode.NONE → eager execution
```

---

## 12. Prefill vs Decode: Why Everything Changes

```
═══════════════════════════════════════════════════════════════════════
PREFILL: Process 2,000 input tokens at once (first request)
═══════════════════════════════════════════════════════════════════════

Token count: 2,000   (too large for CUDA graphs → EAGER execution)
KV cache:    empty → write 2,000 new K/V entries

  Linear layers: [2000, 4096] × [4096, 4096] = COMPUTE BOUND
                 Tensor cores at ~80% utilization
                 Time: ~2 ms per GEMM

  Attention:     Q=[2000, 128], K=[2000, 128], V=[2000, 128]
                 Full causal triangle: 2000 × 2000 / 2 = 2M dot products
                 COMPUTE BOUND (tensor cores doing massive matmul)
                 Time: ~5 ms

  Total: ~300 ms for all layers (GPU fully utilized)

GPU Utilization:
  ████████████████████████████████████████████  ~85% TC utilization
  ████████████████                              ~40% memory BW utilization


═══════════════════════════════════════════════════════════════════════
DECODE: Generate 1 token per request, 32 requests in batch
═══════════════════════════════════════════════════════════════════════

Token count: 32   → CUDA graph (padded to 32)
KV cache:    read 32 × seq_len entries, write 32 new ones

  Linear layers: [32, 4096] × [4096, 4096] = MEMORY BOUND
                 Tensor cores wait for weight loads
                 Weight matrix (4096×4096×0.5B FP4 = 8 MB) loaded from HBM
                 32 tokens of compute can't keep TCs busy while loading
                 Time: ~15 μs per GEMM

  Attention:     Q=[32, 128], must scan K/V for all seq_len tokens
                 32 queries × 4000 KV tokens × 128 dims
                 MEMORY BOUND (loading KV cache dominates)
                 KV load: 32 × 4000 × 128 × 2B × 2 (K+V) = 64 MB from HBM
                 Time: ~15-20 μs

  Total: ~4.8 ms for all layers (GPU limited by memory bandwidth)

GPU Utilization:
  ████                                          ~10% TC utilization
  ████████████████████████████████████████████  ~90% memory BW utilization

THIS IS WHY:
  - CUDA graphs matter more for DECODE (CPU overhead is significant vs small GPU work)
  - NVFP4 matters more for DECODE (smaller weights → less HBM traffic → faster loads)
  - Parallel softmax (16 segments) matters for DECODE (need to fill empty SMs)
  - Batching matters for DECODE (more tokens = more TC work per weight load)
```

### The Crossover

```
Batch size vs Bottleneck (H100, 70B model, seq_len=4096):

Batch    Linear Layer    Attention       Bottleneck
  1      Memory ██░░░    Memory ██░░░    MEMORY (GPU 10% utilized)
  4      Memory ███░░    Memory ███░░    MEMORY (GPU 25% utilized)
  16     Memory ████░    Memory ████░    MEMORY (GPU 60% utilized)
  64     Balanced ████   Memory █████    MIXED  (GPU 75% utilized)
  256    Compute █████   Memory █████    MIXED  (GPU 85% utilized)
  1024   Compute ██████  Compute █████   COMPUTE (GPU 90% utilized)
  2000+  Compute ██████  Compute ██████  COMPUTE (GPU 95% utilized)
                                          ↑ prefill territory
```
