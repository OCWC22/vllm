# vLLM Triton Attention Backend: Office Hours #43 Deep Dive

Tracing every claim from the Feb 12, 2026 vLLM Office Hours presentation
(Burkhard Ringlein / IBM Research, Michael Goin / Red Hat, Sasa Zelenovic / Red Hat)
into the actual vLLM source code, showing exactly what CUDA, ROCm, and Triton do,
where each operation runs on the GPU, and what it means for system design.

---

## Table of Contents

1.  [What Is the Triton Attention Backend?](#1-what-is-the-triton-attention-backend)
2.  [Why Triton? The Performance Portability Argument](#2-why-triton-the-performance-portability-argument)
3.  [Architecture: Three Kernels, One Source](#3-architecture-three-kernels-one-source)
4.  [Q-Blocks: The GQA Optimization (Before vs. After)](#4-q-blocks-the-gqa-optimization)
5.  [Parallel Tiled Softmax: The Two-Kernel Trick](#5-parallel-tiled-softmax-the-two-kernel-trick)
6.  [CUDA Graphs and the Static Launch Grid](#6-cuda-graphs-and-the-static-launch-grid)
7.  [Backend Selection: When Does Triton Run?](#7-backend-selection-when-does-triton-run)
8.  [Sliding Window, ALiBi, Sinks, Multimodal Prefix](#8-sliding-window-alibi-sinks-multimodal-prefix)
9.  [Hardware-Specific Tuning: H100 vs MI300X vs A100](#9-hardware-specific-tuning-h100-vs-mi300x-vs-a100)
10. [GPU Execution Map: Where Every Operation Runs](#10-gpu-execution-map-where-every-operation-runs)
11. [Benchmarks: 100.7% of FlashAttention 3 on H100](#11-benchmarks-1007-of-flashattention-3-on-h100)
12. [Batch Invariance: Deterministic Output Across Batch Sizes](#12-batch-invariance)
13. [Helion: The Future DSL](#13-helion-the-future-dsl)
14. [Future Use Cases and System Design](#14-future-use-cases-and-system-design)

---

## 1. What Is the Triton Attention Backend?

Three files, ~1,100 lines of Triton, written by IBM Research Zurich:

| File | Lines | Purpose |
|------|-------|---------|
| `vllm/attention/ops/triton_unified_attention.py` | ~1,060 | Three `@triton.jit` kernels: 2D, 3D, and reduce |
| `vllm/v1/attention/backends/triton_attn.py` | ~500 | Backend wrapper, metadata builder, CUDA graph support |
| `vllm/attention/ops/triton_reshape_and_cache_flash.py` | ~160 | KV cache store kernel |

Authors (`triton_unified_attention.py:5-8`):
```python
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>
```

**The talk's central claim**: ~800 lines of Triton achieves 100.7% of FlashAttention 3's performance on H100 (end-to-end latency) — replacing ~70,000 lines of CUDA across ~528 kernel variants. And the *same source* runs on AMD MI300X as state-of-the-art.

Reference paper: *The Anatomy of a Triton Attention Kernel* (arxiv.org/abs/2511.11581)

---

## 2. Why Triton? The Performance Portability Argument

The talk defines the problem: vLLM supports 18 attention backends across NVIDIA, AMD, Intel, TPU, etc. Writing separate CUDA/HIP/SYCL kernels for each is unsustainable.

**Triton's abstraction**: You write tiled programs (blocks of threads), Triton compiles to:
- **PTX** → NVIDIA GPUs (A100, H100, B200)
- **GCN/CDNA ISA** → AMD GPUs (MI300X, MI325X)
- **SPIR-V** → Intel GPUs (via triton-cpu/xpu)

The key insight from the talk: *"Triton's tiled programming model allows expressing complicated
optimizations while staying hardware agnostic. Hyperparameters (called 'configurations') allow
simple but effective adaptations for different hardware."*

In the actual code, this manifests as **runtime heuristics** rather than `@triton.autotune`:

```python
# triton_unified_attention.py:896-899 — BLOCK_M/BLOCK_Q computed at runtime
BLOCK_M = (
    16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
)
BLOCK_Q = BLOCK_M // num_queries_per_kv

# triton_unified_attention.py:817-836 — TILE_SIZE selected by platform heuristic
def _get_tile_size(head_size, sliding_window, element_size, is_prefill):
    if _is_gemma3_attention(head_size, sliding_window):
        return 32                                    # Gemma3: always 32
    if is_prefill:
        return 32                                    # Prefill: always 32
    return 16 if element_size >= 2 else 32           # Decode: 16 for bf16, 32 for fp8

# triton_reshape_and_cache_flash.py:146-153 — Platform-specific kernel params
if current_platform.is_rocm() or current_platform.is_xpu():
    num_stages = 4     # AMD/Intel: fewer pipeline stages
    num_warps = 8
else:                  # NVIDIA CUDA
    num_stages = 10    # NVIDIA: deeper software pipeline
    num_warps = 16
    if torch.cuda.get_device_capability(key.device)[0] < 9:   # Pre-Hopper
        TILE_SIZE = min(512, TILE_SIZE)
```

The talk mentions Llama 3 configs: *"H100: BLOCK_M=4, BLOCK_Q=4; MI300: BLOCK_M=64, BLOCK_Q=16"*
— these come from a different version of the tuning (likely with `@triton.autotune`). The current
production code uses the simpler heuristic above, which for Llama 3 (GQA ratio=8) gives:
`BLOCK_M=16, BLOCK_Q=2`.

---

## 3. Architecture: Three Kernels, One Source

### Kernel 1: `kernel_unified_attention_2d` (line 57)

Used for: **all prefill** + **large-batch decode**

```
Launch grid: (total_num_q_blocks, num_kv_heads)
                     ↑                    ↑
         Over-provisioned upper bound    One column per KV head
```

Each thread block:
1. Binary-searches `query_start_len_ptr` to find which sequence it belongs to (line 111)
2. Loads Q tiles for BLOCK_Q query tokens × all GQA heads for one KV head (line 145)
3. Iterates over KV tiles, computing `Q × K^T → softmax → × V → accumulate` (line 230)
4. Writes final output (line 387)

### Kernel 2: `kernel_unified_attention_3d` (line 394)

Used for: **small-batch decode with long context**

```
Launch grid: (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
                                                         ↑
                                              16 segments per sequence
```

Same as 2D, but each thread block only processes `1/16th` of the KV sequence:

```python
# triton_unified_attention.py:468-472
num_segments = NUM_SEGMENTS_PER_SEQ    # 16
tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)
if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
    return  # This segment has no work
```

Each segment writes **partial results** to intermediate buffers:
- `segm_output_ptr`: partial attention output per segment
- `segm_max_ptr`: running softmax maximum per segment
- `segm_expsum_ptr`: running softmax exp-sum per segment

### Kernel 3: `reduce_segments` (line 717)

Merges the 16 partial segment results using the **online softmax** algorithm:

```python
# triton_unified_attention.py:768-792
segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
overall_max = tl.max(segm_max)                          # Global max across all segments

segm_expsum = tl.load(segm_expsum_ptr + segm_offset, ...)
segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)  # Rescale each segment
overall_expsum = tl.sum(segm_expsum)                          # Global normalizer

segm_output = tl.load(segm_output_ptr + segm_output_offset, ...)
segm_output *= tl.exp(segm_max - overall_max)[:, None]       # Rescale outputs
acc_sum = tl.sum(segm_output, axis=0)                          # Sum all segments
acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)
```

### When to Use 2D vs 3D

```python
# triton_unified_attention.py:932-940
if (
    max_seqlen_q > 1           # Any prefill request → 2D (need full causal mask)
    or num_seqs > seq_threshold_3D  # Large batch → 2D (enough parallelism already)
    or <buffers not allocated>
):
    # Launch 2D kernel
else:
    # Launch 3D kernel + reduce_segments

# triton_attn.py:41,151
MIN_LAUNCH_GRID_SIZE_2D = 128
seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
# e.g., Llama 3 with 8 KV heads: threshold = 128 / 8 = 16
# If num_seqs <= 16, use 3D. If num_seqs > 16, use 2D.
```

**The talk explains why**: *"Triton doesn't have a global barrier → two kernels. Trade-off:
additional launch overhead vs more parallelism."*

---

## 4. Q-Blocks: The GQA Optimization

This is the core innovation. The talk's slide says: *"Q-Blocks! Combination of query heads and
multiple query tokens as one 'work item': Load all query heads for one KV head (GQA optimization
→ cache reuse) + fill tiles with multiple query tokens."*

### Before: Naive Per-Head Attention

```
For each KV head h:
  For each query head q in GQA group of h:    ← num_queries_per_kv iterations
    For each query token t:
      Load K[h], V[h]                         ← RELOADS K/V every iteration!
      Compute attention(Q[t,q], K[h], V[h])
```

K and V are loaded `num_queries_per_kv × num_tokens` times from HBM. Massive waste.

### After: Q-Block Tiling (actual code)

```python
# triton_unified_attention.py:127-133
offs_m = tl.arange(0, BLOCK_M)      # e.g., [0, 1, 2, ..., 15] for BLOCK_M=16
query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
#           ↑ which block of tokens          ↑ integer divide maps rows → token positions

query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
#                ↑ base query head for this KV head   ↑ modulo maps rows → specific query heads
```

For **Llama 3** with `num_queries_per_kv = 8` (32 query heads / 4 KV heads = 8):

```
BLOCK_M = 16, BLOCK_Q = 2

Row layout within one BLOCK_M tile:
  Row  0: token 0 (offs_m=0  // 8 = 0), head 0 (offs_m=0  % 8 = 0)
  Row  1: token 0 (offs_m=1  // 8 = 0), head 1 (offs_m=1  % 8 = 1)
  Row  2: token 0 (offs_m=2  // 8 = 0), head 2 (offs_m=2  % 8 = 2)
  Row  3: token 0 (offs_m=3  // 8 = 0), head 3 (offs_m=3  % 8 = 3)
  Row  4: token 0 (offs_m=4  // 8 = 0), head 4 (offs_m=4  % 8 = 4)
  Row  5: token 0 (offs_m=5  // 8 = 0), head 5 (offs_m=5  % 8 = 5)
  Row  6: token 0 (offs_m=6  // 8 = 0), head 6 (offs_m=6  % 8 = 6)
  Row  7: token 0 (offs_m=7  // 8 = 0), head 7 (offs_m=7  % 8 = 7)
  Row  8: token 1 (offs_m=8  // 8 = 1), head 0 (offs_m=8  % 8 = 0)
  Row  9: token 1 (offs_m=9  // 8 = 1), head 1 (offs_m=9  % 8 = 1)
  Row 10: token 1 (offs_m=10 // 8 = 1), head 2 (offs_m=10 % 8 = 2)
  Row 11: token 1 (offs_m=11 // 8 = 1), head 3 (offs_m=11 % 8 = 3)
  Row 12: token 1 (offs_m=12 // 8 = 1), head 4 (offs_m=12 % 8 = 4)
  Row 13: token 1 (offs_m=13 // 8 = 1), head 5 (offs_m=13 % 8 = 5)
  Row 14: token 1 (offs_m=14 // 8 = 1), head 6 (offs_m=14 % 8 = 6)
  Row 15: token 1 (offs_m=15 // 8 = 1), head 7 (offs_m=15 % 8 = 7)
```

K and V are loaded **once** per tile, indexed only by `kv_head_idx`:

```python
# triton_unified_attention.py:245-265
k_offset = (
    physical_block_idx[None, :] * stride_k_cache_0
    + kv_head_idx * stride_k_cache_2          # ← single KV head
    + offs_d[:, None] * stride_k_cache_3
    + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
)
K_load = tl.load(key_cache_ptr + k_offset, ...)
```

Then the dot product `tl.dot(Q, K)` computes scores for **all 8 query heads × 2 tokens** against
the same K tile simultaneously:

```python
# triton_unified_attention.py:318
S += scale * tl.dot(Q, K)  # Q: [16, head_size], K: [head_size, TILE_SIZE]
                            # S: [16, TILE_SIZE] — all 16 Q-rows × all KV positions
```

### GPU Impact

```
Before (per-head):
  K load from HBM: 8 times per tile (once per query head)
  HBM bandwidth consumed: 8x

After (Q-blocks):
  K load from HBM: 1 time per tile (shared across all 8 query heads)
  HBM bandwidth consumed: 1x
  tl.dot() utilization: 16 rows instead of 1-2 → much better tensor core occupancy
```

---

## 5. Parallel Tiled Softmax: The Two-Kernel Trick

The talk's slide explains: *"Long-context requests require additional parallelization. New kernel
that partitions a single request into many tiles per single request. Reduce partial results later
(online softmax). But: Triton doesn't have a global barrier → two kernels."*

### The Problem

During decode, each request generates **1 query token**. With Llama 3 (8 KV heads),
the 2D kernel launches `num_seqs × 8` thread blocks. For batch_size=1, that's only **8 blocks**.
An H100 has 132 SMs — **124 SMs sit idle**.

### The Solution: Segment the KV Sequence

The 3D kernel divides each request's KV sequence into 16 segments. Now batch_size=1 launches
`1 × 8 × 16 = 128` thread blocks — utilizing **128 of 132 SMs**.

```
Sequence length = 32,768 tokens, TILE_SIZE = 16

Without segments (2D):
  1 thread block processes all 2048 tiles sequentially
  1 SM busy, 131 idle

With 16 segments (3D):
  Each segment: 2048 / 16 = 128 tiles
  16 thread blocks process 128 tiles each, IN PARALLEL
  16 SMs busy × 8 KV heads = 128 SMs utilized
```

### The Intermediate Buffers

Pre-allocated at metadata build time (`triton_attn.py:166-187`):

```python
self.softmax_segm_output = torch.empty(
    (seq_threshold_3D, num_heads_q, num_par_softmax_segments, headdim_padded),
    dtype=torch.float32, device=device,
)
self.softmax_segm_max = torch.empty(
    (seq_threshold_3D, num_heads_q, num_par_softmax_segments),
    dtype=torch.float32, device=device,
)
self.softmax_segm_expsum = torch.empty(
    (seq_threshold_3D, num_heads_q, num_par_softmax_segments),
    dtype=torch.float32, device=device,
)
```

For Llama-3.1-8B with 32 query heads, threshold=16, 16 segments, head_size=128:
- `segm_output`: 16 × 32 × 16 × 128 × 4 bytes = **4 MB** (fp32)
- `segm_max`: 16 × 32 × 16 × 4 bytes = **32 KB**
- `segm_expsum`: **32 KB**
- Total overhead: ~4 MB in HBM — negligible

### The Online Softmax Reduction

Each segment computes its own local `max(S)` and `sum(exp(S - max))`. The reduction kernel
combines them using the identity:

```
softmax(concat(S_1, S_2, ..., S_16)) =
  Let M = max(M_1, M_2, ..., M_16)
  Let L_i' = L_i × exp(M_i - M)          # Rescale each segment's exp-sum
  Let O_i' = O_i × exp(M_i - M)          # Rescale each segment's output
  Result = sum(O_i') / sum(L_i')
```

This is mathematically exact — no approximation.

---

## 6. CUDA Graphs and the Static Launch Grid

The talk says: *"CUDA graph 'freezes' the kernel: arguments (& pointers), launch grid. Careful
balance of launch grids for CUDA graphs. Persistent kernels: minimize launch overhead of Triton
kernels (can be up to 200μs). Add outer loop. Metadata stored as tensor on the GPU."*

### How It Works in Code

**Step 1: Over-provisioned grid** (`triton_unified_attention.py:901-910`)

```python
# Upper bound on number of Q-blocks, computed WITHOUT reading actual query_lens from CPU
total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
# q.shape[0] is the max padded token count for the CUDA graph capture size
```

This is batch-size-independent: the grid dimension is determined by the CUDA graph capture
size, not the actual number of sequences.

**Step 2: Early exit guard** (`triton_unified_attention.py:124-125`)

```python
if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
    return  # Thread block has no work — exit immediately
```

Thread blocks for "empty" sequences exit in ~10 cycles (1 register compare + branch).

**Step 3: Binary search for sequence mapping** (`triton_unified_attention.py:34-53`)

```python
@triton.jit
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q, use_q_block_mode):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1
```

Each thread block discovers its own sequence assignment by binary-searching a GPU-side metadata
tensor. No CPU involvement. The metadata tensor is updated each step, but its **address is frozen**
in the CUDA graph — only its **contents** change.

**Step 4: AttentionCGSupport.ALWAYS** (`triton_attn.py:115`)

```python
class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
```

`ALWAYS` (value=3) is the **highest level** — supports mixed prefill+decode within a single
captured graph. Compare:
- Flash Attention: `ALWAYS` with FA3, `UNIFORM_BATCH` with FA2
- FlashInfer: `ALWAYS`
- Triton: `ALWAYS` ← matches the best

**Step 5: 2D/3D threshold snapped to graph capture sizes** (`triton_attn.py:153-164`)

```python
if self.decode_cudagraph_enabled:
    capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
    self.seq_threshold_3D = min(
        capture_sizes,
        key=lambda x: abs(x - self.seq_threshold_3D),
    )
```

This ensures the 2D↔3D transition happens at a batch size that aligns with a CUDA graph
capture boundary, avoiding graph miss when transitioning.

---

## 7. Backend Selection: When Does Triton Run?

The talk says: *"Default on AMD GPUs (rocm platform). On Intel XPU with float32. Models requiring
specific features: ALiBi sqrt (stepfun audio models), Sink Tokens / GPT-OSS. As a Fallback when
dependencies are not installed."*

### ROCm: Default Backend

```python
# vllm/platforms/rocm.py:265-302
if selected_backend is None:
    # Priority 1: AITER Unified Attention (if VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1)
    # Priority 2: AITER MHA Flash Attention (if VLLM_ROCM_USE_AITER_MHA=1 + gfx9)
    # Priority 3: ROCM_ATTN (if VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1)
    # Priority 4: AITER FA (if VLLM_ROCM_USE_AITER=1 + gfx9)
    # → DEFAULT FALLBACK:
    logger.info("Using Triton Attention backend.")
    return AttentionBackendEnum.TRITON_ATTN.get_path()
```

Triton is the **ultimate fallback on AMD**. In practice, many AMD deployments use it as the
primary backend because AITER is optional and may not be installed.

### CUDA: Third Priority

```python
# vllm/platforms/cuda.py:44-82
# Hopper (H100, SM 9.0):
priority = [
    FLASH_ATTN,      # 1st: Flash Attention 2/3
    FLASHINFER,      # 2nd: FlashInfer
    TRITON_ATTN,     # 3rd: Triton ← runs when FA and FI are unavailable
    FLEX_ATTENTION,  # 4th: PyTorch FlexAttention
]

# Blackwell (B200, SM 10.0):
priority = [
    FLASHINFER,      # 1st: FlashInfer (optimized for SM 10.0)
    FLASH_ATTN,      # 2nd: Flash Attention
    TRITON_ATTN,     # 3rd: Triton
    FLEX_ATTENTION,  # 4th: FlexAttention
]
```

### Feature-Gated Selection

Triton is the **only** backend that supports ALL of these simultaneously:
- Sink tokens (`supports_sink = True`)
- ALiBi slopes
- Softcap (`apply_softcap` JIT function)
- Multimodal prefix (bidirectional attention ranges)
- FP8 KV cache
- Head sizes from 32 to 576 (including MLA)
- Sliding window with two-level tile pruning
- Full CUDA graph support (`ALWAYS`)

Flash Attention 3 cannot do ALiBi. FlashInfer requires TRTLLM extension for sinks on pre-Blackwell.
Triton is the only portable option.

---

## 8. Sliding Window, ALiBi, Sinks, Multimodal Prefix

### Sliding Window — Two-Level Optimization

**Level 1: Tile pruning** (`triton_unified_attention.py:206-227`)

```python
if SLIDING_WINDOW > 0 and not USE_MM_PREFIX:
    first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
    last_allowed_key = context_len + qpos_hi
    tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
    tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
```

Entire tiles outside the window are **skipped** — no K/V loads, no computation.

**Level 2: Per-element mask** (`triton_unified_attention.py:288-289`)

```python
if SLIDING_WINDOW > 0:
    seq_mask = seq_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
```

Within boundary tiles, individual elements outside the window are masked to `-inf`.

### ALiBi (Attention with Linear Biases)

```python
# triton_unified_attention.py:172-175 — Load per-head slope
if USE_ALIBI_SLOPES:
    alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0)

# triton_unified_attention.py:327-328 — Apply after softcap, before softmax
if USE_ALIBI_SLOPES:
    S += alibi_slope[:, None] * (seq_offset - context_len)
```

Each query head has a different slope. The bias is `slope × (key_position - context_start)`,
linearly increasing with distance. This is additive to the QK scores.

### Sink Tokens (GPT-OSS / StreamingLLM)

```python
# triton_unified_attention.py:153-160 — Initialize running max with pre-computed sink values
if not USE_SINKS:
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
else:
    M = tl.load(sink_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")).to(tl.float32)
```

Sinks are pre-computed attention values for "anchor tokens" (typically the first few tokens).
By initializing M with these values, the online softmax naturally incorporates them into every
subsequent tile's computation without a separate pass.

In the 3D kernel, sinks only apply to **segment 0** (the segment containing the anchor tokens):

```python
# triton_unified_attention.py:500-510
if USE_SINKS:
    if segm_idx == 0:
        M = tl.load(sink_ptr + query_offset_1, ...)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
```

### Multimodal Prefix (Bidirectional Attention for Images)

```python
# triton_unified_attention.py:291-313
if USE_MM_PREFIX:
    for i in range(MAX_MM_RANGES):
        range_start = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2)
        range_end = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1)
        is_valid = range_start < range_end
        q_in_range = (query_abs_pos >= range_start) & (query_abs_pos <= range_end) & is_valid
        k_in_range = (seq_offset[None, :] >= range_start) & (seq_offset[None, :] <= range_end) & is_valid
        seq_mask |= q_in_range & k_in_range
```

The mask logic follows FlexAttention's convention:
```
final_mask = (causal AND sliding_window) OR multimodal_prefix
```

Image tokens within a range can attend to each other **bidirectionally**, while text tokens
remain causal. Multiple ranges are supported (e.g., multiple images in one prompt).

---

## 9. Hardware-Specific Tuning: H100 vs MI300X vs A100

### TILE_SIZE (the KV iteration block size)

| Scenario | H100 (SM 9.0) | MI300X (CDNA3) | A100 (SM 8.0) |
|----------|---------------|----------------|---------------|
| Prefill, bf16 | 32 | 32 | 32 |
| Decode, bf16 | 16 | 16 | 16 |
| Decode, fp8 | 32 | 32 | 32 |
| Gemma3 (any) | 32 | 32 | 32 |

### Reshape-and-Cache Kernel

| Parameter | NVIDIA (SM ≥ 9.0) | NVIDIA (SM < 9.0) | AMD ROCm | Intel XPU |
|-----------|-------------------|-------------------|----------|-----------|
| num_stages | 10 | 10 | 4 | 4 |
| num_warps | 16 | 16 | 8 | 8 |
| max TILE_SIZE | 2048 | 512 | 2048 | 2048 |

### Decode Attention (Legacy Kernel)

| Parameter | NVIDIA | AMD (is_hip) |
|-----------|--------|-------------|
| BLOCK (MHA) | 64 | 8 |
| BLOCK (GQA) | 32 | 16 (if head≥576) |
| num_warps (GQA) | 2 | 1 |
| num_stages | 2 | 1 |
| Extra kwargs | — | `waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2` |

The AMD-specific kwargs control:
- `waves_per_eu=1`: Limit occupancy to 1 wavefront per execution unit (avoids register spilling)
- `matrix_instr_nonkdim=16`: Use 16-wide matrix instructions (matching CDNA3's matrix core shape)
- `kpack=2`: Pack 2 FP16 values per register load (doubles memory throughput)

---

## 10. GPU Execution Map: Where Every Operation Runs

### Inside the 2D Kernel: One Tile Iteration

```
            ┌─────────────────────────────────────────────────────────────┐
            │  Thread Block (e.g., 128 threads = 4 warps on NVIDIA)      │
            │                                                             │
 HBM ───→  │  1. Load K tile: [HEAD_SIZE × TILE_SIZE] from key_cache     │
            │     physical_block_idx = block_table[seq_idx][tile_j]       │
            │     ↓ streams through L2 cache                              │
  SMEM ←──  │     Triton auto-tiles into shared memory                    │
            │                                                             │
 HBM ───→  │  2. Load V tile: [TILE_SIZE × HEAD_SIZE] from value_cache   │
  SMEM ←──  │     Same block table indirection, same SMEM path            │
            │                                                             │
  Tensor    │  3. S += scale * tl.dot(Q, K)                               │
  Cores ──  │     Q: [BLOCK_M × HEAD_SIZE_PADDED] (in registers/SMEM)    │
            │     K: [HEAD_SIZE_PADDED × TILE_SIZE] (in SMEM)             │
            │     → HMMA instructions on tensor cores                      │
            │     → S: [BLOCK_M × TILE_SIZE] in registers (fp32)          │
            │                                                             │
  CUDA      │  4. Softcap: S = x * tanh(S/x)                             │
  Cores ──  │     exp() and division on CUDA/FP32 ALUs                    │
            │                                                             │
  CUDA      │  5. Causal mask + sliding window                            │
  Cores ──  │     S = where(mask, S, -inf)                                │
            │     Comparison and bitwise ops on INT/CUDA cores             │
            │                                                             │
  CUDA      │  6. Online softmax update:                                  │
  Cores ──  │     m_j = max(M, max(S))     ← register reduction          │
            │     P = exp(S - m_j)          ← exp on CUDA cores           │
            │     alpha = exp(M - m_j)      ← rescale factor              │
            │     acc = acc * alpha + dot(P, V)                            │
            │     ↑                    ↑                                   │
  Tensor    │     register multiply    tensor core HMMA                    │
  Cores ──  │                                                             │
            │     L = L * alpha + sum(P)    ← accumulate normalizer       │
            │     M = m_j                   ← update running max          │
            │                                                             │
 HBM ←───  │  7. Store: acc / L → output_ptr (after all tiles done)      │
            └─────────────────────────────────────────────────────────────┘
```

### Memory Hierarchy Usage

| Data | Size (Llama-3.1-8B, 1 head) | Location |
|------|---------------------------|----------|
| Q tile (per block) | BLOCK_M × 128 × 2B = 4 KB | Registers + SMEM |
| K tile (per iteration) | 128 × TILE_SIZE × 2B = 4 KB | SMEM (loaded from HBM) |
| V tile (per iteration) | TILE_SIZE × 128 × 2B = 4 KB | SMEM (loaded from HBM) |
| S scores (per iteration) | BLOCK_M × TILE_SIZE × 4B = 1 KB | Registers (fp32) |
| Accumulator (running) | BLOCK_M × 128 × 4B = 8 KB | Registers (fp32) |
| M, L (running max/sum) | BLOCK_M × 4B = 64 B | Registers |
| Block table (per seq) | max_blocks × 4B = ~4 KB | HBM → L2 cache |
| KV cache (total model) | 2 × layers × blocks × block_size × heads × dim | HBM |

---

## 11. Benchmarks: 100.7% of FlashAttention 3 on H100

The talk presents end-to-end results for **Llama-3.1-8B, batch_size=1, input_length=500**:

```
H100 SXM (end-to-end latency, lower is better):
  Flash Attention 3 (70K LoC CUDA):  baseline
  Triton Unified Attention:          100.7% of FA3  ← FASTER
  (with static launch grid + full CUDA graphs)

MI300X (end-to-end latency):
  Previous SOTA:                     baseline
  Triton Unified Attention:          5.8x speedup in 6 months
  (same Triton source as H100!)
```

**Why Triton can beat FA3**: Triton's compiler (as of 3.2+) generates highly optimized PTX
with aggressive software pipelining and loop unrolling. The talk's appendix shows: *"Triton
uses twice as many specialized instructions and generates 10x larger binaries (→ software
pipelining, loop unrolling, etc.)"*

The 100.7% result is for a specific scenario (batch_size=1, long decode). FA3 is likely faster
for prefill-heavy workloads where its hand-tuned warp specialization shines. The point is not
that Triton is universally faster — it's that it's **competitive** from a single portable source.

---

## 12. Batch Invariance

The talk mentions *"Supports Batch Invariance"* as a feature of the Triton backend.

Batch invariance is a major vLLM feature (`VLLM_BATCH_INVARIANT=1`) that guarantees **identical
output regardless of batch size or request ordering**. This is critical for:
- Reproducible evaluation benchmarks
- Debugging numerical issues
- Regulatory compliance (auditable AI outputs)

How it works across the stack:

```python
# vllm/model_executor/layers/batch_invariant.py:926 — Monkey-patches PyTorch ops
def enable_batch_invariant_mode():
    # Replaces: mm, bmm, matmul, addmm, linear, softmax, _log_softmax, mean, rms_norm
    # with deterministic equivalents that produce identical results regardless of batch order

# vllm/v1/attention/backends/flash_attn.py:364 — Forces single split for FA
if vllm_is_batch_invariant():
    num_splits = 1  # Prevents non-deterministic parallel reductions

# vllm/model_executor/layers/fused_moe/fused_moe.py:849 — Deterministic expert selection
if vllm_is_batch_invariant():
    sorted = True  # Ensures topk is deterministic
```

The Triton attention backend inherently supports batch invariance because:
1. The binary search in `find_seq_idx` is deterministic
2. The online softmax within each thread block is sequential (no non-deterministic reductions)
3. The 3D kernel's `reduce_segments` is a deterministic summation over a fixed number of segments

---

## 13. Helion: The Future DSL

The talk previews **Helion** — a new DSL from PyTorch described as *"Tiled PyTorch"* or
*"higher-level Triton"*. A draft paged attention kernel was implemented in Helion and
submitted as vLLM draft PR #27293.

**Current status**: Helion is NOT in the vLLM codebase. No references to "helion" or PR #27293
exist in the repository. This is still experimental research — a preview of where the ecosystem
is heading.

The blog post at `pytorch.org/blog/portable-paged-attention-in-helion/` describes the approach:
write attention in a PyTorch-like syntax (no explicit tile management), and Helion compiles it
to optimized Triton which then compiles to PTX/GCN. This would be an additional abstraction layer:

```
Helion (PyTorch-like)
  ↓ compiles to
Triton (tiled programs)
  ↓ compiles to
PTX / GCN / SPIR-V (GPU ISA)
```

---

## 14. Future Use Cases and System Design

### 1. Single-Source Multi-GPU Deployment

The talk's core message is that ONE kernel source can serve:
- **Cloud NVIDIA** (H100, B200) — used by most vLLM deployments
- **On-prem AMD** (MI300X, MI325X) — increasingly adopted for cost efficiency
- **Edge Intel** (Max GPU, Gaudi) — emerging market

System design implication: Stop writing separate kernels per vendor. Write Triton, tune via
heuristics, and ship.

### 2. GPT-OSS and Sink Tokens at Scale

The talk highlights GPT-OSS (GPT Open-Source Scale) models that use attention sinks for
efficient streaming inference. The Triton backend is the primary path for these models,
especially on pre-Hopper GPUs where FA3 isn't available.

### 3. The 200μs Problem: Kernel Launch Overhead

The talk identifies kernel launch overhead (~200μs for Triton kernels) as the main bottleneck
for small/medium decode batches. The solution — persistent kernels with static launch grids
and full CUDA/HIP graphs — eliminates this overhead entirely.

System design implication: For latency-sensitive applications (real-time chat, voice agents),
always enable CUDA graphs (`--enforce-eager=False`) to amortize launch overhead.

### 4. Autotuning as Competitive Advantage

The talk's appendix shows that configuration selection matters enormously — 450 Triton configs
show wide performance variation. The current heuristics are "good enough" but leave performance
on the table.

Future: Offline autotuning per GPU SKU, stored as a config database. When a new GPU launches,
run the autotuner once, and every vLLM deployment on that GPU benefits. This is much cheaper
than writing 528 hand-tuned CUDA kernel variants.

### 5. Performance Portability → Vendor Independence

The deepest implication: if Triton kernels achieve parity with vendor-specific libraries, then
**GPU vendor lock-in for inference disappears**. You serve the same model with the same kernel
on whichever GPU offers the best price/performance at that moment.

This is already happening for attention (100.7% of FA3 on H100, SOTA on MI300X). The next
frontier is FusedMoE, which currently uses platform-specific kernels for AMD (AITER) and
NVIDIA (Triton-based, but with vendor-specific quantization backends like Marlin).

---

*Every claim from the vLLM Office Hours #43 presentation, traced to exact source code with
file paths and line numbers, mapped to GPU silicon, and projected into future system design.*
