# Triton vs CUDA Graphs vs PyTorch vs Omni — What Actually Runs, Where, and Why

**Context**: You're deploying on a B200 GPU, serving a GLM-5 MoE model with NVFP4
quantization. This document traces every technology through the actual code, showing
exactly what runs at each layer of the stack.

---

## The 30-Second Version

| Technology | What It Is | When It Fires | Scope |
|-----------|-----------|---------------|-------|
| **Triton** | GPU kernel language (like CUDA but portable) | Every attention computation | One kernel at a time |
| **CUDA Graphs** | Recording + replaying entire GPU workflows | Every decode step | Whole forward pass |
| **torch.compile** | Fusing multiple ops into fewer kernel launches | Specific fusion patterns | Groups of 2-3 ops |
| **Omni** | Multi-model pipelines (Thinker→Talker→Vocoder) | Multi-stage serving | Across models/GPUs |

They are **not alternatives**. They are **layers of a stack**:

```
┌────────────────────────────────────────────────────┐
│  Omni (orchestrates multiple vLLM instances)       │  ← across GPUs/models
├────────────────────────────────────────────────────┤
│  CUDA Graphs (replays the whole forward pass)      │  ← one forward pass
├────────────────────────────────────────────────────┤
│  torch.compile (fuses adjacent operations)         │  ← groups of ops
├────────────────────────────────────────────────────┤
│  Triton kernels (attention)                        │  ← one GPU kernel
│  CUTLASS kernels (NVFP4 GEMM)                     │  ← one GPU kernel
│  CUDA kernels (FP4 quant, RMSNorm, etc.)          │  ← one GPU kernel
└────────────────────────────────────────────────────┘
     ▼ All execute on ▼
   B200 Tensor Cores + CUDA Cores + HBM
```

---

## Part 1: What Happens When a Token Arrives (GLM-5 + NVFP4 on B200)

Here is the **real execution path** for one decode step. Every file path, line number,
and hardware unit is from the actual codebase.

### Step 1: The Scheduler Picks Requests

```
File: vllm/v1/worker/gpu_model_runner.py:3034
Function: execute_model()

The scheduler outputs a batch: say 32 requests, each generating 1 token.
Total tokens this step: 32.
```

### Step 2: CUDA Graph Dispatch

```
File: vllm/v1/cudagraph_dispatcher.py:143-175
Function: CudagraphDispatcher.dispatch()

Input:  num_tokens=32, uniform_decode=True
Output: CUDAGraphMode.FULL, padded to nearest captured size (e.g., 32)

The dispatcher finds a pre-captured graph that matches batch_size=32.
```

### Step 3: Update Persistent Buffers (CPU → GPU Copy)

```
File: vllm/v1/worker/gpu_model_runner.py:484-561

These tensors have FIXED GPU ADDRESSES (frozen in the graph).
Their CONTENTS are overwritten every step:

  input_ids[0:32]      ← the 32 new token IDs
  positions[0:32]      ← position index for each token
  seq_lens[0:32]       ← current sequence length per request
  block_table[0:32, :] ← which KV cache blocks each request uses
  slot_mapping[0:32]   ← where to write new K/V values
```

### Step 4: Graph Replay (The Whole Forward Pass)

```
File: vllm/compilation/cuda_graph.py:305
Function: CUDAGraphWrapper.__call__()

  entry.cudagraph.replay()   ← ONE CUDA API call replays everything below
```

Everything inside the box below executes from this single `replay()`:

```
┌─────────────────── CUDA Graph Replay ───────────────────┐
│                                                          │
│  ┌─ Embedding Lookup ─────────────────────────────────┐  │
│  │  input_ids → embed_tokens → hidden [32, 4096] BF16 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─ For each of ~60 Transformer Layers ───────────────┐  │
│  │                                                     │  │
│  │  ① RMSNorm (CUDA cores)                            │  │
│  │                                                     │  │
│  │  ② QKV Linear — NVFP4 (see Step 5 below)          │  │
│  │     → scaled_fp4_quant    (CUDA cores)              │  │
│  │     → cutlass_scaled_fp4_mm (TENSOR CORES)          │  │
│  │                                                     │  │
│  │  ③ Triton Attention (see Step 6 below)             │  │
│  │     → KV cache write      (CUDA cores via Triton)   │  │
│  │     → Q·K scores          (TENSOR CORES via Triton) │  │
│  │     → softmax             (CUDA cores via Triton)   │  │
│  │     → P·V output          (TENSOR CORES via Triton) │  │
│  │                                                     │  │
│  │  ④ Output Projection — NVFP4                       │  │
│  │     → scaled_fp4_quant    (CUDA cores)              │  │
│  │     → cutlass_scaled_fp4_mm (TENSOR CORES)          │  │
│  │                                                     │  │
│  │  ⑤ RMSNorm (CUDA cores)                            │  │
│  │                                                     │  │
│  │  ⑥ MoE Layer (see Step 7 below)                    │  │
│  │     GLM-5 uses SharedFusedMoE:                      │  │
│  │     → Router (sigmoid + grouped top-k)              │  │
│  │     → Shared expert MLP (always runs)               │  │
│  │     → Top-K routed expert GEMMs (NVFP4)            │  │
│  │     → Combine: routed * scale + shared              │  │
│  │                                                     │  │
│  │  ⑦ Residual connections                            │  │
│  │                                                     │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌─ Final RMSNorm + LM Head ──────────────────────────┐  │
│  │  → logits [32, vocab_size]                          │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Step 5: NVFP4 Linear — What the Code Actually Does

Every linear layer in GLM-5 with NVFP4 runs this two-step sequence:

```python
# File: vllm/model_executor/layers/quantization/modelopt.py:1353-1403

def apply(self, layer, x, bias):
    # ── STEP A: Quantize activations BF16 → FP4 (CUDA cores) ──
    # x is [32, 4096] BF16
    x_fp4, x_blockscale = scaled_fp4_quant(x, layer.input_scale_inv)
    # x_fp4 is [32, 2048] uint8        (two FP4 values per byte)
    # x_blockscale is [32, 256] FP8     (one scale per 16 elements)

    # ── STEP B: FP4 × FP4 GEMM (5th-gen Tensor Cores) ──
    out = cutlass_scaled_fp4_mm(
        x_fp4,              # [32, 2048] uint8   — quantized activations
        layer.weight,       # [N, 2048] uint8    — quantized weights (loaded at init)
        x_blockscale,       # [32, 256] FP8      — activation block scales
        layer.weight_scale, # [N, 256] FP8       — weight block scales (swizzled)
        layer.alpha,        # scalar FP32         — input_scale × weight_scale_2
        output_dtype,       # BF16
    )
    # out is [32, N] BF16

    return out + bias if bias else out
```

What happens in the CUDA kernel for Step A:

```
File: csrc/quantization/fp4/nvfp4_quant_kernels.cu:49-107

For every group of 16 BF16 values:
  1. Find absolute max across 16 values          (CUDA cores, warp shuffle)
  2. Compute scale = global_scale × (max / 6.0)  (CUDA cores, FP32 arithmetic)
  3. Cast scale to FP8-E4M3                       (CUDA cores)
  4. Divide each value by scale                   (CUDA cores, FP32)
  5. Round to nearest E2M1 (4-bit float)          (PTX: cvt.rn.satfinite.e2m1x2.f32)
  6. Pack two E2M1 values into one uint8 byte     (CUDA cores, bitwise OR)
  7. Write scale to swizzled SF layout            (CUDA cores → HBM)
```

What happens in the CUTLASS kernel for Step B:

```
File: csrc/quantization/fp4/nvfp4_scaled_mm_kernels.cu:66-126

CUTLASS template instantiation for B200 (SM100):
  ElementA     = nv_float4_t<float_e2m1_t>    (FP4 activations)
  ElementB     = nv_float4_t<float_e2m1_t>    (FP4 weights)
  Accumulator  = float                          (FP32)
  OperatorClass = OpClassBlockScaledTensorOp    (block-scaled tensor core ops)
  ArchTag       = Sm100

Tile selection based on M=32 (our batch size):
  M ≤ 16:   TileShape<128,128,256>, Cluster<1,1,1>     ← not us
  M ≤ 256:  TileShape<256,128,256>, Cluster<2,1,1>     ← THIS ONE
  M > 256:  TileShape<256,256,256>, Cluster<2,1,1>     ← prefill

The tensor core executes:
  1. Load FP4 tile from shared memory
  2. Load FP8 block scales from shared memory (swizzled layout)
  3. Hardware dequantizes: FP4 × FP8_scale → higher precision
  4. Matrix multiply-accumulate in FP32
  5. Epilogue: multiply by alpha, convert FP32 → BF16, write to HBM
```

### Step 6: Triton Attention — What the Code Actually Does

For a 32-request decode batch (1 token per request), the Triton backend runs:

```python
# File: vllm/v1/attention/backends/triton_attn.py:372-497

def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
    # ── Write new K/V to the paged cache ──
    triton_reshape_and_cache_flash(key, value, kv_cache, slot_mapping)

    # ── Choose 2D or 3D kernel ──
    # 32 sequences, num_kv_heads=8 → threshold = 128/8 = 16
    # 32 > 16 → use 2D kernel (no segment reduction needed)

    unified_attention(query, output, ...)
```

```python
# File: vllm/attention/ops/triton_unified_attention.py:932-1065

# 2D kernel launch:
grid = (total_q_blocks, num_kv_heads)   # e.g., (32, 8) = 256 thread blocks

kernel_unified_attention_2d[grid](
    query,                    # [32, num_heads, head_dim] in registers
    kv_cache,                 # paged blocks in HBM
    block_table,              # maps (seq, block_idx) → physical page
    query_start_len_ptr,      # cumulative lengths for binary search
    output,                   # [32, num_heads, head_dim] written to HBM
    ...
)
```

Inside the Triton kernel, for each thread block:

```python
# File: vllm/attention/ops/triton_unified_attention.py:57-392

# ── Q-Block: pack multiple query heads into one tile ──
# GQA ratio = 8 (8 query heads per KV head) for GLM-5
# BLOCK_M = 16, so we pack 8 heads × 2 tokens (or 16 heads × 1 token)
# K and V are loaded ONCE and shared across all packed query heads

for each KV tile (TILE_SIZE tokens from the sequence):
    K = tl.load(kv_cache[physical_block, kv_head, :TILE_SIZE, :head_dim])  # HBM → SMEM
    V = tl.load(kv_cache[physical_block, kv_head, :TILE_SIZE, :head_dim])  # HBM → SMEM

    S = scale * tl.dot(Q, K)       # TENSOR CORES: [BLOCK_M, TILE_SIZE]
    # Apply causal mask, sliding window, ALiBi, sinks, softcap as needed
    m_new = tl.max(S, axis=1)      # CUDA CORES: running max
    P = tl.exp(S - m_new)          # CUDA CORES: softmax numerator
    acc = acc * correction + tl.dot(P, V)  # TENSOR CORES: weighted sum
```

**Hardware unit per operation:**

| Operation | Hardware | Triton Instruction |
|-----------|----------|-------------------|
| Load K/V from HBM | Memory controller | `tl.load` |
| Q · K^T (attention scores) | **Tensor Cores** | `tl.dot(Q, K)` |
| exp(S - max) | CUDA Cores (SFU) | `tl.exp(...)` |
| max reduction | CUDA Cores | `tl.max(S, axis=1)` |
| P · V (weighted sum) | **Tensor Cores** | `tl.dot(P, V)` |
| Causal masking | CUDA Cores | `tl.where(mask, S, -inf)` |
| Store output | Memory controller | `tl.store` |

### Step 7: GLM-5 MoE Layer — What the Code Actually Does

GLM-5 uses `SharedFusedMoE` — a fused operation that runs both shared experts and
routed experts, overlapping shared expert computation with all-to-all communication.

```python
# File: vllm/model_executor/models/glm4_moe.py:202-226

def forward(self, hidden_states):   # hidden_states: [32, 4096] BF16
    # ── Router: sigmoid + grouped top-k ──
    router_logits = self.gate(hidden_states.float())  # [32, num_experts] FP32
    # gate has e_score_correction_bias (DeepSeek-V3 style routing)

    # ── SharedFusedMoE: shared + routed in one call ──
    shared_output, routed_output = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )

    # ── Combine ──
    return routed_output * self.routed_scaling_factor + shared_output
```

Inside `SharedFusedMoE` (`vllm/model_executor/layers/fused_moe/shared_fused_moe.py:14-97`):

```
When expert parallelism is active (all-to-all communication):
  ┌───────────────────────────────────────────────┐
  │  Shared expert MLP     │  All-to-all dispatch │  ← RUN IN PARALLEL
  │  (on local GPU)        │  (send tokens to     │
  │  gate_up → SiLU*Mul    │   remote expert GPUs)│
  │  → down_proj           │                      │
  └───────────────────────────────────────────────┘
                            │
                            ▼
                  Routed expert GEMMs
                  (on assigned GPUs)
                            │
                            ▼
                  All-to-all gather
                  (collect results back)
```

Each expert MLP with NVFP4 runs the same `scaled_fp4_quant → cutlass_scaled_fp4_mm`
path as Step 5, but through the MoE group GEMM kernel:

```
File: csrc/quantization/fp4/nvfp4_blockwise_moe_kernel.cu

CUTLASS group GEMM: one sub-problem per expert, each with:
  - FP4 activations (tokens routed to this expert)
  - FP4 expert weights
  - Per-expert block scales and global scales
```

---

## Part 2: torch.compile — The Fusion Layer

`torch.compile` is **not** a separate execution mode. It's an optimization pass that fuses
adjacent operations to reduce kernel launch overhead and memory traffic.

### What It Fuses in GLM-5 + NVFP4

Three fusion patterns fire for NVFP4 models:

**Fusion 1: SiLU + Mul + FP4 Quantization**

```
Before (3 kernel launches, 2 HBM round-trips):
  gate, up = gate_up_proj(x)        # NVFP4 GEMM → HBM
  x = SiLU(gate) * up               # Read from HBM, CUDA cores, write to HBM
  x_fp4, scale = scaled_fp4_quant(x) # Read from HBM, CUDA cores, write to HBM

After (1 kernel launch, 0 extra round-trips):
  gate, up = gate_up_proj(x)                    # NVFP4 GEMM → HBM
  x_fp4, scale = silu_and_mul_nvfp4_quant(gate, up, global_scale)  # SINGLE kernel
```

```
File: vllm/compilation/activation_quant_fusion.py:121-168
Kernel: csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu:66-113
```

**Fusion 2: AllReduce + RMSNorm + FP4 Quantization**

```
Before (3 kernel launches):
  x = all_reduce(x)           # NCCL collective
  x = rmsnorm(x)              # CUDA kernel
  x_fp4 = scaled_fp4_quant(x) # CUDA kernel

After (1 fused call):
  x_fp4 = allreduce_rmsnorm_fp4quant(x)  # Single FlashInfer fused kernel
```

```
File: vllm/compilation/collective_fusion.py:891-985
```

**Fusion 3: Generic Operator Fusion via torch.compile**

`torch.compile` also fuses smaller operations like element-wise adds, residual connections,
and activation functions. The GLM-5 model class is decorated with `@support_torch_compile`:

```
File: vllm/model_executor/models/glm4_moe.py:409
Class: Glm4MoeModel (the backbone is compiled)
```

### The Key Insight

torch.compile operates **between** kernel boundaries. It doesn't change what Triton
or CUTLASS do internally — it reduces the number of times you launch kernels and the
number of times intermediate results bounce through HBM.

```
Without torch.compile:  ~15 kernel launches per layer
With torch.compile:     ~8 kernel launches per layer (fused patterns)
Speedup:                ~10-20% for decode (launch-overhead-bound)
```

---

## Part 3: CUDA Graphs — The Replay Layer

### What a CUDA Graph Is

A CUDA graph is a recording of GPU kernel launches. Instead of the CPU issuing each
kernel launch individually (each costing ~5-10μs of CPU overhead), you record them once,
then "replay" the entire sequence with a **single CPU call** (~1μs).

### How It Works with Variable Batch Sizes

**Problem**: Batch size changes every step. CUDA graphs freeze launch parameters.

**Solution** (used by Triton attention):

```python
# File: vllm/v1/attention/backends/triton_attn.py:189-197

def build_for_cudagraph_capture(self, common_attn_metadata):
    attn_metadata = self.build(0, common_attn_metadata)
    attn_metadata.seq_lens.fill_(1)    # ← Make capture fast (minimal tiles)
    return attn_metadata
```

During capture: Launch with over-provisioned grid, set `seq_lens=1` so kernels exit fast.

During replay: The **same grid** launches, but now the persistent buffers contain real data:

```python
# File: vllm/attention/ops/triton_unified_attention.py:124

# Each thread block checks if it has real work:
if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
    return   # Empty block — exits in ~10 GPU cycles
```

The grid is over-sized. Thread blocks with no work exit immediately. Thread blocks with
work read the updated `seq_lens`, `block_table`, and `slot_mapping` buffers.

### CUDA Graph Support Levels

```python
# File: vllm/v1/attention/backends/utils.py:296-310

AttentionCGSupport.ALWAYS = 3              # Mixed prefill + decode in graph
AttentionCGSupport.UNIFORM_BATCH = 2       # Same query length only
AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE = 1  # Decode only (1 token/req)
AttentionCGSupport.NEVER = 0
```

| Backend | CG Support Level | Implication |
|---------|-----------------|-------------|
| **Triton** | **ALWAYS (3)** | Mixed prefill+decode in a single graph |
| FlashAttention 3 | ALWAYS (3) | Same capability |
| FlashInfer | UNIFORM_SINGLE_TOKEN (1-2) | Separate graphs for prefill vs decode |

### What Gets Frozen vs What Changes

| Frozen in Graph | Changes Every Step |
|----------------|-------------------|
| Tensor GPU **addresses** (pointers) | Tensor **contents** (overwritten in-place) |
| Kernel launch grid dimensions | `seq_lens`, `block_table` values |
| Triton `tl.constexpr` params (HEAD_SIZE, TILE_SIZE) | `query_start_len_ptr` contents |
| CUTLASS tile configuration | `slot_mapping` contents |
| Number of thread blocks | Which blocks have real work vs early-exit |

---

## Part 4: Triton — The Attention Kernel Layer

### What Triton Is (and Isn't)

Triton is a **GPU kernel programming language**. It compiles to the same PTX/SASS that
CUDA produces, but with a higher-level "tile-based" programming model.

**Triton is NOT**: a framework, a serving engine, or an alternative to vLLM.
**Triton IS**: the language that vLLM's attention kernel is written in.

```
CUDA:   You manage threads, warps, shared memory, registers manually.
Triton: You think in tiles (blocks of data), and the compiler handles the rest.
```

### Why vLLM Uses Triton for Attention (Not CUDA)

| Property | Triton Attention | FlashAttention 3 (CUDA) |
|----------|-----------------|------------------------|
| Source lines | ~800 | ~70,000 |
| Portability | NVIDIA + AMD + Intel | NVIDIA only |
| H100 perf | **100.7% of FA3** | Baseline |
| MI300X perf | **5.8x of previous SOTA** | Not available |
| B200 support | Yes (same source) | Separate FA3 build |
| Feature set | Sinks, ALiBi, softcap, sliding window, multimodal prefix | Subset |
| CUDA graph | ALWAYS (level 3) | ALWAYS (level 3) |

### The Three Triton Kernels

```
File: vllm/attention/ops/triton_unified_attention.py

kernel_unified_attention_2d   (line 57)
  Grid: (total_q_blocks, num_kv_heads)
  When: Prefill, or decode with batch > threshold
  Each thread block: processes BLOCK_M query rows × full sequence length

kernel_unified_attention_3d   (line 394)
  Grid: (total_q_blocks, num_kv_heads, 16)
  When: Small-batch decode (few sequences, long contexts)
  Each thread block: processes BLOCK_M query rows × 1/16th of sequence

reduce_segments               (line 717)
  Grid: (num_tokens, num_query_heads)
  When: Only after 3D kernel
  Each thread: merges 16 partial softmax results (online softmax, exact)
```

### Why Two Kernels for Decode?

**Problem**: Batch=1 decode with 8 KV-heads → only 8 thread blocks → 8 of 132 SMs busy (6%).

**Solution**: Split the KV sequence into 16 segments:
```
8 × 16 = 128 thread blocks → 128 of 132 SMs busy (97%)
```

Each segment computes partial `(output, max, exp_sum)`. The `reduce_segments` kernel
merges them using the **online softmax** identity:

```python
# File: vllm/attention/ops/triton_unified_attention.py:717-805

# For each of 16 segments:
#   output_i, max_i, expsum_i = partial attention over 1/16th of KV
#
# Merge:
#   global_max = max(max_0, max_1, ..., max_15)
#   For each segment i:
#     correction = exp(max_i - global_max)
#     global_expsum += expsum_i * correction
#     global_output += output_i * correction
#   final = global_output / global_expsum
```

This is mathematically exact — not an approximation.

---

## Part 5: Omni — The Multi-Model Orchestration Layer

### What's Actually Implemented vs What's Planned

**Implemented in this repo** (building blocks):

| Component | Status | File |
|-----------|--------|------|
| Thinker-only serving for Qwen2.5-Omni, Qwen3-Omni | Working | `vllm/model_executor/models/qwen2_5_omni_thinker.py` |
| Encoder disaggregation (EC transfer) | Working | `vllm/distributed/ec_transfer/` |
| Prefill/Decode disaggregation (KV transfer) | Working | `vllm/distributed/kv_transfer/` |
| 3-instance E→P→D pipeline | Working (example) | `examples/online_serving/disaggregated_encoder/` |

**NOT implemented in this repo** (described in paper/docs only):

| Component | Status | What It Would Be |
|-----------|--------|-----------------|
| `StageGraph` abstraction | Not in codebase | Arbitrary N-stage pipeline DAG |
| `OmniConnector` | Not in codebase | Shared memory / RDMA inter-stage transport |
| `vllm-omni serve --stage-config` | Not in codebase | CLI for multi-stage deployment |
| Talker model (speech codec LLM) | Explicitly skipped | `skip_prefixes=["talker."]` |
| Code2Wav / token2wav (vocoder) | Explicitly skipped | `skip_prefixes=["code2wav."]` |
| Async chunk overlap (Thinker→Talker→DiT) | Not in codebase | Streaming inter-stage pipeline |

The Omni models load with a comment at the top:

```python
# File: vllm/model_executor/models/qwen2_5_omni_thinker.py:1175
skip_prefixes = ["talker.", "token2wav."]
```

This means: vLLM serves the **thinking/reasoning** part only. For the full
Thinker→Talker→Vocoder pipeline described in the Office Hours, you need the
separate `vllm-omni` package (per the arXiv paper 2602.02204) or the future merge.

### What You CAN Deploy Today: The E→P→D Pipeline

```bash
# Instance 1: Vision/Audio Encoder (GPU 2)
vllm serve $MODEL \
  --ec-transfer-config '{"ec_connector":"ECExampleConnector","ec_role":"ec_producer"}'

# Instance 2: Prefill (GPU 2, reuses encoder GPU)
vllm serve $MODEL \
  --ec-transfer-config '{"ec_connector":"ECExampleConnector","ec_role":"ec_consumer"}' \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'

# Instance 3: Decode (GPU 3)
vllm serve $MODEL \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'

# Proxy: Routes requests through the pipeline
python disagg_epd_proxy.py --encoder-url ... --prefill-url ... --decode-url ...
```

This chains **two** connector types:
- EC transfer (Encoder → Prefill): shared filesystem (safetensors)
- KV transfer (Prefill → Decode): NIXL via RDMA (GPU-to-GPU)

---

## Part 6: GLM-5 Architecture — Complete Map

### Layer Structure

```python
# File: vllm/model_executor/models/glm4_moe.py:327-398, 409-634

Glm4MoeForCausalLM
  └── Glm4MoeModel (@support_torch_compile)
        ├── embed_tokens (VocabParallelEmbedding)
        ├── layers[0..first_k_dense_replace-1]     ← DENSE MLP layers
        │     ├── Glm4MoeAttention
        │     │     ├── QKVParallelLinear (NVFP4)
        │     │     ├── RotaryEmbedding (partial_rotary_factor=0.5)
        │     │     ├── Optional: q_norm, k_norm (RMSNorm per-head)
        │     │     ├── Attention (Triton/FA3/FlashInfer backend)
        │     │     └── RowParallelLinear (NVFP4)
        │     └── Glm4MoeMLP
        │           ├── MergedColumnParallelLinear gate_up (NVFP4)
        │           ├── SiluAndMul
        │           └── RowParallelLinear down_proj (NVFP4)
        │
        ├── layers[first_k_dense_replace..N]        ← MoE LAYERS
        │     ├── Glm4MoeAttention (same as above)
        │     └── Glm4MoE
        │           ├── gate (nn.Linear, float32, sigmoid routing)
        │           │     └── e_score_correction_bias (DeepSeek-V3 style)
        │           ├── shared_experts (Glm4MoeMLP — always active)
        │           └── experts (SharedFusedMoE)
        │                 ├── Routed: top-K experts via grouped top-k
        │                 ├── Shared: overlapped with all-to-all dispatch
        │                 └── EPLB: optional expert load balancing
        └── norm (RMSNorm)
```

### GLM-5 Specific Features

| Feature | Implementation | Detail |
|---------|---------------|--------|
| **Partial RoPE** | `partial_rotary_factor=0.5` | RoPE on first 50% of head_dim only |
| **QK-Norm** | Optional per-head RMSNorm on Q and K | Stabilizes large-scale training |
| **Sigmoid routing** | `scoring_func="sigmoid"` | Not softmax — DeepSeek-V3 style |
| **Grouped top-k** | `use_grouped_topk=True` | Experts divided into groups, top-k within selected groups |
| **Score correction bias** | `e_score_correction_bias` | Learned per-expert bias for routing balance |
| **Shared experts** | `Glm4MoeMLP` running in parallel | Always-active expert with intermediate_size × n_shared |
| **Hybrid dense/MoE** | `first_k_dense_replace` | Early layers are dense MLP, later are MoE |
| **MTP speculative** | `Glm4MoeMTP` | Full decoder layers as MTP prediction heads |

### MTP Speculative Decoding

GLM-5 supports Multi-Token Prediction for speculative decoding:

```python
# File: vllm/model_executor/models/glm4_moe_mtp.py:74-121

class Glm4MoeMultiTokenPredictorLayer(nn.Module):
    def forward(self, input_ids, positions, previous_hidden_states, inputs_embeds):
        # 1. Normalize embeddings and previous hidden states
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        # 2. Concatenate and project: [embed || hidden] → hidden_size
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        # 3. Run through a FULL transformer decoder layer (attention + MoE)
        hidden_states, residual = self.mtp_block(positions, hidden_states, residual=None)
        return residual + hidden_states
```

Each MTP layer is a **complete** `Glm4MoeDecoderLayer` — it has its own attention and MoE.
The MTP layers are indexed starting after the main model's layers (`config.num_hidden_layers`).

MTP layers cycle: `spec_step_idx % num_mtp_layers` — so if 2 MTP layers predict 4 tokens
ahead, they alternate: layer 0 → layer 1 → layer 0 → layer 1.

---

## Part 7: The Backend Selection Tree (B200 Specifically)

When vLLM starts on a B200 GPU, this is the exact priority order:

```python
# File: vllm/platforms/cuda.py:69-75

# For B200 (device_capability.major == 10):
priority = [
    FLASHINFER,      # 1st choice
    FLASH_ATTN,      # 2nd choice (FA3)
    TRITON_ATTN,     # 3rd choice
    FLEX_ATTENTION,  # 4th choice
]
```

Each backend's `validate_configuration()` is called. If FlashInfer import fails or doesn't
support the model config (e.g., ALiBi, sinks, multimodal prefix), it falls back to the next.

**To force Triton on B200:**
```bash
VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm serve model
```

**Triton supports every GPU unconditionally:**
```python
# File: vllm/v1/attention/backends/triton_attn.py:313
@classmethod
def supports_compute_capability(cls, capability):
    return True   # Works on SM100 (B200), SM90 (H100), MI300X, anything
```

For NVFP4 GEMMs, the backend selection is separate:

```python
# File: vllm/model_executor/layers/quantization/modelopt.py:1199-1224

# On B200:
if has_flashinfer():
    backend = "flashinfer-cutlass"    # FlashInfer dispatcher + CUTLASS FP4 kernel
elif cutlass_fp4_supported():         # True for SM100+
    backend = "cutlass"               # Native CUTLASS FP4 GEMM
elif is_fp4_marlin_supported():
    backend = "marlin"                # Fallback for older GPUs
```

---

## Part 8: How They All Fit Together — One Decode Step Timeline

```
Time →

CPU:  [schedule]──[update buffers]──[graph.replay()]──────────────[sample tokens]
                        │                    │
GPU:                    │                    ▼
      ┌─────────────────┼─── CUDA Graph Replay ──────────────────────────────┐
      │                 ▼                                                     │
      │  [buffer updates land in HBM]                                        │
      │                                                                      │
      │  Layer 0:                                                            │
      │    [RMSNorm]  ← CUDA core kernel                                    │
      │    [FP4 quant] ← CUDA core kernel (fused with RMSNorm via compile)  │
      │    [FP4 GEMM]  ← CUTLASS on Tensor Cores                           │
      │    [Triton attention 2D] ← Tensor Cores (Q·K, P·V) + CUDA (softmax)│
      │    [FP4 quant] ← CUDA core kernel                                   │
      │    [FP4 GEMM]  ← CUTLASS on Tensor Cores                           │
      │    [RMSNorm]  ← CUDA core kernel                                    │
      │    [Router]   ← CUDA core kernel (sigmoid + top-k)                  │
      │    [SharedFusedMoE] ← CUTLASS group GEMM on Tensor Cores           │
      │    [Residual] ← CUDA core kernel                                    │
      │                                                                      │
      │  Layer 1..N: (same pattern)                                          │
      │                                                                      │
      │  [Final RMSNorm] [LM Head GEMM]                                     │
      └──────────────────────────────────────────────────────────────────────┘
                                              │
CPU:                                          ▼
      ────────────────────────────────[read logits]──[sample]──[send tokens]
```

Total kernel launches per decode step (approximate for GLM-5 with 60 MoE layers):
- Without torch.compile: ~900 kernel launches
- With torch.compile fusion: ~500 kernel launches
- With CUDA graph: **1 CPU call** replays all ~500 kernels

CPU overhead:
- Without CUDA graph: ~500 × 5μs = **2.5ms** of launch overhead
- With CUDA graph: ~**1μs** total (single replay call)

---

## Summary: When to Think About Each Technology

| You're Thinking About... | The Relevant Technology Is... | What to Do |
|--------------------------|------------------------------|------------|
| "Which GPU kernel computes attention?" | **Triton** | Set `VLLM_ATTENTION_BACKEND` or let auto-select |
| "How do I reduce CPU launch overhead?" | **CUDA Graphs** | Enabled by default; tune `cudagraph_capture_sizes` |
| "How do I fuse redundant memory traffic?" | **torch.compile** | Enabled by default; handles SiLU+Mul+FP4Quant etc. |
| "How do I serve Thinker+Talker+Vocoder?" | **Omni** (future) | Today: thinker-only. Future: `vllm-omni` package |
| "How do I shrink model memory 3x?" | **NVFP4** | Use NVFP4-quantized checkpoints on B200 |
| "How do I serve GLM-5 with speculative?" | **MTP** | `--speculative-model` with `num_nextn_predict_layers` |
| "How do I split prefill from decode?" | **KV Transfer** | `--kv-transfer-config` with NixlConnector |
| "How do I run on AMD/Intel?" | **Triton** | Same attention source compiles for ROCm/XPU |
