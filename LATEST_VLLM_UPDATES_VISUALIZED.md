# Latest vLLM Updates — What Actually Changed, Visualized

The five most impactful recent changes, each with before/after code, ASCII diagrams,
and exactly how they affect PyTorch, CUDA Graphs, and Triton.

---

## Table of Contents

1. [Async Scheduling Enabled by Default](#1-async-scheduling-enabled-by-default) — the biggest single change
2. [MoE Refactor: Monolithic → Modular](#2-moe-refactor) — architecture overhaul
3. [CUDA Graph Compound Modes](#3-cuda-graph-compound-modes) — FULL + PIECEWISE in one model
4. [torch.compile Fusion Patterns](#4-torchcompile-fusion-patterns) — 6 new AllReduce fusions
5. [Triton Attention Feature Completeness](#5-triton-attention-feature-completeness) — sinks, softcap, multimodal prefix

---

## 1. Async Scheduling Enabled by Default

**Commit**: `c2ff33c` — PR #27614 — **The single most impactful change**
**Impact**: 5-15% throughput improvement for every vLLM deployment

### The One-Line Change

```python
# vllm/config/scheduler.py:133

# BEFORE:
async_scheduling: bool = False

# AFTER:
async_scheduling: bool = Field(default=None)   # None = auto-enable
```

### What This Means

```
═══════════════════════════════════════════════════════════════════════
BEFORE: Synchronous Scheduling (GPU sits idle while CPU thinks)
═══════════════════════════════════════════════════════════════════════

Step N                                                Step N+1
                                                      (can't start until
                                                       N finishes)
CPU:
├─schedule()──┤                    ├─update()─┤ ├─schedule()──┤
│ Pick reqs   │                    │ Process  │ │ Pick reqs   │
│ Alloc KV    │                    │ outputs  │ │ Alloc KV    │
│ Build batch │                    │ Detok    │ │ Build batch │
└─────────────┘                    └──────────┘ └─────────────┘
               \                  /                            \
GPU:            \                /                              \
                 ├─forward()───┤ ├─sync!─┤                      ├─forward()──
                 │ 60 layers   │ │tolist │                      │
                 │ all GEMMs   │ │GPU→CPU│                      │
                 │ attention   │ │ BLOCK │                      │
                 └─────────────┘ └───────┘                      └────────────

                 ████████████████ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ████████████
                 GPU busy         GPU COMPLETELY IDLE             GPU busy
                                  while CPU schedules
                                  (0.5-2.0 ms wasted)

═══════════════════════════════════════════════════════════════════════
AFTER: Async Scheduling (CPU and GPU overlap)
═══════════════════════════════════════════════════════════════════════

Step N                        Step N+1                      Step N+2
CPU:
├─schedule()──┤ ├─update()──┤ ├─schedule()──┤ ├─update()──┤ ├─schedule()
│ Uses [-1]   │ │ Process   │ │ Uses [-1]   │ │ Process   │ │ ...
│ placeholder │ │ prev step │ │ placeholder │ │ prev step │ │
│ token IDs   │ │           │ │ token IDs   │ │           │ │
└─────────────┘ └───────────┘ └─────────────┘ └───────────┘ └──────────

GPU:
├─forward(N)────────┤ ├─forward(N+1)───────┤ ├─forward(N+2)──────
│ 60 layers          │ │ input_ids from GPU │ │
│ all GEMMs          │ │ (HBM→HBM copy,    │ │
│ attention          │ │  no CPU roundtrip) │ │
└────────────────────┘ └────────────────────┘ └───────────────────

Copy stream:          ↓                      ↓
               [async D2H]            [async D2H]
               token IDs              token IDs
               GPU→CPU on             GPU→CPU on
               separate stream        separate stream

████████████████████████████████████████████████████████████████████████
GPU NEVER IDLE — scheduling overlaps with forward pass
```

### The Key Insight: Why Placeholders Work

The scheduler doesn't need actual token values to schedule the next step:

```python
# vllm/v1/core/sched/async_scheduler.py:13-41

class AsyncScheduler(Scheduler):
    def _update_after_schedule(self, scheduler_output):
        for req in scheduler_output.scheduled_running_reqs:
            if req will generate tokens this step:
                # "Optimistically" tell the KV cache manager
                # a token WILL be produced (we just don't know which one)
                req.num_output_placeholders += 1  # ← placeholder, not real token

# The actual token value is only needed for:
#   1. Building input_ids for the NEXT forward pass → GPU-to-GPU copy (no CPU)
#   2. Sending the token to the user → async D2H on separate stream
```

### Three Code Changes That Enable This

**Change A: Tokens stay on GPU**

```python
# vllm/v1/worker/gpu_model_runner.py:2736-2752

# BEFORE (sync): blocking GPU → CPU copy
sampled_token_ids_list = sampled_token_ids.tolist()  # ← BLOCKS until GPU done

# AFTER (async): tokens stay in HBM
self.input_batch.prev_sampled_token_ids = sampled_token_ids  # stays on GPU
valid_sampled_token_ids = []  # empty! CPU never sees the actual tokens
```

**Change B: Next step's input_ids from GPU memory**

```python
# vllm/v1/worker/gpu_model_runner.py:1138-1229

# BEFORE: CPU writes token IDs to GPU
input_ids_cpu[i] = actual_token_id    # CPU must know the value
input_ids.copy_to_gpu()               # CPU → GPU (PCIe, ~5μs)

# AFTER: GPU copies from its own memory
self.input_ids.gpu[:N].copy_(
    self.input_batch.prev_sampled_token_ids[:N, 0],  # HBM → HBM (~0.1μs)
    non_blocking=True,
)
```

**Change C: Async D2H on separate stream**

```python
# vllm/v1/worker/gpu_model_runner.py:190-224

class AsyncGPUModelRunnerOutput:
    def __init__(self, ...):
        self.async_copy_ready_event = torch.Event()
        with torch.cuda.stream(async_output_copy_stream):     # ← separate stream
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True    # ← non-blocking D2H
            )
            self.async_copy_ready_event.record()

    def get_output(self):
        self.async_copy_ready_event.synchronize()  # blocks only when user needs it
        return self.sampled_token_ids_cpu.tolist()
```

### Impact on CUDA Graphs

CUDA graphs are **unchanged** — the graph replays the same forward pass. The difference
is **when the CPU issues the replay**:

```
BEFORE:                                AFTER:
CPU:  [schedule] [WAIT] [replay]       CPU:  [schedule] [replay]   (no WAIT)
GPU:                    [████████]     GPU:             [████████]
                 ↑                                     ↑
          CPU waited for                         CPU issued replay
          GPU to finish                          while GPU was
          previous step                          still finishing
                                                 previous step
```

### Impact on Triton

Triton kernels are **unchanged** — they execute the same attention computation. The
improvement is that the GPU starts executing Triton kernels sooner (no idle gap
between steps).

### Impact on torch.compile

torch.compile is **unchanged** — the same fusions fire. The improvement is purely
in the scheduling pipeline around the compiled model.

### Auto-Disable Conditions

```python
# vllm/config/vllm.py:570-603

if pipeline_parallel_size > 1:
    async_scheduling = False     # PP needs cross-GPU sync

if speculative_decoding and not EAGLE/MTP:
    async_scheduling = False     # Non-EAGLE spec needs real tokens

if executor not in (mp, uni, external_launcher):
    async_scheduling = False     # Ray executor not supported

# Everything else: ENABLED by default
```

---

## 2. MoE Refactor: Monolithic → Modular

**Commit series**: `[MoE Refactor][1/N]` through `[MoE Refactor][12/N]` (#31499)
**Impact**: Enables any expert backend × any communication backend

### Before: One Giant Function

```python
# BEFORE: vllm/model_executor/layers/fused_moe/fused_moe.py
# ~300 lines, boolean flag explosion

def fused_experts_impl(
    hidden_states, w1, w2, topk_weights, topk_ids,
    use_fp8_w8a8=False,       # ← boolean flags for every format
    use_int8_w8a8=False,
    use_int8_w8a16=False,
    use_int4_w4a16=False,
    per_channel_quant=False,
    ...
):
    # Everything in one function:
    quantize_input(...)           # 1. Quantize
    moe_align_block_size(...)     # 2. Sort tokens
    invoke_fused_moe_kernel(...)  # 3. First GEMM (Triton)
    silu_and_mul(...)             # 4. Activation
    quantize_input(...)           # 5. Re-quantize
    invoke_fused_moe_kernel(...)  # 6. Second GEMM (Triton)
    ops.moe_sum(...)              # 7. Reduce

# To add CUTLASS FP4 MoE: copy all 300 lines, change GEMM calls
# To add DeepEP communication: rewrite all backends
# N backends × M comm mechanisms = N×M implementations
```

### After: Three Composable Interfaces

```python
# AFTER: vllm/model_executor/layers/fused_moe/modular_kernel.py

# ┌────────────────────────────────────────────────────────────────┐
# │                FusedMoEModularKernel                           │
# │                                                                │
# │  ┌──────────────────────┐    ┌──────────────────────────────┐ │
# │  │ PrepareAndFinalize   │    │ PermuteExpertsUnpermute      │ │
# │  │                      │    │                              │ │
# │  │ prepare():           │    │ apply():                     │ │
# │  │   quantize tokens    │──→ │   permute → GEMM1 →         │ │
# │  │   dispatch to experts│    │   activation → GEMM2 →      │ │
# │  │                      │    │   unpermute                  │ │
# │  │ finalize():          │←── │                              │ │
# │  │   apply topk weights │    │ activation_formats:          │ │
# │  │   reduce             │    │   Standard or BatchedExperts │ │
# │  │   combine results    │    │                              │ │
# │  └──────────────────────┘    └──────────────────────────────┘ │
# └────────────────────────────────────────────────────────────────┘

# N backends + M comm mechanisms = N+M implementations (not N×M)
```

### The Full Backend Matrix

```
COMMUNICATION (PrepareAndFinalize)          COMPUTATION (Experts)
┌─────────────────────────────────┐         ┌─────────────────────────────┐
│                                 │         │                             │
│  NoEP (local, no comm)          │←──┐  ┌─→│  TritonExperts              │
│    format: Standard             │   │  │  │  DeepGemmExperts            │
│                                 │   │  │  │  CutlassExpertsFp8         │
│  PPLX All2All (async)           │←──┼──┤  │  CutlassExpertsFp4 (NVFP4) │
│    format: BatchedExperts       │   │  │  │  MarlinExperts              │
│                                 │   │  │  │  FlashInferExperts          │
│  DeepEP High-Throughput         │←──┤  │  │  FlashInferCuteDSLExperts   │
│    format: Standard             │   │  │  │  AiterExperts (ROCm)        │
│                                 │   │  │  │  OAITritonExperts (GPT-OSS) │
│  DeepEP Low-Latency             │←──┘  │  │  TrtLlmGenExperts          │
│    format: BatchedExperts       │      │  │                             │
│                                 │      │  │  BatchedTritonExperts       │
│  FlashInfer All2All             │←─────┤  │  BatchedDeepGemmExperts     │
│    format: Standard             │      │  │  BatchedMarlinExperts       │
│                                 │      │  │  CutlassBatchedExpertsFp8  │
│  FlashInfer AllGather           │←─────┘  │                             │
│    format: Standard             │         │                             │
└─────────────────────────────────┘         └─────────────────────────────┘

        │                                            │
        │         FORMAT NEGOTIATION                 │
        ├───────────────────────────────────────────→│
        │                                            │
        │  Standard ────────→ Standard backends      │
        │  BatchedExperts ──→ Batched* backends      │
        │                                            │
        │  If formats don't match → construction     │
        │  error at init time (not runtime crash)    │
        └────────────────────────────────────────────┘
```

### Concrete Example: PR #31499 (Pure Function Refactor)

```python
# BEFORE: Side-effect mutation
# vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py

def prepare_moe_fp8_layer_for_marlin(layer, ...) -> None:
    weight = ops.gptq_marlin_moe_repack(layer.w13_weight, ...)
    setattr(layer, "w13_weight", weight)           # ← mutates layer
    scales = reshape_w_scales(layer.w13_weight_scale)
    setattr(layer, "w13_weight_scale", scales)     # ← mutates layer
    del layer.w13_input_scale                       # ← deletes attribute
    del layer.w2_input_scale                        # ← deletes attribute
```

```python
# AFTER: Pure function — takes inputs, returns outputs
# vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py

def prepare_moe_fp8_layer_for_marlin(
    layer, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale, ...
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    workspace = ...
    w13_weight = ops.gptq_marlin_moe_repack(w13_weight, ...)
    w13_weight_scale = reshape_w_scales(w13_weight_scale)
    return (workspace, w13_weight, w2_weight,      # ← returns results
            w13_weight_scale, w2_weight_scale)      # caller assigns explicitly
```

### Impact on CUDA Graphs

The modular kernel uses `current_workspace_manager()` to pre-allocate buffers during
capture. CUDA graphs capture the `FusedMoEModularKernel.forward()` call, which internally
dispatches to the right expert backend. The graph replays the exact sequence.

### Impact on Triton

`TritonExperts` and `BatchedTritonExperts` are now first-class `PermuteExpertsUnpermute`
implementations. They plug into the modular kernel via the `apply()` interface. The Triton
MoE kernel code is unchanged — the refactor is in how it's selected and invoked.

### Impact on torch.compile

The `SiluMulNvfp4QuantPattern` fusion fires **inside** the modular MoE pipeline at the
activation step. The modular structure doesn't interfere with torch.compile graph tracing.

---

## 3. CUDA Graph Compound Modes

**PR**: #20059 — Major CUDA graph system redesign
**Impact**: Mixed prefill+decode can use FULL graphs (if backend supports it)

### Before: One Mode, Binary Choice

```python
# BEFORE: config/compilation.py
class CUDAGraphMode(enum.Enum):
    NONE = 0        # No graphs
    PIECEWISE = 1   # Graphs with attention outside
    FULL = 2        # Everything inside graph

# Problem: FULL requires attention backend to support ALL batch types.
# FA2 only supports uniform decode (UNIFORM_BATCH level).
# Result: FA2 users forced to PIECEWISE for all batches,
#         even though uniform decode COULD use FULL.
```

### After: Compound Modes, Best of Both Worlds

```python
# AFTER: config/compilation.py:52-96
class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = (FULL, NONE)           # ← NEW: FULL for decode, nothing for mixed
    FULL_AND_PIECEWISE = (FULL, PIECEWISE)    # ← NEW: FULL for decode, PIECEWISE for mixed

    def decode_mode(self):   # What mode for uniform decode batches
    def mixed_mode(self):    # What mode for prefill/mixed batches
```

### Visual: How Compound Modes Work

```
═══════════════════════════════════════════════════════════════
Default mode: FULL_AND_PIECEWISE (optimization level O2)
═══════════════════════════════════════════════════════════════

UNIFORM DECODE BATCH (all requests generating 1 token each):
┌──── FULL CUDA Graph ──────────────────────────────────┐
│                                                        │
│  RMSNorm → GEMM → RoPE → [ATTENTION] → GEMM → MoE   │
│         → (repeat × 60 layers)                        │
│         → RMSNorm → LM Head                          │
│                                                        │
│  Everything inside, including attention.               │
│  ONE replay() call. Maximum performance.              │
│                                                        │
└────────────────────────────────────────────────────────┘


MIXED PREFILL+DECODE BATCH (some requests prefilling, some decoding):
┌─ Piece 0 ─────────┐          ┌─ Piece 1 ─────────────┐
│ RMSNorm → GEMM    │          │ GEMM → MoE → RMSNorm  │
│ → RoPE            │          │ → GEMM → RoPE         │
└────────────────────┘          └────────────────────────┘
          │                               │
          ▼   EAGER (not graphed)         ▼   EAGER
    ┌ ─ ─ ─ ─ ─ ─ ┐              ┌ ─ ─ ─ ─ ─ ─ ┐
    │  ATTENTION    │              │  ATTENTION    │
    └ ─ ─ ─ ─ ─ ─ ┘              └ ─ ─ ─ ─ ─ ─ ┘
          │                               │
          ▼                               ▼
    (next piece ...)               (next piece ...)

  Attention runs eagerly because mixed batches have
  variable query lengths that FA2 can't graph.
  GEMMs and norms still graphed (75% of launch overhead saved).
```

### The Dispatch Flow

```python
# vllm/v1/cudagraph_dispatcher.py:143-183

dispatch(num_tokens=32, uniform_decode=True):
                │
                ▼
    ┌─ Is batch uniform decode? ─┐
    │           YES               │ NO
    ▼                             ▼
  Try FULL keys               Try PIECEWISE keys
  (exact match)               (relaxed match)
    │                             │
    ▼                             ▼
  Found? ──YES──→ FULL        Found? ──YES──→ PIECEWISE
    │                             │
    NO                            NO
    ▼                             ▼
  Try FULL relaxed            CUDAGraphMode.NONE
    │                         (eager execution)
    ▼
  Found? ──YES──→ FULL
    │
    NO
    ▼
  Try PIECEWISE relaxed
    │
    ▼
  Found? ──YES──→ PIECEWISE
    │
    NO ──→ NONE
```

### The Nested Wrapper Design

```
Model wrapped with TWO layers of CUDAGraphWrapper:

┌─── CUDAGraphWrapper(mode=FULL) ──────────────────────────────────────┐
│                                                                       │
│  If forward_context.mode == FULL:                                    │
│    → replay FULL graph (entire forward pass)                         │
│                                                                       │
│  If forward_context.mode == PIECEWISE:                               │
│    → pass through to inner model (no graph at this level)            │
│                                                                       │
│  ┌─── Compiled Model (torch.compile + Inductor) ─────────────────┐  │
│  │                                                                 │  │
│  │  ┌─ CUDAGraphWrapper(mode=PIECEWISE) ─┐                       │  │
│  │  │ RMSNorm → GEMM → RoPE              │                       │  │
│  │  └────────────────────────────────────┘                        │  │
│  │              │                                                  │  │
│  │              ▼ ATTENTION (eager, splitting op)                  │  │
│  │              │                                                  │  │
│  │  ┌─ CUDAGraphWrapper(mode=PIECEWISE) ─┐                       │  │
│  │  │ GEMM → AllReduce → RMSNorm → MoE   │                       │  │
│  │  └────────────────────────────────────┘                        │  │
│  │              │                                                  │  │
│  │              ▼ ATTENTION (eager)                                │  │
│  │              ... (× 60 layers)                                  │  │
│  │                                                                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘

FULL mode:      Outer wrapper replays → inner wrappers never fire
PIECEWISE mode: Outer wrapper passes through → inner wrappers replay per-piece
NONE mode:      Both wrappers pass through → everything runs eagerly
```

### Backend Compatibility Auto-Resolution

```python
# vllm/v1/worker/gpu_model_runner.py:4933-5086

# User requests FULL_AND_PIECEWISE, but attention backend only supports UNIFORM_BATCH:

min_cg_support = min(all_backend_supports)

if min_cg_support == ALWAYS:           # FA3, Triton
    → keep FULL_AND_PIECEWISE          # both modes work
elif min_cg_support == UNIFORM_BATCH:  # FA2
    → keep FULL_AND_PIECEWISE          # FULL for uniform decode, PIECEWISE for mixed
elif min_cg_support == UNIFORM_SINGLE_TOKEN_DECODE:
    → downgrade to FULL_DECODE_ONLY    # FULL only for single-token decode
elif min_cg_support == NEVER:
    → downgrade to PIECEWISE only      # or raise error
```

---

## 4. torch.compile Fusion Patterns

**Files**: `vllm/compilation/activation_quant_fusion.py`, `vllm/compilation/collective_fusion.py`
**Impact**: Reduces kernel launches by fusing adjacent operations

### Pattern 1: SiLU + Mul + Quantization Fusion

```
═══════════════════════════════════════════════════════════════
BEFORE: 3 separate kernel launches
═══════════════════════════════════════════════════════════════

gate, up = gate_up_proj(x)        # CUTLASS GEMM → write to HBM
                                    │
                                    ▼  HBM read
x = SiLU(gate) * up               # CUDA kernel → write to HBM
                                    │
                                    ▼  HBM read
x_fp4 = scaled_fp4_quant(x)       # CUDA kernel → write to HBM

Kernel launches: 3     (5μs × 3 = 15μs overhead)
HBM round-trips: 2     (read+write × 2 = 4 × bandwidth cost)


═══════════════════════════════════════════════════════════════
AFTER: 1 fused kernel launch (torch.compile pattern match)
═══════════════════════════════════════════════════════════════

gate, up = gate_up_proj(x)        # CUTLASS GEMM → write to HBM
                                    │
                                    ▼  HBM read (ONCE)
x_fp4 = silu_and_mul_nvfp4_quant( # Fused CUDA kernel:
    gate, up, global_scale         #   SiLU + Mul + FP4 quant
)                                  # → write to HBM (ONCE)

Kernel launches: 1     (5μs overhead)
HBM round-trips: 1     (read once, write once)

Savings per MoE layer: 10μs launch + ~30% less HBM traffic for activation
```

```python
# vllm/compilation/activation_quant_fusion.py:121-168

# Pattern registered with torch._inductor.pattern_matcher:
class SiluMulNvfp4QuantPattern:
    # Matches: SiluAndMul(x) → nvfp4_quant(result, scale)
    # Replaces with: silu_and_mul_nvfp4_quant(x, scale)

    # Also registered: SiluMulFp8StaticQuantPattern (for FP8)
```

### Pattern 2: AllReduce + RMSNorm + Quantization Fusion

```
═══════════════════════════════════════════════════════════════
BEFORE: 3 separate operations (AllReduce is expensive)
═══════════════════════════════════════════════════════════════

x = all_reduce(x)                 # NCCL collective (NVLink)
                                    │
                                    ▼  HBM read
residual = x + residual            # CUDA kernel
x = rmsnorm(x, weight)            # CUDA kernel → write to HBM
                                    │
                                    ▼  HBM read
x_fp4 = scaled_fp4_quant(x)       # CUDA kernel → write to HBM

Kernel launches: 4
HBM round-trips: 3


═══════════════════════════════════════════════════════════════
AFTER: 1 fused FlashInfer TRT-LLM kernel
═══════════════════════════════════════════════════════════════

x_fp4, residual = allreduce_fused_add_rmsnorm_nvfp4_quant(
    x, residual, weight, global_scale
)
# AllReduce + residual add + RMSNorm + FP4 quant
# ALL in one kernel, one HBM round-trip

Kernel launches: 1
HBM round-trips: 1
```

### All 12 Fusion Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ACTIVATION + QUANT FUSIONS (activation_quant_fusion.py)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. SiLU+Mul → FP8 static quant       →  silu_and_mul_quant               │
│ 2. SiLU+Mul → NVFP4 quant            →  silu_and_mul_nvfp4_quant         │
├─────────────────────────────────────────────────────────────────────────────┤
│ ASYNC TP FUSIONS (collective_fusion.py — AsyncTPPass)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. mm + reduce_scatter               →  fused_matmul_reduce_scatter       │
│ 4. all_gather + mm                   →  fused_all_gather_matmul           │
│ 5. scaled_mm + reduce_scatter        →  fused_scaled_matmul_reduce_scatter│
│ 6. all_gather + scaled_mm            →  fused_all_gather_scaled_matmul    │
│ 7. cutlass_mm + reduce_scatter       →  (same fused op, CUTLASS variant)  │
│ 8. all_gather + cutlass_mm           →  (same fused op, CUTLASS variant)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ ALLREDUCE FUSIONS (collective_fusion.py — AllReduceFusionPass)              │
│ Uses FlashInfer TRT-LLM allreduce with SM90/SM100-specific size limits     │
├─────────────────────────────────────────────────────────────────────────────┤
│  9. allreduce + RMSNorm                                                     │
│ 10. allreduce + FusedAddRMSNorm (with residual)                            │
│ 11. allreduce + RMSNorm + FP8 quant                                        │
│ 12. allreduce + FusedAddRMSNorm + FP8 quant                               │
│ 13. allreduce + RMSNorm + NVFP4 quant             (SM100+ only)           │
│ 14. allreduce + FusedAddRMSNorm + NVFP4 quant     (SM100+ only)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How torch.compile Sees This

```
torch.compile traces the model's forward() into an FX graph:

┌─ FX Graph (before fusion passes) ────────────────────────────────┐
│                                                                   │
│  %1 = call rmsnorm(hidden, weight)                               │
│  %2 = call cutlass_scaled_fp4_mm(%1, w_qkv, ...)   # QKV GEMM  │
│  %3 = call rotary_embedding(%2, cos, sin)            # RoPE      │
│  %4 = call vllm::unified_attention(%3, kv_cache)     # ATTENTION │ ← splitting op
│  %5 = call cutlass_scaled_fp4_mm(%4, w_o, ...)       # O proj   │
│  %6 = call all_reduce(%5)                            # TP comm   │
│  %7 = call fused_add_rmsnorm(%6, residual, weight)   # Norm     │
│  %8 = call nvfp4_quant(%7, scale)                    # FP4 quant│
│  %9 = call cutlass_scaled_fp4_mm(%8, w_gate_up, ...) # MLP up   │
│  %10= call silu_and_mul(%9)                          # Activation│
│  %11= call nvfp4_quant(%10, scale)                   # FP4 quant│
│  %12= call cutlass_scaled_fp4_mm(%11, w_down, ...)   # MLP down │
│  ...                                                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                    Fusion passes fire:
                              │
                              ▼
┌─ FX Graph (after fusion passes) ─────────────────────────────────┐
│                                                                   │
│  %1 = call rmsnorm(hidden, weight)                               │
│  %2 = call cutlass_scaled_fp4_mm(%1, w_qkv, ...)                │
│  %3 = call rotary_embedding(%2, cos, sin)                        │
│  %4 = call vllm::unified_attention(%3, kv_cache)     ← SPLIT    │
│  %5 = call cutlass_scaled_fp4_mm(%4, w_o, ...)                   │
│  %6 = call allreduce_fused_add_rmsnorm_nvfp4(%5,r,w) ← FUSED   │
│  %7 = call cutlass_scaled_fp4_mm(%6, w_gate_up, ...)             │
│  %8 = call silu_and_mul_nvfp4_quant(%7, scale)       ← FUSED   │
│  %9 = call cutlass_scaled_fp4_mm(%8, w_down, ...)                │
│  ...                                                             │
└──────────────────────────────────────────────────────────────────┘

Operations reduced from 12 to 9 per layer section.
3 fewer kernel launches × 60 layers = 180 fewer launches per step.
```

---

## 5. Triton Attention Feature Completeness

**File**: `vllm/attention/ops/triton_unified_attention.py`
**Impact**: The only backend that supports ALL features on ALL hardware

### Feature Support Matrix

```
                              Triton   FA3     FA2    FlashInfer
                              ──────   ──────  ─────  ──────────
Attention Sinks (GPT-OSS)     ✓        ✓*      ✗       ~†
ALiBi Slopes                   ✓        ✗       ✓       ✓
Softcap (Gemma)                ✓        ✓       ✓       ✓
Sliding Window                 ✓        ✓       ✓       ✓
Multimodal Prefix (bidir)      ✓        ✗       ✗       ✗
FP8 KV Cache                  ✓        ✓*      ✗       ✓
Fused FP8 Output Quant         ✓‡       ✗       ✗       ✓
CUDA Graph: ALWAYS             ✓        ✓*      ✗       ~
Cascade Attention              ✗        ✓       ✓       ✓
NVIDIA GPUs                    ✓        ✓       ✓       ✓
AMD GPUs                       ✓        ✗       ✗       ✗
Intel GPUs                     ✓        ✗       ✗       ✗

✓* = Hopper (SM90) only
✓‡ = FP8 static tensor symmetric only
~† = Via TRT-LLM extension
~  = Varies by version
```

### Sinks: How They Work in Triton

```python
# vllm/attention/ops/triton_unified_attention.py:156-160

# GPT-OSS uses "attention sinks" — learned per-head values
# added to the softmax denominator:
#   softmax(S) = exp(S) / (sum(exp(S)) + exp(sink_value))

if USE_SINKS:
    # Initialize running max with sink values instead of -inf
    M = tl.load(sinks_ptr + cur_head_idx * stride_sks)  # per-head learned value
    l_j = tl.exp(M)   # pre-initialize exp_sum with exp(sink)
else:
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_j = tl.zeros([BLOCK_M], dtype=tl.float32)
```

### Multimodal Prefix: How Bidirectional Ranges Work

```python
# vllm/attention/ops/triton_unified_attention.py:295-320

# For VLMs, image tokens attend to each other bidirectionally
# (not causally). The mm_prefix_range_tensor defines which
# token ranges are bidirectional.

if USE_MM_PREFIX:
    mm_range = tl.load(mm_prefix_range_tensor + seq_idx * 2)  # [start, end]
    is_in_prefix = (token_pos >= mm_range[0]) & (token_pos < mm_range[1])
    # If both query and key are in prefix: no causal mask
    # If either is outside prefix: apply causal mask
    causal_mask = tl.where(is_in_prefix, False, standard_causal_mask)
```

This means a VLM like Qwen3-VL can have:
```
Token 0-5:     text tokens (causal — can only see past)
Token 6-262:   image patch tokens (BIDIRECTIONAL — see all other patches)
Token 263-270: text tokens (causal — see past including all image tokens)
```

Only the Triton backend implements this. FA2/FA3/FlashInfer require a separate
prefix attention pass.

### The 2D → 3D Kernel Switch

```
vllm/v1/attention/backends/triton_attn.py:372-497

Batch size vs kernel selection:

  threshold = MIN_LAUNCH_GRID_SIZE_2D / num_kv_heads
            = 128 / 8 = 16

  batch ≤ 16:  3D kernel (16 parallel segments per sequence)
               grid = (batch, 8 kv_heads, 16 segments) = 2,048 thread blocks
               Then: reduce_segments kernel merges 16 partials

  batch > 16:  2D kernel (no segments)
               grid = (batch, 8 kv_heads) = batch × 8 thread blocks
               Enough parallelism from batch size alone

  Why the switch:
    batch=1, kv_heads=8:
      2D: 1 × 8 = 8 thread blocks → 8 of 132 SMs used (6%)   BAD
      3D: 1 × 8 × 16 = 128 thread blocks → 97% SM usage      GOOD

    batch=64, kv_heads=8:
      2D: 64 × 8 = 512 thread blocks → all SMs busy           GOOD
      3D: 64 × 8 × 16 = 8,192 thread blocks → overkill       WASTEFUL
```

---

## Summary: What Changed Where

```
┌──────────────────────┬──────────────────────────────────────────────────┐
│ Change               │ Effect on Each Layer of the Stack                │
├──────────────────────┼──────────────────────────────────────────────────┤
│                      │                                                  │
│ 1. ASYNC SCHEDULING  │ PyTorch:      unchanged                         │
│    (default=on)      │ CUDA Graphs:  unchanged (replay timing changes) │
│                      │ Triton:       unchanged (executes sooner)       │
│                      │ GPU:          never idle between steps          │
│                      │ Throughput:   +5-15%                            │
│                      │                                                  │
│ 2. MOE REFACTOR      │ PyTorch:      new Module hierarchy              │
│    (modular kernel)  │ CUDA Graphs:  graphs capture modular forward()  │
│                      │ Triton:       TritonExperts now a pluggable impl│
│                      │ GPU:          same kernels, cleaner dispatch    │
│                      │ Extensibility: N+M instead of N×M backends     │
│                      │                                                  │
│ 3. COMPOUND CG MODES │ PyTorch:      torch.compile splits at attention │
│    (FULL+PIECEWISE)  │ CUDA Graphs:  nested wrappers, dual key sets   │
│                      │ Triton:       ALWAYS support enables FULL mode  │
│                      │ GPU:          fewer launches for decode batches │
│                      │ Latency:      -10-20% for decode steps          │
│                      │                                                  │
│ 4. COMPILE FUSIONS   │ PyTorch:      Inductor pattern matcher fires    │
│    (14 patterns)     │ CUDA Graphs:  fewer kernels to capture/replay   │
│                      │ Triton:       not affected (attention stays)    │
│                      │ GPU:          less HBM traffic, fewer launches  │
│                      │ Throughput:   +5-10% for TP configurations      │
│                      │                                                  │
│ 5. TRITON FEATURES   │ PyTorch:      new custom ops registered         │
│    (sinks, mmprefix) │ CUDA Graphs:  ALWAYS support (best level)       │
│                      │ Triton:       ~1060 LoC, all features in one    │
│                      │ GPU:          portable across NVIDIA/AMD/Intel  │
│                      │ Compatibility: only backend with every feature  │
│                      │                                                  │
└──────────────────────┴──────────────────────────────────────────────────┘
```
