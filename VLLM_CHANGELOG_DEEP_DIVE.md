# vLLM Changelog Deep Dive: What Changed, Why, and Where It Runs on the GPU

Every recent change in the vLLM codebase — traced to the exact code, shown before vs. after,
mapped to GPU silicon, and projected forward into future system design.

---

## Table of Contents

1.  [Change 1 — Async Scheduling Enabled by Default](#1-async-scheduling-enabled-by-default)
2.  [Change 2 — FusedMoE Refactor: Marlin FP8 Pure Function](#2-fusedmoe-refactor-marlin-fp8-pure-function)
3.  [Change 3 — ROCm Shared-Expert Accuracy Fix](#3-rocm-shared-expert-accuracy-fix)
4.  [Change 4 — LoRA-Aware Prefix Cache Events](#4-lora-aware-prefix-cache-events)
5.  [Change 5 — LMCache KV-Cache Registration](#5-lmcache-kv-cache-registration)
6.  [Change 6 — AsyncLLM generate/encode Deduplication](#6-asyncllm-generateencode-deduplication)
7.  [Change 7 — Default Chat Template Kwargs](#7-default-chat-template-kwargs)
8.  [Change 8 — continue\_final\_message for Embeddings](#8-continue_final_message-for-embeddings)
9.  [Change 9 — H100 Flash Attention 3 Scheduler Metadata Fix](#9-h100-flash-attention-3-scheduler-metadata-fix)
10. [Change 10 — ROCm GPTQ GEMM Output Zeroing Race Condition](#10-rocm-gptq-gemm-output-zeroing-race-condition)
11. [Change 11 — QKNorm Optimization for MiniMax-M2](#11-qknorm-optimization-for-minimax-m2)
12. [Change 12 — ConvNd Replacement for Torch 2.9 Compatibility](#12-convnd-replacement-for-torch-29-compatibility)
13. [The Qwen3-VL Model: Full Code Walkthrough](#13-the-qwen3-vl-model-full-code-walkthrough)
14. [Attention Backend Selection: 18 Backends, One Hot Path](#14-attention-backend-selection-18-backends-one-hot-path)
15. [KV Cache Architecture: Paged Memory on the GPU](#15-kv-cache-architecture-paged-memory-on-the-gpu)
16. [GPU Execution Map: Where Every Operation Runs](#16-gpu-execution-map-where-every-operation-runs)
17. [Future Use Cases and System Design Implications](#17-future-use-cases-and-system-design-implications)

---

## 1. Async Scheduling Enabled by Default

**Commit**: `c2ff33c` — PR #27614
**Files**: `vllm/config/scheduler.py`, `vllm/config/vllm.py`
**Impact**: 5-15% throughput improvement for decode-heavy workloads

### The Problem

Every vLLM step had this timeline:

```
BEFORE (synchronous):
CPU:  [schedule()] [██ IDLE ██████████████] [update_from_output()] [schedule()] [██ IDLE ██████]
GPU:               [forward() + sample()]                                       [forward() + sample()]
                   ▲                      ▲
                   GPU busy               GPU COMPLETELY IDLE while CPU schedules
```

The scheduler at `vllm/v1/core/sched/scheduler.py:227` iterates every running request,
allocates KV cache blocks, handles preemption, and builds `SchedulerOutput` — all Python,
all CPU, taking hundreds of microseconds to milliseconds. During that time, every SM,
every tensor core, every memory controller on the GPU sits idle.

### Before vs. After Code

**`vllm/config/scheduler.py:133`**

```python
# ──── BEFORE ────
async_scheduling: bool = False

# ──── AFTER ────
async_scheduling: bool = Field(default=None)   # None = auto-enable
```

**`vllm/config/vllm.py:570-603` — auto-enable logic**

```python
# ──── AFTER ──── (new code)
elif self.scheduler_config.async_scheduling is None:
    if self.parallel_config.pipeline_parallel_size > 1:
        logger.warning("Async scheduling not supported with PP > 1, disabling.")
        self.scheduler_config.async_scheduling = False
    elif self.speculative_config is not None:
        if self.speculative_config.method not in get_args(EagleModelTypes):
            self.scheduler_config.async_scheduling = False
        else:
            self.scheduler_config.async_scheduling = False
    # All other cases: enable it
```

### How It Works: The Three-Stream Trick

The insight: the scheduler does NOT need sampled token IDs from step N to schedule step N+1.
It only needs to know _how many_ tokens were generated (always 1 for standard decoding).

**Step 1: Keep sampled tokens on GPU** — `vllm/v1/worker/gpu_model_runner.py:2736-2752`

```python
# ──── BEFORE ────
sampled_token_ids_cpu = sampled_token_ids.tolist()   # GPU→CPU sync! Blocks everything.

# ──── AFTER ────
# Cache the sampled tokens ON THE GPU. Avoid CPU sync entirely.
self.input_batch.prev_sampled_token_ids = sampled_token_ids    # stays in HBM
self.input_batch.prev_req_id_to_index = {
    req_id: i for i, req_id in enumerate(self.input_batch.req_ids)
    if i not in invalid_req_indices_set
}
# Scheduler gets placeholder [-1] values — it doesn't need the actual token
sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
```

**Step 2: Async D2H copy on a separate CUDA stream** — `gpu_model_runner.py:190-224`

```python
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(self, ...):
        self.async_copy_ready_event = torch.Event()

        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):     # SEPARATE stream
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True                       # Non-blocking D2H
            )
            self.async_copy_ready_event.record()               # Event for sync later
```

**Step 3: GPU-to-GPU copy for next step's input_ids** — `gpu_model_runner.py:1138-1221`

```python
def _prepare_input_ids(self, scheduler_output, ...):
    if self.input_batch.prev_sampled_token_ids is None:
        self.input_ids.copy_to_gpu(total_num_scheduled_tokens)  # Normal path
        return

    # Async path: copy prev tokens WITHIN GPU memory (HBM → HBM, no CPU)
    self.input_ids.gpu[:num_common_tokens].copy_(
        self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
        non_blocking=True,
    )
```

### The New Timeline

```
AFTER (async):
         Step N                                Step N+1
CPU:     [schedule(N)] [update] [prep]         [schedule(N+1)] [update] [prep]
                        ↘                       ↘
GPU:                     [forward(N)] [sample]    [forward(N+1)] [sample]
Copy:                               [D2H ──>]                  [D2H ──>]
                                    ▲
         schedule(N+1) OVERLAPS with forward(N) on GPU!
```

### Where on the GPU

| Phase | GPU Resource | Active? |
|-------|-------------|---------|
| `forward()` — QKV projections, MLP | Tensor Cores (FP16/BF16 MMA) | Saturated |
| `forward()` — Flash Attention | Tensor Cores + SM shared memory | Saturated |
| `forward()` — KV cache reads | HBM memory controllers | Streaming |
| `sample()` — vocab projection + sampling | Tensor Cores (GEMM) + CUDA cores | Active |
| **CPU scheduling (BEFORE)** | **ALL GPU resources** | **IDLE** |
| **CPU scheduling (AFTER)** | GPU still running forward(N) | **BUSY** |
| Async D2H copy | PCIe/NVLink DMA engine | Low overhead |
| GPU-to-GPU input prep | HBM → HBM copy engine | Sub-microsecond |

### Use Cases Enabled

- **Small models (7B-13B)**: GPU forward pass is fast (~2ms), scheduling overhead (~0.5ms) was 25% of step time. Now overlapped.
- **High batch sizes**: Scheduling iterates all requests — more requests = more CPU time saved by overlap.
- **Real-time chat**: Every microsecond of per-token latency matters for time-to-first-token (TTFT) and inter-token latency (ITL).

---

## 2. FusedMoE Refactor: Marlin FP8 Pure Function

**Commit**: `9152a30` — PR #31499 (Part 12/N of MoE refactor series)
**Files**: `vllm/model_executor/layers/fused_moe/` (4 files)
**Impact**: Correctness for RL weight reloading, eliminates state mutation bugs

### The Problem

`prepare_moe_fp8_layer_for_marlin()` was an impure function that mutated the layer in-place
through `getattr`/`setattr`/`delattr`. This created ordering bugs, made RL weight reloading
unsafe, and conflated weight preparation with state management.

### Before vs. After Code

**BEFORE** — impure, mutates layer:

```python
def prepare_moe_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    size_k_first: bool = True,           # Ambiguous layout flag
    input_dtype: torch.dtype | None = None,
) -> None:                                # Returns NOTHING
    layer.workspace = marlin_make_workspace_new(device, 4)   # MUTATES
    for name in ["w13_weight", "w2_weight"]:
        weight = getattr(layer, name)       # READS from layer
        # ... repack ...
        setattr(layer, name, weight)        # MUTATES layer
    for name in ["w13", "w2"]:
        scales = getattr(layer, new_name)   # READS
        delattr(layer, new_name)            # DELETES
        setattr(layer, name + "_weight_scale", scales)   # MUTATES
    for name in ["w13_bias", "w2_bias"]:
        bias = getattr(layer, name)
        setattr(layer, name, bias)          # MUTATES

# Caller (fp8.py):
replace_parameter(layer, "w13_weight", w13_weight)
replace_parameter(layer, "w2_weight", w2_weight)
# TODO: we do this AFTER replace_parameter because it uses layer params directly
if self.fp8_backend == Fp8MoeBackend.MARLIN:
    prepare_moe_fp8_layer_for_marlin(layer, False, ...)
    del layer.w13_input_scale    # manual cleanup
    del layer.w2_input_scale     # manual cleanup
```

**AFTER** — pure function, explicit inputs and outputs:

```python
def prepare_moe_fp8_layer_for_marlin(
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,             # Explicit input
    w2_weight: torch.Tensor,              # Explicit input
    w13_weight_scale: torch.Tensor,       # Explicit input
    w2_weight_scale: torch.Tensor,        # Explicit input
    input_dtype: torch.dtype | None = None,
) -> tuple[                               # Returns ALL outputs
    torch.Tensor,  # workspace
    torch.Tensor,  # w13_weight (repacked)
    torch.Tensor,  # w2_weight (repacked)
    torch.Tensor,  # w13_weight_scale (permuted)
    torch.Tensor,  # w2_weight_scale (permuted)
]:
    w13_weight = repack_weight("w13", w13_weight)
    w2_weight = repack_weight("w2", w2_weight)
    w13_weight_scale = permute_scales(w13_weight_scale, "w13")
    w2_weight_scale = permute_scales(w2_weight_scale, "w2")
    return (workspace, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale)

# Caller (fp8.py):
elif self.fp8_backend == Fp8MoeBackend.MARLIN:
    (workspace, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale) = \
        prepare_moe_fp8_layer_for_marlin(
            layer, w13_weight, w2_weight, w13_weight_scale, w2_weight_scale, ...)
    layer.workspace = workspace
# Then ONE replace_parameter pass for ALL backends
replace_parameter(layer, "w13_weight", w13_weight)
replace_parameter(layer, "w2_weight", w2_weight)
```

### What Happens on the GPU: FusedMoE Kernel Execution

The FusedMoE Triton kernel at `vllm/model_executor/layers/fused_moe/fused_moe.py:316`:

```
GPU Execution of One MoE Layer (e.g., Qwen3-VL-235B MoE):

1. Router GEMM (gate):  hidden_states @ gate_weight → router_logits
   ├── Tensor Cores: [batch, hidden] × [hidden, num_experts] → [batch, E]
   └── HBM: Read gate_weight (~1MB), write logits (~tiny)

2. Top-K selection:  router_logits → topk_ids, topk_weights
   ├── CUDA cores: Sort/select top-k per token
   └── Registers: O(batch × top_k) metadata

3. Token permutation:  moe_align_block_size()
   ├── Reorders tokens so all tokens for expert_0 are contiguous, then expert_1, etc.
   ├── CUDA cores: Scatter/gather metadata
   └── HBM: Write sorted_token_ids, expert_ids arrays

4. Fused GEMM #1:  permuted_tokens @ w1[expert] → intermediate
   ├── Tensor Cores: Tiled GEMM with expert selection via sorted_token_ids
   │   Each thread block reads expert_ids[pid_m] to index into w1
   │   w1 shape: [E, 2*intermediate_size, hidden_size]
   ├── Shared Memory: Tiles of A (tokens) and B (expert weights)
   ├── L2 Cache: GROUP_SIZE_M grouping promotes weight reuse across adjacent token blocks
   └── HBM: Read expert weights (the dominant bandwidth consumer)

5. Activation:  SiLU-and-Mul (gate * up projection)
   ├── CUDA cores: Elementwise FP16/BF16 operations
   └── HBM: Read intermediate, write activated

6. Fused GEMM #2:  activated @ w2[expert] → output
   ├── Tensor Cores: Same structure as GEMM #1
   └── HBM: Read w2 expert weights

7. Reduce:  moe_sum() — weighted sum across top-k experts
   ├── CUDA cores: Elementwise weighted addition
   └── HBM: Read partial outputs, write final
```

### What the Marlin FP8 Repack Does

Marlin uses a special weight packing format optimized for Tensor Core access patterns:

```
Standard FP8 layout:    [E, N, K] — contiguous rows
Marlin-repacked layout: [E, N/tile, K/tile, tile_N, tile_K] — 128x128 tiles
                        aligned to Tensor Core's 16x16 MMA tile size

The repack rearranges bytes so that consecutive memory addresses correspond
to consecutive Tensor Core input elements, minimizing bank conflicts in
shared memory and maximizing SMEM→register throughput.
```

---

## 3. ROCm Shared-Expert Accuracy Fix

**Commit**: `1a834df` — PR #31523
**Files**: `vllm/platforms/rocm.py`
**Impact**: Fixes silent accuracy corruption on AMD MI300X when shared expert fusion is enabled

### The Problem

AMD's AITER library fuses shared experts into the MoE kernel by treating them as "always-on"
expert columns appended to the `topk_ids` tensor. This requires the `grouped_topk` custom op
to produce a specific topk format. If `grouped_topk` was disabled, the kernel consumed
incompatible topk data → silent numerical corruption.

### Before vs. After Code

**`vllm/platforms/rocm.py` — BEFORE**: No guard. User could disable `grouped_topk` while `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`.

**`vllm/platforms/rocm.py` — AFTER**: Two new guards added:

```python
# Guard 1: Override explicit user disable
if use_aiter_fused_se and "-grouped_topk" in compilation_config.custom_ops:
    logger.warning_once(
        "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS requires 'grouped_topk'. "
        "Overriding the user-provided '-grouped_topk'."
    )
    compilation_config.custom_ops.remove("-grouped_topk")

# Guard 2: Auto-enable when no user preference
if (use_aiter_fused_moe
    and "+grouped_topk" not in compilation_config.custom_ops
    and "-grouped_topk" not in compilation_config.custom_ops):
    compilation_config.custom_ops.append("+grouped_topk")
```

### How Shared Expert Fusion Works on MI300X

```
Without fusion (standard SharedFusedMoE):
  routed_out = FusedMoE(hidden, topk_ids=[expert_3, expert_7])   # 2 experts per token
  shared_out = shared_expert_mlp(hidden)                          # SEPARATE kernel
  output = routed_out + shared_out                                # elementwise add

With AITER fusion (single kernel):
  total_topk_ids = [expert_3, expert_7, SHARED_0, SHARED_1]      # 4 "experts" per token
  output = AITER_FusedMoE(hidden, total_topk_ids)                 # ONE kernel does everything
  # Shared experts are just extra columns in topk_ids with weight=1.0

The grouped_topk op MUST produce topk_ids in a format where:
  - First top_k columns = routed expert IDs
  - Last n_shared_experts columns = shared expert IDs (always activated)
If grouped_topk is OFF, topk_ids has the wrong shape → AITER reads garbage expert IDs.
```

### Where on the GPU (MI300X)

AMD MI300X uses a chiplet design with 8 XCD (Accelerated Compute Dies):
- Each XCD has 38 CUs (Compute Units), 304 CUs total
- Each CU has 4 SIMD units (64-wide wavefronts)
- 192 GB HBM3, 5.3 TB/s bandwidth
- 256 MB Infinity Cache (L3-equivalent)

The AITER FusedMoE kernel:
- Dispatches wavefronts across CUs, with each wavefront handling a tile of the GEMM
- Expert weight tensors sit in HBM3, streamed through Infinity Cache
- `grouped_topk` output sits in HBM3, read once per token by the dispatch logic
- When `grouped_topk` is wrong: wavefronts read invalid expert IDs → index out of bounds into weight tensor → silent garbage in output

---

## 4. LoRA-Aware Prefix Cache Events

**Commit**: `39512ab` — PR #27577
**Files**: `vllm/distributed/kv_events.py`, `vllm/v1/core/block_pool.py`, `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`
**Impact**: Enables correct distributed KV cache sharing when LoRA adapters are active

### The Problem

The prefix cache INTERNALLY already hashed block contents with the LoRA adapter name.
But the `BlockStored` event — published to external KV cache subscribers (LMCache, NIXL) —
only included `lora_id` (an integer), not `lora_name` (the hash key). External systems
could NOT reconstruct correct block hashes for LoRA requests.

### Before vs. After Code

**`vllm/distributed/kv_events.py`**

```python
# ──── BEFORE ────
class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None        # Only numeric ID
    medium: str | None
    # lora_name MISSING

# ──── AFTER ────
class BlockStored(KVCacheEvent):
    block_hashes: list[ExternalBlockHash]
    parent_block_hash: ExternalBlockHash | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None        # Retained for backward compat (deprecated)
    medium: str | None
    lora_name: str | None      # NEW — the actual hash key
```

**`vllm/v1/core/block_pool.py:281-296` — event emission**

```python
# ──── BEFORE ────
self.kv_event_queue.append(BlockStored(
    ...,
    lora_id=request.lora_request.adapter_id if request.lora_request else None,
    medium=MEDIUM_GPU,
    # lora_name NOT included
))

# ──── AFTER ────
self.kv_event_queue.append(BlockStored(
    ...,
    lora_id=request.lora_request.adapter_id if request.lora_request else None,
    medium=MEDIUM_GPU,
    lora_name=request.lora_request.name if request.lora_request else None,  # NEW
))
```

### Why This Matters: Block Hash Computation

The block hash in `vllm/v1/core/kv_cache_utils.py:525-552` is a **Merkle chain**:

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

Where `extra_keys` includes `lora_name` from `_gen_lora_extra_hash_keys()`:

```
Same tokens [1, 2, 3, ..., 16]:
  No LoRA:       hash(parent, (1..16), None)        = 0xA1B2...
  LoRA "alpaca": hash(parent, (1..16), ("alpaca",)) = 0xC3D4...   ← DIFFERENT
  LoRA "code":   hash(parent, (1..16), ("code",))   = 0xE5F6...   ← DIFFERENT
```

Without `lora_name` in the event, an external LMCache instance would compute the wrong hash
and either (a) fail to find valid cached blocks, or (b) incorrectly share blocks between
different LoRA adapters → numerical corruption.

### Where on the GPU

This change is CPU-side metadata only. But it enables GPU memory savings:

```
Scenario: 100 requests, 50 with LoRA-A, 50 with LoRA-B, all sharing same system prompt.

With correct LoRA-aware hashing:
  LoRA-A requests share prefix blocks among themselves    → 1 copy in HBM
  LoRA-B requests share prefix blocks among themselves    → 1 copy in HBM
  Total HBM for prefix: 2 copies (correct — different K/V values)

Without (broken):
  All 100 requests try to share same blocks               → 1 copy in HBM
  LoRA-B requests read LoRA-A's K/V values                → WRONG OUTPUTS
```

---

## 5. LMCache KV-Cache Registration

**Commit**: `b12cb38` — PR #31397
**Files**: `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`, `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py`
**Impact**: Enables RDMA-based zero-copy KV cache transfer between disaggregated prefill/decode nodes

### Before vs. After

**BEFORE**: KV caches discovered lazily during first `start_load_kv` call. Too late for RDMA memory registration.

```python
# No register_kv_caches method existed.
# Discovery happened inside _init_kv_caches_from_forward_context() during first forward pass.
```

**AFTER**: Explicit registration at model init time.

```python
# vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:110-124
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    """Pre-register KV Caches for RDMA (e.g., NIXL)."""
    if hasattr(self._lmcache_engine, "register_kv_caches"):
        self._lmcache_engine.register_kv_caches(kv_caches)

# vllm/distributed/kv_transfer/kv_connector/v1/lmcache_integration/vllm_v1_adapter.py:785-794
def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
    self.kv_caches = kv_caches
    if self.lmcache_engine is not None:
        kvcaches = list(self.kv_caches.values())
        self.lmcache_engine.post_init(kvcaches=kvcaches)  # Pin GPU memory for RDMA
```

### Why This Matters: Disaggregated Prefill/Decode (P/D)

```
Node A (Prefill):                   Node B (Decode):
  GPU HBM:                           GPU HBM:
  ┌────────────────┐                 ┌────────────────┐
  │ KV Cache Blocks │ ──RDMA──────→  │ KV Cache Blocks │
  │ (REGISTERED)    │  zero-copy     │ (REGISTERED)    │
  └────────────────┘  GPU-to-GPU     └────────────────┘

Without register_kv_caches: RDMA can't pin the buffers → falls back to staged copy:
  GPU → CPU → Network → CPU → GPU  (4 copies, ~10x slower)

With register_kv_caches: RDMA pins GPU HBM regions at init:
  GPU → Network → GPU  (1 copy via GPUDirect RDMA, ~100 Gbps on InfiniBand)
```

### Where on the GPU

- `register_kv_caches` calls `post_init(kvcaches=...)` which internally calls `ibv_reg_mr()` (InfiniBand) or `cuMemGetAddressRange()` (NIXL) to pin the KV cache tensors in GPU HBM
- After registration, RDMA transfers read/write directly from/to GPU HBM via the NIC's DMA engine, bypassing the CPU entirely
- On H100: NVLink + InfiniBand gives ~400 GB/s inter-node; on B200: NVLink 5 gives ~1.8 TB/s intra-node

---

## 6. AsyncLLM generate/encode Deduplication

**Commit**: `e54ee3e` — PR #31510
**Files**: `vllm/v1/engine/async_llm.py`
**Impact**: Eliminates duplicated validation, fixes a subtle truncation validation bug

### Before vs. After Code

**BEFORE**: `generate()` and `encode()` each independently performed:
1. Output handler startup (`self._run_output_handler()`)
2. Pause condition wait (`await self._pause_cond.wait_for(...)`)
3. `tokenization_kwargs` initialization
4. Truncation validation (`_validate_truncation_size(...)`)

```python
# ──── BEFORE in generate() ────
async def generate(self, ...):
    q = None
    try:
        self._run_output_handler()                             # DUPLICATED
        async with self._pause_cond:                           # DUPLICATED
            await self._pause_cond.wait_for(lambda: not self._paused)
        if tokenization_kwargs is None:                        # DUPLICATED
            tokenization_kwargs = {}
            _validate_truncation_size(...)                     # BUG: inside `if`
        # ... build request, send to engine ...

# ──── BEFORE in encode() ────
async def encode(self, ...):
    q = None
    try:
        self._run_output_handler()                             # DUPLICATED
        async with self._pause_cond:                           # DUPLICATED
            await self._pause_cond.wait_for(lambda: not self._paused)
        if tokenization_kwargs is None:                        # DUPLICATED
            tokenization_kwargs = {}
        _validate_truncation_size(...)                         # CORRECT: outside `if`
```

Note the bug: in `generate()`, `_validate_truncation_size` was INSIDE the `if tokenization_kwargs is None` block. If a caller passed `tokenization_kwargs={"padding": True}`, the validation was skipped entirely.

**AFTER**: All logic unified in `add_request()`.

```python
# ──── AFTER in add_request() ────
async def add_request(self, request_id, prompt, params, ...):
    is_pooling = isinstance(params, PoolingParams)

    if self.vllm_config.cache_config.kv_sharing_fast_prefill   # Moved from generate()
        and not is_pooling and params.prompt_logprobs:
        raise ValueError(...)

    if tokenization_kwargs is None:
        tokenization_kwargs = {}
    _validate_truncation_size(...)    # ALWAYS runs — bug fixed

    self._run_output_handler()        # ONE place
    async with self._pause_cond:      # ONE place
        await self._pause_cond.wait_for(lambda: not self._paused)
    queue = RequestOutputCollector(params.output_kind, request.request_id)
    return queue

# ──── AFTER in generate() ──── (now trivial)
async def generate(self, ...):
    q = await self.add_request(request_id, prompt, sampling_params, ...)
    while not finished:
        out = q.get_nowait() or await q.get()
        yield out
```

### Full Request Flow (HTTP → Token Generation)

```
HTTP POST /v1/chat/completions
    │
    ▼
api_server.py:476 → create_chat_completion(request, raw_request)
    │
    ▼
serving_chat.py:220 → create_chat_completion()
    ├── _check_model()
    ├── get_tokenizer()
    ├── _preprocess_chat() → apply chat template (Jinja2)
    ├── to_sampling_params()
    └── engine_client.generate()  ← enters AsyncLLM
         │
         ▼
async_llm.py:391 → generate()
    └── add_request()  ← ALL validation happens here
         ├── _validate_truncation_size()
         ├── _run_output_handler()
         ├── pause_cond.wait()
         ├── input_processor.process_inputs()  → tokenize + multimodal
         └── engine_core.add_request_async()  → IPC to EngineCore process
              │
              ▼ (ZMQ IPC, msgpack serialization)
         EngineCore.step() loop:
              ├── scheduler.schedule()           → CPU
              ├── model_executor.execute_model()  → GPU
              └── scheduler.update_from_output()  → CPU
                   │
                   ▼ (results flow back via IPC)
         output_handler task → pushes to RequestOutputCollector queue
              │
              ▼
         generate() yields → serving_chat formats SSE / JSON
              │
              ▼
         HTTP response streamed to client
```

---

## 7. Default Chat Template Kwargs

**Commit**: `dc837bc` — PR #31343
**Files**: `vllm/entrypoints/openai/cli_args.py`, `serving_chat.py`, `serving_engine.py`
**Impact**: Server operators can set default reasoning model behavior (e.g., `enable_thinking`)

### Before vs. After Code

**`vllm/entrypoints/openai/serving_engine.py:1180-1189`**

```python
# ──── BEFORE ────
_chat_template_kwargs = dict(
    chat_template=chat_template,
    add_generation_prompt=add_generation_prompt,
    continue_final_message=continue_final_message,
    tools=tool_dicts,
    documents=documents,
)
_chat_template_kwargs.update(chat_template_kwargs or {})    # Request only

# ──── AFTER ────
_chat_template_kwargs = dict(
    chat_template=chat_template,
    add_generation_prompt=add_generation_prompt,
    continue_final_message=continue_final_message,
    tools=tool_dicts,
    documents=documents,
)
if default_chat_template_kwargs:                             # NEW: server defaults
    _chat_template_kwargs.update(default_chat_template_kwargs)
_chat_template_kwargs.update(chat_template_kwargs or {})     # Request overrides
```

Three-layer merge: `base < server_defaults < request_overrides`.

### Use Case: Disabling Thinking Mode by Default

```bash
# Server startup — thinking disabled for all requests by default
vllm serve Qwen/Qwen3-32B \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'

# Client can still opt-in per request:
response = client.chat.completions.create(
    model="Qwen/Qwen3-32B",
    messages=[{"role": "user", "content": "Solve this integral..."}],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)
```

---

## 8. continue\_final\_message for Embeddings

**Commit**: `51085c2` — PR #31497
**Files**: `vllm/entrypoints/pooling/embed/protocol.py`, `embed/serving.py`
**Impact**: Enables embedding of partial/in-progress conversations for RAG

### Before vs. After Code

**`vllm/entrypoints/pooling/embed/serving.py:92`**

```python
# ──── BEFORE ────
continue_final_message=False,    # Hardcoded

# ──── AFTER ────
continue_final_message=ctx.request.continue_final_message,   # User-controlled
```

### What This Changes

```
Without continue_final_message (default):
  Messages: [user: "What is ML?", assistant: "ML is"]
  Template output: "What is ML?<|im_end|>ML is<|im_end|>"
                                                    ^^^^^^ EOS added
  Embedding: represents a COMPLETED conversation

With continue_final_message=True:
  Messages: [user: "What is ML?", assistant: "ML is"]
  Template output: "What is ML?<|im_end|>ML is"
                                                 ← NO EOS
  Embedding: represents an IN-PROGRESS conversation
```

Use case: Retrieval-augmented generation where you embed partial assistant completions for semantic search.

---

## 9. H100 Flash Attention 3 Scheduler Metadata Fix

**Commit**: `d63b969` — PR #31187
**Files**: Test infrastructure (affects `vllm/v1/attention/backends/flash_attn.py` behavior)
**Impact**: Fixes incorrect attention output when batch size decreases between CUDA graph iterations

### The Root Cause

Flash Attention 3 (FA3, Hopper-only) uses Ahead-of-Time (AOT) scheduling. The
`scheduler_metadata` tensor is pre-allocated at max batch size for CUDA graph capture.
When the actual batch size shrinks, trailing entries contain **stale data from the previous iteration**.

**`vllm/v1/attention/backends/flash_attn.py:466-473`**

```python
if self.use_full_cuda_graph and scheduler_metadata is not None:
    n = scheduler_metadata.shape[0]
    self.scheduler_metadata[:n] = scheduler_metadata     # Copy current batch metadata
    # CRITICAL: Zero out stale trailing entries
    self.scheduler_metadata[n:] = 0                      # Without this → corruption
    scheduler_metadata = self.scheduler_metadata[:n]
```

### What Goes Wrong Without the Fix

```
Iteration 1: batch_size=8, scheduler_metadata = [s0, s1, s2, s3, s4, s5, s6, s7]
  FA3 kernel launches 8 CTAs, each reads its scheduler_metadata entry. Correct.

Iteration 2: batch_size=4, scheduler_metadata = [s0', s1', s2', s3']
  Pre-alloc buffer after copy: [s0', s1', s2', s3', s4, s5, s6, s7]
                                                        ^^^^^^^^^^^^^^^^ STALE!
  FA3 kernel MAY launch CTAs for entries 4-7 (stale data ≠ 0)
  Those CTAs compute attention for non-existent sequences
  and WRITE GARBAGE into the output buffer at offsets belonging to real sequences

With zeroing: [s0', s1', s2', s3', 0, 0, 0, 0]
  FA3 kernel sees zeros for entries 4-7 → no CTAs launched for them. Correct.
```

### Where on the GPU

- `scheduler_metadata` lives in **HBM**, copied from CPU via `self.scheduler_metadata[:n] = ...`
- The FA3 kernel's **CTA dispatcher** (running on SMs) reads this tensor at launch
- Each CTA uses Tensor Memory Accelerator (TMA) on H100 to asynchronously load Q/K/V tiles from HBM to SMEM
- Stale metadata causes CTAs to compute with wrong Q/K offsets, writing results to wrong output positions via **HBM writes**

### Why H100-Only

FA3 with AOT scheduling is only available on Hopper (SM 9.0). FA2 on Ampere uses a simpler dispatch:

```python
# flash_attn.py:249-253
_cudagraph_support = (
    AttentionCGSupport.ALWAYS       # FA3: full CUDA graph support (needs metadata zeroing)
    if get_flash_attn_version() == 3
    else AttentionCGSupport.UNIFORM_BATCH  # FA2: only for uniform batches (no metadata)
)
```

---

## 10. ROCm GPTQ GEMM Output Zeroing Race Condition

**Commit**: `3ecfdc3` — PR #30719
**Files**: `csrc/quantization/gptq/q_gemm.cu`
**Impact**: Fixes incorrect model outputs for GPTQ-quantized models on AMD GPUs

### The Root Cause

The GPTQ kernel uses a **3D grid** where `gridDim.z` tiles along the K (reduction) dimension.
Multiple thread blocks write to the SAME output `c[m][n]` using `atomicAdd`.

```cuda
// csrc/quantization/gptq/q_gemm.cu:207-211
auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
auto offset_m = blockIdx.y * m_count;
auto offset_k = blockIdx.z * BLOCK_KN_SIZE;    // K-dimension tiling

// Lines 313-321: Atomic accumulation
for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    atomicAdd(out, result01);      // Adds partial sum to output
    atomicAdd(out + 1, result23);
}
```

`atomicAdd` does `*out += partial_result`. This only produces the correct full dot product
if `*out` starts at **exactly zero**. If the output buffer has residual non-zero values,
the result is `residual_garbage + correct_sum`.

### Before vs. After Code

**`csrc/quantization/gptq/q_gemm.cu:1834`**

```cuda
// ──── BEFORE (reconstructed from the fix context) ────
at::Tensor c = torch::empty({a.size(0), b_q_weight.size(1)}, options);
// torch::empty does NOT zero memory. On ROCm/HIP, freed memory often
// retains non-zero values from previous allocations.

// ──── AFTER ────
at::Tensor c = torch::zeros({a.size(0), b_q_weight.size(1)}, options);
// torch::zeros issues hipMemset on the same stream → guaranteed zero before kernel
```

### Why ROCm-Specific

On NVIDIA CUDA, `torch::empty` *sometimes* returns zeroed memory because:
- The CUDA caching allocator may return a freshly-allocated block that was `cudaMemset`-zeroed by the driver
- Or it returns a cached block that happens to be zero from a previous `torch::zeros` call

On ROCm/HIP with AMD GPUs:
- The HIP memory allocator more aggressively recycles memory without zeroing
- `torch::empty` reliably returns non-zero memory → the bug manifests consistently

### Where on the GPU (MI300X)

```
GPTQ Kernel Execution on MI300X:

1. Dequantization:  int32 packed weights → fp16 values
   ├── INT32 ALUs on CUs: Bitwise extraction (shifts, masks)
   └── Registers: Unpacked fp16 values stored in VGPRs

2. Dot Product:  fp16 token activations × fp16 dequantized weights
   ├── FP16 FMA units (Matrix Core): fused multiply-add
   ├── Shared memory (LDS): Tiles of activations
   └── Registers: Accumulation in fp32

3. atomicAdd to output:
   ├── L2 cache → Infinity Cache (256MB): Coalesced atomic operations
   └── HBM3 backing: Final write-back
   ├── With zeros: 0 + sum_block0 + sum_block1 + ... = correct
   └── With garbage: garbage + sum_block0 + sum_block1 + ... = WRONG
```

---

## 11. QKNorm Optimization for MiniMax-M2

**Commit**: `5bc6641` — PR #31493
**Files**: Likely model-specific attention code for MiniMax-M2/M2.1
**Impact**: Fused QK normalization reduces kernel launches for MiniMax models

QK normalization applies RMSNorm to Q and K tensors before attention. The optimization
fuses this into the existing attention preparation, avoiding separate kernel launches
for each norm operation.

---

## 12. ConvNd Replacement for Torch 2.9 Compatibility

**Commit**: `e37e734` — PR #31498
**Files**: Multiple model files using `nn.ConvNd`
**Impact**: Fixes a Torch 2.9 regression by using vLLM's `ConvNdLayer` wrapper

Torch 2.9 introduced a change in `nn.Conv2d`/`nn.Conv3d` behavior that broke some model
loading paths. vLLM's `ConvNdLayer` is a compatibility wrapper that works across Torch versions.

This directly affects **Qwen3-VL** — the vision encoder's `Qwen3_VisionPatchEmbed` uses `Conv3dLayer`:

```python
# vllm/model_executor/models/qwen3_vl.py:155
self.proj = Conv3dLayer(        # ← was nn.Conv3d, now ConvNdLayer wrapper
    in_channels, hidden_size,
    kernel_size=(temporal_patch_size, patch_size, patch_size),
    stride=(temporal_patch_size, patch_size, patch_size),
    bias=True,
)
```

---

## 13. The Qwen3-VL Model: Full Code Walkthrough

**File**: `vllm/model_executor/models/qwen3_vl.py` (2121 lines)

### Architecture at a Glance

```
              ┌─────────────────────────────────────┐
              │        Qwen3-VL Forward Pass         │
              ├─────────────────────────────────────┤
              │                                     │
  Image ──→   │  1. Patch Embed (Conv3D)             │
  [H,W,3]     │     16×16 patches → 1152-dim         │
              │                                     │
              │  2. Position Embed (learned + RoPE)   │
              │     nn.Embedding + bilinear interp    │
              │                                     │
              │  3. ViT Transformer (27 layers)       │
              │     ├─ Layer  8 → DeepStack Merger 0  │
              │     ├─ Layer 16 → DeepStack Merger 1  │
              │     ├─ Layer 24 → DeepStack Merger 2  │
              │     └─ Layer 27 → Main Merger         │
              │                                     │
              │  4. Spatial Merge (2×2 → 4× reduce)   │
              │                                     │
              │  5. Concatenate: [main, ds0, ds1, ds2]│
              │     Output: out_hidden × 4            │
              ├─────────────────────────────────────┤
              │  6. Split into main + deepstack embeds│
              │                                     │
  Text ──→    │  7. LLM Embed + merge main visual     │
  tokens      │                                     │
              │  8. LLM Layers 0-63:                  │
              │     Layer 0: hidden += deepstack[0]   │
              │     Layer 1: hidden += deepstack[1]   │
              │     Layer 2: hidden += deepstack[2]   │
              │     Layers 3-63: standard transformer │
              │                                     │
              │  9. LM Head → logits → sample token   │
              └─────────────────────────────────────┘
```

### DeepStack Extraction — The Key Innovation Over Qwen2-VL

**`qwen3_vl.py:546-594` — ViT forward with feature extraction at layers 8, 16, 24:**

```python
def forward(self, x, grid_thw):
    hidden_states = self.patch_embed(x)
    hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, ...)

        # Extract intermediate features at deepstack layers
        if layer_num in self.deepstack_visual_indexes:   # [8, 16, 24]
            idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)  # Final layer output

    # Concat: [final_features, layer8_features, layer16_features, layer24_features]
    hidden_states = torch.cat([hidden_states] + deepstack_feature_lists, dim=1)
    return hidden_states
```

**`qwen3_vl.py:1134-1183` — DeepStack injection into LLM layers:**

```python
class Qwen3LLMModel(Qwen3Model):
    def forward(self, input_ids, positions, inputs_embeds=None, deepstack_input_embeds=None):
        hidden_states = inputs_embeds or self.embed_input_ids(input_ids)
        residual = None

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)

            # DEEPSTACK INJECTION at LLM layers 0, 1, 2
            if deepstack_input_embeds is not None and layer_idx < len(deepstack_input_embeds):
                hidden_states = hidden_states + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
```

### Key Differences from Qwen2-VL

| Feature | Qwen2-VL | Qwen3-VL |
|---------|----------|----------|
| Visual feature injection | Single depth (input layer only) | Multi-depth: ViT layers 8,16,24 → LLM layers 0,1,2 |
| ViT position embeddings | Rotary only | Learned `nn.Embedding` + bilinear interpolation + Rotary |
| Patch embed bias | `bias=False` | `bias=True` |
| ViT MLP activation | QuickGELU | SiLU |
| Video timestamps | Scalar `second_per_grid_ts` | Explicit `<X.Y seconds>` text tokens per frame |
| torch.compile | Not supported | `@support_torch_compile` on LLM model |
| MoE variant | None | `Qwen3VLMoeForConditionalGeneration` (235B-A22B) |
| EVS (video pruning) | Not supported | Cosine-similarity pruning + MRoPE position recomputation |
| Eagle3 (speculative) | Not supported | Auxiliary hidden state collection at layers 2, mid, n-3 |

---

## 14. Attention Backend Selection: 18 Backends, One Hot Path

**File**: `vllm/attention/backends/registry.py` — 18 registered backends

### CUDA/NVIDIA Backend Priority

```python
# vllm/platforms/cuda.py:44-82
# Blackwell (B200, SM 10.0):
priority = [FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION]

# Hopper (H100, SM 9.0):
priority = [FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION]

# MLA models (DeepSeek-V3) on Blackwell:
priority = [CUTLASS_MLA, FLASHINFER_MLA, FLASH_ATTN_MLA, FLASHMLA, TRITON_MLA]

# MLA models on Hopper:
priority = [FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA]
```

### ROCm/AMD Backend Priority

```python
# vllm/platforms/rocm.py:190-307
if AITER_UNIFIED_ATTENTION:  return ROCM_AITER_UNIFIED_ATTN
elif AITER_MHA + gfx9:       return ROCM_AITER_FA
elif PREFILL_DECODE_SPLIT:   return ROCM_ATTN
elif AITER + gfx9:           return ROCM_AITER_FA
else:                        return TRITON_ATTN    # Default fallback
```

### Flash Attention: Prefill vs. Decode in One Kernel

**`vllm/v1/attention/backends/flash_attn.py:700-722`**

```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens],
    k=key_cache,               # Paged KV cache in HBM
    v=value_cache,
    cu_seqlens_q=cu_seqlens_q, # Cumulative query lengths per sequence
    max_seqlen_q=max_seqlen_q, # 1 for decode, >>1 for prefill
    seqused_k=seqused_k,       # Actual KV lengths (may differ from allocated blocks)
    max_seqlen_k=max_seqlen_k,
    block_table=block_table,   # [batch_size, max_blocks] — paged virtual→physical
    causal=True,
    fa_version=self.vllm_flash_attn_version,  # FA2 or FA3
)
```

The same kernel handles both phases:
- **Prefill**: `max_seqlen_q >> 1`, tiles Q into blocks, uses standard FlashAttention algorithm
- **Decode**: `max_seqlen_q = 1`, uses "FlashDecoding" (split-K across the KV dimension for parallelism)

---

## 15. KV Cache Architecture: Paged Memory on the GPU

### Block Data Structure (CPU metadata)

**`vllm/v1/core/kv_cache_utils.py:107-153`**

```python
@dataclass
class KVCacheBlock:
    block_id: int              # Index into GPU tensor (0..num_gpu_blocks-1)
    ref_cnt: int = 0           # Reference counting for sharing
    _block_hash: BlockHashWithGroupId | None = None  # For prefix cache lookup
    prev_free_block: KVCacheBlock | None = None      # Doubly-linked list
    next_free_block: KVCacheBlock | None = None
```

### GPU Tensor Layout

**`vllm/v1/attention/backends/flash_attn.py:103-112`**

```python
# Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
#         K/V  ~20000     16        8 (GQA)       128
# Total for Qwen3-VL-32B on H100 TP=2:
#   2 × 20000 × 16 × 8 × 128 × 2 bytes (BF16) = ~1.3 GB per layer
#   × 64 layers = ~83 GB across both GPUs (41.5 GB each)
```

### Block Table: Virtual → Physical Mapping

**`vllm/v1/worker/gpu/block_table.py:13-42`**

```python
# Shape: [max_num_reqs, max_num_blocks_per_req], dtype=int32, on GPU
# Example for 3 requests:
# req0: [block_2, block_5, block_8, 0, 0]     ← 3 blocks allocated
# req1: [block_2, block_5, block_12, 0, 0]    ← shares prefix blocks 2,5 with req0!
# req2: [block_7, block_3, block_1, 0, 0]     ← completely different blocks
```

### The Full Memory Hierarchy Path

```
Flash Attention kernel needs K[seq_pos=42] for request 1:

1. Logical block index: 42 // 16 = block 2 within request 1's sequence
2. Read block_table[1][2] from GPU HBM → physical block_id = 12
3. Physical address: key_cache[12, 42 % 16, :, :] = key_cache[12, 10, 8, 128]
4. Load K vector (128 × 2 bytes = 256 bytes) from HBM through L2 cache
5. Store in shared memory tile for Tensor Core consumption
6. Tensor Core computes Q × K^T dot product → attention score
```

---

## 16. GPU Execution Map: Where Every Operation Runs

### Hardware Resources on H100 and B200

| Resource | H100 SXM | B200 HGX | Used By |
|----------|----------|----------|---------|
| SMs | 132 | 148 | All kernel dispatch |
| Tensor Cores | 528 (4th gen) | 528 (5th gen) | GEMM, attention Q×K, S×V |
| CUDA Cores | 16,896 | 16,896 | Activations, sampling, elementwise |
| Shared Memory | 228KB/SM | 228KB/SM | FlashAttention tiles, FusedMoE token tiles |
| L2 Cache | 50MB | ~100MB | KV cache hot blocks, weight reuse |
| HBM | 80GB, 3.35 TB/s | 180GB, 7.7 TB/s | Model weights, KV cache, activations |
| NVLink | 900 GB/s | 1800 GB/s | Tensor Parallel all-reduce |
| Tensor Memory (TMEM) | N/A | 5th gen dedicated | Tensor Core register file |

### Per-Operation GPU Mapping

| Operation | GPU Unit | Memory Tier | Bound By |
|-----------|----------|-------------|----------|
| **Prefill: QKV projection** | Tensor Cores (GEMM) | Weights in HBM, tiles in SMEM | Compute |
| **Prefill: Flash Attention** | Tensor Cores (Q×K, S×V) | Q,K,V tiles in SMEM | Compute |
| **Prefill: Softmax** | SM registers + SMEM | Online algorithm, never materializes N×N | Compute |
| **Prefill: MLP (up+gate, down)** | Tensor Cores (GEMM) | Weights in HBM, tiles in SMEM | Compute |
| **Decode: QKV projection** | Tensor Cores | Single token × weights | Memory BW |
| **Decode: Flash Attention** | Tensor Cores (split-K) | KV cache blocks from HBM | Memory BW |
| **Decode: MLP** | Tensor Cores | Single token × weights | Memory BW |
| **FusedMoE: Token permutation** | CUDA cores | sorted_token_ids in HBM | Latency |
| **FusedMoE: Expert GEMM** | Tensor Cores | Expert weights in HBM, L2 reuse | Memory BW (decode) / Compute (prefill) |
| **FusedMoE: SiLU-and-Mul** | CUDA cores | Intermediate buffers in HBM | Memory BW |
| **FusedMoE: atomicAdd (GPTQ)** | L2 cache atomic unit | Output buffer in HBM | Atomic BW |
| **KV cache write** | DMA (reshape_and_cache) | slot_mapping in HBM | Memory BW |
| **KV cache read** | TMA on H100 / DMA on B200 | Block table + KV blocks | Memory BW |
| **TP all-reduce** | NVLink DMA engines | Inter-GPU via NVLink | NVLink BW |
| **Sampling (top-k/top-p)** | CUDA cores | Logits in HBM | Latency |
| **ViT patch embedding** | Tensor Cores (Conv3D GEMM) | Image pixels in HBM | Compute |
| **DeepStack injection** | CUDA cores (elementwise add) | Hidden states in HBM | Memory BW |
| **MRoPE** | CUDA cores (sin/cos multiply) | Position IDs in HBM | Memory BW |
| **GPTQ dequantization** | INT32 ALUs (bitwise ops) | Packed weights in HBM | Memory BW |
| **RDMA KV transfer** | NIC DMA engine | HBM (pinned) → network | Network BW |

### The Decode Bottleneck: Why Memory Bandwidth Dominates

During decode, each new token requires reading the ENTIRE model weights plus KV cache:

```
Qwen3-VL-32B decode — one token:
  Read model weights: ~64 GB (BF16) ÷ TP_degree
  Read KV cache:      seq_len × 2 × num_layers × num_kv_heads × head_dim × 2 bytes
                      = 4096 × 2 × 64 × 8 × 128 × 2 = 1.07 GB (for 4K context)
  Total read:         ~33 GB (TP=2) for weights + 1 GB for KV = ~34 GB

  H100 at 3.35 TB/s: 34 GB / 3.35 TB/s = ~10 ms per token = ~100 tok/s
  B200 at 7.7 TB/s:  34 GB / 7.7 TB/s  = ~4.4 ms per token = ~230 tok/s

  With FP8: weights = 16.5 GB → ~18 GB total
  H100: 18 GB / 3.35 TB/s = ~5.4 ms = ~185 tok/s
  B200: 18 GB / 7.7 TB/s  = ~2.3 ms = ~430 tok/s

  With FP4 (B200 only): weights = 8.25 GB → ~9.3 GB total
  B200: 9.3 GB / 7.7 TB/s = ~1.2 ms = ~830 tok/s
```

This is why the B200's 2.3x memory bandwidth advantage translates almost directly to 2.3x decode speed — decode is purely memory-bandwidth bound.

---

## 17. Future Use Cases and System Design Implications

### 1. Disaggregated Prefill/Decode (P/D Separation)

**What it is**: Run prefill on one set of GPUs, decode on another.

**Why**: Prefill is compute-bound (batch of tokens × big GEMM). Decode is memory-bandwidth-bound (1 token × full weight read). Mixing them on the same GPU means neither phase runs optimally.

**How vLLM enables it**: The `register_kv_caches` change (#31397) + NIXL connector + prefix cache events (#27577) create the infrastructure for P/D separation:

```
Prefill Node (compute-optimized, e.g., B200 at FP4):
  → Process 128K token prompts at 15K+ tok/s
  → Transfer KV cache via RDMA to decode node

Decode Node (bandwidth-optimized, e.g., H100 with max HBM):
  → Generate tokens at memory-bandwidth speed
  → KV cache pre-registered for zero-copy RDMA receive

Future: B200 handles BOTH better due to 2.3x higher bandwidth AND compute.
Eventually, disaggregation may become less important on Blackwell.
```

### 2. Multi-LoRA at Scale with Correct Prefix Caching

**What it is**: Serve 100+ LoRA adapters simultaneously with shared base model.

**How vLLM enables it**: LoRA-aware prefix cache hashing (#27577) + Punica LoRA wrapper (#31408):

```
Base model: Qwen3-VL-32B (64GB in HBM)
LoRA adapters: 100 adapters × ~100MB each = 10GB (CPU memory, loaded on demand)

System prompt: "You are a helpful assistant" (shared across ALL LoRA requests)
  → BUT KV cache differs per LoRA (different attention projections)
  → LoRA-aware hashing ensures correct prefix sharing within same adapter

Future: Dynamic LoRA multiplexing — hundreds of concurrent users, each with
different fine-tuned behavior, served from ONE GPU. This is the path to
SaaS-model-as-a-service at scale.
```

### 3. The Stage Graph: Any-to-Any Multimodal Pipelines

**What it is**: vLLM-Omni's stage graph decomposes multi-model pipelines into independently served stages.

**Future system design**:

```
Example: Customer Support AI Pipeline

Stage 1: Qwen3-VL-32B (Vision + Text Understanding)
  GPU allocation: 2x B200 (TP=2)
  Input: Customer screenshot + text complaint
  Output: Understanding tokens → routing

Stage 2: Qwen3-32B (Reasoning + Response)
  GPU allocation: 1x B200
  Input: Understanding tokens
  Output: Text response + action commands

Stage 3: Qwen3-TTS (Voice Synthesis)
  GPU allocation: 1x B200
  Input: Text response
  Output: Audio stream (RTF 0.60)

Stage 4: FLUX.1-dev + Brand LoRA (Visual Response)
  GPU allocation: 1x B200 (with CPU offloading)
  Input: Action commands
  Output: Generated product image / diagram

All 4 stages run simultaneously via async chunk pipeline overlap.
End-to-end latency: NOT sum of all stages, but approximately max(stage latencies).
```

### 4. MLPerf-Scale VLM Serving

**What**: Qwen3-VL-235B-A22B as the MLPerf v6.0 reference, processing 40M products/day (Shopify).

**System design for 40M products/day**:

```
Requirement: 40M products ÷ 86,400 seconds = 463 requests/second average
Peak (Black Friday): ~10x = 4,630 requests/second
p99 latency: ≤ 12 seconds

With 8xB200 (FP8, TP=8):
  Estimated throughput: ~50 requests/second sustained (complex VLM, long outputs)
  Clusters needed: 463 / 50 = ~10 clusters for average
  Peak: ~93 clusters (or auto-scale on cloud)
  Total GPUs: 80 B200s (average) to 744 B200s (peak)
  Monthly cost: ~$500K (average) to ~$4.6M (peak Black Friday)

vs. Human annotation:
  200 annotators × 200 products/day = 40K/day → need 200x MORE humans
  Or: 200,000 annotators × $40K/year = $8B/year
  AI cost: $6M/year (average) → 1,333x cheaper
```

### 5. FP4 on Blackwell: The Precision Frontier

**What**: B200's NVFP4 delivers 18 petaFLOPS — 4.5x more than H100's FP8.

**System design implication**: Models that were "too big" become servable:

```
Qwen3-VL-32B at FP4 on single B200:
  Weights: ~16GB (vs 64GB BF16)
  Remaining HBM: 164GB for KV cache
  Decode speed: ~830 tok/s (vs ~100 tok/s on H100 BF16)

The quality impact of FP4 is model-dependent. The 2nd-gen Transformer Engine
auto-selects FP4 for layers where it's safe and FP8/FP16 for sensitive layers.
This per-layer granularity means quality loss is minimal for well-trained models.

Future: FP4-aware training (quantization-aware fine-tuning) will close the gap further.
```

### 6. Async Everything: The Scheduling Paradigm Shift

**What it means for system design**: The async scheduling default (#27614) is the beginning of
a broader trend — eliminating ALL synchronization points between CPU and GPU.

```
Today's async path:
  schedule(N+1) overlaps with forward(N) on GPU

Future async path:
  schedule(N+2) overlaps with forward(N+1) overlaps with D2H(N) overlaps with ...
  The CPU is NEVER idle. The GPU is NEVER idle. Every cycle is productive.

  This is essentially pipelining for inference — the same concept that made
  CPUs fast in the 1990s, applied to GPU scheduling.

Incompatible features (currently):
  - Pipeline parallelism (PP > 1) — has its own pipeline
  - Most speculative decoding — needs actual token values on CPU
  - Exception: Eagle-based spec decode already works with async scheduling

Future: Async scheduling + speculative decoding + disaggregated P/D + stage graphs
= a fully pipelined, fully disaggregated, fully overlapped inference system.
```

### 7. The Multi-Backend World

**What it means**: 18 attention backends mean vLLM runs everywhere — NVIDIA, AMD, Intel, Huawei.

```
Backend selection is already automatic and priority-based:
  B200 → FlashInfer (optimized for SM 10.0)
  H100 → Flash Attention (optimized for SM 9.0, FA3 AOT scheduling)
  MI300X → AITER Flash Attention (optimized for CDNA3, Infinity Cache)
  Intel Max → IPEX Attention (optimized for Xe cores)

Future: Each hardware vendor optimizes their backend independently.
vLLM's abstraction layer means model code NEVER changes — only the backend selection.
This is the "write once, run everywhere" promise for AI inference.
```

---

*This document analyzes every significant recent commit in the vLLM repository, traces each change
to the exact source code (with file paths and line numbers), shows before/after diffs, maps execution
to GPU hardware, and projects implications for future system design.*
