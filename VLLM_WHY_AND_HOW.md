# vLLM: Why It Exists and How It Works

**A code-grounded, why-driven guide for CEOs and engineers**

---

## Table of Contents
- [A) WHY vLLM Exists](#a-why-vllm-exists)
- [B) Foundational Concepts](#b-foundational-concepts)
- [C) Repo Map](#c-repo-map)
- [D) One Request Walkthrough](#d-one-request-walkthrough)
- [E) ASCII Diagrams](#e-ascii-diagrams)
- [F) Practical Takeaways](#f-practical-takeaways)

---

# Core Questions Answered First

## 1) Why is vLLM needed at all? What breaks with "normal" inference stacks?

**The Problem**: LLM inference is fundamentally different from traditional ML serving:

| Traditional ML | LLM Inference |
|---------------|---------------|
| Fixed input size | Variable-length sequences |
| Single forward pass | Hundreds of autoregressive steps |
| Memory proportional to batch | Memory proportional to `batch × sequence_length × layers` |
| Compute-bound | Memory-bandwidth bound (decode) |

**What Breaks**:
1. **Memory Explosion**: Each token requires storing Key/Value vectors across ALL layers. A 70B model with 4K context needs ~1.3GB KV cache per request.
2. **Fragmentation**: Variable-length sequences + dynamic batching = massive memory waste (60-80% in naive implementations).
3. **Underutilization**: GPUs sit idle during decode because they're waiting for memory, not compute.

**Why Normal Stacks Collapse**:
```
Without vLLM:                           With vLLM:
┌─────────────────────────────┐        ┌─────────────────────────────┐
│ Request 1: Reserve 4K slots │        │ Request 1: 3 blocks (48 tok)│
│ (using only 200 tokens)     │        │ Request 2: 2 blocks (32 tok)│
│ Request 2: Reserve 4K slots │        │ Request 3: 4 blocks (64 tok)│
│ (using only 150 tokens)     │        │ Free: 41 blocks             │
│ Request 3: OOM - no memory! │        │ ... can fit 20+ more reqs   │
└─────────────────────────────┘        └─────────────────────────────┘
  95% of reserved memory wasted          ~4% fragmentation
```

## 2) What is the *real bottleneck* in LLM serving?

**Answer: Memory bandwidth during decode, amplified by fragmentation.**

```
DECODE STEP ANALYSIS (per token generated):
┌────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Operation                    | Compute (FLOPs) | Memory (bytes)  │
│   ─────────────────────────────┼─────────────────┼─────────────────│
│   Load KV cache (N tokens)     | 0               | N × 2 × L × H   │
│   Q·K attention scores         | N × H           | small           │
│   Softmax + V weighted sum     | N × H           | small           │
│   FFN forward pass             | 8 × H²          | weight load     │
│                                                                     │
│   For Llama-70B at 2K context:                                     │
│   - Memory bandwidth: ~60 GB per token (KV + weights)              │
│   - Compute: ~140 GFLOPs per token                                 │
│   - A100 80GB: 2TB/s bandwidth, 312 TFLOPs                        │
│   - Memory time: 30ms, Compute time: 0.5ms → 98% memory-bound!    │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

**The Cascade Effect**:
1. Pre-allocated memory → fewer concurrent requests → low batch size
2. Low batch size → can't amortize weight loading → even more memory-bound
3. Memory-bound decode → GPU utilization <20%
4. Low utilization → poor cost efficiency → $$$

## 3) What does PagedAttention change about the memory model?

**Core Innovation**: Apply OS virtual memory concepts to KV cache.

| Traditional | PagedAttention |
|-------------|----------------|
| Contiguous allocation per request | Non-contiguous blocks |
| Pre-allocate max length | Allocate on-demand |
| No sharing possible | Share blocks via reference counting |
| Internal + external fragmentation | Near-zero fragmentation (~4%) |

**From the paper** ([arXiv:2309.06180](https://arxiv.org/abs/2309.06180)):
> "We find that existing systems waste 60%-80% of memory due to fragmentation and over-reservation."

**Implementation** (from `vllm/v1/core/block_pool.py`):
```python
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    The free_block_queue stores free blocks in eviction order.
    The cached_block_hash_to_block maps block hash → cached block
    for prefix caching.
    """
    def __init__(self, num_gpu_blocks, enable_caching, ...):
        # All blocks pre-allocated at startup
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        # LRU queue for eviction (O(1) operations)
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # Hash table for prefix cache lookup (O(1))
        self.cached_block_hash_to_block = BlockHashToBlockMap()
```

## 4) How does vLLM compare to alternatives?

### vLLM vs TensorRT-LLM / FasterTransformer

| Aspect | TensorRT-LLM | vLLM |
|--------|--------------|------|
| **Focus** | Kernel-level optimization | System-level memory management |
| **KV Cache** | Contiguous, pre-allocated | Paged, on-demand |
| **Batching** | Static or semi-dynamic | Continuous (iteration-level) |
| **Prefix Caching** | Limited | First-class, hash-based |
| **Flexibility** | Compile-time optimization | Runtime flexibility |
| **Best For** | Known workloads, max kernel perf | Variable workloads, multi-tenant |

**TL;DR**: TensorRT optimizes *within* a batch; vLLM optimizes *across* batches and time.

### vLLM vs SGLang

| Aspect | SGLang | vLLM |
|--------|--------|------|
| **Prefix Reuse** | RadixAttention (radix tree) | Hash-based block cache |
| **Granularity** | Token-level radix tree | Block-level (16 tokens default) |
| **Memory Model** | Radix tree + paging | Pure paging |
| **Scheduling** | Prefix-aware | Cache-aware + continuous batching |
| **Strength** | Complex multi-turn, branching | High concurrency, diverse workloads |

**Key Difference**: SGLang's radix tree excels at structured prefix reuse (e.g., tool definitions). vLLM's hash-based approach is simpler, faster for lookups, but coarser-grained.

### vLLM vs "Manual" PyTorch Serving

| Aspect | PyTorch (naive) | vLLM |
|--------|-----------------|------|
| **Batching** | Request-level | Iteration-level (continuous) |
| **KV Management** | Manual, contiguous | Automatic, paged |
| **Concurrency** | 1-10 requests | 100-1000 requests |
| **Memory Efficiency** | 20-40% | 95%+ |
| **GPU Utilization** | 10-30% | 50-80% |

**Why PyTorch Collapses**: 
- No native KV cache management
- Padding to max length in batch
- Requests must wait for entire batch to complete

## 5) Trade-offs: When vLLM is NOT the best choice

| Scenario | Better Alternative | Why |
|----------|-------------------|-----|
| Single long sequence (1M+ tokens) | Custom kernel optimization | Paging overhead, single-sequence bandwidth |
| Fixed, predictable workload | TensorRT-LLM | Compile-time optimization wins |
| Complex multi-turn with branching | SGLang | Radix tree more efficient |
| Embedding/encoding only | ONNX/TensorRT | No decode phase |
| Training/fine-tuning | PyTorch/DeepSpeed | vLLM is inference-only |
| Edge deployment (tiny models) | llama.cpp / ONNX | vLLM overhead not worth it |

---

# A) WHY vLLM Exists (CEO-Level Summary)

## The 15 Bullets

1. **LLM inference is memory-bound, not compute-bound** during decode. GPUs wait for memory 90%+ of the time.

2. **KV cache is the bottleneck**: Every token generated requires loading ALL previous KV vectors. At 70B scale, this is ~320KB per token per layer.

3. **Traditional systems waste 60-80% of GPU memory** due to pre-allocation and fragmentation.

4. **PagedAttention solves fragmentation** by treating KV cache like OS virtual memory—non-contiguous blocks allocated on-demand.

5. **Result: 2-4x more concurrent requests** on the same hardware = 2-4x cost efficiency.

6. **Continuous batching eliminates idle time**: New requests start immediately when capacity frees, no waiting for batch completion.

7. **Prefix caching skips redundant compute**: Shared system prompts, RAG documents, tool definitions compute once, reuse many times.

8. **TTFT (Time to First Token) drops 50-90%** for workloads with shared prefixes.

9. **Throughput increases 2-4x** through better memory utilization and batching.

10. **Works with any HuggingFace model** via a unified interface—no custom optimization needed.

11. **Production-proven at scale**: Used by Anyscale, Databricks, Modal, and many LLM API providers.

12. **OpenAI-compatible API** for drop-in replacement of existing integrations.

13. **Supports multi-GPU (TP/PP)**, quantization, LoRA, speculative decoding, and structured output.

14. **Open source (Apache 2.0)** with active community and frequent releases.

15. **Not a silver bullet**: Overhead for simple workloads, radix approaches may win for specific patterns.

## Comparison Table

```
┌─────────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐
│ Feature             │ vLLM           │ TensorRT-LLM   │ SGLang         │ PyTorch Manual │
├─────────────────────┼────────────────┼────────────────┼────────────────┼────────────────┤
│ Memory Efficiency   │ ★★★★★ 95%+    │ ★★★☆☆ 60-70%  │ ★★★★☆ 85-90%  │ ★★☆☆☆ 20-40%  │
│ Max Throughput      │ ★★★★☆ High    │ ★★★★★ Highest │ ★★★★☆ High    │ ★★☆☆☆ Low     │
│ Prefix Reuse        │ ★★★★☆ Block   │ ★★☆☆☆ Limited │ ★★★★★ Token   │ ☆☆☆☆☆ None    │
│ Flexibility         │ ★★★★★ High    │ ★★☆☆☆ Low     │ ★★★★☆ Medium  │ ★★★★★ Highest │
│ Setup Complexity    │ ★★★★☆ Easy    │ ★★☆☆☆ Hard    │ ★★★★☆ Easy    │ ★★★☆☆ Manual  │
│ Multi-tenant        │ ★★★★★ Best    │ ★★☆☆☆ Poor    │ ★★★★☆ Good    │ ★☆☆☆☆ Hard    │
│ Variable Workloads  │ ★★★★★ Best    │ ★★☆☆☆ Poor    │ ★★★★☆ Good    │ ★★☆☆☆ Poor    │
└─────────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘
```

## Decision Rules

**Choose vLLM when:**
- ✅ Serving multiple concurrent users with variable-length requests
- ✅ Multi-tenant environment (API service, chatbot)
- ✅ Shared prefixes across requests (system prompts, RAG)
- ✅ Need to maximize requests/GPU/dollar
- ✅ Want drop-in OpenAI-compatible API
- ✅ Using HuggingFace models without custom optimization

**Consider alternatives when:**
- ⚠️ Fixed, predictable batch sizes → TensorRT-LLM
- ⚠️ Complex multi-turn with branching → SGLang
- ⚠️ Single very long sequence → custom optimization
- ⚠️ Latency-critical single request → dedicated instance
- ⚠️ Edge deployment with tiny models → llama.cpp

---

# B) Foundational Concepts

## Prefill vs Decode: What Happens, Why It Matters

### Prefill Phase (Prompt Processing)

```
INPUT: "What is the capital of France?"
       ↓ Tokenize
       [1234, 567, 89, 1011, 12, 1314]  (6 tokens)
       ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         PREFILL                                      │
├─────────────────────────────────────────────────────────────────────┤
│  • Process ALL 6 tokens in ONE forward pass                         │
│  • Compute attention: each token attends to all previous            │
│  • Generate KV cache: K[6×heads×dim], V[6×heads×dim] per layer     │
│  • Compute-bound: matrix multiplications dominate                   │
│  • Output: first generated token + full KV cache                    │
└─────────────────────────────────────────────────────────────────────┘
       ↓
OUTPUT: Token "Paris" + KV cache for 6 tokens
```

### Decode Phase (Token Generation)

```
FOR EACH NEW TOKEN:
┌─────────────────────────────────────────────────────────────────────┐
│                         DECODE STEP N                                │
├─────────────────────────────────────────────────────────────────────┤
│  • Process 1 new token                                               │
│  • Load KV cache for ALL N-1 previous tokens (HUGE memory read)     │
│  • Compute attention: new token attends to all previous             │
│  • Append new K, V to cache                                         │
│  • Memory-bound: waiting for KV cache load dominates                │
│  • Output: next token                                                │
└─────────────────────────────────────────────────────────────────────┘
```

### Why This Matters

| Phase | Tokens/Step | Compute/Memory | Optimization Focus |
|-------|------------|----------------|-------------------|
| Prefill | Many (100-10K) | Compute-bound | Kernel efficiency, tensor parallelism |
| Decode | One | Memory-bound | Batching (amortize weight load), KV efficiency |

**The vLLM Insight**: 
- Prefill is efficient naturally (parallel)
- Decode needs BATCHING to amortize memory loads
- Batching requires MEMORY EFFICIENCY to fit more requests
- Memory efficiency requires PAGING to avoid fragmentation

## KV Cache: What It Is, Why It Explodes

### What Is Stored

For each layer `l`, each attention head `h`, and each token position `t`:
```
K[l][h][t] = key_projection(hidden_state[t])    # shape: [head_dim]
V[l][h][t] = value_projection(hidden_state[t])  # shape: [head_dim]
```

### Size Calculation

```
KV Cache Size = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes × num_tokens

Example: Llama-2-70B at 4K context
  = 2 × 80 × 8 × 128 × 2 bytes × 4096 tokens
  = 1.34 GB per request

Example: Llama-3-70B at 128K context  
  = 2 × 80 × 8 × 128 × 2 bytes × 131072 tokens
  = 42.9 GB per request (!)
```

### Why It's the Limiting Factor

```
Memory Budget Breakdown (A100 80GB):
┌────────────────────────────────────────────────────────────────┐
│ Component           │ Size        │ What It Limits             │
├─────────────────────┼─────────────┼────────────────────────────┤
│ Model Weights       │ ~35 GB      │ Fixed (Llama-70B FP16)     │
│ Activations/Grad    │ ~5 GB       │ Per forward pass           │
│ CUDA Overhead       │ ~2 GB       │ Fixed                      │
│ ─────────────────── │ ─────────── │ ────────────────────────── │
│ Available for KV    │ ~38 GB      │ = 28 requests at 4K        │
│                     │             │ = 0.9 requests at 128K (!!)│
└────────────────────────────────────────────────────────────────┘
```

**Without paging**: Reserve 128K per request → <1 concurrent request possible
**With paging**: Allocate as needed → many requests can share the 38GB pool

## Fragmentation: What It Is, Production Symptoms

### Internal Fragmentation

```
Request allocates 4K slots, uses only 500:
┌──────────────────────────────────────────────────────────────────┐
│ ███████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ ← 500 used →   ← 3500 wasted (reserved but empty) →             │
└──────────────────────────────────────────────────────────────────┘
                      87.5% waste!
```

### External Fragmentation

```
After many requests complete, memory looks like:
┌────┬────────┬────┬──────────┬────┬────────────┬────┬───────────┐
│Used│  Free  │Used│   Free   │Used│    Free    │Used│   Free    │
│100 │  200   │150 │   300    │200 │    400     │100 │   250     │
└────┴────────┴────┴──────────┴────┴────────────┴────┴───────────┘
         Total free: 1150 tokens... but largest contiguous: 400
         Can't fit a 500-token request even though 1150 are "free"!
```

### Production Symptoms

| Symptom | Likely Cause |
|---------|-------------|
| OOM despite low KV cache "usage" | External fragmentation |
| Throughput drops over time | Memory becomes fragmented |
| Latency variance increases | Waiting for defragmentation |
| Low GPU util with high memory | Internal fragmentation |

## Paged KV: How It Avoids Fragmentation

### The Solution

```
Instead of:  Request → Contiguous 4K slot reservation
Use:         Request → Block Table → Non-contiguous 16-token blocks

Block Table (per request):
┌─────────────────────────────────────────────────────────────────┐
│ Logical Block 0 → Physical Block 42                            │
│ Logical Block 1 → Physical Block 17                            │
│ Logical Block 2 → Physical Block 89                            │
│ ... (allocated on demand as tokens generated)                  │
└─────────────────────────────────────────────────────────────────┘

GPU Memory:
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ B0   │ B1   │ B2   │ B3   │ B4   │ B5   │ B6   │ B7   │ ...  │
│ Req1 │ Free │ Req2 │ Req1 │ Req3 │ Free │ Req2 │ Free │      │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
        ↑                ↑
        These free blocks can be used by ANY new request
        No contiguous requirement!
```

### Why This Enables Dynamic Scheduling

1. **No over-reservation**: Allocate blocks only when tokens generated
2. **No fragmentation**: Any free block can serve any request
3. **Block sharing**: Identical prefixes can share physical blocks
4. **LRU eviction**: Least-recently-used cached blocks freed first

---

# C) Repo Map (Code-Driven)

## Directory Structure Overview

```
vllm/
├── v1/                          # V1 engine (current production)
│   ├── core/                    # Scheduling + KV cache management
│   │   ├── sched/               # Scheduler implementation
│   │   │   └── scheduler.py     # Main scheduling loop
│   │   ├── kv_cache_manager.py  # High-level KV cache API
│   │   ├── kv_cache_coordinator.py  # Coordinates KV groups
│   │   ├── kv_cache_utils.py    # Block hashing, LRU queue
│   │   ├── block_pool.py        # Low-level block management
│   │   └── single_type_kv_cache_manager.py  # Per-attention-type manager
│   ├── worker/                  # GPU execution
│   │   └── gpu/
│   │       ├── model_runner.py  # Model execution orchestration
│   │       ├── input_batch.py   # Batch preparation
│   │       └── sample/          # Sampling implementation
│   ├── engine/
│   │   └── core.py              # EngineCore (main loop)
│   └── executor/                # Distributed execution
├── attention/                   # Attention backends
│   └── ops/
│       └── paged_attn.py        # PagedAttention kernel interface
├── model_executor/              # Model loading + layers
└── entrypoints/                 # API servers
    └── openai/
        └── api_server.py        # OpenAI-compatible API
```

## 1) Block/Page Manager (KV Paging)

### File: `vllm/v1/core/block_pool.py`

**What it does**: Manages the physical block pool and LRU eviction queue.

**Why it exists**: 
- Need O(1) allocation/deallocation of fixed-size blocks
- Need O(1) LRU eviction for prefix cache
- Need O(1) prefix cache lookup by hash

**Key Data Structures**:

```python
# From vllm/v1/core/kv_cache_utils.py
@dataclass
class KVCacheBlock:
    block_id: int                           # Physical block ID (0 to N-1)
    ref_cnt: int = 0                        # Reference count for sharing
    _block_hash: BlockHashWithGroupId | None = None  # For prefix cache
    prev_free_block: "KVCacheBlock | None" = None    # Doubly-linked list
    next_free_block: "KVCacheBlock | None" = None    # for LRU queue
```

```python
# From vllm/v1/core/block_pool.py
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, ...):
        # Pre-allocate ALL blocks at startup (no Python object creation later)
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        
        # LRU queue: doubly-linked list for O(1) eviction
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        
        # Hash table: block_hash → KVCacheBlock for O(1) prefix lookup
        self.cached_block_hash_to_block = BlockHashToBlockMap()
```

**Key Operations**:

```python
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    """Pop blocks from free queue, evicting cached blocks if needed."""
    ret = self.free_block_queue.popleft_n(num_blocks)
    for block in ret:
        self._maybe_evict_cached_block(block)  # Remove from hash table
        block.ref_cnt += 1
    return ret

def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    """Mark blocks as used (cache hit). Remove from eviction queue."""
    for block in blocks:
        if block.ref_cnt == 0:  # Was evictable
            self.free_block_queue.remove(block)  # O(1) via doubly-linked list
        block.ref_cnt += 1

def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    """Return blocks to free queue. Added in REVERSE order for LRU."""
    for block in ordered_blocks:
        block.ref_cnt -= 1
    # Blocks with ref_cnt=0 added to tail (LRU = evict from head)
    self.free_block_queue.append_n([b for b in ordered_blocks if b.ref_cnt == 0])
```

### File: `vllm/v1/core/kv_cache_manager.py`

**What it does**: High-level KV cache API for scheduler.

**Why it exists**: Abstract block management from scheduling logic.

```python
class KVCacheManager:
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Find prefix cache hits for a request."""
        # Calls coordinator.find_longest_cache_hit()
        computed_blocks, num_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes,    # Pre-computed hashes
                max_cache_hit_length     # Don't hit on last token (need logits)
            )
        )
        return self.create_kv_cache_blocks(computed_blocks), num_computed_tokens

    def allocate_slots(self, request, num_new_tokens, ...):
        """Allocate KV blocks for new tokens. Returns None if OOM."""
        # 1. Check if enough blocks available
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            return None  # Signal scheduler to preempt
        
        # 2. Touch cached blocks (prevent eviction)
        # 3. Allocate new blocks
        new_blocks = self.coordinator.allocate_new_blocks(...)
        
        # 4. Cache newly full blocks
        self.coordinator.cache_blocks(request, num_tokens_to_cache)
        return new_blocks
```

## 2) Scheduler / Continuous Batching

### File: `vllm/v1/core/sched/scheduler.py`

**What it does**: Decides which requests to run each step, manages preemption.

**Why it exists**: 
- Continuous batching: start/stop requests at iteration granularity
- Memory-aware: don't schedule if KV allocation would fail
- Priority-aware: preempt low-priority requests when needed

**The Main Loop** (simplified):

```python
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """Called every iteration. Returns work for model runner."""
        
        token_budget = self.max_num_scheduled_tokens
        scheduled_requests = []
        
        # === PHASE 1: Schedule RUNNING requests (decode) ===
        for request in self.running:
            num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget)
            
            # Try to allocate KV blocks for new tokens
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, ...
            )
            
            if new_blocks is None:
                # OOM! Preempt lowest-priority request
                preempted = self._preempt_lowest_priority()
                # Retry allocation...
            
            scheduled_requests.append(request)
            token_budget -= num_new_tokens
        
        # === PHASE 2: Schedule WAITING requests (new prefills) ===
        while self.waiting and token_budget > 0:
            request = self.waiting.peek()
            
            # Check prefix cache for computed blocks
            computed_blocks, num_computed = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            
            num_new_tokens = request.num_tokens - num_computed
            num_new_tokens = min(num_new_tokens, token_budget)
            
            # Allocate blocks
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens,
                new_computed_blocks=computed_blocks,
                num_new_computed_tokens=num_computed,
            )
            
            if new_blocks is None:
                break  # No more memory
            
            # Move to running
            self.waiting.pop()
            self.running.append(request)
            request.num_computed_tokens = num_computed  # Skip cached portion!
            
        return SchedulerOutput(...)
```

**Key Insight**: `num_computed_tokens` lets the scheduler skip prefill for cached portions. The model runner only computes tokens beyond `num_computed_tokens`.

## 3) Prefix Cache (Hash-Based, Not Radix Tree)

### File: `vllm/v1/core/kv_cache_utils.py`

**What it does**: Compute block hashes for prefix matching.

**Why it exists**: 
- O(1) lookup instead of O(prefix_length) radix traversal
- Block-granular (16 tokens) for efficiency

**Hash Computation**:

```python
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """
    Hash structure:
      hash(parent_hash, current_tokens, extra_keys)
    
    This creates a CHAIN: Block N's hash depends on Block N-1's hash,
    which depends on Block N-2's... back to the root.
    
    Why chain hashing? Two requests with same tokens 0-31 but different
    tokens 32-47 will have DIFFERENT hashes for block 2, even though
    block 2's tokens are identical. The parent_hash encodes the prefix.
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    
    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

**Extra Keys** (for isolation):

```python
def generate_block_hash_extra_keys(request, start_idx, end_idx, ...):
    """Include LoRA name, MM hashes, cache_salt in hash."""
    extra_keys = []
    extra_keys.extend(_gen_lora_extra_hash_keys(request))    # LoRA isolation
    extra_keys.extend(_gen_mm_extra_hash_keys(request, ...)) # Image/video hashes
    if request.cache_salt:
        extra_keys.append(request.cache_salt)  # Multi-tenant isolation
    return tuple(extra_keys)
```

### File: `vllm/v1/core/block_pool.py` - Cache Lookup

```python
def get_cached_block(self, block_hash: BlockHash, ...) -> list[KVCacheBlock] | None:
    """O(1) hash table lookup for prefix cache hit."""
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        key = make_block_hash_with_group_id(block_hash, group_id)
        block = self.cached_block_hash_to_block.get_one_block(key)
        if not block:
            return None  # Miss! Can't use partial match
        cached_blocks.append(block)
    return cached_blocks
```

### File: `vllm/v1/core/single_type_kv_cache_manager.py` - Finding Longest Hit

```python
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, ...):
    """
    Find longest prefix of block_hashes that ALL hit the cache.
    Returns early on first miss (chain property ensures no later hits).
    """
    computed_blocks = []
    for block_hash in block_hashes:
        if len(computed_blocks) * block_size >= max_length:
            break
        cached = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
        if cached is None:
            break  # Miss! Chain broken, no further hits possible
        computed_blocks.append(cached)
    return computed_blocks, len(computed_blocks) * block_size
```

**What Is Reused**:
- ✅ KV blocks (physical memory with computed K/V values)
- ✅ Prefill compute (skipped entirely for cached blocks)
- ❌ Activations (recomputed; not cached)

**What Cannot Be Reused**:
- Different LoRA adapter → different hash
- Different images in same placeholder → different hash
- Different `cache_salt` → different hash

## 4) Model Execution + Kernel Integration

### File: `vllm/v1/worker/gpu/model_runner.py`

**What it does**: Prepare inputs, call model, extract outputs.

**Why it exists**: Bridge between scheduler output and GPU execution.

**Key Method** (simplified):

```python
class GPUModelRunner:
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        # 1. Prepare input tensors
        input_ids, positions, block_tables = self._prepare_inputs(scheduler_output)
        
        # 2. Build attention metadata (tells kernels where KV lives)
        attn_metadata = build_attn_metadata(
            block_tables=block_tables,
            seq_lens=seq_lens,
            ...
        )
        
        # 3. Forward pass through model
        with set_forward_context(attn_metadata):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=self.kv_caches,  # Pre-allocated GPU tensors
                attn_metadata=attn_metadata,
            )
        
        # 4. Sample next tokens
        logits = self.model.compute_logits(hidden_states)
        sampler_output = self.sampler(logits, sampling_metadata)
        
        return ModelRunnerOutput(sampled_token_ids=sampler_output.token_ids, ...)
```

### File: `vllm/attention/ops/paged_attn.py`

**What it does**: Interface to PagedAttention CUDA kernels.

```python
class PagedAttention:
    @staticmethod
    def write_to_paged_cache(key, value, key_cache, value_cache, slot_mapping, ...):
        """Write new K/V to their paged locations."""
        ops.reshape_and_cache(
            key, value,
            key_cache, value_cache,
            slot_mapping.flatten(),  # Maps token position → physical slot
            ...
        )
```

**The slot_mapping** is the key insight:
- Scheduler computes: logical position → block table → physical block ID × block_size + offset
- Kernel uses slot_mapping to scatter K/V writes and gather reads

## 5) Sampling/Output Pipeline

### File: `vllm/v1/worker/gpu/sample/sampler.py`

**What it does**: Convert logits to token IDs.

```python
class Sampler:
    def forward(self, logits: torch.Tensor, metadata: SamplingMetadata) -> SamplerOutput:
        # Apply temperature, top-p, top-k, repetition penalty, etc.
        probs = self._apply_sampling_params(logits, metadata)
        
        # Sample token IDs
        token_ids = self._sample(probs, metadata)
        
        # Compute logprobs if requested
        logprobs = self._compute_logprobs(logits, token_ids, metadata)
        
        return SamplerOutput(token_ids=token_ids, logprobs=logprobs)
```

---

# D) One Request Walkthrough

## Tracing a Single Request Through the Code

```
User Request: "What is the capital of France?"
System Prompt: "You are a helpful assistant."  (shared with other requests)
```

### Step 1: API Entry

**File**: `vllm/entrypoints/openai/api_server.py`

```python
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # Tokenize input
    prompt_ids = tokenizer.encode(full_prompt)  # [sys_prompt + user_prompt]
    
    # Create engine request
    engine_request = EngineCoreRequest(
        request_id=uuid4(),
        prompt_token_ids=prompt_ids,
        sampling_params=request.sampling_params,
        ...
    )
    
    # Submit to engine
    await engine.add_request(engine_request)
```

### Step 2: Request Enters Engine Core

**File**: `vllm/v1/engine/core.py`

```python
class EngineCore:
    def add_request(self, request: EngineCoreRequest):
        # Compute block hashes for prefix caching
        block_hashes = self.request_block_hasher(request)
        request.block_hashes = block_hashes
        
        # Add to scheduler's waiting queue
        self.scheduler.add_request(Request.from_engine_request(request))
```

### Step 3: Scheduler Processes Request

**File**: `vllm/v1/core/sched/scheduler.py`

```python
def schedule(self) -> SchedulerOutput:
    # ... schedule running requests first ...
    
    # Process waiting request
    request = self.waiting.pop()
    
    # ┌──────────────────────────────────────────────────────────────┐
    # │ PREFIX CACHE LOOKUP                                          │
    # └──────────────────────────────────────────────────────────────┘
    computed_blocks, num_computed = self.kv_cache_manager.get_computed_blocks(request)
    # If system prompt was cached: num_computed = 64 (4 blocks × 16 tokens)
    # computed_blocks = [Block 42, Block 17, Block 89, Block 23]
    
    # ┌──────────────────────────────────────────────────────────────┐
    # │ KV BLOCK ALLOCATION                                          │
    # └──────────────────────────────────────────────────────────────┘
    # Request has 80 tokens total, 64 cached → need slots for 16 new
    new_blocks = self.kv_cache_manager.allocate_slots(
        request,
        num_new_tokens=16,
        new_computed_blocks=computed_blocks,
        num_new_computed_tokens=64,
    )
    # new_blocks = [Block 156]  (1 new block for remaining 16 tokens)
    
    # ┌──────────────────────────────────────────────────────────────┐
    # │ UPDATE REQUEST STATE                                         │
    # └──────────────────────────────────────────────────────────────┘
    request.num_computed_tokens = 64  # Skip cached portion!
    self.running.append(request)
    
    return SchedulerOutput(
        scheduled_requests=[request],
        num_scheduled_tokens={request.id: 16},  # Only 16, not 80!
        ...
    )
```

### Step 4: Model Runner Executes

**File**: `vllm/v1/worker/gpu/model_runner.py`

```python
def execute_model(self, scheduler_output: SchedulerOutput):
    # ┌──────────────────────────────────────────────────────────────┐
    # │ PREPARE INPUTS                                               │
    # └──────────────────────────────────────────────────────────────┘
    # Only tokens 64-79 need computation (16 tokens)
    input_ids = request.prompt_ids[64:80]  # Last 16 tokens
    positions = torch.arange(64, 80)       # Positions 64-79
    
    # Block table: [42, 17, 89, 23, 156] (4 cached + 1 new)
    block_table = build_block_table(request.blocks)
    
    # ┌──────────────────────────────────────────────────────────────┐
    # │ ATTENTION METADATA                                           │
    # └──────────────────────────────────────────────────────────────┘
    attn_metadata = build_attn_metadata(
        block_tables=[block_table],
        seq_lens=[80],  # Total sequence length for attention
        num_prefill_tokens=16,
        ...
    )
    
    # ┌──────────────────────────────────────────────────────────────┐
    # │ FORWARD PASS (PREFILL FOR 16 TOKENS)                         │
    # └──────────────────────────────────────────────────────────────┘
    with set_forward_context(attn_metadata):
        # Model computes attention:
        #   - Query: from 16 new tokens (positions 64-79)
        #   - Keys/Values: load from blocks [42,17,89,23] for positions 0-63
        #                  + compute new K/V for positions 64-79 → store in block 156
        hidden_states = self.model(input_ids, positions, kv_caches, attn_metadata)
    
    # ┌──────────────────────────────────────────────────────────────┐
    # │ SAMPLING                                                     │
    # └──────────────────────────────────────────────────────────────┘
    logits = self.model.compute_logits(hidden_states[-1:])  # Last token only
    next_token = self.sampler(logits)  # → "Paris"
    
    return ModelRunnerOutput(sampled_token_ids=[next_token], ...)
```

### Step 5: Decode Loop

```python
# SUBSEQUENT ITERATIONS (decode phase):
while not stop_condition:
    # Scheduler: request has num_tokens_with_spec = 81 (80 + 1 generated)
    #            but num_computed_tokens = 80
    #            → schedule 1 new token
    
    # Model Runner:
    #   - input_ids = [token_id of "Paris"]
    #   - positions = [80]
    #   - Load KV from all 5 blocks for attention
    #   - Generate new K/V → store in block 156 (slot 0)
    #   - Sample next token
    
    # Eventually: allocate new block when block 156 fills up
```

### Step 6: Request Completion

**File**: `vllm/v1/core/sched/scheduler.py`

```python
def _free_request(self, request: Request):
    # Free KV blocks (added to free queue in REVERSE order)
    self.kv_cache_manager.free(request)
    # Block 156 → free queue tail
    # Blocks 23, 89, 17, 42 → ref_cnt decremented
    #   If ref_cnt == 0 (no other requests using): → free queue tail
    #   If ref_cnt > 0 (shared with other requests): stay allocated
```

---

# E) ASCII Diagrams

## 1) GPU Mental Model: HBM, SMs, Memory Bandwidth Bottleneck

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NVIDIA A100 GPU                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                     HBM2e (80 GB)                                  │    │
│   │  ┌──────────────┬──────────────┬───────────────────────────────┐  │    │
│   │  │ Model        │ KV Cache     │ Activations                   │  │    │
│   │  │ Weights      │ (dynamic)    │ + Workspace                   │  │    │
│   │  │ ~35 GB       │ ~38 GB avail │ ~5 GB                         │  │    │
│   │  └──────────────┴──────────────┴───────────────────────────────┘  │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                          ▲                                                   │
│                          │ Memory Bandwidth: 2.0 TB/s                       │
│                          │ (THE BOTTLENECK for decode)                      │
│                          ▼                                                   │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                 108 Streaming Multiprocessors                       │    │
│   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐           ┌─────┐                │    │
│   │  │ SM0 │ │ SM1 │ │ SM2 │ │ SM3 │    ...    │SM107│                │    │
│   │  └─────┘ └─────┘ └─────┘ └─────┘           └─────┘                │    │
│   │  Compute: 312 TFLOPS (FP16)                                        │    │
│   │  (Often IDLE during decode - waiting for memory!)                  │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│   BOTTLENECK ANALYSIS (Decode, Llama-70B, 2K context):                      │
│   ─────────────────────────────────────────────────────                     │
│   Memory load per token:  ~60 GB  (KV cache + weights)                      │
│   Time at 2 TB/s:         30 ms                                             │
│   Compute per token:      ~140 GFLOPs                                       │
│   Time at 312 TFLOPs:     0.5 ms                                            │
│                                                                              │
│   → 98% of time waiting for memory!                                         │
│   → Solution: BATCH MORE REQUESTS to amortize weight loads                  │
│   → But batching requires MEMORY EFFICIENCY → PagedAttention               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2) Memory Layout WITHOUT Paging (Fragmentation)

```
GPU HBM - Traditional KV Cache Allocation
═════════════════════════════════════════════════════════════════════════════

Time T=0: 3 requests arrive
┌────────────────────────────────────────────────────────────────────────────┐
│ Request 1 (reserved 4K)          │ Request 2 (reserved 4K)                │
│ ██████████░░░░░░░░░░░░░░░░░░░░░░│ ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ using 512 tokens                 │ using 384 tokens                       │
│ (87% wasted)                     │ (90% wasted)                           │
├──────────────────────────────────┼────────────────────────────────────────┤
│ Request 3 (reserved 4K)          │ FREE SPACE                             │
│ ████████████████░░░░░░░░░░░░░░░░│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ using 1024 tokens                │ 4K contiguous                          │
│ (75% wasted)                     │                                        │
└──────────────────────────────────┴────────────────────────────────────────┘
                Total: 12K reserved, 1.9K used = 84% waste!

Time T=1: Request 2 completes, Request 4 arrives (needs 5K tokens)
┌────────────────────────────────────────────────────────────────────────────┐
│ Request 1 (reserved 4K)          │ FREE                                   │
│ ██████████████░░░░░░░░░░░░░░░░░░│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ now using 768 tokens             │ 4K contiguous                          │
├──────────────────────────────────┼────────────────────────────────────────┤
│ Request 3 (reserved 4K)          │ FREE SPACE                             │
│ ████████████████████████░░░░░░░░│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ now using 1536 tokens            │ 4K contiguous                          │
└──────────────────────────────────┴────────────────────────────────────────┘
        Request 4 needs 5K contiguous... but largest free chunk is 4K!
        → FRAGMENTATION → OOM despite 8K total free!

External Fragmentation:  Can't use free space (non-contiguous)
Internal Fragmentation:  Reserved space unused within requests
```

## 3) Memory Layout WITH Paged KV Blocks

```
GPU HBM - PagedAttention Block Pool
═════════════════════════════════════════════════════════════════════════════

Block Size: 16 tokens each
Total Blocks: 20 (320 tokens capacity)

Block Pool:
┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │ B8 │ B9 │B10 │B11 │B12 │B13 │... │
├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
│ R1 │ R1 │ R3 │ R1 │Free│ R3 │ R2 │ R2 │Free│ R3 │Free│Free│ R1 │Free│... │
│████│████│████│████│░░░░│████│████│███░│░░░░│██░░│░░░░│░░░░│██░░│░░░░│    │
│full│full│full│full│    │full│full│12tk│    │8tok│    │    │10tk│    │    │
└────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

Block Tables (per request):
┌─────────────────────────────────────────────────────────────────────────┐
│ Request 1: logical [0,1,2,3] → physical [B0, B1, B3, B12]              │
│            Tokens: 64 + 10 = 74 tokens (no wasted reservation!)        │
│                                                                         │
│ Request 2: logical [0,1] → physical [B6, B7]                           │
│            Tokens: 16 + 12 = 28 tokens                                 │
│                                                                         │
│ Request 3: logical [0,1,2] → physical [B2, B5, B9]                     │
│            Tokens: 32 + 8 = 40 tokens                                  │
└─────────────────────────────────────────────────────────────────────────┘

Free Blocks: [B4, B8, B10, B11, B13, ...]  (available for ANY request)

Benefits:
✓ No contiguous requirement → ANY free block can serve ANY request
✓ On-demand allocation → no wasted reservations
✓ ~4% fragmentation (last block may be partial) vs 60-80%
✓ 3-4x more concurrent requests on same memory!
```

## 4) Software Pipeline: API → Scheduler → KV Manager → Kernels → Sampler

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         vLLM REQUEST FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  Client Request
        │
        ▼
┌───────────────────┐
│   API Server      │  vllm/entrypoints/openai/api_server.py
│   (OpenAI compat) │  - Tokenize prompt
│                   │  - Create EngineCoreRequest
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Engine Core     │  vllm/v1/engine/core.py
│                   │  - Compute block hashes (for prefix cache)
│                   │  - Submit to scheduler
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   Scheduler       │  vllm/v1/core/sched/scheduler.py
│                   │  - Prefix cache lookup (get_computed_blocks)
│                   │  - KV block allocation (allocate_slots)
│   ┌───────────┐   │  - Preemption if OOM
│   │KV Cache   │   │  - Build SchedulerOutput
│   │Manager    │◄──┤
│   └───────────┘   │
└───────────────────┘
        │ SchedulerOutput
        ▼
┌───────────────────┐
│   Executor        │  vllm/v1/executor/*.py
│   (TP/PP/DP)      │  - Distribute to workers
└───────────────────┘
        │
        ▼
┌───────────────────┐
│   GPU Worker      │  vllm/v1/worker/gpu_worker.py
│                   │  - Receive scheduled work
└───────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────────┐
│                      GPU Model Runner                                  │
│                      vllm/v1/worker/gpu/model_runner.py                │
│                                                                        │
│   1. Prepare Inputs     ─────────────────────────────────────────►    │
│      - Build input_ids, positions from SchedulerOutput                │
│      - Build block_tables for attention                               │
│      - Build attention metadata                                        │
│                                                                        │
│   2. Forward Pass       ─────────────────────────────────────────►    │
│      ┌──────────────────────────────────────────────────────────┐     │
│      │               Model Layers (GPU)                          │     │
│      │  ┌─────────┐  ┌─────────────────────┐  ┌─────────┐      │     │
│      │  │ Embed   │→ │ Attention + FFN     │→ │ LM Head │      │     │
│      │  │         │  │ (×80 layers)        │  │         │      │     │
│      │  └─────────┘  └─────────────────────┘  └─────────┘      │     │
│      │                       │                                   │     │
│      │                       ▼                                   │     │
│      │              ┌────────────────┐                          │     │
│      │              │ PagedAttention │  vllm/attention/         │     │
│      │              │ CUDA Kernels   │  ops/paged_attn.py       │     │
│      │              │                │                          │     │
│      │              │ - Read KV from │                          │     │
│      │              │   paged blocks │                          │     │
│      │              │ - Write new KV │                          │     │
│      │              │   to blocks    │                          │     │
│      │              └────────────────┘                          │     │
│      └──────────────────────────────────────────────────────────┘     │
│                                                                        │
│   3. Sampling           ─────────────────────────────────────────►    │
│      ┌──────────────────────────────────────────────────────────┐     │
│      │ Sampler                vllm/v1/worker/gpu/sample/        │     │
│      │ - Apply temperature, top-p, top-k                        │     │
│      │ - Sample token IDs                                       │     │
│      │ - Compute logprobs (if requested)                        │     │
│      └──────────────────────────────────────────────────────────┘     │
│                                                                        │
└───────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ModelRunnerOutput (sampled tokens, logprobs, ...)
        │
        ▼
  Stream to client via SSE
```

## 5) Prefill vs Decode Timeline with Continuous Batching

```
═══════════════════════════════════════════════════════════════════════════════
                    CONTINUOUS BATCHING TIMELINE
═══════════════════════════════════════════════════════════════════════════════

Traditional Batching (requests wait for batch to complete):
──────────────────────────────────────────────────────────────────────────────
Time:     0    100   200   300   400   500   600   700   800   900  1000
          ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤

Batch 1:  [PPPPPPPP][D][D][D][D][D][D]  (wait for all to finish...)
Req A:    [████████][█][█][█][█][█][DONE]
Req B:    [████    ][█][█][█][█][█][█][█][█][WAIT...     ] (padded, waiting)
Req C:    [████████████][█][█][█][█][█][WAIT...         ] (padded, waiting)

          ↑ Req B finishes early but waits
          ↑ Req D can't start until entire batch completes

Batch 2:                                            [PPPPP][D][D][D]...
Req D:                                              [█████][█][█][█]...

──────────────────────────────────────────────────────────────────────────────
Continuous Batching (iteration-level scheduling):
──────────────────────────────────────────────────────────────────────────────
Time:     0    100   200   300   400   500   600   700   800   900  1000
          ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤

Req A:    [PPPP][D][D][D][D][D][DONE]
                ↑
Req B:         [PP][D][D][D][D][D][D][DONE]   ← Starts iteration 1
                   ↑
Req C:            [PPPPPP][D][D][D][D][D][D][DONE]  ← Starts iteration 2
                          ↑
Req D:                   [PPP][D][D][D][D][D][DONE]  ← Starts iteration 3

Iteration │ Tokens Processed    │ Requests Active
──────────┼────────────────────┼─────────────────
    0     │ A:prefill(100)     │ {A}
    1     │ A:1, B:prefill(50) │ {A,B}
    2     │ A:1, B:1, C:pf(80) │ {A,B,C}
    3     │ A:1, B:1, C:1, D:pf│ {A,B,C,D}
    ...   │ ...                │ ...

Benefits:
✓ New requests start IMMEDIATELY (no waiting for batch)
✓ Finished requests FREE memory immediately (no padding)
✓ GPU always has work (mix prefill + decode in same step)
✓ Higher utilization → higher throughput → lower latency
═══════════════════════════════════════════════════════════════════════════════
```

## 6) Prefix Cache: Hash-Based Block Lookup

```
═══════════════════════════════════════════════════════════════════════════════
                    PREFIX CACHE: HASH-BASED LOOKUP
═══════════════════════════════════════════════════════════════════════════════

Block Size: 16 tokens

EXAMPLE: System prompt "You are a helpful assistant..." = 64 tokens = 4 blocks

Request 1 arrives (first with this system prompt):
─────────────────────────────────────────────────────────────────────────────

Tokens:     [You are a helpful] [assistant that h] [elps users with ] [their questions.]
Block:           Block 0             Block 1            Block 2            Block 3

Hash Computation (chain hashing):
┌─────────────────────────────────────────────────────────────────────────────┐
│ H₀ = hash(NONE_HASH, [You,are,a,helpful])           → 0xAB12...           │
│ H₁ = hash(H₀, [assistant,that,h])                   → 0xCD34...           │
│ H₂ = hash(H₁, [elps,users,with,])                   → 0xEF56...           │
│ H₃ = hash(H₂, [their,questions,.])                  → 0x7890...           │
└─────────────────────────────────────────────────────────────────────────────┘

After prefill, cache state:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Hash Table:                                                                  │
│   0xAB12... → Block 42   (physical block ID)                                │
│   0xCD34... → Block 17                                                       │
│   0xEF56... → Block 89                                                       │
│   0x7890... → Block 23                                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Request 2 arrives (SAME system prompt + different user question):
─────────────────────────────────────────────────────────────────────────────

Tokens:     [You are a helpful] [assistant that h] [elps users with ] [their questions.] [User: What is]...
                 Block 0             Block 1            Block 2            Block 3         Block 4 (new)

Hash Computation:
┌─────────────────────────────────────────────────────────────────────────────┐
│ H₀ = hash(NONE_HASH, [You,are,a,helpful]) → 0xAB12... → CACHE HIT!         │
│   └→ Reuse Block 42, skip prefill for tokens 0-15                          │
│                                                                              │
│ H₁ = hash(H₀, [assistant,that,h])         → 0xCD34... → CACHE HIT!         │
│   └→ Reuse Block 17, skip prefill for tokens 16-31                         │
│                                                                              │
│ H₂ = hash(H₁, [elps,users,with,])         → 0xEF56... → CACHE HIT!         │
│   └→ Reuse Block 89, skip prefill for tokens 32-47                         │
│                                                                              │
│ H₃ = hash(H₂, [their,questions,.])        → 0x7890... → CACHE HIT!         │
│   └→ Reuse Block 23, skip prefill for tokens 48-63                         │
│                                                                              │
│ H₄ = hash(H₃, [User:,What,is,...])        → 0x1111... → CACHE MISS         │
│   └→ Allocate new Block 156, must prefill tokens 64-79                     │
└─────────────────────────────────────────────────────────────────────────────┘

Result:
  • Request 2 only prefills 16 tokens (vs 80 without caching)
  • 80% of prefill compute SKIPPED!
  • TTFT reduced by ~80%

─────────────────────────────────────────────────────────────────────────────
PARTIAL MATCH EXAMPLE:
─────────────────────────────────────────────────────────────────────────────

Request 3: Same first 48 tokens, different block 3

Tokens:     [You are a helpful] [assistant that h] [elps users with ] [code and debugs.] [...]
                 Block 0             Block 1            Block 2           ← DIFFERENT!

Hash Computation:
  H₀ → 0xAB12... → HIT (reuse Block 42)
  H₁ → 0xCD34... → HIT (reuse Block 17)
  H₂ → 0xEF56... → HIT (reuse Block 89)
  H₃ = hash(H₂, [code,and,debugs,.]) → 0xFFFF... → MISS!
       └→ Chain broken! Must compute Block 3 onward.

Note: Even though Block 4+ tokens might match another cached sequence,
      chain hashing means H₄ depends on H₃, so no match possible.
      (This is the trade-off vs token-level radix trees)

═══════════════════════════════════════════════════════════════════════════════
```

## 7) Comparison Diagram: vLLM vs TensorRT vs SGLang

```
═══════════════════════════════════════════════════════════════════════════════
              OPTIMIZATION FOCUS COMPARISON
═══════════════════════════════════════════════════════════════════════════════

              ┌─────────────────────────────────────────────────────────────┐
              │                    INFERENCE STACK                          │
              ├─────────────────────────────────────────────────────────────┤
              │                                                             │
Application   │   API / User Requests                                      │
              │        │                                                    │
              │        ▼                                                    │
              │   ┌─────────────────────────────────────────────────────┐  │
Scheduling    │   │  Request Scheduling + Batching                      │  │
Layer         │   │  ┌───────────────────────────────────────────────┐ │  │
              │   │  │ vLLM: Continuous batching + cache-aware       │ │  │
              │   │  │ SGLang: Prefix-aware scheduling               │ │  │
              │   │  │ TensorRT: Static/semi-dynamic batching        │ │  │
              │   │  └───────────────────────────────────────────────┘ │  │
              │   └─────────────────────────────────────────────────────┘  │
              │        │                                                    │
              │        ▼                                                    │
              │   ┌─────────────────────────────────────────────────────┐  │
Memory        │   │  KV Cache Memory Management                         │  │
Layer         │   │  ┌───────────────────────────────────────────────┐ │  │
              │   │  │ vLLM: PagedAttention (block pool, LRU)        │◄┼──┼── MAIN INNOVATION
              │   │  │ SGLang: RadixAttention (radix tree + paging)  │ │  │
              │   │  │ TensorRT: Contiguous pre-allocation           │ │  │
              │   │  └───────────────────────────────────────────────┘ │  │
              │   └─────────────────────────────────────────────────────┘  │
              │        │                                                    │
              │        ▼                                                    │
              │   ┌─────────────────────────────────────────────────────┐  │
Prefix        │   │  Prefix Reuse / Caching                             │  │
Reuse         │   │  ┌───────────────────────────────────────────────┐ │  │
              │   │  │ vLLM: Hash-based block cache (16-tok blocks)  │ │  │
              │   │  │ SGLang: Radix tree (token-level granularity)  │◄┼──┼── SGLang wins here
              │   │  │ TensorRT: Limited / manual                    │ │  │
              │   │  └───────────────────────────────────────────────┘ │  │
              │   └─────────────────────────────────────────────────────┘  │
              │        │                                                    │
              │        ▼                                                    │
              │   ┌─────────────────────────────────────────────────────┐  │
Kernel        │   │  GPU Kernel Execution                               │  │
Layer         │   │  ┌───────────────────────────────────────────────┐ │  │
              │   │  │ vLLM: Flash/PagedAttn kernels, flexible       │ │  │
              │   │  │ SGLang: Similar kernel set                     │ │  │
              │   │  │ TensorRT: Highly optimized fused kernels      │◄┼──┼── TensorRT wins here
              │   │  └───────────────────────────────────────────────┘ │  │
              │   └─────────────────────────────────────────────────────┘  │
              │                                                             │
              └─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                    WHEN TO USE EACH
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  vLLM is BEST when:                                                         │
│  ├─ High concurrency, variable workloads                                    │
│  ├─ Multi-tenant API service                                                │
│  ├─ Memory efficiency is critical                                           │
│  └─ Using HuggingFace models without custom optimization                    │
│                                                                              │
│  TensorRT-LLM is BEST when:                                                 │
│  ├─ Fixed, predictable batch sizes                                          │
│  ├─ Maximum single-request latency matters                                  │
│  ├─ Willing to invest in compile-time optimization                          │
│  └─ NVIDIA hardware with specific model support                             │
│                                                                              │
│  SGLang is BEST when:                                                       │
│  ├─ Complex multi-turn conversations with branching                         │
│  ├─ Tool-use with shared tool definitions                                   │
│  ├─ Maximum prefix reuse granularity needed                                 │
│  └─ Python-native programming model preferred                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
═══════════════════════════════════════════════════════════════════════════════
```

---

# F) Practical Takeaways

## Strengths and Weaknesses

### Strengths

| Strength | Why It Matters |
|----------|---------------|
| **Memory efficiency (95%+)** | 3-4x more concurrent requests per GPU |
| **Continuous batching** | No idle time, always processing |
| **Prefix caching** | Skip redundant compute, faster TTFT |
| **HuggingFace compatible** | Drop-in for any HF model |
| **OpenAI API compatible** | Easy integration |
| **Production proven** | Used by major LLM providers |
| **Active development** | Frequent releases, good community |

### Weaknesses

| Weakness | When It Hurts |
|----------|--------------|
| **Block-level prefix caching** | Loses efficiency vs token-level for partial blocks |
| **Python scheduling overhead** | Very high request rates (>1000 req/s) |
| **Not compile-optimized** | Maximum single-request latency |
| **Memory overhead for paging** | Very small models where fixed alloc is fine |
| **Learning curve** | Understanding paging for debugging |

## Common Failure Modes

### 1. OOM Despite "Enough" Memory

**Symptom**: CUDA OOM error when KV cache usage shows 60-70%

**Cause**: 
- Model weights larger than expected (quantization mismatch)
- CUDA context overhead not accounted for
- Fragmentation in non-KV memory

**Fix**:
```bash
# Reduce GPU memory utilization
vllm serve model --gpu-memory-utilization 0.85  # Default is 0.90

# Or limit max_model_len
vllm serve model --max-model-len 4096
```

### 2. Tail Latency Spikes

**Symptom**: p99 latency 5-10x higher than p50

**Causes**:
- Long prefills blocking decode steps
- Cache eviction storms
- GC pauses (Python)

**Fixes**:
```bash
# Enable chunked prefill to not block decode
vllm serve model --enable-chunked-prefill --max-num-batched-tokens 8192

# Limit concurrent partial prefills
vllm serve model --max-num-partial-prefills 1
```

### 3. Low Throughput Despite High GPU%

**Symptom**: GPU utilization 80%+, but TPS lower than expected

**Causes**:
- Memory bandwidth bound (too few concurrent requests)
- Inefficient batching (many short sequences)
- Prefix cache misses

**Fixes**:
```bash
# Increase max concurrent sequences
vllm serve model --max-num-seqs 512

# Check prefix cache hit rate in metrics
# If low, ensure consistent system prompts
```

### 4. Cache Thrash / Low Prefix Hit Rate

**Symptom**: Prefix cache hit rate <10%, high eviction rate

**Causes**:
- Diverse prompts with no shared prefix
- Cache too small for working set
- Unique content at start of prompts (timestamps, IDs)

**Fixes**:
```bash
# Increase GPU memory for larger cache
vllm serve model --gpu-memory-utilization 0.95

# Restructure prompts: static content FIRST
# BAD:  "[Timestamp: 2024-01-01] System prompt..."
# GOOD: "System prompt... [Timestamp: 2024-01-01]"
```

## Diagnostic Patterns

| Metric Pattern | Usually Means |
|---------------|---------------|
| High KV usage + low throughput | Memory bandwidth bound, need more batching |
| Low KV usage + OOM | Non-KV memory issue (weights, activations) |
| High TTFT + low cache hits | Prefix cache not helping, diverse prompts |
| Low TTFT + high TTFT p99 | Occasional long prefills or eviction |
| GPU util low + memory high | Scheduling issue or waiting for CPU |

## Top 5 Tuning Knobs

### 1. `--gpu-memory-utilization` (default: 0.90)

```bash
# More memory for KV cache = more concurrent requests
vllm serve model --gpu-memory-utilization 0.95

# Less if getting OOM on complex models
vllm serve model --gpu-memory-utilization 0.80
```

### 2. `--max-num-batched-tokens` (default: 2048)

```bash
# Higher = more tokens per step = better GPU utilization
# But: increases memory, may increase latency variance
vllm serve model --max-num-batched-tokens 8192
```

### 3. `--max-num-seqs` (default: 256)

```bash
# More sequences = better batching for decode
vllm serve model --max-num-seqs 512

# But: each sequence has overhead, diminishing returns past ~1000
```

### 4. `--enable-prefix-caching` (default: True in v1)

```bash
# Essential for workloads with shared prefixes
vllm serve model --enable-prefix-caching

# Disable if unique prompts (saves hash computation overhead)
vllm serve model --no-enable-prefix-caching
```

### 5. `--enable-chunked-prefill` (default: True in v1)

```bash
# Prevents long prefills from blocking decode
vllm serve model --enable-chunked-prefill

# Tune the chunk size
vllm serve model --max-num-batched-tokens 4096  # Chunks prefill to 4K
```

## 1-Week Onboarding Plan

### Day 1: Conceptual Foundation
- [ ] Read this document (sections A, B, E diagrams)
- [ ] Read [vLLM paper](https://arxiv.org/abs/2309.06180) sections 1-4
- [ ] Run basic example: `vllm serve facebook/opt-125m`
- [ ] Send requests via curl, observe responses

### Day 2: Hands-On Exploration
- [ ] Run `examples/offline_inference/prefix_caching.py`
- [ ] Observe TTFT difference with/without caching
- [ ] Check `/metrics` endpoint, understand key metrics
- [ ] Try different `--gpu-memory-utilization` values

### Day 3: Code Reading (Scheduler)
- [ ] Read `vllm/v1/core/sched/scheduler.py` `schedule()` method
- [ ] Trace how `get_computed_blocks()` does prefix lookup
- [ ] Understand how `allocate_slots()` manages memory
- [ ] Set breakpoints, trace a single request

### Day 4: Code Reading (KV Cache)
- [ ] Read `vllm/v1/core/block_pool.py`
- [ ] Understand `FreeKVCacheBlockQueue` (LRU queue)
- [ ] Read `vllm/v1/core/kv_cache_utils.py` (hash computation)
- [ ] Trace how blocks are allocated and freed

### Day 5: Code Reading (Execution)
- [ ] Read `vllm/v1/worker/gpu/model_runner.py` `execute_model()`
- [ ] Understand how `SchedulerOutput` becomes GPU execution
- [ ] Read `vllm/attention/ops/paged_attn.py`
- [ ] Understand slot_mapping for paged reads/writes

### Day 6: Benchmarking
- [ ] Run `benchmarks/benchmark_latency.py` with different configs
- [ ] Run `benchmarks/benchmark_throughput.py`
- [ ] Create a workload representative of your use case
- [ ] Measure TTFT, TPS, p50/p95/p99 latency

### Day 7: Optimization & Contribution
- [ ] Tune knobs for your workload
- [ ] Document findings
- [ ] Identify a small improvement or bug fix
- [ ] Read CONTRIBUTING.md, set up dev environment
- [ ] Submit a small PR (even docs/typo is valuable)

---

## Summary: Why vLLM in One Sentence

**vLLM exists because LLM inference is memory-bound, traditional systems waste 60-80% of GPU memory, and PagedAttention + continuous batching unlocks 2-4x better utilization by treating KV cache like OS virtual memory.**

---

## References

1. **vLLM Paper**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

2. **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai/)

3. **Prefix Caching Design**: [docs.vllm.ai/en/latest/design/automatic_prefix_caching.html](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html)

4. **Source Code**: This document references actual code from the vLLM repository at [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

---

*Document based on vLLM codebase analysis, December 2024*
