# vLLM Inference Deep Dive: PagedAttention, KV Cache, Prefix Caching, and Continuous Batching

**A comprehensive guide for CEOs seeking intuition and engineers seeking implementation details.**

---

## Table of Contents
1. [CEO Explanation](#1-ceo-explanation)
2. [Intern Onboarding Explanation](#2-intern-onboarding-explanation)
3. [Engineering Deep Dive](#3-engineering-deep-dive)
4. [Practical Playbook](#4-practical-playbook)
5. [Minimal Starter Snippets](#5-minimal-starter-snippets)
6. [What I Would Do Next Week](#6-what-i-would-do-next-week)

---

# 1. CEO Explanation

## The Story of LLM Inference: Prefill vs Decode

Imagine you're running a restaurant. When a customer orders, there are two phases:
1. **Prefill (Reading the Order)**: The chef reads the entire order at once—this is fast per-item because they can read in parallel.
2. **Decode (Cooking One Dish at a Time)**: The chef cooks each dish sequentially, waiting for each to finish before starting the next.

In LLM inference:
- **Prefill**: The model reads your entire prompt at once (parallelizable, compute-bound)
- **Decode**: The model generates tokens one-by-one (sequential, memory-bound)

**Key Insight**: Decode is slow not because of math, but because the GPU spends most of its time *waiting for memory*. The model must re-read previously computed information (the "KV cache") for every single new token.

## What Makes Inference Expensive: The KV Cache Problem

For every token the model has seen, it stores two vectors: a **Key (K)** and a **Value (V)**. These grow linearly with context length. For a 70B model with 128K context:
- **~40-80GB of KV cache per request** (depending on model architecture)

**The Traditional Problem (Before vLLM)**:
- Systems pre-allocated maximum possible memory for each request
- If a user might generate up to 2048 tokens, you reserve space for 2048—even if they only need 50
- Result: 60-80% of GPU memory sits **unused but reserved** ([vLLM Paper](https://arxiv.org/abs/2309.06180))

## PagedAttention: Why Paging Beats Monolithic Allocation

**The vLLM Innovation**: Borrow ideas from operating system virtual memory.

```
Traditional Approach:                  PagedAttention:
┌──────────────────────────┐          ┌──────┬──────┬──────┐
│ Request 1: Reserved 2048 │          │ Req1 │ Req2 │ Req3 │  ← Block 0
│ (using only 200 tokens)  │          ├──────┼──────┼──────┤
├──────────────────────────┤          │ Req1 │ Req3 │ Free │  ← Block 1
│ Request 2: Reserved 2048 │          ├──────┼──────┼──────┤
│ (using only 150 tokens)  │          │ Req2 │ Free │ Free │  ← Block 2
├──────────────────────────┤          └──────┴──────┴──────┘
│ ❌ GPU FULL - No Room!   │          ✓ Blocks allocated on-demand
└──────────────────────────┘          ✓ 3-4x more concurrent requests
```

**Business Impact**:
- **2-4x higher throughput** on the same hardware
- **Near-zero memory waste** (only ~4% fragmentation vs 60-80%)
- **More users served per GPU dollar**

## Automatic Prefix Caching: Sharing Work Across Requests

Many requests share common prefixes:
- System prompts ("You are a helpful assistant...")
- Document/context preambles
- Multi-turn conversation history

**Without Prefix Caching**: Recompute the same KV values for every request  
**With Prefix Caching**: Compute once, reuse many times

```
Request 1: "System prompt... User: What is 2+2?"
Request 2: "System prompt... User: What is the capital of France?"
                ↑
        Shared prefix! Cache it.
```

**Business Impact**:
- **TTFT (Time to First Token) reduction**: 50-90% for workloads with shared prefixes
- **Cost reduction**: Less compute = lower cloud bills
- **Better user experience**: Faster responses

## When Should You Choose vLLM?

| Your Workload Looks Like... | Expected Gains | Choose vLLM? |
|----------------------------|----------------|--------------|
| Many concurrent users, short-medium outputs | **3-5x throughput** | ✅ Strongly yes |
| Chatbot with system prompt | **50-80% TTFT reduction** (prefix cache) | ✅ Yes |
| RAG with shared document context | **60-90% TTFT reduction** | ✅ Yes |
| Agent loops with tool descriptions | **40-70% TTFT reduction** | ✅ Yes |
| Batch processing, no shared context | **2-3x throughput** | ✅ Yes |
| Single user, interactive, unique prompts | **Modest gains** | ⚠️ Okay |
| Training/fine-tuning | **N/A** | ❌ Wrong tool |

## Decision Table: What Gains to Expect

| Feature | Best Case | Average Case | Worst Case |
|---------|-----------|--------------|------------|
| **PagedAttention** | 4x throughput | 2-3x throughput | 1.5x (very long sequences) |
| **Prefix Caching** | 90% TTFT reduction | 40-60% | 0% (no shared prefixes) |
| **Continuous Batching** | 10x utilization | 3-5x | 1x (single request) |

**Bottom Line**: If you're serving LLM inference at scale, vLLM is likely your best open-source option. The memory efficiency from PagedAttention alone can cut your GPU costs by 50-75%.

---

# 2. Intern Onboarding Explanation

Welcome! This section will build your mental model of how vLLM works, step by step.

## Step 1: Tokenize → Prefill → KV Cache Growth → Decode Loop

### The Complete Request Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        REQUEST LIFECYCLE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. TOKENIZE           2. PREFILL              3. DECODE (loop)         │
│  ┌─────────┐          ┌─────────────┐         ┌─────────────┐          │
│  │ "Hello" │ ──────►  │ Compute KV  │ ──────► │ Generate    │          │
│  │ "world" │          │ for ALL     │         │ ONE token   │ ◄──┐     │
│  └─────────┘          │ tokens at   │         │ at a time   │ ───┘     │
│       │               │ once        │         └─────────────┘          │
│       ▼               └─────────────┘               │                  │
│  [15496, 995]              │                        ▼                  │
│  (token IDs)               ▼                   Token by token:         │
│                      KV Cache Created:         "!" → "How" → "are"...  │
│                      K: [2 x heads x dim]                              │
│                      V: [2 x heads x dim]                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### What Happens in Each Phase:

**1. Tokenization**: Convert text to token IDs
```python
"Hello world" → [15496, 995]
```

**2. Prefill (Prompt Processing)**:
- Process ALL prompt tokens in parallel
- Compute attention: every token attends to all previous tokens
- Generate K and V tensors for each layer (the "KV Cache")
- Output: first generated token + KV cache

**3. Decode (Token-by-Token Generation)**:
- Process ONE new token at a time
- New token attends to ALL previous KV values
- Append new K, V to the cache
- Repeat until stop condition

### Why Decode is Slow

```
Decode Step for Token N:
┌────────────────────────────────────────────────────────────────┐
│ 1. Load query vector for new token    (small read)             │
│ 2. Load ALL previous K vectors        (HUGE read: N-1 tokens)  │
│ 3. Compute Q·K attention scores       (small compute)          │
│ 4. Load ALL previous V vectors        (HUGE read: N-1 tokens)  │
│ 5. Weighted sum of V                  (small compute)          │
│ 6. Append new K, V to cache           (small write)            │
└────────────────────────────────────────────────────────────────┘
         ↑
    Memory bandwidth is the bottleneck!
    GPU is idle 90% of the time waiting for memory.
```

## Step 2: What is the KV Cache (K/V Tensors)?

For each layer in the transformer, and for each token, we store:
- **K (Key)**: Used to compute attention scores
- **V (Value)**: Used to compute weighted outputs

### Size Calculation

```
KV Cache Size per Token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element

Example (Llama 2 70B):
- 80 layers, 8 KV heads (GQA), 128 head_dim, FP16
- Per token: 2 × 80 × 8 × 128 × 2 = 327,680 bytes = 320 KB

For 4096 tokens: 320 KB × 4096 = 1.28 GB per request!
```

### Why Fragmentation Matters

**Traditional Approach (Contiguous Allocation)**:
```
┌────────────────────────────────────────────────────────────────┐
│   Request 1 (reserved 1000)   │   Request 2 (reserved 1000)    │
│   ████████░░░░░░░░░░░░░░░░░░  │   ████░░░░░░░░░░░░░░░░░░░░░░   │
│   (using 400)                 │   (using 200)                   │
└────────────────────────────────────────────────────────────────┘
     60% wasted!                      80% wasted!
```

**PagedAttention Approach**:
```
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ R1  │ R1  │ R1  │ R1  │ R2  │ R2  │free │free │free │free │
│ B0  │ B1  │ B2  │ B3  │ B0  │ B1  │     │     │     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  Only allocate blocks as needed. Free blocks available for new requests!
```

## Step 3: PagedAttention Mental Model

### Blocks/Pages, Allocation, Reuse, Eviction

Think of KV cache memory like a parking lot with numbered spaces:

```
                        BLOCK POOL (GPU Memory)
┌─────────────────────────────────────────────────────────────────┐
│  Block 0  │  Block 1  │  Block 2  │  Block 3  │  Block 4  │ ... │
│  16 slots │  16 slots │  16 slots │  16 slots │  16 slots │     │
│ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │ ┌───────┐ │     │
│ │K₀..K₁₅│ │ │K₁₆.K₃₁│ │ │ FREE  │ │ │ FREE  │ │ │K₀..K₁₅│ │     │
│ │V₀..V₁₅│ │ │V₁₆.V₃₁│ │ │       │ │ │       │ │ │V₀..V₁₅│ │     │
│ └───────┘ │ └───────┘ │ └───────┘ │ └───────┘ │ └───────┘ │     │
│  Request1 │  Request1 │  Available│  Available│  Request2 │     │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
            Block Table (per request, maps logical → physical)
            ┌─────────────────────────────────────────┐
            │ Request 1: [0, 1]        (2 blocks)     │
            │ Request 2: [4]           (1 block)      │
            └─────────────────────────────────────────┘
```

**Key Operations**:
1. **Allocate**: Pop blocks from free queue as tokens grow
2. **Append**: Write new K/V to current block's next slot
3. **Reuse**: If prefix matches (hash lookup), skip recomputation
4. **Evict**: When memory pressure, free LRU cached blocks

## Step 4: Automatic Prefix Caching Mental Model

### The Radix Tree / Hash-Based Approach

vLLM uses **hash-based prefix caching** (not a traditional radix tree):

```
Block Hash = hash(parent_block_hash, block_tokens, extra_keys)

Example:
┌─────────────────────────────────────────────────────────────────────────┐
│                    "You are a helpful assistant. User: "                │
│                                                                         │
│ Block 0: [You, are, a, helpful]    hash = H₀ = hash(NONE_HASH, tokens)  │
│ Block 1: [assistant, ., User, :]   hash = H₁ = hash(H₀, tokens)         │
│                                                                         │
│ When new request arrives with same prefix:                              │
│   1. Compute hash for Block 0 → matches H₀ → CACHE HIT!                │
│   2. Compute hash for Block 1 → matches H₁ → CACHE HIT!                │
│   3. Skip prefill for these blocks, reuse existing KV                   │
└─────────────────────────────────────────────────────────────────────────┘
```

Source: [vLLM Prefix Caching Design Doc](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html)

## ASCII Diagrams

### a) KV Cache as Pages/Blocks Across Requests

```
     GPU MEMORY KV CACHE (Block Pool)
     ═══════════════════════════════════════════════════════════════

     Block 0    Block 1    Block 2    Block 3    Block 4    Block 5
    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
    │Request │ │Request │ │Request │ │Request │ │ FREE   │ │Request │
    │   A    │ │   A    │ │   B    │ │   B    │ │(LRU    │ │   A    │
    │Block 0 │ │Block 1 │ │Block 0 │ │Block 1 │ │Evict)  │ │Block 2 │
    │tokens  │ │tokens  │ │tokens  │ │tokens  │ │        │ │tokens  │
    │[0:16]  │ │[16:32] │ │[0:16]  │ │[16:32] │ │        │ │[32:48] │
    └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘
         │          │          │          │                    │
         └──────────┴──────────│──────────│────────────────────┘
                               │          │
              Request A        │          │
              Block Table:     │          │
              [0, 1, 5]        │          │
                               └──────────┘
                               Request B Block Table: [2, 3]
```

### b) Continuous Batching Timeline

```
    Time →  t0    t1    t2    t3    t4    t5    t6    t7    t8
    ════════════════════════════════════════════════════════════

    Req A   [PPPP] [D]   [D]   [D]   [D]   [DONE]
            prefill decode decode decode decode

    Req B         [PP]  [D]   [D]   [D]   [D]   [D]   [DONE]
                  prefill (starts when A decoding)

    Req C               [PPPPPP]    [D]   [D]   [D]   [D]   [DONE]
                        long prefill    decoding

    Req D                           [PP]  [D]   [D]   [DONE]
                                    prefill (starts mid-batch)

    ┌─────────────────────────────────────────────────────────────┐
    │ Key Insight: No waiting for batches to complete!            │
    │ New requests start immediately when capacity available.     │
    │ GPU utilization maximized by mixing prefill + decode.       │
    └─────────────────────────────────────────────────────────────┘
```

### c) Prefix Cache Lookup and Partial Match Example

```
    CACHED BLOCK HASHES (in memory)
    ════════════════════════════════════════════════════════════════

    Hash Table:
    ┌─────────────────────────────────────────────────────────────┐
    │  H("You are a help")        →  Block ID: 42                 │
    │  H(H(...), "ful assistant") →  Block ID: 17                 │
    │  H(H(...), ". User: Tell")  →  Block ID: 89                 │
    └─────────────────────────────────────────────────────────────┘

    NEW REQUEST: "You are a helpful assistant. User: What is..."
    ════════════════════════════════════════════════════════════════

    Block 0: "You are a help"       → hash matches H₀ → CACHE HIT ✓
    Block 1: "ful assistant"        → hash matches H₁ → CACHE HIT ✓
    Block 2: ". User: What"         → hash computed   → CACHE MISS ✗
                                      (H₂ not in table)

    Result: Reuse blocks 42, 17. Allocate new block for ". User: What"
            Skip prefill for first 2 blocks (32 tokens saved!)
```

### d) How Paged KV + Prefix Caching Compose

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    COMPLETE WORKFLOW                             │
    └─────────────────────────────────────────────────────────────────┘

    1. REQUEST ARRIVES
       │
       ▼
    2. HASH COMPUTATION (for each block's worth of tokens)
       │  hash(parent_hash, block_tokens, extra_keys)
       │
       ▼
    3. PREFIX CACHE LOOKUP ──────────────────────────────────────┐
       │                                                          │
       │  Cache Hit?                                              │
       │  ┌────┐                                                  │
       │  │ Y  │──► Touch block (move to end of LRU queue)       │
       │  │    │    Increment ref_cnt                             │
       │  │    │    Skip prefill for this block                   │
       │  └────┘                                                  │
       │  ┌────┐                                                  │
       │  │ N  │──► Continue to allocation                        │
       │  └────┘                                                  │
       │                                                          │
       ▼                                                          │
    4. BLOCK ALLOCATION (from free queue)                         │
       │  Pop from head of free queue                             │
       │  If cached block → evict (remove from hash table)        │
       │  Increment ref_cnt                                       │
       │                                                          │
       ▼                                                          │
    5. PREFILL (for non-cached blocks only)                       │
       │  Compute K, V for new tokens                             │
       │  Write to allocated blocks                               │
       │                                                          │
       ▼                                                          │
    6. CACHE NEW FULL BLOCKS                                      │
       │  Compute block hash                                      │
       │  Insert into hash table                                  │
       │                                                          │
       ▼                                                          │
    7. DECODE LOOP ◄──────────────────────────────────────────────┘
       │  Generate token
       │  Append K, V to current block
       │  If block full → cache it
       │  If stop condition → free blocks
       │
       ▼
    8. FREE ON COMPLETION
       Decrement ref_cnt for all blocks
       If ref_cnt == 0 → add to free queue (tail, reverse order)
       Block stays in cache until evicted
```

---

# 3. Engineering Deep Dive

## PagedAttention: The Problem It Solves

### Fragmentation and Dynamic Sequences

**Problem Statement** (from [vLLM Paper](https://arxiv.org/abs/2309.06180)):
1. **Internal Fragmentation**: Pre-allocating max sequence length wastes memory
2. **External Fragmentation**: Variable-length sequences create memory gaps
3. **Memory Sharing Impossible**: Duplicate prefixes recomputed for each request

**PagedAttention Solution**:
- Divide KV cache into fixed-size blocks (default: 16 tokens/block)
- Maintain block table per request mapping logical → physical blocks
- Allocate blocks on-demand as sequences grow
- Enable memory sharing through reference counting

### Implementation in vLLM v1

From `vllm/v1/core/block_pool.py`:

```python
class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.
    """

    def __init__(self, num_gpu_blocks: int, enable_caching: bool, ...):
        # All KV-cache blocks, pre-allocated at startup
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Doubly-linked list for O(1) eviction
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # Hash table for prefix cache lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
```

## Block Manager: Allocation Strategy, Reuse, Eviction

### Allocation Strategy

From `vllm/v1/core/kv_cache_manager.py`:

```python
def allocate_slots(self, request, num_new_tokens, ...):
    """
    Blocks layout:
    ----------------------------------------------------------------------
    | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
    ----------------------------------------------------------------------
                                              |   < to be computed >     |
    ----------------------------------------------------------------------

    The allocation has three stages:
    - Free unnecessary blocks in `comp` and check sufficient free blocks
    - Handle prefix tokens: free unnecessary, allocate for external
    - Allocate new blocks for tokens to be computed
    """
```

### LRU Eviction with Prefix Awareness

From `vllm/v1/core/kv_cache_utils.py`:

```python
class FreeKVCacheBlockQueue:
    """Doubly linked list of free blocks ordered for LRU eviction:
    1. Least recently used block at the front (LRU).
    2. If two blocks have same access time, the one with more hash tokens
       (tail of a block chain) is at the front.

    This order is maintained by reversing block order when freeing a request.
    """
```

**Key Insight**: Blocks are freed in reverse order so that the last blocks of a sequence (most unique, least likely to be reused) are evicted first.

## Prefix Caching: Matching Granularity

### Token Sequence → Prefix Match

From `vllm/v1/core/kv_cache_utils.py`:

```python
def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes hash for prefix caching.
    Args:
        parent_block_hash: Hash of parent block (None if first block)
        curr_block_token_ids: Tokens in current block (must be full)
        extra_keys: LoRA IDs, multimodal hashes, cache salts
    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    return BlockHash(
        hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
    )
```

### Insert/Lookup/Update Costs

| Operation | Complexity | Details |
|-----------|-----------|---------|
| **Lookup** | O(1) | Hash table lookup by block hash |
| **Insert** | O(1) amortized | Hash table insertion when block becomes full |
| **Update** | N/A | Blocks are immutable once cached |
| **Eviction** | O(1) | Pop from head of LRU queue |
| **Touch (hit)** | O(1) | Remove from free queue, increment ref_cnt |

### Eviction Policy and Memory Accounting

From `vllm/v1/core/block_pool.py`:

```python
def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
    """When allocating a block that's cached, evict it first."""
    block_hash = block.block_hash
    if block_hash is None:
        return False  # Not cached, no eviction needed

    # Remove from hash table
    self.cached_block_hash_to_block.pop(block_hash, block.block_id)
    block.reset_hash()
    return True
```

**What Gets Dropped First**:
1. Blocks with `ref_cnt == 0` (not in use by any request)
2. Among those, LRU ordering: least recently used first
3. Within same access time: blocks with more prefix tokens first (more unique)

## Scheduler Interactions: Continuous Batching Mechanics

### How Scheduling Affects TTFT vs Throughput

From `vllm/v1/core/sched/scheduler.py`:

```python
def schedule(self) -> SchedulerOutput:
    """
    NOTE: There's no "decoding phase" nor "prefill phase" in the scheduler.
    Each request just has num_computed_tokens and num_tokens_with_spec.
    At each step, the scheduler assigns tokens to requests so that
    num_computed_tokens can catch up to num_tokens_with_spec.

    This is general enough to cover:
    - Chunked prefills
    - Prefix caching
    - Speculative decoding
    - "Jump decoding" optimization
    """
```

**Key Trade-offs**:

| Setting | Effect on TTFT | Effect on Throughput |
|---------|---------------|---------------------|
| `max_num_batched_tokens` ↑ | ↓ TTFT (more tokens/step) | ↑ Throughput |
| `max_num_seqs` ↑ | ↑ TTFT (more contention) | ↑ Throughput |
| `enable_chunked_prefill=True` | ↓ TTFT (prefills don't block) | Slight ↓ |
| `enable_prefix_caching=True` | ↓↓ TTFT (skip prefill) | ↑ (reuse compute) |

### Cache-Aware Batching Ideas

**What Increases Hit Rate**:
- Consistent system prompts across requests
- Request ordering that groups similar prefixes
- Longer block size (more tokens per hash, but coarser granularity)

**What Destroys Hit Rate**:
- Unique per-request context (e.g., random document IDs)
- High memory pressure causing frequent evictions
- Very short requests that don't fill blocks

## Strengths, Weaknesses, and Failure Modes

### Best-Case Speedups

| Scenario | Speedup | Why |
|----------|---------|-----|
| Chatbot with system prompt | 3-5x TTFT | 90%+ prefix cache hit |
| RAG with shared documents | 2-4x TTFT | Document context cached |
| High concurrency serving | 3-4x throughput | Memory efficiency |

### Worst-Case Overheads / Churn Scenarios

| Scenario | Overhead | Cause |
|----------|----------|-------|
| Unique prompts, no sharing | ~5% overhead | Hash computation, block management |
| Very long sequences | Memory pressure | KV cache still grows linearly |
| Rapid cache turnover | Eviction thrashing | Cache too small for working set |

### Multi-Tenant Pitfalls

**Noisy Neighbor Problem**:
- One tenant with large, unique requests can evict cached blocks for others
- Solution: Use `cache_salt` for tenant isolation

**Cache Thrash**:
- Many tenants with different prefixes competing for cache
- Solution: Increase `gpu_memory_utilization` or partition by tenant

### Correctness/Isolation Concerns

**LoRA/Adapters**:
- LoRA name is included in block hash via `extra_keys`
- Different LoRA adapters will NOT share cached blocks

**Per-User Prompts**:
- Use `cache_salt` in request to isolate user caches

**Privacy**:
- Cache hit timing attacks possible without `cache_salt`
- Production deployments should use tenant-specific salts

## Common Failure Modes + Symptoms + Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| OOM during startup | `gpu_memory_utilization` too high | Lower to 0.85-0.90 |
| High TTFT despite prefix caching | Cache misses (check hit rate) | Align prompts to block boundaries |
| Throughput plateaus at low batch size | Memory pressure | Increase GPU memory or reduce `max_model_len` |
| Inconsistent latencies | Cache eviction churn | Monitor eviction rate, increase cache |
| Wrong outputs with LoRA | LoRA not in hash (old version) | Upgrade vLLM |
| Prefix cache hit rate = 0 | Caching disabled or no shared prefixes | Check `enable_prefix_caching=True` |

---

# 4. Practical Playbook: How to Get Wins Fast

## Checklist to Maximize Performance

### Prompt/Prefix Design Patterns That Raise Hit Rate

✅ **DO**:
- Put system prompt FIRST (before user input)
- Use consistent system prompts across requests
- Pad prompts to block boundaries (default: 16 tokens) for predictable caching
- Batch requests with similar prefixes together

❌ **DON'T**:
- Include timestamps or unique IDs at the start of prompts
- Randomize system prompt ordering
- Use very short prompts that don't fill a block

### Request Shaping for Batching

```python
# Good: Length-bucket your requests
short_requests = [r for r in requests if len(r.prompt) < 512]
medium_requests = [r for r in requests if 512 <= len(r.prompt) < 2048]
long_requests = [r for r in requests if len(r.prompt) >= 2048]

# Process in order to maximize batching efficiency
for batch in [short_requests, medium_requests, long_requests]:
    llm.generate(batch, ...)
```

### When to Separate Prefill-Heavy vs Decode-Heavy Traffic

**Prefill-Heavy** (long prompts, short outputs):
- RAG queries
- Document summarization
- Code completion with large context

**Decode-Heavy** (short prompts, long outputs):
- Story generation
- Code generation
- Multi-turn chat with history

**For production at scale**: Consider disaggregated serving with separate prefill and decode clusters (vLLM supports this via P/D disaggregation).

## Metrics to Instrument

### Core Latency Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **TTFT** (Time to First Token) | Prefill latency + queue time | <500ms interactive |
| **TPS** (Tokens per Second) | Decode throughput | Model-dependent |
| **Latency P50/P95/P99** | End-to-end request latency | P99 < 2-3x P50 |
| **ITL** (Inter-Token Latency) | Time between tokens | <50ms interactive |

### Memory Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **GPU Memory Usage** | Total VRAM used | <95% to avoid OOM |
| **KV Cache Usage** | `kv_cache_usage` | 70-90% (lower = headroom) |
| **Block Utilization** | Blocks in use / total | Monitor for pressure |
| **Eviction Rate** | Blocks evicted / time | Low = good cache efficiency |

### Prefix Cache Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Prefix Cache Hit Rate** | Cached tokens / total prompt tokens | >50% for chatbots |
| **Cache Churn** | Evictions per request | Low = stable cache |
| **Compute Saved** | FLOPS avoided via caching | Higher = better |

### How to Access Metrics

```python
# Via OpenAI-compatible server
# GET /metrics (Prometheus format)

# Via LLM class
from vllm import LLM
llm = LLM(model="...", enable_prefix_caching=True)
# Metrics are logged when log_stats=True (default in server)
```

## Benchmark Plan

### Microbenchmark 1: Prefill-Only

```bash
# Measure pure prefill performance
python -m vllm.benchmarks.benchmark_latency \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 2048 \
    --output-len 1 \
    --batch-size 1 8 16 32 \
    --num-iters 100
```

### Microbenchmark 2: Decode-Only

```bash
# Measure pure decode performance
python -m vllm.benchmarks.benchmark_latency \
    --model meta-llama/Llama-2-7b-hf \
    --input-len 128 \
    --output-len 512 \
    --batch-size 1 8 16 32 \
    --num-iters 100
```

### Microbenchmark 3: Mixed Workload

```bash
# Measure realistic mixed workload
python -m vllm.benchmarks.benchmark_throughput \
    --model meta-llama/Llama-2-7b-hf \
    --dataset-path your_dataset.json \
    --num-prompts 1000 \
    --request-rate 10
```

### Realistic Workload 1: Multi-Turn Chat

```python
# benchmark_chat.py
import time
from vllm import LLM, SamplingParams

system_prompt = "You are a helpful assistant. " * 50  # ~200 tokens

conversations = [
    [system_prompt + "User: Hello!", "User: How are you?", "User: Tell me a joke"],
    [system_prompt + "User: Hi there!", "User: What's the weather?", "User: Thanks!"],
    # ... 100 more conversations
]

llm = LLM(model="meta-llama/Llama-2-7b-hf", enable_prefix_caching=True)
sampling = SamplingParams(max_tokens=100)

# Warm up cache with system prompt
llm.generate([system_prompt + "User: Warmup"], sampling)

# Measure with cache
start = time.time()
for conv in conversations:
    for turn in conv:
        llm.generate([turn], sampling)
print(f"Time with caching: {time.time() - start:.2f}s")
```

### Realistic Workload 2: Agent/Tool Loops

```python
# benchmark_agent.py
tool_descriptions = """
Available tools:
1. search(query) - Search the web
2. calculate(expr) - Evaluate math expressions
3. get_weather(city) - Get current weather
... [500+ tokens of tool descriptions]
"""

agent_requests = [
    tool_descriptions + "User: Search for Python tutorials",
    tool_descriptions + "User: Calculate 15% tip on $45.50",
    tool_descriptions + "User: What's the weather in NYC?",
    # ... shared tool prefix, different user queries
]
```

## Tuning Knobs (Actual vLLM Flags)

### Memory Configuration

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --gpu-memory-utilization 0.90 \       # Default 0.90, lower if OOM
    --max-model-len 4096 \                 # Limit context length
    --block-size 16                        # Tokens per block (default 16)
```

### Batching Configuration

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --max-num-batched-tokens 8192 \        # Tokens per step
    --max-num-seqs 256 \                   # Max concurrent sequences
    --enable-chunked-prefill               # Enable by default in v1
```

### Prefix Caching

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enable-prefix-caching \              # Enable APC
    --prefix-caching-hash-algo sha256      # Use SHA256 for multi-tenant
```

### Scheduling

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --scheduler-policy fcfs \              # First-come-first-served
    --enable-chunked-prefill \             # Don't block on long prefills
    --max-num-partial-prefills 1           # Concurrent chunked prefills
```

### Performance

```bash
vllm serve meta-llama/Llama-2-7b-hf \
    --enforce-eager \                      # Disable CUDA graphs (debug)
    --disable-log-stats                    # Reduce logging overhead
```

---

# 5. Minimal Starter Snippets

## Example: Running vLLM with Optimal Settings

### Server Mode (Recommended for Production)

```bash
# Start vLLM server with prefix caching and sensible defaults
vllm serve meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 256 \
    --enable-chunked-prefill
```

### Offline Mode (For Batch Processing)

```python
from vllm import LLM, SamplingParams

# Initialize with prefix caching
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    gpu_memory_utilization=0.90,
    enable_prefix_caching=True,
    max_num_batched_tokens=8192,
    max_num_seqs=256,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)
```

## Example: Client Request Flow Benefiting from Prefix Caching

```python
import requests

API_URL = "http://localhost:8000/v1/chat/completions"

# Shared system prompt (will be cached after first request)
SYSTEM_PROMPT = """You are an expert customer service agent for Acme Corp.
You have access to the following tools:
- check_order_status(order_id): Returns order status
- initiate_refund(order_id): Starts refund process
- escalate_to_human(reason): Transfers to human agent

Always be helpful, professional, and concise.
"""  # ~80 tokens, 5 blocks

def chat(user_message: str, cache_salt: str = None):
    """Send a chat request. First request caches system prompt."""
    payload = {
        "model": "meta-llama/Llama-2-7b-hf",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }
    if cache_salt:
        payload["cache_salt"] = cache_salt  # For multi-tenant isolation

    response = requests.post(API_URL, json=payload)
    return response.json()

# First request: ~500ms TTFT (prefills system prompt)
response1 = chat("What's the status of order #12345?")

# Subsequent requests: ~100ms TTFT (reuses cached system prompt!)
response2 = chat("I'd like a refund for order #12345")
response3 = chat("Connect me with a human agent")
```

## Pseudo-Code: Where Key Operations Happen

### Where Cache Lookup Happens

```python
# vllm/v1/core/kv_cache_manager.py
class KVCacheManager:
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """
        Called by scheduler BEFORE allocating slots for a new request.
        Returns (cached_blocks, num_computed_tokens).
        """
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # HERE: The cache lookup happens!
        computed_blocks, num_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes,  # Pre-computed hashes for request blocks
                max_cache_hit_length=request.num_tokens - 1
            )
        )
        return self.create_kv_cache_blocks(computed_blocks), num_computed_tokens
```

### Where KV Blocks Are Allocated

```python
# vllm/v1/core/block_pool.py
class BlockPool:
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """
        Called when allocating slots for new/extended requests.
        Pops from free queue, evicting cached blocks if necessary.
        """
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        if self.enable_caching:
            for block in ret:
                # If this block was cached, evict it first
                self._maybe_evict_cached_block(block)
                block.ref_cnt += 1
        else:
            for block in ret:
                block.ref_cnt += 1

        return ret
```

### Where Scheduling Decisions Happen

```python
# vllm/v1/core/sched/scheduler.py
class Scheduler:
    def schedule(self) -> SchedulerOutput:
        """
        Main scheduling loop. Called every step.
        """
        scheduled_new_reqs = []
        scheduled_running_reqs = []
        token_budget = self.max_num_scheduled_tokens

        # 1. Schedule RUNNING requests (decode step)
        for request in self.running:
            num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens
            num_new_tokens = min(num_new_tokens, token_budget)

            # Allocate KV blocks for new tokens
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens, ...
            )

            if new_blocks is not None:
                scheduled_running_reqs.append(request)
                token_budget -= num_new_tokens
            else:
                # Memory pressure: preempt lowest priority request
                self._preempt_request(...)

        # 2. Schedule WAITING requests (new requests)
        while self.waiting and token_budget > 0:
            request = self.waiting.peek_request()

            # Get cached blocks (prefix cache lookup)
            computed_blocks, num_computed = (
                self.kv_cache_manager.get_computed_blocks(request)
            )

            # Allocate remaining blocks
            new_blocks = self.kv_cache_manager.allocate_slots(
                request, num_new_tokens,
                new_computed_blocks=computed_blocks,
                num_new_computed_tokens=num_computed,
            )

            if new_blocks is not None:
                scheduled_new_reqs.append(request)
                request.num_computed_tokens = num_computed  # Skip cached portion!
```

---

# 6. What I Would Do Next Week

## Onboarding Plan for Engineers New to vLLM Inference Optimization

### Day 1-2: Build Intuition

- [ ] Read this document end-to-end
- [ ] Read the [vLLM Paper](https://arxiv.org/abs/2309.06180) (focus on Sections 3-4)
- [ ] Read [Prefix Caching Design Doc](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html)
- [ ] Run the prefix caching example: `examples/offline_inference/prefix_caching.py`

### Day 3-4: Get Hands-On

- [ ] Deploy vLLM server locally with a small model (e.g., `facebook/opt-125m`)
- [ ] Send requests via curl/Python, observe TTFT differences with prefix caching
- [ ] Run microbenchmarks from Section 4
- [ ] Monitor metrics via `/metrics` endpoint

### Day 5: Dive into Code

- [ ] Trace a request through the codebase:
  1. Start at `vllm/entrypoints/openai/api_server.py`
  2. Follow to `vllm/v1/core/sched/scheduler.py` (scheduling)
  3. Examine `vllm/v1/core/kv_cache_manager.py` (block allocation)
  4. Look at `vllm/v1/core/block_pool.py` (prefix caching)

- [ ] Set breakpoints and trace a single request's lifecycle
- [ ] Understand the block hash computation in `kv_cache_utils.py`

### Day 6-7: Experiment and Measure

- [ ] Create a workload representative of your production traffic
- [ ] Benchmark with and without prefix caching
- [ ] Tune `max_num_batched_tokens`, `max_num_seqs`, `gpu_memory_utilization`
- [ ] Document findings and share with team

### Ongoing: Stay Current

- [ ] Watch vLLM GitHub for new releases and features
- [ ] Join vLLM Discord/Slack for community discussions
- [ ] Experiment with speculative decoding, disaggregated serving
- [ ] Consider contributing optimizations back to the project

---

## References

1. **vLLM Paper**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

2. **vLLM Documentation**: [docs.vllm.ai](https://docs.vllm.ai/)

3. **Prefix Caching Design Doc**: [docs.vllm.ai/en/latest/design/automatic_prefix_caching.html](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html)

4. **vLLM GitHub Repository**: [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

---

*Document Version: 1.0*  
*Based on vLLM codebase as of December 2024*  
*Author: Generated via comprehensive codebase analysis*
