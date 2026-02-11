# Qwen3-VL-32B Inference Walkthrough: H100 vs B200 with vLLM

A real, numbers-grounded walkthrough of what actually happens when you run
`vllm serve Qwen/Qwen3-VL-32B-Instruct` and send it a request with an image.

Every number ties back to actual weight matrix dimensions from the Qwen3-VL-32B
config and actual GPU specs.

---

## Table of Contents

1. [The Hardware](#the-hardware-what-youre-working-with)
2. [The Model Architecture](#the-model-qwen3-vl-32b-layer-by-layer)
3. [Step 0: Model Loading — GPU Memory Layout](#step-0-model-loading--what-goes-where-in-gpu-memory)
4. [Step 1: A Request Arrives](#step-1-a-request-arrives)
5. [Step 2: Scheduler Allocates KV Cache Blocks](#step-2-scheduler-allocates-kv-cache-blocks)
6. [Step 3: Prefill — Vision Encoder](#step-3-prefill--the-vision-encoder-runs-first)
7. [Step 4: Prefill — 64-Layer LLM Forward Pass](#step-4-prefill--the-64-layer-llm-forward-pass)
8. [Step 5: Decode — Generating Tokens One at a Time](#step-5-decode--generating-tokens-one-at-a-time)
9. [Step 6: How Batching Saves Decode](#step-6-how-batching-saves-decode)
10. [Step 7: KV Cache Memory Layout](#step-7-what-the-kv-cache-actually-looks-like-in-memory)
11. [Summary: Full Picture Timeline](#summary-the-full-picture)

---

## The Hardware: What You're Working With

```
 ┌───────────── H100 SXM (ONE GPU) ──────────────┐
 │                                                 │
 │  132 SMs × 4 Tensor Cores each = 528 TCs       │
 │  FP16 peak: ~990 TFLOPS                        │
 │  FP8 peak:  ~1,979 TFLOPS                      │
 │                                                 │
 │  ┌───────────────────────────────────────────┐  │
 │  │         HBM3: 80 GB                       │  │
 │  │         Bandwidth: 3.35 TB/s              │  │
 │  │                                           │  │
 │  │   This is where weights + KV cache live   │  │
 │  └───────────────────────────────────────────┘  │
 │                                                 │
 │  L2 Cache: 50 MB                                │
 │  Shared Mem per SM: 228 KB                      │
 └─────────────────────────────────────────────────┘

 ┌───────────── B200 (ONE GPU) ───────────────────┐
 │                                                 │
 │  192 SMs                                        │
 │  FP16 peak: ~2,250 TFLOPS                      │
 │  FP8 peak:  ~4,500 TFLOPS                      │
 │                                                 │
 │  ┌───────────────────────────────────────────┐  │
 │  │         HBM3e: 192 GB                     │  │
 │  │         Bandwidth: 8.0 TB/s               │  │
 │  │                                           │  │
 │  │   2.4x more memory, 2.4x more bandwidth  │  │
 │  └───────────────────────────────────────────┘  │
 │                                                 │
 └─────────────────────────────────────────────────┘
```

---

## The Model: Qwen3-VL-32B Layer by Layer

```
 Qwen3-VL-32B-Instruct — Exact Architecture
 ════════════════════════════════════════════

 VISION ENCODER (Qwen3_VisionTransformer)
 ─────────────────────────────────────────
   Patch embed:  3D Conv (patch=14, temporal=2)
   Layers:       32 transformer blocks
   Hidden dim:   1280
   Heads:        16
   MLP hidden:   5120
   Merge:        spatial_merge_size=2 (4 patches → 1 token)
   Output dim:   5120 (= out_hidden_size, matches LLM hidden)
   ~0.6B parameters, ~1.2 GB in BF16

 LLM DECODER (Qwen3 dense transformer)
 ──────────────────────────────────────
   Layers:          64
   Hidden dim:      5120
   Attention heads: 64 (query)
   KV heads:        8 (grouped query attention, 8:1 ratio)
   Head dim:        128
   FFN hidden:      25600
   Vocab:           151,936
   ~32.2B parameters, ~64.4 GB in BF16, ~32.2 GB in FP8

 TOTAL: ~32.8B params → ~65.6 GB BF16 → ~32.8 GB FP8
```

---

## Step 0: Model Loading — What Goes Where in GPU Memory

### Single H100, FP8 quantized

```
 vllm serve Qwen/Qwen3-VL-32B-Instruct --quantization fp8

 H100 HBM (80 GB total)
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  MODEL WEIGHTS (FP8): ~32.8 GB                              │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  Embedding table:  151,936 × 5,120 × 1 byte = 0.74 GB│  │
 │  │                                                        │  │
 │  │  64 × Decoder Layer, each containing:                  │  │
 │  │  ┌──────────────────────────────────────────────────┐  │  │
 │  │  │ Q proj: 5120 × 8192  × 1B = 40 MB  (64 heads)  │  │  │
 │  │  │ K proj: 5120 × 1024  × 1B =  5 MB  (8 KV heads)│  │  │
 │  │  │ V proj: 5120 × 1024  × 1B =  5 MB  (8 KV heads)│  │  │
 │  │  │ O proj: 8192 × 5120  × 1B = 40 MB              │  │  │
 │  │  │ Gate:   5120 × 25600 × 1B = 125 MB             │  │  │
 │  │  │ Up:     5120 × 25600 × 1B = 125 MB             │  │  │
 │  │  │ Down:  25600 × 5120  × 1B = 125 MB             │  │  │
 │  │  │ Norms: ~20 KB (negligible)                      │  │  │
 │  │  │                                   ≈ 465 MB/layer│  │  │
 │  │  └──────────────────────────────────────────────────┘  │  │
 │  │  64 layers × 465 MB = ~29.0 GB                         │  │
 │  │                                                        │  │
 │  │  Vision encoder: ~0.6 GB (FP8)                         │  │
 │  │  LM head + other: ~0.7 GB                              │  │
 │  │  Projector/merger: ~0.2 GB                             │  │
 │  │                                         TOTAL ≈ 31.3 GB│  │
 │  └────────────────────────────────────────────────────────┘  │
 │                                                              │
 │  KV CACHE: ~42 GB (fills remaining memory)                  │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  Organized as BLOCKS of 16 tokens each                 │  │
 │  │                                                        │  │
 │  │  Per token per layer:                                  │  │
 │  │    K: 8 heads × 128 dim × 2 bytes (FP16) = 2,048 B    │  │
 │  │    V: 8 heads × 128 dim × 2 bytes (FP16) = 2,048 B    │  │
 │  │    Total: 4,096 bytes per token per layer              │  │
 │  │                                                        │  │
 │  │  Per token ALL 64 layers:                              │  │
 │  │    64 × 4,096 = 262,144 bytes = 256 KB per token      │  │
 │  │                                                        │  │
 │  │  42 GB / 256 KB = ~168,000 tokens of KV cache          │  │
 │  │  = ~5 concurrent 32K-context requests                  │  │
 │  │  = ~21 concurrent 8K-context requests                  │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                                                              │
 │  CUDA overhead + activations: ~5-7 GB                        │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘
      ▲
      │ 80 GB total, tightly packed
```

### Single B200, BF16 full precision

```
 vllm serve Qwen/Qwen3-VL-32B-Instruct    (BF16 default)

 B200 HBM (192 GB total)
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  MODEL WEIGHTS (BF16): ~65.6 GB                             │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  Same structure, but every weight is 2 bytes not 1     │  │
 │  │  64 layers × ~930 MB/layer = ~58 GB                    │  │
 │  │  + embeddings, vision, head ≈ 65.6 GB total            │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                                                              │
 │  KV CACHE: ~118 GB                                          │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  Same 256 KB per token (KV cache is always FP16)       │  │
 │  │                                                        │  │
 │  │  118 GB / 256 KB = ~472,000 tokens of KV cache         │  │
 │  │  = ~14 concurrent 32K-context requests                 │  │
 │  │  = ~59 concurrent 8K-context requests                  │  │
 │  │                                                        │  │
 │  │  MUCH more headroom than H100                          │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                                                              │
 │  CUDA overhead + activations: ~7 GB                          │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘
      ▲
      │ 192 GB total, comfortable fit
```

---

## Step 1: A Request Arrives

```
 User sends:
 POST /v1/chat/completions
 {
   "messages": [{
     "role": "user",
     "content": [
       {"type": "image_url", "image_url": {"url": "photo.jpg"}},  ← 1024×768 JPEG
       {"type": "text", "text": "What is in this image?"}
     ]
   }]
 }

 API Server (vllm/entrypoints/openai/api_server.py)
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  1. Parse the JSON request                                   │
 │  2. Download the image from URL                              │
 │  3. Pass to the multimodal processor                         │
 │                                                              │
 │  ┌─────────── Multimodal Processor ───────────────────────┐  │
 │  │  Qwen3VLMultiModalProcessor (qwen3_vl.py)              │  │
 │  │                                                        │  │
 │  │  Image: 1024×768 pixels                                │  │
 │  │    / patch_size(14) = 73×54 patches                    │  │
 │  │    / spatial_merge(2×2) = 36×27 merged patches         │  │
 │  │    = 972 vision tokens (placeholder count)              │  │
 │  │                                                        │  │
 │  │  Token sequence built:                                 │  │
 │  │  [BOS] <|vision_start|> [IMG×972] <|vision_end|>       │  │
 │  │  What is in this image? [EOS]                          │  │
 │  │                                                        │  │
 │  │  Total: ~980 tokens                                    │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                                                              │
 │  ──▶ Send EngineCoreRequest to Engine Core                   │
 └──────────────────────────────────────────────────────────────┘
```

---

## Step 2: Scheduler Allocates KV Cache Blocks

```
 Engine Core — Scheduler (vllm/v1/core/sched/scheduler.py)
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  New request: ~980 tokens to prefill                         │
 │                                                              │
 │  Step 1: Check KV cache budget                               │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  Need: ceil(980 / 16) = 62 blocks                      │  │
 │  │  Each block = 16 tokens × 256 KB/token = 4 MB           │  │
 │  │  Total: 62 × 4 MB = 248 MB for this request            │  │
 │  │                                                        │  │
 │  │  KV Cache Manager (kv_cache_manager.py):                │  │
 │  │    Free blocks: 10,500 (on H100 FP8)                   │  │
 │  │    After alloc: 10,438 free                             │  │
 │  │    ✓ Approved                                          │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                                                              │
 │  Step 2: Check token budget                                  │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  max_num_batched_tokens = 8192 (default)               │  │
 │  │  This request wants 980 tokens for prefill              │  │
 │  │  Other running requests: 12 decodes × 1 token = 12     │  │
 │  │  Total: 980 + 12 = 992 tokens  ✓ Under 8192            │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                                                              │
 │  Step 3: Build SchedulerOutput                               │
 │  ┌────────────────────────────────────────────────────────┐  │
 │  │  Batch for this step:                                  │  │
 │  │    Req_new: prefill 980 tokens, blocks [B100..B161]    │  │
 │  │    Req_1:   decode 1 token                             │  │
 │  │    Req_2:   decode 1 token                             │  │
 │  │    ...                                                 │  │
 │  │    Req_12:  decode 1 token                             │  │
 │  └────────────────────────────────────────────────────────┘  │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘
```

---

## Step 3: Prefill — The Vision Encoder Runs First

```
 GPU Model Runner — execute_model() (gpu_model_runner.py)
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  Phase A: VISION ENCODER                                             │
 │  ═══════════════════════                                             │
 │  Only runs for the new request (has image). Decode requests skip.    │
 │                                                                      │
 │  Input image pixels: [972 patches × 3 channels × 14 × 14]           │
 │                                                                      │
 │  ┌──── Qwen3_VisionTransformer (qwen3_vl.py:320) ────────────────┐  │
 │  │                                                                │  │
 │  │  3D Conv Patch Embed:                                          │  │
 │  │    [972, 3, 2, 14, 14] ──Conv3D──▶ [972, 1280]                │  │
 │  │    972 patches → 972 embeddings of dim 1280                    │  │
 │  │                                                                │  │
 │  │         ┌─── 32 Vision Transformer Blocks ──────────────────┐  │  │
 │  │         │                                                   │  │  │
 │  │         │  For each of the 32 layers:                       │  │  │
 │  │         │                                                   │  │  │
 │  │         │   ┌──────────────────────────────────┐            │  │  │
 │  │         │   │  LayerNorm                       │            │  │  │
 │  │         │   │  [972, 1280] → [972, 1280]       │            │  │  │
 │  │         │   └───────────────┬──────────────────┘            │  │  │
 │  │         │                   ▼                               │  │  │
 │  │         │   ┌──────────────────────────────────┐            │  │  │
 │  │         │   │  Self-Attention (16 heads)       │            │  │  │
 │  │         │   │  Q,K,V: [972, 1280] each         │            │  │  │
 │  │         │   │  head_dim = 80                   │            │  │  │
 │  │         │   │  All patches attend to each other│            │  │  │
 │  │         │   │  + RoPE positional encoding      │            │  │  │
 │  │         │   └───────────────┬──────────────────┘            │  │  │
 │  │         │                   ▼                               │  │  │
 │  │         │   ┌──────────────────────────────────┐            │  │  │
 │  │         │   │  LayerNorm                       │            │  │  │
 │  │         │   └───────────────┬──────────────────┘            │  │  │
 │  │         │                   ▼                               │  │  │
 │  │         │   ┌──────────────────────────────────┐            │  │  │
 │  │         │   │  MLP (SiLU activation)           │            │  │  │
 │  │         │   │  1280 → 5120 → 1280              │            │  │  │
 │  │         │   └───────────────┬──────────────────┘            │  │  │
 │  │         │                   ▼                               │  │  │
 │  │         │             [972, 1280]                           │  │  │
 │  │         │                                                   │  │  │
 │  │         └───────────────────┬───────────────────────────────┘  │  │
 │  │                             ▼                                  │  │
 │  │  Patch Merger (spatial_merge_size=2):                          │  │
 │  │    Groups of 2×2 = 4 adjacent patches → 1 merged token        │  │
 │  │    [972, 1280] → reshape to [243, 4×1280] = [243, 5120]       │  │
 │  │    → LayerNorm → Linear(5120→5120) → GELU → Linear(5120→5120) │  │
 │  │    → [243, 5120]                                               │  │
 │  │                                                                │  │
 │  │  * The 972 was pre-merge placeholder count. After merge:       │  │
 │  │    972 / 4 = 243 actual vision tokens, each dim 5120           │  │
 │  │    (matches LLM hidden_size, ready to inject)                  │  │
 │  │                                                                │  │
 │  └────────────────────────────────────────────────────────────────┘  │
 │                                                                      │
 │  Output: 243 vision embeddings of dim 5120                           │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘

 What's happening on the HARDWARE during this:
 ──────────────────────────────────────────────

 H100:                              B200:
 ┌────────────────────────┐         ┌────────────────────────┐
 │ 132 SMs active         │         │ 192 SMs active         │
 │                        │         │                        │
 │ Vision encoder is      │         │ Same ops, but:         │
 │ COMPUTE-BOUND:         │         │ - 2.3× more TFLOPS    │
 │ - 972 patches means    │         │ - So runs ~2x faster   │
 │   decent batch size    │         │                        │
 │ - Matrix multiplies    │         │ Vision encoder:        │
 │   keep tensor cores    │         │ ~0.5ms (estimate)      │
 │   reasonably busy      │         │                        │
 │                        │         │                        │
 │ Vision encoder:        │         │                        │
 │ ~1ms (estimate)        │         │                        │
 └────────────────────────┘         └────────────────────────┘
```

---

## Step 4: Prefill — The 64-Layer LLM Forward Pass

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Phase B: LLM PREFILL (all ~980 tokens in one pass)                  │
 │  ═══════════════════════════════════════════════════                  │
 │                                                                      │
 │  Step 1: Build input embeddings                                      │
 │  ┌────────────────────────────────────────────────────────────────┐  │
 │  │                                                                │  │
 │  │  Token IDs → Embedding table lookup → [980, 5120] BF16/FP16   │  │
 │  │                                                                │  │
 │  │  Then REPLACE vision placeholder positions with encoder output:│  │
 │  │                                                                │  │
 │  │  Position: 0    1     2    ...   243   244  245  ...  979      │  │
 │  │  Before: [BOS] [IMG_0] [IMG_1]... [IMG_242] [What] [is].. [?] │  │
 │  │  After:  [BOS] [VIS_0] [VIS_1]... [VIS_242] [What] [is].. [?] │  │
 │  │                ▲                     ▲                         │  │
 │  │                └─────────────────────┘                         │  │
 │  │          243 positions filled with vision encoder output       │  │
 │  │          (each is a 5120-dim vector)                           │  │
 │  └────────────────────────────────────────────────────────────────┘  │
 │                                                                      │
 │  Step 2: Run through 64 decoder layers                               │
 │  ┌────────────────────────────────────────────────────────────────┐  │
 │  │                                                                │  │
 │  │  For EACH of the 64 layers:                                    │  │
 │  │                                                                │  │
 │  │  ┌── RMSNorm ──────────────────────────────────────────────┐   │  │
 │  │  │  [980, 5120] → [980, 5120]  (elementwise, fast)        │   │  │
 │  │  └─────────────────────────┬───────────────────────────────┘   │  │
 │  │                            ▼                                   │  │
 │  │  ┌── Grouped Query Attention ──────────────────────────────┐   │  │
 │  │  │                                                         │   │  │
 │  │  │  Q projection: [980, 5120] × [5120, 8192] = [980, 8192]│   │  │
 │  │  │    → reshape to [980, 64 heads, 128 dim]                │   │  │
 │  │  │                                                         │   │  │
 │  │  │  K projection: [980, 5120] × [5120, 1024] = [980, 1024]│   │  │
 │  │  │    → reshape to [980, 8 heads, 128 dim]                 │   │  │
 │  │  │                                                         │   │  │
 │  │  │  V projection: [980, 5120] × [5120, 1024] = [980, 1024]│   │  │
 │  │  │    → reshape to [980, 8 heads, 128 dim]                 │   │  │
 │  │  │                                                         │   │  │
 │  │  │  ┌─ Multi-Head Rotary Position Embedding (M-RoPE) ──┐  │   │  │
 │  │  │  │ Qwen3-VL uses 3D positions: (time, height, width) │  │   │  │
 │  │  │  │ Vision tokens get spatial (h,w) positions          │  │   │  │
 │  │  │  │ Text tokens get sequential positions               │  │   │  │
 │  │  │  │ Applied to Q and K before attention                │  │   │  │
 │  │  │  └────────────────────────────────────────────────────┘  │   │  │
 │  │  │                                                         │   │  │
 │  │  │  WRITE K,V to KV cache blocks:                          │   │  │
 │  │  │  ┌──────────────────────────────────────────────────┐   │   │  │
 │  │  │  │  PagedAttention.write_to_paged_cache()            │   │   │  │
 │  │  │  │  (vllm/attention/ops/paged_attn.py:32)            │   │   │  │
 │  │  │  │                                                   │   │   │  │
 │  │  │  │  slot_mapping tells which block+offset to write:  │   │   │  │
 │  │  │  │  Token 0  → Block 100, slot 0                    │   │   │  │
 │  │  │  │  Token 1  → Block 100, slot 1                    │   │   │  │
 │  │  │  │  ...                                             │   │   │  │
 │  │  │  │  Token 15 → Block 100, slot 15 (block full!)     │   │   │  │
 │  │  │  │  Token 16 → Block 101, slot 0                    │   │   │  │
 │  │  │  │  ...                                             │   │   │  │
 │  │  │  │  Token 979 → Block 161, slot 3                   │   │   │  │
 │  │  │  └──────────────────────────────────────────────────┘   │   │  │
 │  │  │                                                         │   │  │
 │  │  │  Compute attention (FlashAttention kernel):             │   │  │
 │  │  │                                                         │   │  │
 │  │  │    For each of 64 query heads:                          │   │  │
 │  │  │      Which KV head? head_idx // 8 (GQA: 8 Q share 1 K) │   │  │
 │  │  │                                                         │   │  │
 │  │  │    scores = Q × K^T / sqrt(128)                         │   │  │
 │  │  │    [980, 980] attention matrix (causal mask applied)    │   │  │
 │  │  │    weights = softmax(scores)                            │   │  │
 │  │  │    output = weights × V  → [980, 128] per head          │   │  │
 │  │  │                                                         │   │  │
 │  │  │  O projection: [980, 8192] × [8192, 5120] = [980, 5120]│   │  │
 │  │  │  + residual connection                                  │   │  │
 │  │  └─────────────────────────┬───────────────────────────────┘   │  │
 │  │                            ▼                                   │  │
 │  │  ┌── RMSNorm ──────────────────────────────────────────────┐   │  │
 │  │  │  [980, 5120] → [980, 5120]                              │   │  │
 │  │  └─────────────────────────┬───────────────────────────────┘   │  │
 │  │                            ▼                                   │  │
 │  │  ┌── Feed-Forward Network (SwiGLU) ───────────────────────┐   │  │
 │  │  │                                                         │   │  │
 │  │  │  Gate proj: [980, 5120] × [5120, 25600] = [980, 25600] │   │  │
 │  │  │  Up proj:   [980, 5120] × [5120, 25600] = [980, 25600] │   │  │
 │  │  │  hidden = SiLU(gate) * up                               │   │  │
 │  │  │  Down proj: [980, 25600] × [25600, 5120] = [980, 5120] │   │  │
 │  │  │  + residual connection                                  │   │  │
 │  │  │                                                         │   │  │
 │  │  └─────────────────────────┬───────────────────────────────┘   │  │
 │  │                            ▼                                   │  │
 │  │                   [980, 5120]  → feed to next layer             │  │
 │  │                                                                │  │
 │  └────────────────────────────────────────────────────────────────┘  │
 │                                                                      │
 │  After all 64 layers:                                                │
 │  Final RMSNorm → [980, 5120]                                        │
 │  LM Head: [980, 5120] × [5120, 151936] = [980, 151936]             │
 │  Take LAST token's logits → sample → first generated token          │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘
```

### Prefill Compute Math

```
 Per layer, the big matmuls (batch=980 tokens):
   Q proj:   980 × 5120 × 8192  × 2 = 82.4 GFLOP
   K proj:   980 × 5120 × 1024  × 2 = 10.3 GFLOP
   V proj:   980 × 5120 × 1024  × 2 = 10.3 GFLOP
   O proj:   980 × 8192 × 5120  × 2 = 82.4 GFLOP
   Gate:     980 × 5120 × 25600 × 2 = 257.0 GFLOP
   Up:       980 × 5120 × 25600 × 2 = 257.0 GFLOP
   Down:     980 × 25600 × 5120 × 2 = 257.0 GFLOP
   Attention: ~980^2 × 128 × 64 heads  ≈ 9.9 GFLOP
                              Layer total ≈ 966 GFLOP

 64 layers × 966 = ~61,800 GFLOP = ~62 TFLOP

 H100 at FP8 (1,979 TFLOPS peak): 62 / 1979 ≈ 31 ms (compute time)
 H100 at FP16 (990 TFLOPS peak):  62 / 990  ≈ 63 ms (compute time)
 B200 at FP8 (4,500 TFLOPS peak): 62 / 4500 ≈ 14 ms (compute time)

 But weight reads also matter:
   Must read ~32 GB of weights from HBM (FP8)
   H100 (3.35 TB/s): 32 / 3350 ≈ 9.6 ms
   B200 (8.0 TB/s):  32 / 8000 ≈ 4.0 ms

 Actual prefill time (the LARGER of compute vs memory):
   H100 FP8:  ~31 ms (compute-bound with 980 tokens)
   B200 FP8:  ~14 ms (compute-bound with 980 tokens)

 (Real-world: ~50-70% of peak, so H100 ~50ms, B200 ~25ms)
```

---

## Step 5: Decode — Generating Tokens One at a Time

```
 Now we generate "The image shows a cat sitting on a windowsill."
 Each step processes just 1 NEW token per request.

 ┌──────────────────────────────────────────────────────────────────────┐
 │  DECODE STEP (one new token, e.g., generating "image")               │
 │  ══════════════════════════════════════════════════════               │
 │                                                                      │
 │  Input: just the PREVIOUS token "The" → token_id = [576]             │
 │                                                                      │
 │  Embedding: [1, 5120]                                                │
 │                                                                      │
 │  For EACH of 64 layers:                                              │
 │  ┌────────────────────────────────────────────────────────────────┐  │
 │  │  RMSNorm: [1, 5120]                                           │  │
 │  │                                                                │  │
 │  │  Q proj: [1, 5120] × [5120, 8192] = [1, 8192]                 │  │
 │  │  K proj: [1, 5120] × [5120, 1024] = [1, 1024]  → CACHE IT    │  │
 │  │  V proj: [1, 5120] × [5120, 1024] = [1, 1024]  → CACHE IT    │  │
 │  │                                                                │  │
 │  │  Attention: Q(new) attends to ALL 981 cached K,V tokens:       │  │
 │  │    ┌──────────────────────────────────────────────┐            │  │
 │  │    │  READ from KV cache (PagedAttention kernel): │            │  │
 │  │    │                                              │            │  │
 │  │    │  Block table: [B100, B101, ..., B161]        │            │  │
 │  │    │                                              │            │  │
 │  │    │  For each of 64 query heads:                 │            │  │
 │  │    │    k_head = head_idx // 8 (GQA mapping)      │            │  │
 │  │    │    score = Q[head] . K_cached[k_head, 0:981] │            │  │
 │  │    │    weight = softmax(score / sqrt(128))       │            │  │
 │  │    │    out = weight . V_cached[k_head, 0:981]    │            │  │
 │  │    │                                              │            │  │
 │  │    │  This reads cached KV for ALL 981 tokens     │            │  │
 │  │    │  from scattered blocks in HBM — the main     │            │  │
 │  │    │  memory bottleneck of decode!                 │            │  │
 │  │    └──────────────────────────────────────────────┘            │  │
 │  │                                                                │  │
 │  │  O proj: [1, 8192] × [8192, 5120] = [1, 5120]                 │  │
 │  │  FFN: gate+up [1, 5120]→[1, 25600], down→[1, 5120]            │  │
 │  └────────────────────────────────────────────────────────────────┘  │
 │                                                                      │
 │  LM Head: [1, 5120] × [5120, 151936] → logits → sample → "image"   │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘
```

### Decode Compute Math

```
 Compute per token per layer (batch=1):
   Q proj:  1 × 5120 × 8192  × 2 = 83.9 MFLOP
   K proj:  1 × 5120 × 1024  × 2 = 10.5 MFLOP
   V proj:  1 × 5120 × 1024  × 2 = 10.5 MFLOP
   O proj:  1 × 8192 × 5120  × 2 = 83.9 MFLOP
   Gate:    1 × 5120 × 25600 × 2 = 262.1 MFLOP
   Up:      1 × 5120 × 25600 × 2 = 262.1 MFLOP
   Down:    1 × 25600 × 5120 × 2 = 262.1 MFLOP
   Attn:    1 × 981 × 128 × 2 × 64 ≈ 16.1 MFLOP
                              Layer total ≈ 991 MFLOP

 64 layers: 64 × 991 = ~63.4 GFLOP = 0.063 TFLOP

 Data read per token (MUST read all weights from HBM):
   ~32 GB (FP8) or ~64 GB (BF16)

 Arithmetic intensity = 0.063 TFLOP / 32 GB = ~2 FLOP/byte
 This is DEEP in memory-bandwidth territory!

 Time per token (limited by reading weights from HBM):

 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  H100 (FP8 model, 3.35 TB/s):                               │
 │    Read 32 GB weights: 32 / 3350 = 9.6 ms per token         │
 │    → ~104 tokens/sec for a SINGLE request                   │
 │    (Compute takes 0.032 ms — 300x faster than memory!)      │
 │                                                              │
 │  B200 (BF16 model, 8.0 TB/s):                               │
 │    Read 64 GB weights: 64 / 8000 = 8.0 ms per token         │
 │    → ~125 tokens/sec for a SINGLE request                   │
 │                                                              │
 │  B200 (FP8 model, 8.0 TB/s):                                │
 │    Read 32 GB weights: 32 / 8000 = 4.0 ms per token         │
 │    → ~250 tokens/sec for a SINGLE request                   │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘
```

---

## Step 6: How Batching Saves Decode

```
 THE KEY INSIGHT: batch multiple requests to amortize weight reads
 ══════════════════════════════════════════════════════════════════

 Single request decode:
 ┌──────────────┐
 │  Read 32 GB  │ weights from HBM
 │  Do 0.06 TF  │ of compute       ← GPU is 99.7% idle!
 │  Output: 1   │ token
 └──────────────┘
 Arithmetic intensity: ~2 FLOP/byte (terrible)

 Batch of 32 requests:
 ┌──────────────┐
 │  Read 32 GB  │ weights from HBM (SAME amount!)
 │  Do 2.0 TF   │ of compute (32×)  ← GPU much busier
 │  Output: 32  │ tokens
 └──────────────┘
 Arithmetic intensity: ~64 FLOP/byte (much better)

 Time to generate 32 tokens (one per request):

 H100 FP8:  still ~9.6 ms (memory-bound, same read)
   → 32 tokens / 9.6 ms = 3,333 tokens/sec throughput!
   vs 104 tok/s for single request. 32× improvement.

 B200 FP8:  still ~4.0 ms
   → 32 tokens / 4.0 ms = 8,000 tokens/sec throughput!

 ┌───────────────────────────────────────────────────────┐
 │        THROUGHPUT vs BATCH SIZE (tokens/sec)          │
 │                                                       │
 │  Batch   H100 FP8    B200 BF16    B200 FP8            │
 │  ──────  ─────────   ──────────   ──────────          │
 │     1      104         125          250               │
 │     8      833        1,000        2,000               │
 │    32    3,333        4,000        8,000               │
 │    64    6,666        8,000       16,000               │
 │   128    limited by   16,000      limited by          │
 │          KV cache                  compute             │
 │                                                       │
 │  *Theoretical peaks; real numbers ~60-80% of these    │
 │   due to attention, sampling, scheduling overhead     │
 └───────────────────────────────────────────────────────┘

 The limit: KV cache memory determines max batch size!
 Each request at 8K context uses 8192 × 256 KB = 2 GB of KV cache.
   H100 FP8:  ~42 GB for KV → ~21 concurrent 8K requests
   B200 BF16: ~118 GB for KV → ~59 concurrent 8K requests
   B200 FP8:  ~152 GB for KV → ~76 concurrent 8K requests
```

---

## Step 7: What the KV Cache Actually Looks Like in Memory

```
 KV Cache layout for ONE layer, ONE request (980 tokens):
 ═════════════════════════════════════════════════════════

 Block 100 (tokens 0-15):
 ┌─────────────────────────────────────────────────────────────┐
 │ K heads:  [8 heads × 16 tokens × 128 dim] = 32 KB         │
 │ V heads:  [8 heads × 16 tokens × 128 dim] = 32 KB         │
 │                                            Total: 64 KB    │
 └─────────────────────────────────────────────────────────────┘
 Block 101 (tokens 16-31):
 ┌─────────────────────────────────────────────────────────────┐
 │ K: [8 × 16 × 128] = 32 KB   V: [8 × 16 × 128] = 32 KB    │
 └─────────────────────────────────────────────────────────────┘
 ...
 Block 161 (tokens 976-979, partially filled):
 ┌─────────────────────────────────────────────────────────────┐
 │ K: [8 × 4 × 128] = 8 KB     V: [8 × 4 × 128] = 8 KB      │
 │ (only 4 of 16 slots used)    (12 slots empty for growth)   │
 └─────────────────────────────────────────────────────────────┘

 This repeats for ALL 64 layers → 62 blocks × 64 layers = 3,968 blocks

 PagedAttention reads from SCATTERED blocks:
 ┌────────────────────────────────────────────────────────────────┐
 │  HBM (GPU memory)                                              │
 │                                                                │
 │  ░░░[B100]░░░░░░[B101]░░[B102]░░░░░░░░░[B103]░░░░░░[B104]░░ │
 │       │          │        │               │          │        │
 │       ▼          ▼        ▼               ▼          ▼        │
 │  The PagedAttention CUDA kernel follows the block_table       │
 │  to gather K,V from non-contiguous memory locations.          │
 │  This is like OS page table lookups but on the GPU.           │
 └────────────────────────────────────────────────────────────────┘
```

---

## Summary: The Full Picture

```
 REQUEST LIFECYCLE ON H100 (FP8) — Qwen3-VL-32B
 ════════════════════════════════════════════════

 Time ──▶

 ├── ~1ms ──┼───── ~50ms ─────┼── ~10ms ─┼── ~10ms ─┼── ~10ms ─┼ ... ─┤
 │           │                 │           │           │           │      │
 │  Vision   │   LLM           │  Decode   │  Decode   │  Decode   │      │
 │  Encoder  │   Prefill       │  "The"    │  "image"  │  "shows"  │ ...  │
 │  (32 ViT  │   (64 layers    │  1 token  │  1 token  │  1 token  │      │
 │  layers)  │   × 980 tokens) │  read     │  read     │  read     │      │
 │           │                 │  32GB     │  32GB     │  32GB     │      │
 │           │                 │  weights  │  weights  │  weights  │      │
 │ COMPUTE   │  COMPUTE        │ MEMORY    │ MEMORY    │ MEMORY    │      │
 │ BOUND     │  BOUND          │ BOUND     │ BOUND     │ BOUND     │      │
 │           │                 │           │           │           │      │

 Generating "The image shows a cat sitting on a windowsill."
 = 10 tokens × ~10ms = ~100ms decode
 + ~51ms prefill
 = ~151ms total (first token at ~51ms, last token at ~151ms)


 REQUEST LIFECYCLE ON B200 (FP8) — Qwen3-VL-32B
 ════════════════════════════════════════════════

 Time ──▶

 ├ ~0.5ms ┼── ~25ms ──┼ ~4ms ┼ ~4ms ┼ ~4ms ┼ ... ─┤
 │         │            │      │      │      │      │
 │ Vision  │  Prefill   │  D   │  D   │  D   │ ...  │
 │ Enc     │            │      │      │      │      │

 = 10 × 4ms + 25ms = ~65ms total (~2.3× faster than H100)


 REQUEST LIFECYCLE ON B200 (BF16) — Qwen3-VL-32B
 ════════════════════════════════════════════════

 Time ──▶

 ├ ~0.5ms ┼── ~45ms ──┼ ~8ms ┼ ~8ms ┼ ~8ms ┼ ... ─┤
 │         │            │      │      │      │      │
 │ Vision  │  Prefill   │  D   │  D   │  D   │ ...  │
 │ Enc     │ (BF16 uses │      │      │      │      │
 │         │  FP16 TC)  │      │      │      │      │

 = 10 × 8ms + 45ms = ~125ms total (slower than FP8 on B200)
 But: more KV cache → higher concurrent throughput
```

---

## Key Takeaways

```
 ┌──────────────────────────────────────────────────────────────────┐
 │                    DECISION MATRIX                               │
 ├─────────────────────┬──────────────────┬────────────────────────┤
 │                     │     H100 SXM     │        B200            │
 ├─────────────────────┼──────────────────┼────────────────────────┤
 │ Precision           │ FP8 (must use)   │ FP8 or BF16 (choice)  │
 │ Fits on 1 GPU?      │ Yes (FP8 only)   │ Yes (either)          │
 │ Weights in memory   │ ~32 GB           │ ~33 or ~66 GB         │
 │ KV cache available  │ ~42 GB           │ ~152 or ~118 GB       │
 │ Max 8K-ctx reqs     │ ~21              │ ~76 or ~59            │
 │ Single-req tok/s    │ ~104             │ ~250 or ~125          │
 │ 32-batch tok/s      │ ~3,333           │ ~8,000 or ~4,000      │
 │ Time-to-first-token │ ~50ms            │ ~25ms or ~45ms        │
 │ Best for            │ Cost-effective   │ Max throughput         │
 │                     │ single-GPU       │ or max concurrency    │
 └─────────────────────┴──────────────────┴────────────────────────┘
```

---

*All architecture numbers from Qwen3-VL-32B-Instruct config: hidden_size=5120,
num_hidden_layers=64, num_attention_heads=64, num_key_value_heads=8, head_dim=128,
intermediate_size=25600, vision depth=32, vision hidden_size=1280.*

*GPU specs: H100 SXM (80GB HBM3, 3.35 TB/s, 1979 FP8 TFLOPS);
B200 (192GB HBM3e, 8.0 TB/s, 4500 FP8 TFLOPS).*
