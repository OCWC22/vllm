# Qwen2-VL vs Qwen3-VL: Complete Architecture Guide

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Qwen2-VL Architecture Deep Dive](#qwen2-vl-architecture)
3. [Qwen3-VL Architecture Deep Dive](#qwen3-vl-architecture)
4. [Component-by-Component Comparison](#component-comparison)
5. [Hardware Optimization Guide](#hardware-optimization)
6. [Performance Benchmarks](#performance-benchmarks)

---

## High-Level Overview

```
╔═══════════════════════════════════════════════════════════════════════════════════════╗
║                    QWEN VISION-LANGUAGE MODEL EVOLUTION                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                       ║
║   QWEN2-VL (Late 2024)                    QWEN3-VL (2025)                             ║
║   ══════════════════                      ══════════════                              ║
║                                                                                       ║
║   ┌─────────────────────┐                 ┌─────────────────────┐                     ║
║   │    Image/Video      │                 │    Image/Video      │                     ║
║   │      Inputs         │                 │      Inputs         │                     ║
║   └──────────┬──────────┘                 └──────────┬──────────┘                     ║
║              │                                       │                                ║
║              ▼                                       ▼                                ║
║   ┌─────────────────────┐                 ┌─────────────────────┐                     ║
║   │  Conv3D Patch Embed │                 │ Conv3D Patch Embed  │                     ║
║   │    (no bias)        │                 │   (WITH bias)       │                     ║
║   └──────────┬──────────┘                 └──────────┬──────────┘                     ║
║              │                                       │                                ║
║              │                            ┌──────────▼──────────┐                     ║
║              │                            │  Learned Position   │                     ║
║              │                            │  Embed + Interpolate│                     ║
║              │                            └──────────┬──────────┘                     ║
║              │                                       │                                ║
║              ▼                                       ▼                                ║
║   ┌─────────────────────┐                 ┌─────────────────────┐                     ║
║   │   Vision Blocks     │                 │   Vision Blocks     │────────┐            ║
║   │   (N layers)        │                 │   (N layers)        │        │            ║
║   │   QuickGELU         │                 │   SiLU activation   │        ▼            ║
║   └──────────┬──────────┘                 └──────────┬──────────┘  DeepStack          ║
║              │                                       │             Mergers            ║
║              ▼                                       ▼               │                ║
║   ┌─────────────────────┐                 ┌─────────────────────┐    │                ║
║   │   Single Merger     │                 │   Main Merger       │◄───┘                ║
║   └──────────┬──────────┘                 └──────────┬──────────┘                     ║
║              │                                       │                                ║
║              ▼                                       ▼                                ║
║   ┌─────────────────────┐                 ┌─────────────────────┐                     ║
║   │    Qwen2 LLM        │                 │    Qwen3 LLM        │                     ║
║   │  (Qwen2ForCausalLM) │                 │ (DeepStack inject)  │                     ║
║   └─────────────────────┘                 └─────────────────────┘                     ║
║                                                                                       ║
╚═══════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Qwen2-VL Architecture

### Vision Encoder Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          QWEN2-VL VISION ENCODER                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  INPUT: Image (H × W × 3) or Video (T × H × W × 3)                                  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 1: PATCH EMBEDDING (Qwen2VisionPatchEmbed)                            │   │
│  │  ═══════════════════════════════════════════════                            │   │
│  │                                                                             │   │
│  │    Input: (L, C) where C = 3 × temporal_patch × patch² = 3 × 2 × 14² = 1176│   │
│  │                                                                             │   │
│  │    ┌─────────────────────────────────────────────────────────┐             │   │
│  │    │  Conv3D(in=3, out=embed_dim, kernel=(2,14,14))         │             │   │
│  │    │         │                                               │             │   │
│  │    │         ▼                                               │             │   │
│  │    │  Reshape: (L, embed_dim) = (L, 1152)                   │             │   │
│  │    └─────────────────────────────────────────────────────────┘             │   │
│  │                                                                             │   │
│  │    Output: (num_patches, 1152)                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 2: ROTARY POSITION EMBEDDING (3D RoPE)                                │   │
│  │  ═══════════════════════════════════════════                                │   │
│  │                                                                             │   │
│  │    ┌──────────┐    ┌──────────┐                                             │   │
│  │    │ h_pos_ids│    │ w_pos_ids│  ← Computed from grid_thw                   │   │
│  │    └────┬─────┘    └────┬─────┘                                             │   │
│  │         │               │                                                    │   │
│  │         └───────┬───────┘                                                    │   │
│  │                 ▼                                                            │   │
│  │    ┌─────────────────────────────────────────────────────────┐             │   │
│  │    │  cos, sin = rotary_pos_emb.get_cos_sin(max_grid_size)  │             │   │
│  │    │  cos_combined = cos[pos_ids].flatten(1)                 │             │   │
│  │    │  sin_combined = sin[pos_ids].flatten(1)                 │             │   │
│  │    └─────────────────────────────────────────────────────────┘             │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 3: VISION TRANSFORMER BLOCKS × N (Qwen2VisionBlock)                   │   │
│  │  ═══════════════════════════════════════════════════════                    │   │
│  │                                                                             │   │
│  │    for each block in self.blocks:                                           │   │
│  │                                                                             │   │
│  │    ┌────────────────────────────────────────────────────────────────┐      │   │
│  │    │  x                                                             │      │   │
│  │    │  │                                                             │      │   │
│  │    │  ├──▶ LayerNorm ──▶ Attention ──▶ Add ─┐                       │      │   │
│  │    │  │    (eps=1e-6)    (with RoPE)        │                       │      │   │
│  │    │  │◀────────────────────────────────────┘                       │      │   │
│  │    │  │                                                             │      │   │
│  │    │  ├──▶ LayerNorm ──▶ MLP ──▶ Add ───────┐                       │      │   │
│  │    │  │    (eps=1e-6)    (QuickGELU)        │                       │      │   │
│  │    │  │◀────────────────────────────────────┘                       │      │   │
│  │    │  │                                                             │      │   │
│  │    │  ▼                                                             │      │   │
│  │    │  x (updated)                                                   │      │   │
│  │    └────────────────────────────────────────────────────────────────┘      │   │
│  │                                                                             │   │
│  │    MLP Structure:                                                           │   │
│  │    ┌───────────────────────────────────────────────────────────────┐       │   │
│  │    │  fc1: Linear(embed_dim, embed_dim * mlp_ratio)                │       │   │
│  │    │  act: QuickGELU (x * sigmoid(1.702 * x))                      │       │   │
│  │    │  fc2: Linear(embed_dim * mlp_ratio, embed_dim)                │       │   │
│  │    └───────────────────────────────────────────────────────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 4: PATCH MERGER (Qwen2VisionPatchMerger)                              │   │
│  │  ═══════════════════════════════════════════                                │   │
│  │                                                                             │   │
│  │    spatial_merge_size = 2 → Merge 2×2 patches into 1                        │   │
│  │                                                                             │   │
│  │    ┌───────────────────────────────────────────────────────────────┐       │   │
│  │    │  ln_q: LayerNorm(embed_dim)                                   │       │   │
│  │    │        │                                                       │       │   │
│  │    │        ▼                                                       │       │   │
│  │    │  reshape: (N, embed_dim) → (N/4, embed_dim × 4)               │       │   │
│  │    │        │                                                       │       │   │
│  │    │        ▼                                                       │       │   │
│  │    │  mlp[0]: Linear(embed_dim × 4, embed_dim × 4)                 │       │   │
│  │    │  mlp[1]: GELU()                                                │       │   │
│  │    │  mlp[2]: Linear(embed_dim × 4, hidden_size)                   │       │   │
│  │    └───────────────────────────────────────────────────────────────┘       │   │
│  │                                                                             │   │
│  │    Output: (num_patches / 4, hidden_size) = merged visual tokens            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  OUTPUT: Visual embeddings ready for LLM integration                               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Attention Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      QWEN2-VL VISION ATTENTION                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│    Input: x (seq_len, batch, embed_dim)                                             │
│                                                                                     │
│    ┌───────────────────────────────────────────────────────────────────────────┐   │
│    │  1. QKV Projection                                                        │   │
│    │     qkv = Linear(embed_dim, 3 * embed_dim)(x)                             │   │
│    │                                                                           │   │
│    │  2. Split into Q, K, V                                                    │   │
│    │     q, k, v = qkv.chunk(3, dim=-1)                                        │   │
│    │     Reshape: (seq, batch, num_heads, head_dim)                            │   │
│    │                                                                           │   │
│    │  3. Apply Rotary Embedding                                                │   │
│    │     qk = cat([q, k], dim=0)                                               │   │
│    │     qk_rotated = apply_rotary_emb(qk, cos, sin)                           │   │
│    │     q, k = qk_rotated.chunk(2, dim=0)                                     │   │
│    │                                                                           │   │
│    │  4. Compute Attention                                                     │   │
│    │     ┌─────────────────────────────────────────────────────────┐          │   │
│    │     │ if FlashAttention available:                            │          │   │
│    │     │   out = flash_attn_varlen_func(q, k, v, cu_seqlens)     │          │   │
│    │     │ else:                                                   │          │   │
│    │     │   out = torch.nn.functional.scaled_dot_product_attention│          │   │
│    │     └─────────────────────────────────────────────────────────┘          │   │
│    │                                                                           │   │
│    │  5. Output Projection                                                     │   │
│    │     out = Linear(embed_dim, embed_dim)(out)                               │   │
│    └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│    Supported Backends:                                                              │
│    ├─ FLASH_ATTN (SM ≥ 8.0, fastest)                                                │
│    ├─ TORCH_SDPA (universal fallback)                                               │
│    ├─ XFORMERS (SM ≥ 7.0)                                                           │
│    └─ ROCM_AITER_FA (AMD GPUs)                                                      │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Qwen3-VL Architecture

### Vision Encoder Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          QWEN3-VL VISION ENCODER                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  INPUT: Image (H × W × 3) or Video (T × H × W × 3)                                  │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 1: PATCH EMBEDDING (Qwen3_VisionPatchEmbed)                           │   │
│  │  ══════════════════════════════════════════════                             │   │
│  │                                                                             │   │
│  │    ┌─────────────────────────────────────────────────────────┐             │   │
│  │    │  Conv3D(in=3, out=hidden_size, kernel=(2,14,14))       │             │   │
│  │    │  **WITH BIAS** (unlike Qwen2-VL)                        │◄── KEY DIFF │   │
│  │    └─────────────────────────────────────────────────────────┘             │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 2: LEARNED POSITION EMBEDDING + INTERPOLATION                         │   │
│  │  ═════════════════════════════════════════════════════                      │   │
│  │  **NEW IN QWEN3-VL**                                                        │   │
│  │                                                                             │   │
│  │    ┌─────────────────────────────────────────────────────────┐             │   │
│  │    │  pos_embed = nn.Embedding(num_pos_embeddings, hidden)   │             │   │
│  │    │                                                          │             │   │
│  │    │  BILINEAR INTERPOLATION for variable resolutions:       │             │   │
│  │    │                                                          │             │   │
│  │    │  for each (t, h, w) in grid_thw:                        │             │   │
│  │    │    h_idxs = linspace(0, num_grid-1, h)                  │             │   │
│  │    │    w_idxs = linspace(0, num_grid-1, w)                  │             │   │
│  │    │                                                          │             │   │
│  │    │    # Bilinear weights                                    │             │   │
│  │    │    w00 = (1 - dh) * (1 - dw)                            │             │   │
│  │    │    w01 = (1 - dh) * dw                                  │             │   │
│  │    │    w10 = dh * (1 - dw)                                  │             │   │
│  │    │    w11 = dh * dw                                        │             │   │
│  │    │                                                          │             │   │
│  │    │    # Interpolated embedding                              │             │   │
│  │    │    embeds = sum(weights * pos_embed[indices])           │             │   │
│  │    └─────────────────────────────────────────────────────────┘             │   │
│  │                                                                             │   │
│  │    hidden_states = patch_embed + interpolated_pos_embed                     │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 3: VISION TRANSFORMER BLOCKS + DEEPSTACK                              │   │
│  │  ═════════════════════════════════════════════                              │   │
│  │  **DEEPSTACK IS NEW IN QWEN3-VL**                                           │   │
│  │                                                                             │   │
│  │    deepstack_visual_indexes = [layer_k, layer_m, ...]                       │   │
│  │                                                                             │   │
│  │    for layer_num, block in enumerate(self.blocks):                          │   │
│  │                                                                             │   │
│  │    ┌────────────────────────────────────────────────────────────────┐      │   │
│  │    │  hidden = block(hidden, cu_seqlens, cos, sin, max_seqlen)      │      │   │
│  │    │                                                                 │      │   │
│  │    │  if layer_num in deepstack_visual_indexes:                     │      │   │
│  │    │    ┌───────────────────────────────────────────────────────┐   │      │   │
│  │    │    │  deepstack_feature = deepstack_merger[idx](hidden)    │   │      │   │
│  │    │    │  deepstack_features.append(deepstack_feature)         │   │      │   │
│  │    │    └───────────────────────────────────────────────────────┘   │      │   │
│  │    └────────────────────────────────────────────────────────────────┘      │   │
│  │                                                                             │   │
│  │    Block Structure (different from Qwen2-VL):                               │   │
│  │    ┌───────────────────────────────────────────────────────────────┐       │   │
│  │    │  MLP:                                                         │       │   │
│  │    │    linear_fc1: Linear(dim, mlp_hidden_dim, bias=False)       │       │   │
│  │    │    act_fn: SiLU (x * sigmoid(x))  ◄── Different from QuickGELU│       │   │
│  │    │    linear_fc2: Linear(mlp_hidden_dim, dim, bias=False)       │       │   │
│  │    └───────────────────────────────────────────────────────────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 4: MULTI-SCALE FEATURE MERGING                                        │   │
│  │  ════════════════════════════════════                                       │   │
│  │                                                                             │   │
│  │    # Main merger (same as Qwen2-VL)                                         │   │
│  │    main_features = self.merger(hidden_states)                               │   │
│  │                                                                             │   │
│  │    # Concatenate with DeepStack features                                    │   │
│  │    output = torch.cat([main_features] + deepstack_features, dim=1)          │   │
│  │                                                                             │   │
│  │    ┌─────────────────────────────────────────────────────────────────┐     │   │
│  │    │  OUTPUT SHAPE:                                                  │     │   │
│  │    │  (seq_len, hidden_size * (1 + num_deepstack_levels))            │     │   │
│  │    │                                                                  │     │   │
│  │    │  Example: if hidden_size=3584 and 2 deepstack levels:           │     │   │
│  │    │  (seq_len, 3584 * 3) = (seq_len, 10752)                         │     │   │
│  │    └─────────────────────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### DeepStack Feature Injection into LLM

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL DEEPSTACK LLM INTEGRATION                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Vision Output: [main_emb | ds_emb_0 | ds_emb_1 | ...]                              │
│                 └─────────────────┬─────────────────┘                               │
│                                   │                                                 │
│                                   ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  SPLIT INTO COMPONENTS:                                                     │   │
│  │                                                                             │   │
│  │  multimodal_embeddings_main = output[:, :visual_dim]                        │   │
│  │  multimodal_embeddings_multiscale = output[:, visual_dim:]                  │   │
│  │                                                                             │   │
│  └────────────────────────────────┬────────────────────────────────────────────┘   │
│                                   │                                                 │
│                                   ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  COMPUTE DEEPSTACK INPUT EMBEDS:                                            │   │
│  │                                                                             │   │
│  │  deepstack_input_embeds = zeros(seq_len, num_levels * hidden_size)          │   │
│  │                                                                             │   │
│  │  # Merge multiscale features at multimodal positions                        │   │
│  │  deepstack_input_embeds[is_multimodal] = multimodal_embeddings_multiscale   │   │
│  │                                                                             │   │
│  │  # Reshape for per-layer injection                                          │   │
│  │  deepstack_input_embeds = reshape(seq_len, num_levels, visual_dim)          │   │
│  │  deepstack_input_embeds = permute(num_levels, seq_len, visual_dim)          │   │
│  │                                                                             │   │
│  └────────────────────────────────┬────────────────────────────────────────────┘   │
│                                   │                                                 │
│                                   ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  LLM FORWARD WITH DEEPSTACK INJECTION (Qwen3LLMModel):                      │   │
│  │                                                                             │   │
│  │  for layer_idx, layer in enumerate(self.layers):                            │   │
│  │      hidden_states, residual = layer(positions, hidden_states, residual)    │   │
│  │                                                                             │   │
│  │      if layer_idx < len(deepstack_input_embeds):                            │   │
│  │          ┌──────────────────────────────────────────────────────────────┐  │   │
│  │          │  # ADD deepstack features to hidden states                   │  │   │
│  │          │  hidden_states = hidden_states +                              │  │   │
│  │          │                  deepstack_input_embeds[layer_idx]            │  │   │
│  │          │                                                               │  │   │
│  │          │  This injects multi-scale visual features into EARLY layers! │  │   │
│  │          └──────────────────────────────────────────────────────────────┘  │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  BENEFIT: Multi-scale visual features are available throughout the LLM,            │
│           not just at the embedding level. Improves fine-grained understanding!    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### EVS (Efficient Video Sampling) - Qwen3-VL Exclusive

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                QWEN3-VL EFFICIENT VIDEO SAMPLING (EVS)                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  PROBLEM: Long videos = Many tokens = Slow inference + OOM                          │
│                                                                                     │
│  SOLUTION: Content-aware pruning of video tokens                                    │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  WITHOUT EVS (Qwen2-VL):                                                    │   │
│  │                                                                             │   │
│  │  Video (100 frames) → ████████████████████████████████████████████████████  │   │
│  │                       ALL tokens passed to LLM (expensive!)                 │   │
│  │                       ~16,000 tokens for 100 frames                         │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │  WITH EVS (Qwen3-VL, video_pruning_rate > 0):                               │   │
│  │                                                                             │   │
│  │  Step 1: Compute retention mask                                             │   │
│  │  ────────────────────────────                                               │   │
│  │    retention_mask = compute_retention_mask(                                 │   │
│  │        embeddings,                                                          │   │
│  │        grid_thw,                                                            │   │
│  │        spatial_merge_size,                                                  │   │
│  │        q=video_pruning_rate  # e.g., 0.5 = keep 50%                         │   │
│  │    )                                                                        │   │
│  │                                                                             │   │
│  │  Step 2: Apply mask                                                         │   │
│  │  ────────────────────                                                       │   │
│  │    pruned_embeddings = embeddings[retention_mask]                           │   │
│  │    pruned_positions = positions[retention_mask]                             │   │
│  │                                                                             │   │
│  │  Video (100 frames) → ████████░░░░████████░░░░████████░░░░████████░░░░████  │   │
│  │                       Only IMPORTANT tokens kept (~8,000)                   │   │
│  │                       50% reduction with minimal quality loss!              │   │
│  │                                                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
│  Step 3: Recompute M-RoPE positions                                                 │
│  ───────────────────────────────────                                                │
│    # Original positions become invalid after pruning                               │
│    # Must recompute to maintain correct spatial relationships                      │
│                                                                                     │
│    positions, delta = recompute_mrope_positions(                                    │
│        input_ids,                                                                   │
│        mm_embeddings_pos,  # Contains pre-computed positions                       │
│        mrope_positions,                                                             │
│        num_computed_tokens,                                                         │
│        vision_start_token_id,                                                       │
│        image_token_id,                                                              │
│        video_token_id,                                                              │
│    )                                                                                │
│                                                                                     │
│  PERFORMANCE IMPACT:                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────┐    │
│  │  Pruning Rate │ Token Reduction │ Quality Loss │ Speed Improvement         │    │
│  │  ─────────────┼─────────────────┼──────────────┼─────────────────          │    │
│  │  0.3 (30%)    │ ~30%            │ Minimal      │ ~1.4x faster              │    │
│  │  0.5 (50%)    │ ~50%            │ Slight       │ ~2x faster                │    │
│  │  0.7 (70%)    │ ~70%            │ Noticeable   │ ~3x faster                │    │
│  └────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Comparison

```
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                    COMPONENT-BY-COMPONENT COMPARISON                                ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  ┌──────────────────────┬─────────────────────────┬────────────────────────────┐   ║
║  │ Component            │ Qwen2-VL                │ Qwen3-VL                   │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Patch Embedding      │ Conv3D (no bias)        │ Conv3D (WITH bias)         │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Position Encoding    │ 3D RoPE only            │ Learned + RoPE + Interp    │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ MLP Activation       │ QuickGELU               │ SiLU (configurable)        │   ║
║  │                      │ x * σ(1.702x)           │ x * σ(x)                   │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ MLP Bias             │ Has bias                │ No bias (linear_fc1/fc2)   │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Feature Merging      │ Single merger           │ Main + DeepStack mergers   │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Multi-Scale          │ ❌ None                 │ ✅ DeepStack               │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Video Pruning        │ ❌ None                 │ ✅ EVS                     │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Max Video Frames     │ 14                      │ 24,576                     │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Attention Backend    │ FA, SDPA, Xformers,     │ FA, SDPA, ROCm only       │   ║
║  │                      │ ROCm, etc.              │ (stricter requirement)     │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Speculative Decode   │ Basic                   │ Eagle3 native support      │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ MoE Variants         │ ❌ None                 │ ✅ Qwen3-VL-30B-A3B        │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ LLM Backbone         │ Qwen2ForCausalLM        │ Qwen3LLMForCausalLM        │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Torch Compile        │ Limited                 │ Decorated support          │   ║
║  └──────────────────────┴─────────────────────────┴────────────────────────────┘   ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Hardware Optimization

```
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                        GPU OPTIMIZATION MATRIX                                      ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐   ║
║  │                              T4 (16GB, SM 7.5)                              │   ║
║  ├─────────────────────────────────────────────────────────────────────────────┤   ║
║  │  Qwen2-VL-2B: ✅ AWQ quantization, SDPA attention, 2K context              │   ║
║  │  Qwen2-VL-7B: ⚠️  4-bit required, very limited                             │   ║
║  │  Qwen3-VL-4B: ✅ 4-bit quantization, small batch, EVS helps!               │   ║
║  │  Qwen3-VL-8B: ⚠️  Very tight, needs aggressive quantization                │   ║
║  │                                                                             │   ║
║  │  Best Config: enforce_eager=True, max_num_seqs=4, max_pixels=512000        │   ║
║  └─────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                     ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐   ║
║  │                            A100 (40/80GB, SM 8.0)                           │   ║
║  ├─────────────────────────────────────────────────────────────────────────────┤   ║
║  │  Qwen2-VL-*:  ✅ Full precision (80GB), FlashAttention v2                  │   ║
║  │  Qwen3-VL-4B: ✅ Excellent, full precision, high throughput                │   ║
║  │  Qwen3-VL-8B: ✅ Full precision (80GB), good concurrency                   │   ║
║  │  Qwen3-VL-30B-A3B: ✅ MoE fits in 80GB                                     │   ║
║  │                                                                             │   ║
║  │  Best Config: dtype=bfloat16, max_num_seqs=32, enable_prefix_caching=True  │   ║
║  └─────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                     ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐   ║
║  │                             H100 (80GB, SM 9.0)                             │   ║
║  ├─────────────────────────────────────────────────────────────────────────────┤   ║
║  │  Qwen2-VL-*:  ✅ Excellent with FP8 quantization available                 │   ║
║  │  Qwen3-VL-*:  ✅ All models optimal, FlashAttention 3                      │   ║
║  │               ✅ DeepStack + EVS leverage H100's bandwidth                 │   ║
║  │                                                                             │   ║
║  │  Best Config: kv_cache_dtype=fp8, max_num_seqs=64, max_model_len=32768     │   ║
║  └─────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                     ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐   ║
║  │                            B200 (192GB, SM 10.0)                            │   ║
║  ├─────────────────────────────────────────────────────────────────────────────┤   ║
║  │  Qwen2-VL-72B: ✅ Full precision on single GPU!                            │   ║
║  │  Qwen3-VL-*:   ✅ All models, massive batch sizes, ultra-long context      │   ║
║  │               ✅ FP4 quantization when available (4x throughput)           │   ║
║  │                                                                             │   ║
║  │  Best Config: max_num_seqs=128, max_model_len=131072, max_pixels=4147200   │   ║
║  └─────────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Performance Benchmarks

```
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                    EXPECTED PERFORMANCE (Image Inference)                           ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  Single Image (1024×1024), 256 output tokens:                                       ║
║                                                                                     ║
║  ┌────────────────────┬─────────────────┬─────────────────┬───────────────────┐    ║
║  │ Model              │ T4 (16GB)       │ H100 (80GB)     │ B200 (192GB)      │    ║
║  ├────────────────────┼─────────────────┼─────────────────┼───────────────────┤    ║
║  │ Qwen2-VL-2B        │ ~800ms, 20 t/s  │ ~150ms, 120 t/s │ ~80ms, 200 t/s    │    ║
║  │ Qwen2-VL-7B        │ ~2000ms*, 8 t/s │ ~250ms, 80 t/s  │ ~120ms, 150 t/s   │    ║
║  ├────────────────────┼─────────────────┼─────────────────┼───────────────────┤    ║
║  │ Qwen3-VL-4B        │ ~1000ms, 18 t/s │ ~120ms, 150 t/s │ ~60ms, 280 t/s    │    ║
║  │ Qwen3-VL-8B        │ ~1800ms*, 10 t/s│ ~180ms, 100 t/s │ ~90ms, 200 t/s    │    ║
║  │ Qwen3-VL-30B-A3B   │ ❌ OOM          │ ~300ms, 60 t/s  │ ~150ms, 120 t/s   │    ║
║  └────────────────────┴─────────────────┴─────────────────┴───────────────────┘    ║
║  * = Requires 4-bit quantization                                                    ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝

╔═════════════════════════════════════════════════════════════════════════════════════╗
║                    EXPECTED PERFORMANCE (Video Inference)                           ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  60-second video (30 fps → 1800 frames → sampled to ~100 frames):                   ║
║                                                                                     ║
║  ┌────────────────────┬─────────────────┬─────────────────┬───────────────────┐    ║
║  │ Model              │ A100 (80GB)     │ H100 (80GB)     │ B200 (192GB)      │    ║
║  ├────────────────────┼─────────────────┼─────────────────┼───────────────────┤    ║
║  │ Qwen2-VL-7B        │ ~5s, 16K tokens │ ~2s, 16K tokens │ ~1s, 16K tokens   │    ║
║  │                    │ (no pruning)    │ (no pruning)    │ (no pruning)      │    ║
║  ├────────────────────┼─────────────────┼─────────────────┼───────────────────┤    ║
║  │ Qwen3-VL-8B        │ ~3s, 8K tokens  │ ~1.2s, 8K tokens│ ~0.6s, 8K tokens  │    ║
║  │ (EVS 50%)          │ (50% pruned!)   │ (50% pruned!)   │ (50% pruned!)     │    ║
║  └────────────────────┴─────────────────┴─────────────────┴───────────────────┘    ║
║                                                                                     ║
║  EVS IMPACT: ~2x speedup on long videos with minimal quality degradation           ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
```

