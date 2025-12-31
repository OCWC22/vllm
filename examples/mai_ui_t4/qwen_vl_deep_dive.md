# Qwen2-VL vs Qwen3-VL: Deep Technical Analysis

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Vision Encoder Deep Dive](#vision-encoder-deep-dive)
3. [Position Encoding Systems](#position-encoding-systems)
4. [DeepStack Multi-Scale Features (Qwen3-VL Only)](#deepstack-multi-scale-features)
5. [EVS: Efficient Video Sampling (Qwen3-VL Only)](#evs-efficient-video-sampling)
6. [LLM Backbone Integration](#llm-backbone-integration)
7. [GPU-Specific Optimization Analysis](#gpu-specific-optimization-analysis)
8. [SGLang/vLLM Layer Compatibility](#sglang-vllm-layer-compatibility)
9. [Practical Code Examples](#practical-code-examples)

---

## Executive Summary

```
╔═════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              QWEN2-VL vs QWEN3-VL ARCHITECTURE EVOLUTION                            ║
╠═════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                     ║
║  QWEN2-VL (2024)                                    QWEN3-VL (2025)                                 ║
║  ══════════════                                     ══════════════                                  ║
║                                                                                                     ║
║  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐              ║
║  │ Image/Video Input               │               │ Image/Video Input               │              ║
║  └───────────────┬─────────────────┘               └───────────────┬─────────────────┘              ║
║                  │                                                 │                                ║
║                  ▼                                                 ▼                                ║
║  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐              ║
║  │ Conv3D Patch Embed (NO BIAS)    │               │ Conv3D Patch Embed (WITH BIAS)  │              ║
║  │ kernel=(2,14,14), stride=same   │               │ kernel=(2,14,14), stride=same   │              ║
║  │ bias=False                      │               │ bias=True ◄─── KEY DIFFERENCE   │              ║
║  └───────────────┬─────────────────┘               └───────────────┬─────────────────┘              ║
║                  │                                                 │                                ║
║                  │                                                 ▼                                ║
║                  │                                 ┌─────────────────────────────────┐              ║
║                  │                                 │ Learned Position Embedding      │              ║
║                  │                                 │ nn.Embedding(num_pos, hidden)   │              ║
║                  │                                 │ + Bilinear Interpolation        │              ║
║                  │                                 │ for variable resolutions        │              ║
║                  │                                 └───────────────┬─────────────────┘              ║
║                  │                                                 │                                ║
║                  ▼                                                 ▼                                ║
║  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐              ║
║  │ 3D RoPE (Rotary Position Embed) │               │ RoPE (Half Rotary)              │              ║
║  │ Full rotation on Q and K        │               │ partial_rotary_factor=0.5       │              ║
║  └───────────────┬─────────────────┘               └───────────────┬─────────────────┘              ║
║                  │                                                 │                                ║
║                  ▼                                                 ▼                                ║
║  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐              ║
║  │ Vision Transformer Blocks       │               │ Vision Transformer Blocks       │──┐           ║
║  │ × N layers                      │               │ × N layers                      │  │           ║
║  │                                 │               │                                 │  │           ║
║  │ ┌─────────────────────────────┐ │               │ ┌─────────────────────────────┐ │  │           ║
║  │ │ LayerNorm(eps=1e-6)        │ │               │ │ LayerNorm(eps=1e-6)        │ │  │           ║
║  │ │       ↓                    │ │               │ │       ↓                    │ │  │           ║
║  │ │ Attention (QKV + RoPE)     │ │               │ │ Attention (QKV + RoPE)     │ │  │DeepStack  ║
║  │ │       ↓                    │ │               │ │       ↓                    │ │  │Mergers    ║
║  │ │ LayerNorm(eps=1e-6)        │ │               │ │ LayerNorm(eps=1e-6)        │ │  │at layers  ║
║  │ │       ↓                    │ │               │ │       ↓                    │ │  │specified  ║
║  │ │ MLP:                       │ │               │ │ MLP:                       │ │  │in config  ║
║  │ │  fc1 → QuickGELU → fc2     │ │               │ │  fc1 → SiLU → fc2          │ │  │           ║
║  │ │  (WITH bias)               │ │               │ │  (NO bias on fc1/fc2)      │ │  │           ║
║  │ └─────────────────────────────┘ │               │ └─────────────────────────────┘ │  │           ║
║  └───────────────┬─────────────────┘               └───────────────┬─────────────────┘  │           ║
║                  │                                                 │                    │           ║
║                  ▼                                                 ▼                    │           ║
║  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐  │           ║
║  │ Single Patch Merger             │               │ Main Patch Merger               │◄─┘           ║
║  │ ln_q → reshape → MLP            │               │ norm → fc1 → GELU → fc2         │              ║
║  │ (2×2 spatial merge)             │               │ + DeepStack Mergers concat      │              ║
║  └───────────────┬─────────────────┘               └───────────────┬─────────────────┘              ║
║                  │                                                 │                                ║
║                  │                                                 │                                ║
║                  │ Output: (seq, hidden)                           │ Output: (seq, hidden * (1+k))  ║
║                  │                                                 │ where k = num deepstack levels ║
║                  ▼                                                 ▼                                ║
║  ┌─────────────────────────────────┐               ┌─────────────────────────────────┐              ║
║  │ Qwen2 LLM Backbone              │               │ Qwen3 LLM Backbone              │              ║
║  │ (Qwen2ForCausalLM)              │               │ (Qwen3ForCausalLM)              │              ║
║  │                                 │               │                                 │              ║
║  │ Visual tokens embedded at       │               │ Main visual tokens embedded     │              ║
║  │ <|image_pad|> positions         │               │ DeepStack features INJECTED     │              ║
║  │                                 │               │ into early LLM layers           │              ║
║  └─────────────────────────────────┘               └─────────────────────────────────┘              ║
║                                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Vision Encoder Deep Dive

### Patch Embedding Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PATCH EMBEDDING COMPARISON                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  QWEN2-VL: Qwen2VisionPatchEmbed                                                                    │
│  ═══════════════════════════════                                                                    │
│                                                                                                     │
│  class Qwen2VisionPatchEmbed(nn.Module):                                                            │
│      def __init__(self, patch_size=14, temporal_patch_size=2, in_channels=3, embed_dim=1152):      │
│          self.proj = Conv3dLayer(                                                                   │
│              in_channels,                                                                           │
│              embed_dim,                                                                             │
│              kernel_size=(temporal_patch_size, patch_size, patch_size),  # (2, 14, 14)              │
│              stride=(temporal_patch_size, patch_size, patch_size),                                  │
│              bias=False  ◄───────────────────────────────────────── NO BIAS                         │
│          )                                                                                          │
│                                                                                                     │
│  Input:  (L, C) where C = 3 × 2 × 14 × 14 = 1176                                                    │
│  Output: (L, embed_dim) = (L, 1152)                                                                 │
│                                                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                     │
│  QWEN3-VL: Qwen3_VisionPatchEmbed                                                                   │
│  ═══════════════════════════════                                                                    │
│                                                                                                     │
│  class Qwen3_VisionPatchEmbed(nn.Module):                                                           │
│      def __init__(self, patch_size=14, temporal_patch_size=2, in_channels=3, hidden_size=1152):    │
│          self.proj = Conv3dLayer(                                                                   │
│              in_channels,                                                                           │
│              hidden_size,                                                                           │
│              kernel_size=(temporal_patch_size, patch_size, patch_size),                             │
│              stride=(temporal_patch_size, patch_size, patch_size),                                  │
│              bias=True  ◄────────────────────────────────────────── HAS BIAS (more expressive)     │
│          )                                                                                          │
│                                                                                                     │
│  WHY THE DIFFERENCE?                                                                                │
│  ─────────────────────                                                                              │
│  • Qwen2-VL: Relies purely on LayerNorm for centering, simpler computation                          │
│  • Qwen3-VL: Bias allows learnable offset per output channel, captures low-frequency                │
│              patterns in patches (like average brightness per region)                               │
│                                                                                                     │
│  MEMORY IMPACT: Minimal (hidden_size additional parameters = ~4.6KB for 1152-dim)                   │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### MLP Activation Functions

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MLP ACTIVATION FUNCTION COMPARISON                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  QWEN2-VL: QuickGELU                                                                                │
│  ═══════════════════                                                                                │
│                                                                                                     │
│  class QuickGELU(nn.Module):                                                                        │
│      def forward(self, x):                                                                          │
│          return x * torch.sigmoid(1.702 * x)                                                        │
│                                                                                                     │
│  Characteristics:                                                                                   │
│  • Approximation of GELU that's faster to compute                                                   │
│  • 1.702 coefficient tuned for approximation accuracy                                               │
│  • Single sigmoid operation vs GELU's erf function                                                  │
│                                                                                                     │
│                    QuickGELU(x) = x × σ(1.702x)                                                     │
│                                                                                                     │
│         │   ╭────────────                                                                           │
│       2 │  ╱                                                                                        │
│         │ ╱                                                                                         │
│       1 │╱                                                                                          │
│         ├──────────────────►                                                                        │
│       0 ╰───────                                                                                    │
│      -1 │                                                                                           │
│         │                                                                                           │
│                                                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                     │
│  QWEN3-VL: SiLU (Swish)                                                                             │
│  ══════════════════════                                                                             │
│                                                                                                     │
│  # From _ACTIVATION_REGISTRY[vision_config.hidden_act] → typically F.silu                           │
│  def silu(x):                                                                                       │
│      return x * torch.sigmoid(x)                                                                    │
│                                                                                                     │
│  Characteristics:                                                                                   │
│  • Smoother than QuickGELU (no 1.702 scaling)                                                       │
│  • Better gradient flow for deeper networks                                                         │
│  • Used in modern LLMs (LLaMA, Qwen3, etc.)                                                         │
│                                                                                                     │
│                    SiLU(x) = x × σ(x)                                                               │
│                                                                                                     │
│         │   ╭────────────                                                                           │
│       2 │  ╱                                                                                        │
│         │ ╱                                                                                         │
│       1 │╱                                                                                          │
│         ├──────────────────►                                                                        │
│       0 ╰───╲                                                                                       │
│      -1 │    ╲_____                                                                                 │
│         │                                                                                           │
│                                                                                                     │
│  PERFORMANCE DIFFERENCE:                                                                            │
│  • QuickGELU: ~5% faster due to pre-computed scaling                                                │
│  • SiLU: Slightly better accuracy, standard in modern architectures                                 │
│  • On T4: No meaningful difference                                                                  │
│  • On H100/B200: SiLU benefits from fused CUDA kernels                                              │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Position Encoding Systems

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              POSITION ENCODING COMPARISON                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  QWEN2-VL: Pure 3D RoPE                                                                             │
│  ═════════════════════                                                                              │
│                                                                                                     │
│  Position IDs computed from grid_thw (temporal, height, width):                                     │
│                                                                                                     │
│    ┌───────────────────────────────────────────────────────────────────────────┐                   │
│    │  for each image/video:                                                     │                   │
│    │    hpos_ids = broadcast(arange(h), shape=(h, w))  # Height positions      │                   │
│    │    wpos_ids = broadcast(arange(w), shape=(h, w))  # Width positions       │                   │
│    │                                                                            │                   │
│    │    # Reshape for 2×2 spatial merging                                       │                   │
│    │    hpos_ids = reshape(h_div, merge_size, w_div, merge_size).transpose()   │                   │
│    │    wpos_ids = reshape(h_div, merge_size, w_div, merge_size).transpose()   │                   │
│    │                                                                            │                   │
│    │    pos_ids = stack([hpos_ids, wpos_ids])  # Shape: (num_patches, 2)       │                   │
│    └───────────────────────────────────────────────────────────────────────────┘                   │
│                                                                                                     │
│    # Apply RoPE                                                                                     │
│    cos, sin = rotary_pos_emb.get_cos_sin(max_grid_size)                                             │
│    cos_combined = cos[pos_ids].flatten(1)                                                           │
│    sin_combined = sin[pos_ids].flatten(1)                                                           │
│                                                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                     │
│  QWEN3-VL: Learned + Interpolated + RoPE                                                            │
│  ═══════════════════════════════════════                                                            │
│                                                                                                     │
│  1. LEARNED POSITION EMBEDDING:                                                                     │
│     ┌───────────────────────────────────────────────────────────────────────────┐                  │
│     │  self.pos_embed = nn.Embedding(num_position_embeddings, hidden_size)     │                  │
│     │                                                                           │                  │
│     │  # Trained embedding table for base resolution                            │                  │
│     │  # num_position_embeddings = num_grid_per_side² (e.g., 40×40 = 1600)     │                  │
│     └───────────────────────────────────────────────────────────────────────────┘                  │
│                                                                                                     │
│  2. BILINEAR INTERPOLATION for variable resolutions:                                                │
│     ┌───────────────────────────────────────────────────────────────────────────┐                  │
│     │  def fast_pos_embed_interpolate(grid_thw):                               │                  │
│     │      for t, h, w in grid_thw:                                             │                  │
│     │          # Create floating-point indices                                  │                  │
│     │          h_idxs = linspace(0, num_grid-1, h)  # e.g., [0, 0.5, 1, ...]   │                  │
│     │          w_idxs = linspace(0, num_grid-1, w)                              │                  │
│     │                                                                           │                  │
│     │          # Floor/ceil for bilinear interpolation                          │                  │
│     │          h_floor, h_ceil = floor(h_idxs), clamp(ceil(h_idxs), max=n-1)   │                  │
│     │          w_floor, w_ceil = floor(w_idxs), clamp(ceil(w_idxs), max=n-1)   │                  │
│     │                                                                           │                  │
│     │          # Compute weights                                                 │                  │
│     │          dh = h_idxs - h_floor  # Fractional part                         │                  │
│     │          dw = w_idxs - w_floor                                            │                  │
│     │                                                                           │                  │
│     │          # Bilinear interpolation weights                                 │                  │
│     │          w00 = (1 - dh) * (1 - dw)  # Top-left                           │                  │
│     │          w01 = (1 - dh) * dw        # Top-right                          │                  │
│     │          w10 = dh * (1 - dw)        # Bottom-left                        │                  │
│     │          w11 = dh * dw              # Bottom-right                       │                  │
│     │                                                                           │                  │
│     │          # Weighted sum of nearest embeddings                             │                  │
│     │          embeds = w00*E[h0,w0] + w01*E[h0,w1] + w10*E[h1,w0] + w11*E[h1,w1]│                  │
│     │      return embeds                                                        │                  │
│     └───────────────────────────────────────────────────────────────────────────┘                  │
│                                                                                                     │
│  3. COMBINED with RoPE (partial rotation):                                                          │
│     ┌───────────────────────────────────────────────────────────────────────────┐                  │
│     │  self.rotary_pos_emb = get_rope(                                         │                  │
│     │      head_size=head_dim,                                                  │                  │
│     │      max_position=8192,                                                   │                  │
│     │      rope_parameters={"partial_rotary_factor": 0.5}  ◄── Only 50% rotated │                  │
│     │  )                                                                        │                  │
│     │                                                                           │                  │
│     │  # Apply: first half rotated, second half unchanged                       │                  │
│     │  hidden_states = patch_embed + interpolated_pos_embed                     │                  │
│     │  # RoPE applied during attention                                          │                  │
│     └───────────────────────────────────────────────────────────────────────────┘                  │
│                                                                                                     │
│  WHY THIS MATTERS:                                                                                  │
│  ─────────────────                                                                                  │
│  • Qwen2-VL: Fixed RoPE frequencies, good for standard resolutions                                  │
│  • Qwen3-VL: Learned embeddings capture complex spatial patterns                                    │
│              Interpolation enables smooth handling of ANY resolution                                │
│              Partial RoPE preserves some absolute position info                                     │
│                                                                                                     │
│  GPU IMPACT:                                                                                        │
│  • T4: Interpolation adds ~2-3ms overhead per image                                                 │
│  • H100/B200: Negligible with fused kernels (<1ms)                                                  │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## DeepStack Multi-Scale Features

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           DEEPSTACK: MULTI-SCALE FEATURE EXTRACTION                                │
│                                  (QWEN3-VL EXCLUSIVE FEATURE)                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  CONCEPT: Extract features from INTERMEDIATE vision encoder layers and inject them                  │
│           into EARLY language model layers for better fine-grained understanding.                   │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          VISION ENCODER                                                      │   │
│  │                                                                                              │   │
│  │    Layer 0  ──────────────────────────────────────────────────────────────────────►         │   │
│  │    Layer 1  ──────────────────────────────────────────────────────────────────────►         │   │
│  │    ...                                                                                       │   │
│  │    Layer k  ──────────────► DeepStack Merger [0] ──────────────────┐                        │   │
│  │    ...                                  ▼                          │                        │   │
│  │    Layer m  ──────────────► DeepStack Merger [1] ──────────────────┼─┐                      │   │
│  │    ...                                  ▼                          │ │                      │   │
│  │    Layer N  ──────────────► Main Merger ────────────────────────┐  │ │                      │   │
│  │                                         ▼                       │  │ │                      │   │
│  └─────────────────────────────────────────────────────────────────┼──┼─┼──────────────────────┘   │
│                                                                    │  │ │                          │
│                              CONCATENATE                           │  │ │                          │
│                                  ▼                                 │  │ │                          │
│                     ┌──────────────────────────┐                   │  │ │                          │
│                     │ [main_emb | ds_0 | ds_1] │◄──────────────────┴──┴─┘                          │
│                     │ (seq, hidden * 3)        │                                                    │
│                     └────────────┬─────────────┘                                                    │
│                                  │                                                                  │
│                                  ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          LANGUAGE MODEL (Qwen3LLMModel)                                      │   │
│  │                                                                                              │   │
│  │    # Split embeddings                                                                        │   │
│  │    main_embeds = output[:, :visual_dim]                                                      │   │
│  │    multiscale_embeds = output[:, visual_dim:]                                                │   │
│  │                                                                                              │   │
│  │    # Reshape for per-layer injection                                                         │   │
│  │    deepstack_embeds = reshape(seq, num_levels, visual_dim)                                   │   │
│  │    deepstack_embeds = permute(num_levels, seq, visual_dim)                                   │   │
│  │                                                                                              │   │
│  │    Layer 0: hidden_states = layer(x) + deepstack_embeds[0]  ◄── INJECTION                   │   │
│  │    Layer 1: hidden_states = layer(x) + deepstack_embeds[1]  ◄── INJECTION                   │   │
│  │    Layer 2: hidden_states = layer(x)  (no injection)                                         │   │
│  │    ...                                                                                       │   │
│  │    Layer N: output                                                                           │   │
│  │                                                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                     │
│  CONFIGURATION (from vision_config):                                                                │
│  ───────────────────────────────────                                                                │
│    deepstack_visual_indexes = [layer_k, layer_m, ...]  # Which vision layers to extract from       │
│    num_deepstack_levels = len(deepstack_visual_indexes)  # Number of scale levels                  │
│                                                                                                     │
│  BENEFITS:                                                                                          │
│  ─────────                                                                                          │
│    ✅ Early layers capture low-level features (edges, textures)                                     │
│    ✅ Middle layers capture mid-level features (shapes, parts)                                      │
│    ✅ Final layers capture high-level features (objects, semantics)                                 │
│    ✅ LLM gets multi-resolution visual information for better grounding                             │
│                                                                                                     │
│  MEMORY OVERHEAD:                                                                                   │
│  ─────────────────                                                                                  │
│    Additional memory = num_levels × seq_len × hidden_size × 2 bytes (FP16)                         │
│                                                                                                     │
│    Example for 1024×1024 image:                                                                     │
│      seq_len ≈ 1024 (after 2×2 merge)                                                               │
│      hidden_size = 3584                                                                             │
│      num_levels = 2                                                                                 │
│      Overhead = 2 × 1024 × 3584 × 2 = ~14 MB per image                                              │
│                                                                                                     │
│    T4 (16GB):  Significant overhead, may need to reduce batch size                                  │
│    A100 (80GB): Negligible                                                                          │
│    H100 (80GB): Negligible                                                                          │
│    B200 (192GB): Negligible                                                                         │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## EVS: Efficient Video Sampling

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      EVS: EFFICIENT VIDEO SAMPLING (QWEN3-VL EXCLUSIVE)                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  PROBLEM: Long videos generate MASSIVE token counts                                                 │
│  ═════════════════════════════════════════════════                                                  │
│                                                                                                     │
│    Example: 60-second video at 2 FPS                                                                │
│    • 120 frames × ~160 tokens/frame = 19,200 tokens                                                 │
│    • Context window: Only 32K on H100                                                               │
│    • Video alone consumes 60% of context!                                                           │
│                                                                                                     │
│  SOLUTION: Content-aware token pruning based on frame similarity                                    │
│  ═════════════════════════════════════════════════════════════════                                  │
│                                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                          EVS ALGORITHM (compute_retention_mask)                              │   │
│  │                                                                                              │   │
│  │  Input: video_embeds (T × H × W, hidden_size)                                                │   │
│  │         video_size_thw (temporal, height, width)                                             │   │
│  │         q = pruning_rate (0.0 to 1.0)                                                        │   │
│  │                                                                                              │   │
│  │  Step 1: Reshape to (T, H, W, hidden_size)                                                   │   │
│  │                                                                                              │   │
│  │  Step 2: Compute frame-to-frame similarity                                                   │   │
│  │          ┌─────────────────────────────────────────────────────────────────────────────┐    │   │
│  │          │  similarity = cosine_similarity(frame[t], frame[t-1])                       │    │   │
│  │          │  dissimilarity = 1 - similarity                                              │    │   │
│  │          │                                                                              │    │   │
│  │          │  Frame 0:  Always keep (set dissimilarity = 255)                             │    │   │
│  │          │  Frame 1:  Compare to Frame 0, keep if different                             │    │   │
│  │          │  Frame 2:  Compare to Frame 1, keep if different                             │    │   │
│  │          │  ...                                                                          │    │   │
│  │          └─────────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                              │   │
│  │  Step 3: Rank all tokens by dissimilarity (higher = more different = more important)        │   │
│  │          order = argsort(dissimilarity, descending=True)                                     │   │
│  │                                                                                              │   │
│  │  Step 4: Keep top-k tokens where k = total_tokens × (1 - q)                                  │   │
│  │          retain_num = max(tokens_per_frame, int(total × (1 - q)))                            │   │
│  │          topk_indices = order[:retain_num]                                                   │   │
│  │                                                                                              │   │
│  │  Step 5: Create boolean mask                                                                 │   │
│  │          retention_mask[topk_indices] = True                                                 │   │
│  │                                                                                              │   │
│  │  Output: retention_mask (T × H × W)                                                          │   │
│  │                                                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                     │
│  VISUAL EXAMPLE:                                                                                    │
│  ═══════════════                                                                                    │
│                                                                                                     │
│    Original video (100 frames, ~16,000 tokens):                                                     │
│    ████████████████████████████████████████████████████████████████████████████████████████████████│
│    ████████████████████████████████████████████████████████████████████████████████████████████████│
│                                                                                                     │
│    After EVS (q=0.5, 50% pruning, ~8,000 tokens):                                                   │
│    ████████░░░░████████░░░░████████░░░░████████░░░░████████░░░░████████░░░░████████░░░░████████░░░░│
│    ████░░░░████████░░░░████████░░░░████████░░░░████████░░░░████████░░░░████████░░░░████████░░░░████│
│                                                                                                     │
│    ████ = Kept tokens (high dissimilarity / different from previous frame)                         │
│    ░░░░ = Pruned tokens (low dissimilarity / similar to previous frame)                            │
│                                                                                                     │
│  M-ROPE POSITION RECOMPUTATION:                                                                     │
│  ═══════════════════════════════                                                                    │
│                                                                                                     │
│    After pruning, positional encodings must be recomputed!                                          │
│                                                                                                     │
│    positions, delta = recompute_mrope_positions(                                                    │
│        input_ids,                                                                                   │
│        mm_embeddings_pos,     # Pre-computed positions with pruning info                            │
│        mrope_positions,       # Original M-RoPE positions                                           │
│        num_computed_tokens,   # Tokens processed so far                                             │
│        vision_start_token_id, # <|vision_start|>                                                    │
│        image_token_id,        # <|image_pad|>                                                       │
│        video_token_id,        # <|video_pad|>                                                       │
│    )                                                                                                │
│                                                                                                     │
│  CONFIGURATION:                                                                                     │
│  ═══════════════                                                                                    │
│                                                                                                     │
│    # In vLLM engine config                                                                          │
│    mm_processor_kwargs = {                                                                          │
│        "video_pruning_rate": 0.5,  # Prune 50% of video tokens                                      │
│    }                                                                                                │
│                                                                                                     │
│    # Or via multimodal_config                                                                       │
│    self.video_pruning_rate = multimodal_config.video_pruning_rate                                   │
│                                                                                                     │
│  PERFORMANCE IMPACT BY GPU:                                                                         │
│  ══════════════════════════                                                                         │
│                                                                                                     │
│    ┌─────────────────────────────────────────────────────────────────────────────────────────┐     │
│    │  GPU   │ Without EVS          │ With EVS (q=0.5)     │ Speedup │ Quality Loss          │     │
│    ├─────────────────────────────────────────────────────────────────────────────────────────┤     │
│    │  T4    │ OOM for >30s video   │ Handles 60s video    │ ∞       │ Minimal               │     │
│    │  A100  │ ~5s for 60s video    │ ~2.5s for 60s video  │ 2x      │ Minimal               │     │
│    │  H100  │ ~2s for 60s video    │ ~1s for 60s video    │ 2x      │ Minimal               │     │
│    │  B200  │ ~1s for 60s video    │ ~0.5s for 60s video  │ 2x      │ Minimal               │     │
│    └─────────────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                                     │
│    FIRST FRAME ALWAYS PRESERVED: Guarantees context for scene understanding                        │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## LLM Backbone Integration

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              LLM BACKBONE COMPARISON                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  QWEN2-VL: Uses Qwen2ForCausalLM                                                                    │
│  ═══════════════════════════════                                                                    │
│                                                                                                     │
│    # From qwen2_vl.py                                                                               │
│    from .qwen2 import Qwen2ForCausalLM, Qwen2Model                                                  │
│                                                                                                     │
│    class Qwen2VLForConditionalGeneration:                                                           │
│        def __init__(self, ...):                                                                     │
│            self.model = Qwen2Model(...)       # LLM backbone                                        │
│            self.visual = Qwen2VisionTransformer(...)  # Vision encoder                              │
│                                                                                                     │
│        def forward(self, ...):                                                                      │
│            # Get visual embeddings                                                                  │
│            vision_embeds = self.visual(pixel_values, grid_thw)                                      │
│                                                                                                     │
│            # Merge into input embeddings at placeholder positions                                   │
│            inputs_embeds = _merge_multimodal_embeddings(                                            │
│                input_ids, inputs_embeds, vision_embeds, placeholder_range                           │
│            )                                                                                        │
│                                                                                                     │
│            # Run LLM                                                                                │
│            return self.model(inputs_embeds=inputs_embeds, ...)                                      │
│                                                                                                     │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                     │
│  QWEN3-VL: Uses Qwen3ForCausalLM with DeepStack injection                                           │
│  ═══════════════════════════════════════════════════════                                            │
│                                                                                                     │
│    # From qwen3_vl.py                                                                               │
│    from .qwen3 import Qwen3ForCausalLM, Qwen3Model                                                  │
│                                                                                                     │
│    class Qwen3LLMModel(Qwen3Model):                                                                 │
│        """Modified Qwen3Model with DeepStack support"""                                             │
│                                                                                                     │
│        def forward(self, ..., deepstack_input_embeds=None):                                         │
│            hidden_states = self.embed_input_ids(input_ids)                                          │
│            residual = None                                                                          │
│                                                                                                     │
│            for layer_idx, layer in enumerate(self.layers):                                          │
│                hidden_states, residual = layer(positions, hidden_states, residual)                  │
│                                                                                                     │
│                # ▼▼▼ DEEPSTACK INJECTION ▼▼▼                                                        │
│                if deepstack_input_embeds is not None:                                               │
│                    if layer_idx < len(deepstack_input_embeds):                                      │
│                        hidden_states = hidden_states +                                              │
│                            deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]            │
│                # ▲▲▲ DEEPSTACK INJECTION ▲▲▲                                                        │
│                                                                                                     │
│            return self.norm(hidden_states, residual)                                                │
│                                                                                                     │
│    class Qwen3VLForConditionalGeneration:                                                           │
│        def __init__(self, ...):                                                                     │
│            self.visual = Qwen3_VisionTransformer(...)                                               │
│            self.language_model = Qwen3LLMForCausalLM(...)                                           │
│                                                                                                     │
│            # DeepStack configuration                                                                │
│            self.use_deepstack = True                                                                │
│            self.deepstack_num_level = len(config.vision_config.deepstack_visual_indexes)            │
│            self.visual_dim = config.vision_config.out_hidden_size                                   │
│            self.multiscale_dim = self.visual_dim * self.deepstack_num_level                         │
│                                                                                                     │
│        def forward(self, ...):                                                                      │
│            # Get visual embeddings (includes DeepStack features concatenated)                       │
│            vision_output = self.visual(pixel_values, grid_thw)                                      │
│            # Shape: (seq_len, visual_dim + multiscale_dim)                                          │
│                                                                                                     │
│            # Split main and DeepStack embeddings                                                    │
│            main_embeds = vision_output[:, :self.visual_dim]                                         │
│            multiscale_embeds = vision_output[:, self.visual_dim:]                                   │
│                                                                                                     │
│            # Prepare DeepStack for injection                                                        │
│            deepstack_embeds = self._prepare_deepstack(multiscale_embeds)                            │
│                                                                                                     │
│            # Merge main embeddings into input                                                       │
│            inputs_embeds = _merge_multimodal_embeddings(...)                                        │
│                                                                                                     │
│            # Run LLM with DeepStack injection                                                       │
│            return self.language_model(                                                              │
│                inputs_embeds=inputs_embeds,                                                         │
│                deepstack_input_embeds=deepstack_embeds,  # ◄── PASSED TO LLM                        │
│                ...                                                                                  │
│            )                                                                                        │
│                                                                                                     │
│  KEY DIFFERENCES:                                                                                   │
│  ═════════════════                                                                                  │
│                                                                                                     │
│    ┌────────────────────────┬──────────────────────────┬────────────────────────────────────────┐   │
│    │ Aspect                 │ Qwen2-VL                 │ Qwen3-VL                               │   │
│    ├────────────────────────┼──────────────────────────┼────────────────────────────────────────┤   │
│    │ Visual injection       │ Embedding level only     │ Embedding + Early LLM layers           │   │
│    │ Multi-scale features   │ No                       │ Yes (DeepStack)                        │   │
│    │ LLM modification       │ None                     │ Modified forward() for injection       │   │
│    │ Memory during forward  │ Lower                    │ Higher (DeepStack buffers)             │   │
│    │ Torch compile support  │ Limited                  │ @support_torch_compile decorator       │   │
│    │ Eagle3 speculative     │ No                       │ Yes (SupportsEagle3 interface)         │   │
│    │ Multimodal pruning     │ No                       │ Yes (SupportsMultiModalPruning)        │   │
│    └────────────────────────┴──────────────────────────┴────────────────────────────────────────┘   │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## GPU-Specific Optimization Analysis

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          GPU-SPECIFIC OPTIMIZATION ANALYSIS                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                           T4 (TURING SM 7.5, 16GB GDDR6)                                     ║  │
│  ╠══════════════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                              ║  │
│  ║  ARCHITECTURE LIMITATIONS:                                                                   ║  │
│  ║  • No FlashAttention v2 (requires SM 8.0+)                                                   ║  │
│  ║  • No BF16 support (FP16 only)                                                               ║  │
│  ║  • No FP8 quantization                                                                       ║  │
│  ║  • Limited tensor core utilization for attention                                             ║  │
│  ║                                                                                              ║  │
│  ║  ATTENTION BACKEND: TORCH_SDPA (only option)                                                 ║  │
│  ║  ┌────────────────────────────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  # From qwen3_vl.py                                                                    │ ║  │
│  ║  │  if self.attn_backend not in {                                                         │ ║  │
│  ║  │      AttentionBackendEnum.FLASH_ATTN,                                                  │ ║  │
│  ║  │      AttentionBackendEnum.TORCH_SDPA,      ◄── T4 uses this                            │ ║  │
│  ║  │      AttentionBackendEnum.ROCM_AITER_FA,                                               │ ║  │
│  ║  │  }:                                                                                    │ ║  │
│  ║  │      raise RuntimeError("Qwen3-VL does not support...")                                │ ║  │
│  ║  └────────────────────────────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                                              ║  │
│  ║  MEMORY CONSTRAINTS:                                                                         ║  │
│  ║  ┌────────────────────────────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  Component             │ Qwen2-VL-2B      │ Qwen3-VL-4B (4-bit)                        │ ║  │
│  ║  │  ─────────────────────┼──────────────────┼─────────────────────────────────           │ ║  │
│  ║  │  Model weights         │ ~5 GB            │ ~3 GB (quantized)                          │ ║  │
│  ║  │  KV Cache (2K ctx)     │ ~2 GB            │ ~2.5 GB                                    │ ║  │
│  ║  │  Activations           │ ~1 GB            │ ~1.5 GB                                    │ ║  │
│  ║  │  CUDA context          │ ~1.5 GB          │ ~1.5 GB                                    │ ║  │
│  ║  │  Vision encoder temp   │ ~2 GB            │ ~3 GB (DeepStack overhead)                 │ ║  │
│  ║  │  ─────────────────────┼──────────────────┼─────────────────────────────────           │ ║  │
│  ║  │  TOTAL                 │ ~11.5 GB         │ ~11.5 GB                                   │ ║  │
│  ║  │  FREE                  │ ~4.5 GB          │ ~4.5 GB                                    │ ║  │
│  ║  └────────────────────────────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                                              ║  │
│  ║  RECOMMENDED SETTINGS:                                                                       ║  │
│  ║  • enforce_eager=True (saves ~500MB by disabling CUDA graphs)                                ║  │
│  ║  • enable_prefix_caching=False (memory constraint)                                           ║  │
│  ║  • enable_chunked_prefill=False (simpler execution path)                                     ║  │
│  ║  • max_num_seqs=4 (limited batch size)                                                       ║  │
│  ║  • max_pixels=512000 (~720×720 max resolution)                                               ║  │
│  ║  • video_pruning_rate=0.5 (aggressive EVS for Qwen3-VL)                                      ║  │
│  ║                                                                                              ║  │
│  ║  EXPECTED PERFORMANCE:                                                                       ║  │
│  ║  • Latency: 800-1200ms per image                                                             ║  │
│  ║  • Throughput: 15-25 tok/s                                                                   ║  │
│  ║  • Batch: 4 concurrent requests max                                                          ║  │
│  ║                                                                                              ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                          A100 (AMPERE SM 8.0, 40/80GB HBM2e)                                 ║  │
│  ╠══════════════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                              ║  │
│  ║  ARCHITECTURE ADVANTAGES:                                                                    ║  │
│  ║  • FlashAttention v2 supported                                                               ║  │
│  ║  • BF16 native support                                                                       ║  │
│  ║  • 2 TB/s memory bandwidth (40GB) / 2.4 TB/s (80GB)                                          ║  │
│  ║  • 312 TFLOPS FP16 tensor core performance                                                   ║  │
│  ║                                                                                              ║  │
│  ║  ATTENTION BACKEND: FLASH_ATTN (preferred)                                                   ║  │
│  ║  ┌────────────────────────────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  # FlashAttention automatically selected for SM 8.0+                                   │ ║  │
│  ║  │  attn_backend = get_vit_attn_backend(head_size, dtype, attn_backend_override=None)     │ ║  │
│  ║  │  # Returns AttentionBackendEnum.FLASH_ATTN                                             │ ║  │
│  ║  └────────────────────────────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                                              ║  │
│  ║  MEMORY BUDGET (80GB):                                                                       ║  │
│  ║  ┌────────────────────────────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  Component             │ Qwen2-VL-7B      │ Qwen3-VL-8B       │ Qwen3-VL-30B-A3B       │ ║  │
│  ║  │  ─────────────────────┼──────────────────┼───────────────────┼────────────────        │ ║  │
│  ║  │  Model weights         │ ~15 GB           │ ~17 GB            │ ~65 GB (MoE)           │ ║  │
│  ║  │  KV Cache (16K ctx)    │ ~20 GB           │ ~22 GB            │ ~8 GB (sparse)         │ ║  │
│  ║  │  Activations           │ ~8 GB            │ ~10 GB            │ ~5 GB                  │ ║  │
│  ║  │  DeepStack buffers     │ N/A              │ ~2 GB             │ ~2 GB                  │ ║  │
│  ║  │  Prefix cache          │ ~10 GB           │ ~10 GB            │ N/A                    │ ║  │
│  ║  │  ─────────────────────┼──────────────────┼───────────────────┼────────────────        │ ║  │
│  ║  │  TOTAL                 │ ~53 GB           │ ~61 GB            │ ~80 GB                 │ ║  │
│  ║  └────────────────────────────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                                              ║  │
│  ║  RECOMMENDED SETTINGS:                                                                       ║  │
│  ║  • dtype="bfloat16" (numerical stability + tensor core efficiency)                           ║  │
│  ║  • enable_prefix_caching=True (leverage abundant memory)                                     ║  │
│  ║  • enable_chunked_prefill=True (better long-sequence handling)                               ║  │
│  ║  • max_num_seqs=32 (good concurrency)                                                        ║  │
│  ║  • max_pixels=1572864 (~1280×1280 resolution)                                                ║  │
│  ║  • video_pruning_rate=0.3 (moderate EVS for Qwen3-VL)                                        ║  │
│  ║                                                                                              ║  │
│  ║  EXPECTED PERFORMANCE:                                                                       ║  │
│  ║  • Latency: 200-400ms per image                                                              ║  │
│  ║  • Throughput: 80-120 tok/s                                                                  ║  │
│  ║  • Batch: 32 concurrent requests                                                             ║  │
│  ║                                                                                              ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                           H100 (HOPPER SM 9.0, 80GB HBM3)                                    ║  │
│  ╠══════════════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                              ║  │
│  ║  ARCHITECTURE ADVANTAGES:                                                                    ║  │
│  ║  • FlashAttention v3 (Hopper-optimized)                                                      ║  │
│  ║  • FP8 hardware support (2x throughput)                                                      ║  │
│  ║  • 3.35 TB/s memory bandwidth (10x T4!)                                                      ║  │
│  ║  • 1,979 TFLOPS FP16 / 3,958 TFLOPS FP8                                                      ║  │
│  ║  • Transformer Engine (automatic mixed precision)                                            ║  │
│  ║                                                                                              ║  │
│  ║  FP8 QUANTIZATION BENEFITS:                                                                  ║  │
│  ║  ┌────────────────────────────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  # For maximum throughput on H100                                                      │ ║  │
│  ║  │  config = {                                                                            │ ║  │
│  ║  │      "quantization": "fp8",        # 2x throughput vs FP16                             │ ║  │
│  ║  │      "kv_cache_dtype": "fp8",      # 2x KV cache capacity                              │ ║  │
│  ║  │      "max_model_len": 65536,       # 64K context possible!                             │ ║  │
│  ║  │      "max_num_seqs": 128,          # Massive concurrency                               │ ║  │
│  ║  │  }                                                                                     │ ║  │
│  ║  └────────────────────────────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                                              ║  │
│  ║  RECOMMENDED SETTINGS (Qwen3-VL-8B):                                                         ║  │
│  ║  • dtype="bfloat16"                                                                          ║  │
│  ║  • max_model_len=32768 (32K context)                                                         ║  │
│  ║  • max_num_seqs=64                                                                           ║  │
│  ║  • max_pixels=2073600 (1920×1080)                                                            ║  │
│  ║  • video_pruning_rate=0.3                                                                    ║  │
│  ║  • enable_prefix_caching=True                                                                ║  │
│  ║                                                                                              ║  │
│  ║  EXPECTED PERFORMANCE:                                                                       ║  │
│  ║  • Latency: 100-200ms per image                                                              ║  │
│  ║  • Throughput: 120-180 tok/s                                                                 ║  │
│  ║  • Batch: 64+ concurrent requests                                                            ║  │
│  ║                                                                                              ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                                     │
│  ╔══════════════════════════════════════════════════════════════════════════════════════════════╗  │
│  ║                         B200 (BLACKWELL SM 10.0, 192GB HBM3e)                                ║  │
│  ╠══════════════════════════════════════════════════════════════════════════════════════════════╣  │
│  ║                                                                                              ║  │
│  ║  ARCHITECTURE ADVANTAGES:                                                                    ║  │
│  ║  • FP4 hardware support (4x throughput when available)                                       ║  │
│  ║  • 8 TB/s memory bandwidth (25x T4!)                                                         ║  │
│  ║  • ~4,000 TFLOPS FP16 / ~8,000 TFLOPS FP8 / ~16,000 TFLOPS FP4                               ║  │
│  ║  • 2nd Gen Transformer Engine                                                                ║  │
│  ║  • Decompression Engine (direct compressed loading)                                          ║  │
│  ║                                                                                              ║  │
│  ║  UNIQUE CAPABILITIES:                                                                        ║  │
│  ║  ┌────────────────────────────────────────────────────────────────────────────────────────┐ ║  │
│  ║  │  • Run Qwen2-VL-72B on SINGLE GPU (full precision!)                                    │ ║  │
│  ║  │  • 128K context with Qwen3-VL-8B                                                       │ ║  │
│  ║  │  • 4K resolution images natively                                                       │ ║  │
│  ║  │  • Process 10+ minute videos without OOM                                               │ ║  │
│  ║  └────────────────────────────────────────────────────────────────────────────────────────┘ ║  │
│  ║                                                                                              ║  │
│  ║  RECOMMENDED SETTINGS (Qwen3-VL-8B):                                                         ║  │
│  ║  • dtype="bfloat16"                                                                          ║  │
│  ║  • max_model_len=131072 (128K context!)                                                      ║  │
│  ║  • max_num_seqs=128                                                                          ║  │
│  ║  • max_pixels=4147200 (4K resolution)                                                        ║  │
│  ║  • video_pruning_rate=0.3                                                                    ║  │
│  ║  • enable_prefix_caching=True                                                                ║  │
│  ║                                                                                              ║  │
│  ║  EXPECTED PERFORMANCE:                                                                       ║  │
│  ║  • Latency: 50-100ms per image                                                               ║  │
│  ║  • Throughput: 180-300 tok/s                                                                 ║  │
│  ║  • Batch: 128+ concurrent requests                                                           ║  │
│  ║                                                                                              ║  │
│  ╚══════════════════════════════════════════════════════════════════════════════════════════════╝  │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## SGLang/vLLM Layer Compatibility

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          vLLM LAYER COMPATIBILITY ANALYSIS                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                     │
│  SUPPORTED INTERFACES:                                                                              │
│  ═════════════════════                                                                              │
│                                                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Class                            │ Qwen2-VL │ Qwen3-VL │ Description                       │    │
│  ├────────────────────────────────────────────────────────────────────────────────────────────┤    │
│  │  SupportsMultiModal               │ ✅       │ ✅       │ Basic multimodal support          │    │
│  │  SupportsLoRA                     │ ✅       │ ✅       │ LoRA fine-tuning                  │    │
│  │  SupportsPP                       │ ✅       │ ✅       │ Pipeline parallelism              │    │
│  │  SupportsMRoPE                    │ ✅       │ ✅       │ Multi-dimensional RoPE            │    │
│  │  SupportsEagle3                   │ ❌       │ ✅       │ Eagle3 speculative decode         │    │
│  │  SupportsMultiModalPruning        │ ❌       │ ✅       │ EVS token pruning                 │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
│  ATTENTION BACKENDS:                                                                                │
│  ═══════════════════                                                                                │
│                                                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Backend               │ Qwen2-VL │ Qwen3-VL │ GPU Requirement                            │    │
│  ├────────────────────────────────────────────────────────────────────────────────────────────┤    │
│  │  FLASH_ATTN            │ ✅       │ ✅       │ SM 8.0+ (A100, H100, B200)                 │    │
│  │  TORCH_SDPA            │ ✅       │ ✅       │ Any (fallback for T4)                      │    │
│  │  XFORMERS              │ ✅       │ ❌       │ SM 7.0+ (not supported in Qwen3-VL)        │    │
│  │  ROCM_AITER_FA         │ ✅       │ ✅       │ AMD ROCm GPUs                              │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
│  TORCH.COMPILE SUPPORT:                                                                             │
│  ══════════════════════                                                                             │
│                                                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │  # Qwen3-VL has explicit torch.compile support                                            │    │
│  │  @support_torch_compile(                                                                  │    │
│  │      dynamic_arg_dims={                                                                   │    │
│  │          "input_ids": 0,                                                                  │    │
│  │          "positions": -1,  # (3, seq_len) for M-RoPE                                      │    │
│  │          "intermediate_tensors": 0,                                                       │    │
│  │          "inputs_embeds": 0,                                                              │    │
│  │          "deepstack_input_embeds": 0,  # ◄── Qwen3-VL specific                            │    │
│  │      }                                                                                    │    │
│  │  )                                                                                        │    │
│  │  class Qwen3LLMModel(Qwen3Model):                                                         │    │
│  │      ...                                                                                  │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
│  TENSOR PARALLELISM:                                                                                │
│  ═══════════════════                                                                                │
│                                                                                                     │
│  Both support:                                                                                      │
│  • ColumnParallelLinear / RowParallelLinear for LLM layers                                          │
│  • Vision encoder can use TP or Data Parallel mode:                                                 │
│                                                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │  # Data parallel mode for vision encoder (faster for multi-image batches)                │    │
│  │  use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"                       │    │
│  │                                                                                           │    │
│  │  self.linear_fc1 = ColumnParallelLinear(                                                  │    │
│  │      in_features,                                                                         │    │
│  │      hidden_features,                                                                     │    │
│  │      disable_tp=use_data_parallel,  # ◄── Disable TP if using DP                          │    │
│  │  )                                                                                        │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
│  QUANTIZATION SUPPORT:                                                                              │
│  ══════════════════════                                                                             │
│                                                                                                     │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │  Method         │ Qwen2-VL │ Qwen3-VL │ GPU Requirement │ Notes                           │    │
│  ├────────────────────────────────────────────────────────────────────────────────────────────┤    │
│  │  FP16           │ ✅       │ ✅       │ Any             │ Default                          │    │
│  │  BF16           │ ✅       │ ✅       │ SM 8.0+         │ Recommended for A100+            │    │
│  │  INT8 (W8A8)    │ ✅       │ ✅       │ Any             │ Via bitsandbytes                 │    │
│  │  INT4 (W4A16)   │ ✅       │ ✅       │ Any             │ AWQ, GPTQ, bitsandbytes          │    │
│  │  FP8 (E4M3)     │ ✅       │ ✅       │ SM 9.0+         │ H100+ only, 2x throughput        │    │
│  │  FP4            │ ❌       │ ⚠️        │ SM 10.0+        │ B200 only, future support        │    │
│  └────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Practical Code Examples

### T4 Configuration

```python
# Optimal T4 configuration for Qwen2-VL-2B (full precision)
T4_QWEN2_VL_CONFIG = {
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "trust_remote_code": True,
    "dtype": "half",  # FP16 only (no BF16 on Turing)
    "gpu_memory_utilization": 0.90,
    "max_model_len": 2048,
    "enforce_eager": True,  # Saves ~500MB
    "max_num_seqs": 4,
    "limit_mm_per_prompt": {"image": 1, "video": 0},
    "mm_processor_kwargs": {
        "min_pixels": 784,
        "max_pixels": 512000,  # ~720×720
    },
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
}

# Optimal T4 configuration for Qwen3-VL-4B (4-bit quantized)
T4_QWEN3_VL_CONFIG = {
    "model": "Qwen/Qwen3-VL-4B-Instruct",
    "trust_remote_code": True,
    "dtype": "half",
    "quantization": "bitsandbytes",
    "load_format": "bitsandbytes",
    "gpu_memory_utilization": 0.92,
    "max_model_len": 2048,
    "enforce_eager": True,
    "max_num_seqs": 4,
    "limit_mm_per_prompt": {"image": 1, "video": 1},
    "mm_processor_kwargs": {
        "min_pixels": 784,
        "max_pixels": 512000,
        "video_pruning_rate": 0.5,  # EVS: Keep 50%
    },
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
}
```

### H100 Configuration

```python
# Optimal H100 configuration for Qwen3-VL-8B (full precision)
H100_QWEN3_VL_CONFIG = {
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.95,
    "max_model_len": 32768,  # 32K context
    "enforce_eager": False,  # Enable CUDA graphs
    "max_num_seqs": 64,
    "limit_mm_per_prompt": {"image": 16, "video": 8},
    "mm_processor_kwargs": {
        "min_pixels": 784,
        "max_pixels": 2073600,  # 1920×1080
        "video_pruning_rate": 0.3,  # EVS: Keep 70%
    },
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
}

# High-throughput H100 with FP8
H100_QWEN3_VL_FP8_CONFIG = {
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "quantization": "fp8",
    "kv_cache_dtype": "fp8",  # 2x KV capacity
    "gpu_memory_utilization": 0.95,
    "max_model_len": 65536,  # 64K context!
    "enforce_eager": False,
    "max_num_seqs": 128,  # Massive batch
    "limit_mm_per_prompt": {"image": 32, "video": 16},
    "mm_processor_kwargs": {
        "min_pixels": 784,
        "max_pixels": 2073600,
        "video_pruning_rate": 0.3,
    },
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
}
```

### B200 Configuration

```python
# Optimal B200 configuration for Qwen3-VL-8B (ultra-long context)
B200_QWEN3_VL_CONFIG = {
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.95,
    "max_model_len": 131072,  # 128K context!
    "enforce_eager": False,
    "max_num_seqs": 128,
    "limit_mm_per_prompt": {"image": 32, "video": 16},
    "mm_processor_kwargs": {
        "min_pixels": 784,
        "max_pixels": 4147200,  # 4K resolution
        "video_pruning_rate": 0.3,
    },
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
}

# B200 running Qwen2-VL-72B on SINGLE GPU
B200_QWEN2_VL_72B_CONFIG = {
    "model": "Qwen/Qwen2-VL-72B-Instruct",
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.95,
    "max_model_len": 32768,
    "enforce_eager": False,
    "max_num_seqs": 32,
    "limit_mm_per_prompt": {"image": 16, "video": 8},
    "mm_processor_kwargs": {
        "min_pixels": 784,
        "max_pixels": 4147200,
    },
    "enable_prefix_caching": True,
    "enable_chunked_prefill": True,
}
```

