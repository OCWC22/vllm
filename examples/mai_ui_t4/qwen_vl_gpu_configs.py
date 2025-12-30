# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen2-VL and Qwen3-VL GPU-Optimized Configurations

This module provides production-ready configurations for running Qwen Vision-Language
models across different NVIDIA GPU architectures with vLLM.

Architecture Comparison:
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    QWEN2-VL vs QWEN3-VL KEY DIFFERENCES                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Feature              │ Qwen2-VL                  │ Qwen3-VL                        │
│  ─────────────────────┼───────────────────────────┼─────────────────────────        │
│  Position Encoding    │ 3D RoPE only              │ Learned + RoPE + Interpolate    │
│  MLP Activation       │ QuickGELU                 │ SiLU (configurable)             │
│  Multi-Scale Features │ ❌                        │ ✅ DeepStack                    │
│  Video Token Pruning  │ ❌                        │ ✅ EVS (Efficient Video Samp)   │
│  Max Video Frames     │ 14                        │ 24,576                          │
│  Speculative Decode   │ Basic                     │ ✅ Eagle3 native                │
│  MoE Variants         │ ❌                        │ ✅ Qwen3-VL-30B-A3B             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

GPU Memory Requirements:
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  Model              │ FP16 Size │ 4-bit Size │ Recommended GPU                     │
│  ───────────────────┼───────────┼────────────┼──────────────────────               │
│  Qwen2-VL-2B        │ ~5 GB     │ ~2 GB      │ T4 (16GB) ✅                        │
│  Qwen2-VL-7B        │ ~15 GB    │ ~5 GB      │ A100 (40GB) ✅, T4 (4-bit)          │
│  Qwen2-VL-72B       │ ~150 GB   │ ~40 GB     │ Multi-GPU or B200                   │
│  ───────────────────┼───────────┼────────────┼──────────────────────               │
│  Qwen3-VL-4B        │ ~9 GB     │ ~3 GB      │ T4 (16GB) with quant                │
│  Qwen3-VL-8B        │ ~17 GB    │ ~5 GB      │ A100 (40GB) ✅, T4 (4-bit)          │
│  Qwen3-VL-30B-A3B   │ ~65 GB    │ ~20 GB     │ A100/H100 (80GB) ✅                 │
└─────────────────────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GPUArchitecture(str, Enum):
    """Supported GPU architectures."""
    TURING = "turing"        # T4, RTX 20xx (SM 7.5)
    AMPERE = "ampere"        # A100, RTX 30xx (SM 8.0)
    ADA = "ada"              # L4, L40, RTX 40xx (SM 8.9)
    HOPPER = "hopper"        # H100, H200 (SM 9.0)
    BLACKWELL = "blackwell"  # B100, B200 (SM 10.0)


class ModelFamily(str, Enum):
    """Qwen VL model families."""
    QWEN2_VL = "qwen2_vl"
    QWEN3_VL = "qwen3_vl"


class ModelSize(str, Enum):
    """Model size variants."""
    # Qwen2-VL
    QWEN2_VL_2B = "Qwen/Qwen2-VL-2B-Instruct"
    QWEN2_VL_7B = "Qwen/Qwen2-VL-7B-Instruct"
    QWEN2_VL_72B = "Qwen/Qwen2-VL-72B-Instruct"
    # Qwen3-VL
    QWEN3_VL_4B = "Qwen/Qwen3-VL-4B-Instruct"
    QWEN3_VL_8B = "Qwen/Qwen3-VL-8B-Instruct"
    QWEN3_VL_30B_A3B = "Qwen/Qwen3-VL-30B-A3B-Instruct"  # MoE


class QuantizationMethod(str, Enum):
    """Quantization methods by GPU support."""
    NONE = "none"              # FP16/BF16, all GPUs
    BITSANDBYTES = "bitsandbytes"  # 4-bit, all GPUs
    AWQ = "awq"                # 4-bit, Ampere+ preferred
    GPTQ = "gptq"              # 4-bit, all GPUs
    FP8 = "fp8"                # FP8, Hopper+ only (SM 9.0+)
    FP4 = "fp4"                # FP4, Blackwell only (SM 10.0+)


@dataclass
class QwenVLConfig:
    """
    GPU-specific optimization config for Qwen Vision-Language models.
    
    This config handles both Qwen2-VL and Qwen3-VL with appropriate
    architecture-specific optimizations.
    """
    
    # Model identification
    model: ModelSize
    model_family: ModelFamily = ModelFamily.QWEN2_VL
    
    # Precision
    quantization: QuantizationMethod = QuantizationMethod.NONE
    dtype: str = "half"  # "half", "bfloat16", or "auto"
    kv_cache_dtype: str = "auto"  # "auto", "fp8", "fp8_e5m2"
    
    # Memory management
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    enforce_eager: bool = False
    
    # Batching
    max_num_seqs: int = 8
    
    # Vision settings
    min_pixels: int = 28 * 28  # 784
    max_pixels: int = 1003520  # ~1000x1000
    limit_images: int = 4
    limit_videos: int = 1
    
    # Qwen3-VL specific: EVS (Efficient Video Sampling)
    video_pruning_rate: float | None = None  # 0.0-1.0, None = disabled
    
    # Advanced optimizations
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    attention_backend: str | None = None  # Auto-detect
    
    def to_engine_args(self) -> dict[str, Any]:
        """Convert to vLLM LLM() constructor arguments."""
        args = {
            "model": self.model.value,
            "trust_remote_code": True,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "max_num_seqs": self.max_num_seqs,
            "limit_mm_per_prompt": {
                "image": self.limit_images,
                "video": self.limit_videos,
            },
            "mm_processor_kwargs": {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            },
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
        }
        
        # Quantization
        if self.quantization != QuantizationMethod.NONE:
            args["quantization"] = self.quantization.value
            if self.quantization == QuantizationMethod.BITSANDBYTES:
                args["load_format"] = "bitsandbytes"
        
        # KV cache dtype
        if self.kv_cache_dtype != "auto":
            args["kv_cache_dtype"] = self.kv_cache_dtype
        
        # Qwen3-VL specific: EVS
        if self.video_pruning_rate is not None and self.model_family == ModelFamily.QWEN3_VL:
            args["mm_processor_kwargs"]["video_pruning_rate"] = self.video_pruning_rate
        
        return args
    
    def get_architecture_info(self) -> str:
        """Return architecture diagram for this config."""
        if self.model_family == ModelFamily.QWEN2_VL:
            return QWEN2_VL_ARCH_DIAGRAM
        return QWEN3_VL_ARCH_DIAGRAM


# =============================================================================
# ARCHITECTURE DIAGRAMS
# =============================================================================

QWEN2_VL_ARCH_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          QWEN2-VL ARCHITECTURE                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Image/Video ──▶ Conv3D Patch Embed (no bias) ──▶ 3D RoPE Position Encoding        │
│                                                        │                            │
│                                                        ▼                            │
│                                              Vision Transformer Blocks              │
│                                              ┌─────────────────────────┐            │
│                                              │ LayerNorm               │            │
│                                              │     ↓                   │            │
│                                              │ Attention (QKV + RoPE)  │            │
│                                              │     ↓                   │            │
│                                              │ LayerNorm               │            │
│                                              │     ↓                   │            │
│                                              │ MLP (QuickGELU)         │            │
│                                              └─────────────────────────┘            │
│                                                        │                            │
│                                                        ▼                            │
│                                              Single Patch Merger                    │
│                                                        │                            │
│                                                        ▼                            │
│                                              Qwen2 LLM Backbone                     │
│                                                                                     │
│  Key Features:                                                                      │
│  • QuickGELU activation: x * sigmoid(1.702 * x)                                     │
│  • Spatial merge size: 2×2 patches → 1 token                                        │
│  • Max 14 video frames (default)                                                    │
│  • Supports FlashAttn, SDPA, Xformers                                               │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
"""

QWEN3_VL_ARCH_DIAGRAM = """
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          QWEN3-VL ARCHITECTURE                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  Image/Video ──▶ Conv3D Patch Embed (WITH bias) ──▶ Learned + Interpolated PosEmb  │
│                                                             │                       │
│                                                             ▼                       │
│                                              Vision Transformer Blocks              │
│                                              ┌─────────────────────────┐            │
│                                              │ LayerNorm               │            │
│                                              │     ↓                   │────────┐   │
│                                              │ Attention (QKV + RoPE)  │        │   │
│                                              │     ↓                   │   DeepStack│
│                                              │ LayerNorm               │   Mergers  │
│                                              │     ↓                   │   (multi-  │
│                                              │ MLP (SiLU)              │   scale)   │
│                                              └─────────────────────────┘        │   │
│                                                        │                        │   │
│                                                        ▼                        │   │
│                                              Main Patch Merger ←────────────────┘   │
│                                                        │                            │
│                                                        ▼                            │
│                                              Qwen3 LLM (DeepStack injection)        │
│                                                                                     │
│  Key NEW Features:                                                                  │
│  • SiLU activation: x * sigmoid(x)                                                  │
│  • DeepStack: Multi-scale features from intermediate layers                         │
│  • EVS: Efficient Video Sampling (token pruning)                                    │
│  • Max 24,576 video frames                                                          │
│  • Eagle3 speculative decoding support                                              │
│  • MoE variant: Qwen3-VL-30B-A3B                                                    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
"""


# =============================================================================
# T4 CONFIGURATIONS (TURING - SM 7.5, 16GB)
# =============================================================================
# Constraints: No FlashAttention v2, no BF16, no FP8, limited memory
# Strategy: Aggressive quantization, reduced context, single image

T4_QWEN2_VL_2B = QwenVLConfig(
    model=ModelSize.QWEN2_VL_2B,
    model_family=ModelFamily.QWEN2_VL,
    quantization=QuantizationMethod.NONE,
    dtype="half",  # FP16 only (no BF16 on Turing)
    gpu_memory_utilization=0.90,
    max_model_len=2048,
    enforce_eager=True,  # Disable CUDA graphs (saves ~500MB)
    max_num_seqs=4,
    max_pixels=512000,  # ~720×720
    limit_images=1,
    limit_videos=0,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
)

T4_QWEN2_VL_7B_4BIT = QwenVLConfig(
    model=ModelSize.QWEN2_VL_7B,
    model_family=ModelFamily.QWEN2_VL,
    quantization=QuantizationMethod.BITSANDBYTES,
    dtype="half",
    gpu_memory_utilization=0.92,
    max_model_len=1024,
    enforce_eager=True,
    max_num_seqs=2,
    max_pixels=256000,
    limit_images=1,
    limit_videos=0,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
)

T4_QWEN3_VL_4B_4BIT = QwenVLConfig(
    model=ModelSize.QWEN3_VL_4B,
    model_family=ModelFamily.QWEN3_VL,
    quantization=QuantizationMethod.BITSANDBYTES,
    dtype="half",
    gpu_memory_utilization=0.92,
    max_model_len=2048,
    enforce_eager=True,
    max_num_seqs=4,
    max_pixels=512000,
    limit_images=1,
    limit_videos=1,
    video_pruning_rate=0.5,  # EVS: Keep 50% of video tokens
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
)


# =============================================================================
# A100 CONFIGURATIONS (AMPERE - SM 8.0, 40/80GB)
# =============================================================================
# Advantages: FlashAttention v2, BF16, high bandwidth
# Strategy: Full precision when possible, moderate batching

A100_40GB_QWEN2_VL_7B = QwenVLConfig(
    model=ModelSize.QWEN2_VL_7B,
    model_family=ModelFamily.QWEN2_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    enforce_eager=False,
    max_num_seqs=16,
    max_pixels=1003520,
    limit_images=4,
    limit_videos=1,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

A100_80GB_QWEN2_VL_7B = QwenVLConfig(
    model=ModelSize.QWEN2_VL_7B,
    model_family=ModelFamily.QWEN2_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=16384,
    enforce_eager=False,
    max_num_seqs=32,
    max_pixels=1572864,  # ~1280×1280
    limit_images=8,
    limit_videos=2,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

A100_80GB_QWEN3_VL_8B = QwenVLConfig(
    model=ModelSize.QWEN3_VL_8B,
    model_family=ModelFamily.QWEN3_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=16384,
    enforce_eager=False,
    max_num_seqs=32,
    max_pixels=1572864,
    limit_images=8,
    limit_videos=4,
    video_pruning_rate=0.3,  # EVS: Keep 70% of video tokens
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

A100_80GB_QWEN3_VL_30B_MOE = QwenVLConfig(
    model=ModelSize.QWEN3_VL_30B_A3B,
    model_family=ModelFamily.QWEN3_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=8192,
    enforce_eager=False,
    max_num_seqs=16,
    max_pixels=1003520,
    limit_images=4,
    limit_videos=2,
    video_pruning_rate=0.4,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)


# =============================================================================
# H100 CONFIGURATIONS (HOPPER - SM 9.0, 80GB HBM3)
# =============================================================================
# Advantages: FlashAttention 3, FP8, 3.35 TB/s bandwidth, Transformer Engine
# Strategy: Maximum performance, FP8 for throughput

H100_QWEN2_VL_7B = QwenVLConfig(
    model=ModelSize.QWEN2_VL_7B,
    model_family=ModelFamily.QWEN2_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=32768,
    enforce_eager=False,
    max_num_seqs=64,
    max_pixels=2073600,  # 1920×1080
    limit_images=16,
    limit_videos=4,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

H100_QWEN3_VL_8B = QwenVLConfig(
    model=ModelSize.QWEN3_VL_8B,
    model_family=ModelFamily.QWEN3_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=32768,
    enforce_eager=False,
    max_num_seqs=64,
    max_pixels=2073600,
    limit_images=16,
    limit_videos=8,
    video_pruning_rate=0.3,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

H100_QWEN3_VL_8B_FP8 = QwenVLConfig(
    model=ModelSize.QWEN3_VL_8B,
    model_family=ModelFamily.QWEN3_VL,
    quantization=QuantizationMethod.FP8,
    dtype="bfloat16",
    kv_cache_dtype="fp8",  # FP8 KV cache for 2x capacity
    gpu_memory_utilization=0.95,
    max_model_len=65536,  # 64K context with FP8
    enforce_eager=False,
    max_num_seqs=128,
    max_pixels=2073600,
    limit_images=32,
    limit_videos=16,
    video_pruning_rate=0.3,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

H100_QWEN3_VL_30B_MOE = QwenVLConfig(
    model=ModelSize.QWEN3_VL_30B_A3B,
    model_family=ModelFamily.QWEN3_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=16384,
    enforce_eager=False,
    max_num_seqs=32,
    max_pixels=1572864,
    limit_images=8,
    limit_videos=4,
    video_pruning_rate=0.4,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)


# =============================================================================
# B200 CONFIGURATIONS (BLACKWELL - SM 10.0, 192GB HBM3e)
# =============================================================================
# Advantages: FP4 (future), 8 TB/s bandwidth, 2nd gen Transformer Engine
# Strategy: Maximum everything, next-gen features

B200_QWEN2_VL_72B = QwenVLConfig(
    model=ModelSize.QWEN2_VL_72B,
    model_family=ModelFamily.QWEN2_VL,
    quantization=QuantizationMethod.NONE,  # Full precision on single GPU!
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=32768,
    enforce_eager=False,
    max_num_seqs=32,
    max_pixels=4147200,  # 4K resolution
    limit_images=16,
    limit_videos=8,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

B200_QWEN3_VL_8B = QwenVLConfig(
    model=ModelSize.QWEN3_VL_8B,
    model_family=ModelFamily.QWEN3_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=131072,  # 128K context
    enforce_eager=False,
    max_num_seqs=128,
    max_pixels=4147200,  # 4K
    limit_images=32,
    limit_videos=16,
    video_pruning_rate=0.3,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)

B200_QWEN3_VL_30B_MOE = QwenVLConfig(
    model=ModelSize.QWEN3_VL_30B_A3B,
    model_family=ModelFamily.QWEN3_VL,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=65536,
    enforce_eager=False,
    max_num_seqs=64,
    max_pixels=4147200,
    limit_images=32,
    limit_videos=16,
    video_pruning_rate=0.3,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
)


# =============================================================================
# CONFIG REGISTRY
# =============================================================================

CONFIGS = {
    # T4 (Turing, 16GB)
    "t4_qwen2vl_2b": T4_QWEN2_VL_2B,
    "t4_qwen2vl_7b_4bit": T4_QWEN2_VL_7B_4BIT,
    "t4_qwen3vl_4b_4bit": T4_QWEN3_VL_4B_4BIT,
    
    # A100 (Ampere, 40/80GB)
    "a100_40gb_qwen2vl_7b": A100_40GB_QWEN2_VL_7B,
    "a100_80gb_qwen2vl_7b": A100_80GB_QWEN2_VL_7B,
    "a100_80gb_qwen3vl_8b": A100_80GB_QWEN3_VL_8B,
    "a100_80gb_qwen3vl_30b_moe": A100_80GB_QWEN3_VL_30B_MOE,
    
    # H100 (Hopper, 80GB HBM3)
    "h100_qwen2vl_7b": H100_QWEN2_VL_7B,
    "h100_qwen3vl_8b": H100_QWEN3_VL_8B,
    "h100_qwen3vl_8b_fp8": H100_QWEN3_VL_8B_FP8,
    "h100_qwen3vl_30b_moe": H100_QWEN3_VL_30B_MOE,
    
    # B200 (Blackwell, 192GB HBM3e)
    "b200_qwen2vl_72b": B200_QWEN2_VL_72B,
    "b200_qwen3vl_8b": B200_QWEN3_VL_8B,
    "b200_qwen3vl_30b_moe": B200_QWEN3_VL_30B_MOE,
}


def detect_gpu_and_get_config(
    prefer_qwen3: bool = True,
    prefer_larger_model: bool = False,
) -> QwenVLConfig:
    """
    Auto-detect GPU and return optimal Qwen VL configuration.
    
    Args:
        prefer_qwen3: If True, prefer Qwen3-VL over Qwen2-VL (recommended)
        prefer_larger_model: If True, choose larger model even if it requires quantization
    
    Returns:
        Optimal QwenVLConfig for the detected GPU
    """
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available")
        
        gpu_name = torch.cuda.get_device_name(0).lower()
        cc = torch.cuda.get_device_capability()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Detect architecture
        if cc[0] >= 10:  # Blackwell
            if prefer_qwen3:
                return B200_QWEN3_VL_8B if not prefer_larger_model else B200_QWEN3_VL_30B_MOE
            return B200_QWEN2_VL_72B
        
        elif cc[0] >= 9:  # Hopper (H100)
            if prefer_qwen3:
                return H100_QWEN3_VL_8B if not prefer_larger_model else H100_QWEN3_VL_30B_MOE
            return H100_QWEN2_VL_7B
        
        elif cc[0] >= 8:  # Ampere (A100) or Ada
            if vram_gb >= 70:
                if prefer_qwen3:
                    return A100_80GB_QWEN3_VL_8B if not prefer_larger_model else A100_80GB_QWEN3_VL_30B_MOE
                return A100_80GB_QWEN2_VL_7B
            else:
                return A100_40GB_QWEN2_VL_7B
        
        else:  # Turing (T4) or older
            if prefer_qwen3:
                return T4_QWEN3_VL_4B_4BIT
            return T4_QWEN2_VL_2B
            
    except Exception as e:
        print(f"GPU detection failed: {e}, defaulting to T4 config")
        return T4_QWEN2_VL_2B


def print_config_comparison():
    """Print configuration comparison table."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                    QWEN VL GPU CONFIGURATION COMPARISON                                  ║
╠═════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                         ║
║  Config                    │ Model          │ VRAM  │ Context │ Batch │ Resolution     ║
║  ─────────────────────────┼────────────────┼───────┼─────────┼───────┼───────────     ║
║  T4 (16GB)                                                                              ║
║    t4_qwen2vl_2b          │ Qwen2-VL-2B    │ ~5GB  │ 2K      │ 4     │ 720×720        ║
║    t4_qwen2vl_7b_4bit     │ Qwen2-VL-7B    │ ~5GB  │ 1K      │ 2     │ 500×500        ║
║    t4_qwen3vl_4b_4bit     │ Qwen3-VL-4B    │ ~4GB  │ 2K      │ 4     │ 720×720 +EVS   ║
║  ─────────────────────────┼────────────────┼───────┼─────────┼───────┼───────────     ║
║  A100 (80GB)                                                                            ║
║    a100_80gb_qwen2vl_7b   │ Qwen2-VL-7B    │ ~15GB │ 16K     │ 32    │ 1280×1280      ║
║    a100_80gb_qwen3vl_8b   │ Qwen3-VL-8B    │ ~17GB │ 16K     │ 32    │ 1280×1280 +EVS ║
║    a100_80gb_qwen3vl_30b  │ Qwen3-VL-30B   │ ~65GB │ 8K      │ 16    │ 1000×1000 +EVS ║
║  ─────────────────────────┼────────────────┼───────┼─────────┼───────┼───────────     ║
║  H100 (80GB)                                                                            ║
║    h100_qwen2vl_7b        │ Qwen2-VL-7B    │ ~15GB │ 32K     │ 64    │ 1920×1080      ║
║    h100_qwen3vl_8b        │ Qwen3-VL-8B    │ ~17GB │ 32K     │ 64    │ 1920×1080 +EVS ║
║    h100_qwen3vl_8b_fp8    │ Qwen3-VL-8B FP8│ ~9GB  │ 64K     │ 128   │ 1920×1080 +EVS ║
║  ─────────────────────────┼────────────────┼───────┼─────────┼───────┼───────────     ║
║  B200 (192GB)                                                                           ║
║    b200_qwen2vl_72b       │ Qwen2-VL-72B   │ ~150GB│ 32K     │ 32    │ 4K             ║
║    b200_qwen3vl_8b        │ Qwen3-VL-8B    │ ~17GB │ 128K    │ 128   │ 4K +EVS        ║
║    b200_qwen3vl_30b_moe   │ Qwen3-VL-30B   │ ~65GB │ 64K     │ 64    │ 4K +EVS        ║
║                                                                                         ║
║  EVS = Efficient Video Sampling (Qwen3-VL only, reduces video tokens by 30-70%)        ║
║                                                                                         ║
╚═════════════════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_config_comparison()
    
    print("\nAuto-detected configuration:")
    config = detect_gpu_and_get_config()
    print(f"  Model: {config.model.value}")
    print(f"  Quantization: {config.quantization.value}")
    print(f"  Max Context: {config.max_model_len}")
    print(f"  Max Batch: {config.max_num_seqs}")
    print(f"  Max Pixels: {config.max_pixels}")
    if config.video_pruning_rate:
        print(f"  EVS Rate: {config.video_pruning_rate}")
    
    print("\n" + config.get_architecture_info())

