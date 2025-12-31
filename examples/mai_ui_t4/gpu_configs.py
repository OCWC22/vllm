# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GPU-Optimized Configurations for MAI-UI (Qwen2-VL based GUI Agent)

This module provides optimized configurations for running MAI-UI models
across different NVIDIA GPU architectures:

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        GPU ARCHITECTURE COMPARISON                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  GPU        │  Arch      │  SM    │  VRAM    │  Bandwidth │  Key Features       │
│  ───────────┼────────────┼────────┼──────────┼────────────┼──────────────────── │
│  T4         │  Turing    │  7.5   │  16 GB   │  320 GB/s  │  FP16, INT8         │
│  H100 SXM   │  Hopper    │  9.0   │  80 GB   │  3.35 TB/s │  FP8, FlashAttn3    │
│  H100 PCIe  │  Hopper    │  9.0   │  80 GB   │  2.0 TB/s  │  FP8, FlashAttn3    │
│  B200       │  Blackwell │  10.0  │  192 GB  │  8.0 TB/s  │  FP4, Next-gen      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

Performance Expectations:
- T4:   ~1000ms latency, ~20 tok/s, memory-constrained
- H100: ~200ms latency, ~150 tok/s, compute-optimized
- B200: ~100ms latency, ~300 tok/s, next-gen performance
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GPUArchitecture(str, Enum):
    """Supported GPU architectures."""
    TURING = "turing"      # T4, RTX 20xx (SM 7.5)
    AMPERE = "ampere"      # A100, RTX 30xx (SM 8.0)
    ADA = "ada"            # L4, L40, RTX 40xx (SM 8.9)
    HOPPER = "hopper"      # H100, H200 (SM 9.0)
    BLACKWELL = "blackwell"  # B100, B200 (SM 10.0)


class ModelVariant(str, Enum):
    """MAI-UI model variants."""
    MAI_UI_2B = "2b"
    MAI_UI_8B = "8b"


class QuantizationMethod(str, Enum):
    """Quantization methods by GPU support."""
    NONE = "none"           # FP16/BF16, all GPUs
    BITSANDBYTES = "bitsandbytes"  # 4-bit, all GPUs (software)
    AWQ = "awq"             # 4-bit, Ampere+ preferred
    GPTQ = "gptq"           # 4-bit, all GPUs
    FP8 = "fp8"             # FP8, Ada/Hopper+ only (SM 8.9+)
    FP4 = "fp4"             # FP4, Blackwell only (SM 10.0+)


@dataclass
class GPUOptimizationConfig:
    """
    GPU-specific optimization parameters.
    
    Each configuration is tuned for the specific GPU architecture's
    strengths and limitations.
    """
    
    # Model configuration
    model_variant: ModelVariant = ModelVariant.MAI_UI_2B
    quantization: QuantizationMethod = QuantizationMethod.NONE
    
    # Precision: "half" (FP16), "bfloat16", or "auto"
    dtype: str = "half"
    
    # Memory settings
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    enforce_eager: bool = False  # Enable CUDA graphs by default for modern GPUs
    
    # Batching
    max_num_seqs: int = 8
    
    # Vision settings
    min_pixels: int = 28 * 28  # 784
    max_pixels: int = 1003520  # Default Qwen2-VL max
    limit_mm_per_prompt: int = 4  # Images per request
    
    # Advanced settings
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    
    # Architecture-specific
    attention_backend: str | None = None  # Auto-detect
    kv_cache_dtype: str = "auto"
    
    def to_engine_args(self) -> dict[str, Any]:
        """Convert to vLLM engine arguments."""
        
        model_paths = {
            ModelVariant.MAI_UI_2B: "Tongyi-MAI/MAI-UI-2B",
            ModelVariant.MAI_UI_8B: "Tongyi-MAI/MAI-UI-8B",
        }
        
        args = {
            "model": model_paths[self.model_variant],
            "trust_remote_code": True,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "max_num_seqs": self.max_num_seqs,
            "limit_mm_per_prompt": {"image": self.limit_mm_per_prompt, "video": 0},
            "mm_processor_kwargs": {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            },
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
        }
        
        # Add quantization if specified
        if self.quantization != QuantizationMethod.NONE:
            args["quantization"] = self.quantization.value
            if self.quantization == QuantizationMethod.BITSANDBYTES:
                args["load_format"] = "bitsandbytes"
        
        # Add KV cache dtype if not auto
        if self.kv_cache_dtype != "auto":
            args["kv_cache_dtype"] = self.kv_cache_dtype
        
        return args


# =============================================================================
# T4 CONFIGURATION (TURING - SM 7.5)
# =============================================================================
# Constraints: 16GB VRAM, no FlashAttn2, no BF16, no FP8
# Strategy: Memory-conservative, FP16 Tensor Cores

T4_CONFIG = GPUOptimizationConfig(
    model_variant=ModelVariant.MAI_UI_2B,
    quantization=QuantizationMethod.NONE,
    dtype="half",  # FP16 only (no BF16 support)
    gpu_memory_utilization=0.90,
    max_model_len=2048,  # Reduced for memory
    enforce_eager=True,  # Disable CUDA graphs (saves ~500MB)
    max_num_seqs=4,  # Limited concurrent requests
    min_pixels=784,
    max_pixels=512000,  # Aggressive reduction (~720x720)
    limit_mm_per_prompt=1,  # Single image only
    enable_prefix_caching=False,  # Memory constraint
    enable_chunked_prefill=False,  # Simpler execution
    attention_backend="TORCH_SDPA",  # Only option for SM 7.5
    kv_cache_dtype="auto",
)

T4_8B_CONFIG = GPUOptimizationConfig(
    model_variant=ModelVariant.MAI_UI_8B,
    quantization=QuantizationMethod.BITSANDBYTES,  # 4-bit required
    dtype="half",
    gpu_memory_utilization=0.92,
    max_model_len=1024,  # Very limited
    enforce_eager=True,
    max_num_seqs=2,
    min_pixels=784,
    max_pixels=256000,  # Very aggressive
    limit_mm_per_prompt=1,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
    attention_backend="TORCH_SDPA",
    kv_cache_dtype="auto",
)


# =============================================================================
# H100 CONFIGURATION (HOPPER - SM 9.0)
# =============================================================================
# Advantages: 80GB HBM3, FlashAttention 2/3, FP8, 3.35 TB/s bandwidth
# Strategy: Maximum performance, leverage all Hopper features

"""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        H100 ARCHITECTURE (HOPPER)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MEMORY: 80 GB HBM3 @ 3.35 TB/s (10x T4 bandwidth!)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ████████████████████████████████████████████████████████████████████   │   │
│  │  ←──────────────────────── 80 GB Total ────────────────────────────→   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  COMPUTE: 132 SMs × 128 CUDA Cores = 16,896 CUDA Cores                          │
│           132 SMs × 4 Tensor Cores = 528 Tensor Cores (4th Gen)                 │
│                                                                                  │
│  PERFORMANCE:                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │  FP16: 1,979 TFLOPS  │  FP8: 3,958 TFLOPS  │  INT8: 3,958 TOPS        │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  H100 EXCLUSIVE FEATURES:                                                       │
│  ✅ FlashAttention 2/3 (2-4x faster attention)                                  │
│  ✅ FP8 Tensor Cores (2x throughput vs FP16)                                    │
│  ✅ Transformer Engine (auto FP8 training)                                       │
│  ✅ HBM3 Memory (10x bandwidth vs T4)                                           │
│  ✅ NVLink 4.0 (multi-GPU at 900 GB/s)                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
"""

H100_CONFIG = GPUOptimizationConfig(
    model_variant=ModelVariant.MAI_UI_2B,
    quantization=QuantizationMethod.NONE,  # FP16 is fast enough
    dtype="bfloat16",  # BF16 for better numerical stability
    gpu_memory_utilization=0.95,  # More aggressive with 80GB
    max_model_len=32768,  # Full context length
    enforce_eager=False,  # Enable CUDA graphs (plenty of memory)
    max_num_seqs=64,  # High concurrent throughput
    min_pixels=784,
    max_pixels=2073600,  # Full resolution (1920x1080)
    limit_mm_per_prompt=8,  # Multiple images supported
    enable_prefix_caching=True,  # Leverage HBM3 bandwidth
    enable_chunked_prefill=True,  # Better prefill efficiency
    attention_backend=None,  # Auto-select FlashAttention
    kv_cache_dtype="auto",  # Could use FP8 for more capacity
)

H100_8B_CONFIG = GPUOptimizationConfig(
    model_variant=ModelVariant.MAI_UI_8B,
    quantization=QuantizationMethod.NONE,  # Full precision, 80GB is enough
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=16384,
    enforce_eager=False,
    max_num_seqs=32,
    min_pixels=784,
    max_pixels=1572864,  # ~1280x1280
    limit_mm_per_prompt=4,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    attention_backend=None,
    kv_cache_dtype="auto",
)

H100_8B_FP8_CONFIG = GPUOptimizationConfig(
    model_variant=ModelVariant.MAI_UI_8B,
    quantization=QuantizationMethod.FP8,  # FP8 for 2x throughput
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=32768,  # Full context with FP8
    enforce_eager=False,
    max_num_seqs=64,  # Double throughput
    min_pixels=784,
    max_pixels=2073600,
    limit_mm_per_prompt=8,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    attention_backend=None,
    kv_cache_dtype="fp8",  # FP8 KV cache for 2x capacity
)


# =============================================================================
# B200 CONFIGURATION (BLACKWELL - SM 10.0)
# =============================================================================
# Advantages: 192GB HBM3e, FP4, 8 TB/s bandwidth, 2nd gen Transformer Engine
# Strategy: Maximum everything, next-gen features

"""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        B200 ARCHITECTURE (BLACKWELL)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MEMORY: 192 GB HBM3e @ 8.0 TB/s (25x T4 bandwidth!)                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ████████████████████████████████████████████████████████████████████   │   │
│  │  ←────────────────────────── 192 GB Total ──────────────────────────→   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  COMPUTE: Next-gen architecture                                                 │
│           5th Generation Tensor Cores                                           │
│           2nd Generation Transformer Engine                                     │
│                                                                                  │
│  PERFORMANCE (Expected):                                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │  FP16: ~4,000 TFLOPS  │  FP8: ~8,000 TFLOPS  │  FP4: ~16,000 TFLOPS   │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  B200 EXCLUSIVE FEATURES:                                                       │
│  ✅ FP4 Tensor Cores (4x throughput vs FP16)                                    │
│  ✅ 2nd Gen Transformer Engine                                                   │
│  ✅ HBM3e Memory (24x bandwidth vs T4)                                          │
│  ✅ NVLink 5.0 (1.8 TB/s multi-GPU)                                             │
│  ✅ Enhanced sparsity support                                                    │
│  ✅ Decompression Engine (direct compressed data loading)                       │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
"""

B200_CONFIG = GPUOptimizationConfig(
    model_variant=ModelVariant.MAI_UI_2B,
    quantization=QuantizationMethod.NONE,
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=131072,  # 128K context possible
    enforce_eager=False,
    max_num_seqs=128,  # Massive throughput
    min_pixels=784,
    max_pixels=4147200,  # 4K resolution (3840x1080)
    limit_mm_per_prompt=16,  # Many images
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    attention_backend=None,  # Latest FlashAttention
    kv_cache_dtype="auto",
)

B200_8B_CONFIG = GPUOptimizationConfig(
    model_variant=ModelVariant.MAI_UI_8B,
    quantization=QuantizationMethod.NONE,  # Full precision, 192GB is massive
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=65536,  # 64K context
    enforce_eager=False,
    max_num_seqs=64,
    min_pixels=784,
    max_pixels=4147200,  # 4K resolution
    limit_mm_per_prompt=16,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    attention_backend=None,
    kv_cache_dtype="auto",
)

B200_8B_FP4_CONFIG = GPUOptimizationConfig(
    model_variant=ModelVariant.MAI_UI_8B,
    quantization=QuantizationMethod.FP4,  # FP4 for 4x throughput (when available)
    dtype="bfloat16",
    gpu_memory_utilization=0.95,
    max_model_len=131072,  # 128K context with FP4
    enforce_eager=False,
    max_num_seqs=256,  # Massive batch size
    min_pixels=784,
    max_pixels=4147200,
    limit_mm_per_prompt=32,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
    attention_backend=None,
    kv_cache_dtype="fp8",  # FP8 KV (FP4 KV when available)
)


# =============================================================================
# GPU CONFIGURATION REGISTRY
# =============================================================================

GPU_CONFIGS = {
    # T4 (Turing) - Memory constrained
    "t4_2b": T4_CONFIG,
    "t4_8b": T4_8B_CONFIG,
    
    # H100 (Hopper) - High performance
    "h100_2b": H100_CONFIG,
    "h100_8b": H100_8B_CONFIG,
    "h100_8b_fp8": H100_8B_FP8_CONFIG,
    
    # B200 (Blackwell) - Next-gen
    "b200_2b": B200_CONFIG,
    "b200_8b": B200_8B_CONFIG,
    "b200_8b_fp4": B200_8B_FP4_CONFIG,
}


def get_config_for_gpu(gpu_name: str) -> GPUOptimizationConfig:
    """
    Auto-detect GPU and return optimal configuration.
    
    Args:
        gpu_name: GPU name string (e.g., "NVIDIA H100")
    
    Returns:
        Optimal GPUOptimizationConfig for the detected GPU
    """
    gpu_name_lower = gpu_name.lower()
    
    if "b200" in gpu_name_lower or "b100" in gpu_name_lower:
        return B200_CONFIG
    elif "h100" in gpu_name_lower or "h200" in gpu_name_lower:
        return H100_CONFIG
    elif "l40" in gpu_name_lower or "l4" in gpu_name_lower:
        # Ada Lovelace - similar to H100 but less memory
        return GPUOptimizationConfig(
            model_variant=ModelVariant.MAI_UI_2B,
            dtype="bfloat16",
            max_model_len=8192,
            max_num_seqs=16,
            max_pixels=1003520,
        )
    elif "a100" in gpu_name_lower:
        # Ampere A100
        return GPUOptimizationConfig(
            model_variant=ModelVariant.MAI_UI_2B,
            dtype="bfloat16",
            max_model_len=16384,
            max_num_seqs=32,
            max_pixels=1572864,
            enable_prefix_caching=True,
        )
    elif "t4" in gpu_name_lower or "turing" in gpu_name_lower:
        return T4_CONFIG
    else:
        # Default to T4-like conservative settings
        return T4_CONFIG


def print_gpu_comparison():
    """Print GPU comparison table."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════╗
║                        GPU PERFORMANCE COMPARISON                               ║
╠═════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  Metric              │  T4 (Turing)  │  H100 (Hopper)  │  B200 (Blackwell)     ║
║  ────────────────────┼───────────────┼─────────────────┼───────────────────    ║
║  VRAM                │  16 GB        │  80 GB          │  192 GB               ║
║  Bandwidth           │  320 GB/s     │  3,350 GB/s     │  8,000 GB/s           ║
║  FP16 TFLOPS         │  65           │  1,979          │  ~4,000               ║
║  FP8 TFLOPS          │  ❌           │  3,958          │  ~8,000               ║
║  FP4 TFLOPS          │  ❌           │  ❌             │  ~16,000              ║
║  ────────────────────┼───────────────┼─────────────────┼───────────────────    ║
║  FlashAttention      │  ❌ SDPA      │  ✅ FA2/FA3     │  ✅ FA4 (expected)    ║
║  BF16                │  ❌           │  ✅             │  ✅                   ║
║  FP8 Quant           │  ❌           │  ✅             │  ✅                   ║
║  FP4 Quant           │  ❌           │  ❌             │  ✅                   ║
║  ────────────────────┼───────────────┼─────────────────┼───────────────────    ║
║  MAI-UI-2B Latency   │  ~1000ms      │  ~200ms         │  ~100ms               ║
║  MAI-UI-2B Tok/s     │  ~20          │  ~150           │  ~300                 ║
║  Max Concurrent      │  4            │  64             │  128                  ║
║  Max Context         │  2K           │  32K            │  128K                 ║
║                                                                                 ║
╚═════════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_gpu_comparison()
    
    print("\n" + "=" * 70)
    print("Available Configurations:")
    print("=" * 70)
    for name, config in GPU_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Model: {config.model_variant.value}")
        print(f"  Dtype: {config.dtype}")
        print(f"  Quant: {config.quantization.value}")
        print(f"  Max Len: {config.max_model_len}")
        print(f"  Max Seqs: {config.max_num_seqs}")
        print(f"  Max Pixels: {config.max_pixels}")




