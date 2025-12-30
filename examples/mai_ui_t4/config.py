# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
T4-Optimized Configuration for MAI-UI (Qwen2-VL based GUI Agent)

This module provides optimized configurations for running MAI-UI models
on NVIDIA T4 GPUs (16GB VRAM, Turing architecture, SM 7.5).

Architecture Considerations:
- T4 supports FP16 Tensor Cores (65 TFLOPS) but NOT FlashAttention 2
- T4 uses TORCH_SDPA for attention (not FlashAttn2 which requires SM 8.0+)
- T4 supports INT8/INT4 quantization via BitsAndBytes, AWQ, GPTQ
- Memory bandwidth: 320 GB/s (decode phase is memory-bound)

Memory Budget (16GB):
- MAI-UI-2B FP16: ~10GB used, ~6GB headroom ✅
- MAI-UI-8B 4-bit: ~13.5GB used, ~2.5GB headroom ⚠️
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm import EngineArgs


class ModelVariant(str, Enum):
    """Supported MAI-UI model variants."""
    
    MAI_UI_2B = "2b"
    MAI_UI_8B = "8b"


class QuantizationMethod(str, Enum):
    """Supported quantization methods for T4."""
    
    NONE = "none"  # FP16 only
    BITSANDBYTES = "bitsandbytes"  # 4-bit, on-the-fly
    AWQ = "awq"  # 4-bit, requires pre-quantized model
    GPTQ = "gptq"  # 4-bit, requires pre-quantized model


@dataclass
class T4OptimizationConfig:
    """
    T4-specific optimization parameters.
    
    These parameters are tuned for NVIDIA T4 (16GB VRAM, SM 7.5).
    Adjust based on your specific workload requirements.
    """
    
    # Model selection
    model_variant: ModelVariant = ModelVariant.MAI_UI_2B
    quantization: QuantizationMethod = QuantizationMethod.NONE
    
    # Memory optimization
    # Why: T4 has only 16GB VRAM - every MB counts
    gpu_memory_utilization: float = 0.90  # Reserve 10% for safety
    max_model_len: int = 2048  # ✅ Reduced for T4 (saves KV cache memory)
    enforce_eager: bool = True  # Disable CUDA graphs to save ~500MB
    
    # Vision optimization
    # Why: Fewer vision tokens = less memory + faster processing
    # Formula: tokens ≈ (max_pixels / 14²) / 4 (with spatial_merge_size=2)
    max_pixels: int = 512000  # ~650 vision tokens vs ~1000 default
    min_pixels: int = 784  # 28×28 minimum
    
    # Batching optimization
    # Why: T4 is memory-limited; too many concurrent requests = OOM
    max_num_seqs: int = 4  # Max concurrent requests
    limit_mm_per_prompt: int = 1  # One image per request
    
    # Performance tuning
    dtype: str = "half"  # FP16 for Tensor Core acceleration
    
    def to_engine_args(self) -> dict[str, Any]:
        """Convert to vLLM EngineArgs compatible dict."""
        
        # Select model path based on variant
        # ✅ CORRECT: Using official Tongyi-MAI model paths
        model_paths = {
            ModelVariant.MAI_UI_2B: "Tongyi-MAI/MAI-UI-2B",
            ModelVariant.MAI_UI_8B: "Tongyi-MAI/MAI-UI-8B",
        }
        
        args = {
            "model": model_paths[self.model_variant],
            "trust_remote_code": True,  # ✅ REQUIRED for Qwen2-VL based models
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "max_num_seqs": self.max_num_seqs,
            "limit_mm_per_prompt": {"image": self.limit_mm_per_prompt},
            "mm_processor_kwargs": {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            },
        }
        
        # Add quantization if specified
        if self.quantization != QuantizationMethod.NONE:
            args["quantization"] = self.quantization.value
            if self.quantization == QuantizationMethod.BITSANDBYTES:
                args["load_format"] = "bitsandbytes"
        
        return args


# Pre-configured profiles for common use cases
T4_PROFILES: dict[str, T4OptimizationConfig] = {
    # Best for latency-sensitive single requests
    "2b_latency": T4OptimizationConfig(
        model_variant=ModelVariant.MAI_UI_2B,
        quantization=QuantizationMethod.NONE,
        max_num_seqs=1,
        max_pixels=256000,  # Smaller images for speed
        max_model_len=2048,
    ),
    
    # Best for throughput with 2B model
    "2b_throughput": T4OptimizationConfig(
        model_variant=ModelVariant.MAI_UI_2B,
        quantization=QuantizationMethod.NONE,
        max_num_seqs=4,
        max_pixels=512000,
        max_model_len=4096,
    ),
    
    # Best quality with 8B model (requires quantization)
    "8b_quality": T4OptimizationConfig(
        model_variant=ModelVariant.MAI_UI_8B,
        quantization=QuantizationMethod.BITSANDBYTES,
        max_num_seqs=2,
        max_pixels=256000,  # Smaller for memory
        max_model_len=2048,  # Reduced for memory
        gpu_memory_utilization=0.95,  # More aggressive
    ),
    
    # Balanced 2B configuration (recommended default)
    "2b_balanced": T4OptimizationConfig(
        model_variant=ModelVariant.MAI_UI_2B,
        quantization=QuantizationMethod.NONE,
        max_num_seqs=2,
        max_pixels=512000,
        max_model_len=4096,
    ),
}


def get_t4_engine_args(
    profile: str = "2b_balanced",
    custom_config: T4OptimizationConfig | None = None,
) -> EngineArgs:
    """
    Get T4-optimized EngineArgs for MAI-UI.
    
    Args:
        profile: Pre-configured profile name. Options:
            - "2b_latency": Fastest single-request latency
            - "2b_throughput": Best throughput for batching
            - "8b_quality": Best quality (requires 4-bit quantization)
            - "2b_balanced": Recommended default
        custom_config: Override with custom T4OptimizationConfig
    
    Returns:
        EngineArgs configured for T4 optimization
    
    Example:
        >>> args = get_t4_engine_args("2b_balanced")
        >>> llm = LLM(**asdict(args))
    """
    if custom_config is not None:
        config = custom_config
    elif profile in T4_PROFILES:
        config = T4_PROFILES[profile]
    else:
        raise ValueError(
            f"Unknown profile: {profile}. "
            f"Available: {list(T4_PROFILES.keys())}"
        )
    
    return EngineArgs(**config.to_engine_args())


def get_mai_ui_prompt(instruction: str, modality: str = "image") -> str:
    """
    Format instruction as MAI-UI prompt.
    
    MAI-UI uses Qwen2-VL's ChatML format with vision tokens.
    
    Args:
        instruction: The GUI action instruction (e.g., "Click the submit button")
        modality: "image" or "video"
    
    Returns:
        Formatted prompt string
    """
    placeholder = "<|image_pad|>" if modality == "image" else "<|video_pad|>"
    
    # MAI-UI system prompt (from the paper)
    system_prompt = (
        "You are a GUI agent. Based on the screenshot and instruction, "
        "output the action to perform. Format: pyautogui.ACTION(parameters)"
    )
    
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def validate_t4_compatibility() -> dict[str, bool]:
    """
    Check T4 compatibility and available optimizations.
    
    Returns:
        Dict with compatibility flags for various features
    """
    import torch
    
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    capability = (props.major, props.minor)
    
    is_t4 = "T4" in props.name
    is_turing = capability[0] == 7 and capability[1] >= 5
    
    return {
        "cuda_available": True,
        "device_name": props.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_gb": props.total_memory / (1024**3),
        "is_t4": is_t4,
        "is_turing": is_turing,
        # Feature support based on compute capability
        "supports_fp16": True,
        "supports_bf16": capability[0] >= 8,  # Ampere+
        "supports_fp8": capability[0] >= 9,  # Hopper+
        "supports_flash_attn2": capability[0] >= 8,  # Ampere+
        "supports_int8_tensor_cores": is_turing or capability[0] >= 8,
        "supports_int4_bnb": True,  # Software-based
        "supports_awq": True,  # Software-based
        "attention_backend": "TORCH_SDPA" if is_turing else "FLASH_ATTN",
    }


if __name__ == "__main__":
    # Print T4 compatibility info
    print("=" * 60)
    print("T4 COMPATIBILITY CHECK")
    print("=" * 60)
    
    compat = validate_t4_compatibility()
    for key, value in compat.items():
        status = "✅" if value else "❌" if isinstance(value, bool) else ""
        print(f"  {key}: {status} {value}")
    
    print("\n" + "=" * 60)
    print("AVAILABLE PROFILES")
    print("=" * 60)
    
    for name, config in T4_PROFILES.items():
        print(f"\n  [{name}]")
        print(f"    Model: {config.model_variant.value}")
        print(f"    Quantization: {config.quantization.value}")
        print(f"    Max Sequences: {config.max_num_seqs}")
        print(f"    Max Pixels: {config.max_pixels:,}")
        print(f"    Max Context: {config.max_model_len:,}")

