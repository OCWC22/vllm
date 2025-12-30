# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen2-VL vs Qwen3-VL: Unified Notebook Code for All GPUs

This module provides notebook-ready code for running Qwen Vision-Language models
optimized for T4, A100, H100, and B200 GPUs.

Usage in Colab/Jupyter:
    # Cell 1: Setup
    from qwen_vl_notebook import setup_environment, create_engine
    
    # Cell 2: Create engine
    llm = create_engine()  # Auto-detects GPU
    
    # Cell 3: Run inference
    from qwen_vl_notebook import run_image_inference
    result = run_image_inference(llm, image, "Describe this image")
"""

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# =============================================================================
# CELL 1: ENVIRONMENT SETUP
# =============================================================================

SETUP_CODE = '''
# ============================================================================
# CELL 1: GPU Detection & Environment Setup
# ============================================================================
import subprocess
import sys

print("=" * 80)
print("🔍 GPU DETECTION & ENVIRONMENT SETUP")
print("=" * 80)

# Check GPU
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv"],
        capture_output=True, text=True
    )
    print(result.stdout)
except Exception as e:
    print(f"nvidia-smi failed: {e}")

import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cc = torch.cuda.get_device_capability()
    
    print(f"✅ GPU: {gpu_name}")
    print(f"✅ Memory: {gpu_memory:.1f} GB")
    print(f"✅ Compute Capability: SM {cc[0]}.{cc[1]}")
    
    # Determine architecture
    if cc[0] >= 10:
        arch = "Blackwell (B200)"
        features = "FP4, 8 TB/s, 192GB"
    elif cc[0] >= 9:
        arch = "Hopper (H100)"
        features = "FP8, FlashAttn3, 3.35 TB/s, 80GB"
    elif cc[0] >= 8:
        arch = "Ampere (A100)"
        features = "BF16, FlashAttn2"
    else:
        arch = "Turing (T4)"
        features = "FP16, SDPA"
    
    print(f"✅ Architecture: {arch}")
    print(f"✅ Features: {features}")
else:
    print("❌ No GPU available!")
    sys.exit(1)

print("\\n📦 Installing dependencies...")
%pip install -q vllm>=0.6.0 pillow requests jinja2 numpy
print("✅ Dependencies installed!")
'''


def setup_environment():
    """Run environment setup and return GPU info."""
    import torch
    
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    cc = torch.cuda.get_device_capability()
    
    info = {
        "name": gpu_name,
        "memory_gb": gpu_memory,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "sm_version": cc[0],
    }
    
    # Determine architecture
    if cc[0] >= 10:
        info["architecture"] = "Blackwell"
        info["recommended_model"] = "Qwen3-VL-8B"
    elif cc[0] >= 9:
        info["architecture"] = "Hopper"
        info["recommended_model"] = "Qwen3-VL-8B"
    elif cc[0] >= 8:
        info["architecture"] = "Ampere"
        info["recommended_model"] = "Qwen3-VL-8B" if gpu_memory >= 70 else "Qwen2-VL-7B"
    else:
        info["architecture"] = "Turing"
        info["recommended_model"] = "Qwen2-VL-2B or Qwen3-VL-4B (4-bit)"
    
    return info


# =============================================================================
# CELL 2: ENGINE CREATION
# =============================================================================

def get_engine_config(
    gpu_info: dict | None = None,
    model_preference: str = "auto",  # "qwen2", "qwen3", or "auto"
    force_quantization: str | None = None,
) -> dict[str, Any]:
    """
    Get optimal vLLM engine configuration based on GPU.
    
    Args:
        gpu_info: GPU info dict from setup_environment()
        model_preference: "qwen2", "qwen3", or "auto" (prefers qwen3)
        force_quantization: Override quantization ("bitsandbytes", "awq", None)
    
    Returns:
        Configuration dict for vLLM LLM()
    """
    if gpu_info is None:
        gpu_info = setup_environment()
    
    sm = gpu_info["sm_version"]
    vram = gpu_info["memory_gb"]
    
    # Prefer Qwen3-VL unless explicitly asked for Qwen2
    prefer_qwen3 = model_preference != "qwen2"
    
    # =========================================================================
    # T4 / TURING (SM 7.5, 16GB)
    # =========================================================================
    if sm < 8:
        if prefer_qwen3:
            # Qwen3-VL-4B with 4-bit quantization
            config = {
                "model": "Qwen/Qwen3-VL-4B-Instruct",
                "trust_remote_code": True,
                "dtype": "half",  # No BF16 on Turing
                "quantization": force_quantization or "bitsandbytes",
                "load_format": "bitsandbytes" if not force_quantization else None,
                "gpu_memory_utilization": 0.92,
                "max_model_len": 2048,
                "enforce_eager": True,  # Disable CUDA graphs (saves memory)
                "max_num_seqs": 4,
                "limit_mm_per_prompt": {"image": 1, "video": 1},
                "mm_processor_kwargs": {
                    "min_pixels": 784,
                    "max_pixels": 512000,  # ~720×720
                    "video_pruning_rate": 0.5,  # EVS: Keep 50%
                },
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
            }
        else:
            # Qwen2-VL-2B full precision
            config = {
                "model": "Qwen/Qwen2-VL-2B-Instruct",
                "trust_remote_code": True,
                "dtype": "half",
                "gpu_memory_utilization": 0.90,
                "max_model_len": 2048,
                "enforce_eager": True,
                "max_num_seqs": 4,
                "limit_mm_per_prompt": {"image": 1, "video": 0},
                "mm_processor_kwargs": {
                    "min_pixels": 784,
                    "max_pixels": 512000,
                },
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
            }
    
    # =========================================================================
    # A100 / AMPERE (SM 8.0, 40-80GB)
    # =========================================================================
    elif sm < 9:
        if vram >= 70:  # A100-80GB
            if prefer_qwen3:
                config = {
                    "model": "Qwen/Qwen3-VL-8B-Instruct",
                    "trust_remote_code": True,
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.95,
                    "max_model_len": 16384,
                    "enforce_eager": False,
                    "max_num_seqs": 32,
                    "limit_mm_per_prompt": {"image": 8, "video": 4},
                    "mm_processor_kwargs": {
                        "min_pixels": 784,
                        "max_pixels": 1572864,  # ~1280×1280
                        "video_pruning_rate": 0.3,
                    },
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                }
            else:
                config = {
                    "model": "Qwen/Qwen2-VL-7B-Instruct",
                    "trust_remote_code": True,
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.95,
                    "max_model_len": 16384,
                    "enforce_eager": False,
                    "max_num_seqs": 32,
                    "limit_mm_per_prompt": {"image": 8, "video": 2},
                    "mm_processor_kwargs": {
                        "min_pixels": 784,
                        "max_pixels": 1572864,
                    },
                    "enable_prefix_caching": True,
                    "enable_chunked_prefill": True,
                }
        else:  # A100-40GB or similar
            config = {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.95,
                "max_model_len": 8192,
                "enforce_eager": False,
                "max_num_seqs": 16,
                "limit_mm_per_prompt": {"image": 4, "video": 1},
                "mm_processor_kwargs": {
                    "min_pixels": 784,
                    "max_pixels": 1003520,
                },
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
            }
    
    # =========================================================================
    # H100 / HOPPER (SM 9.0, 80GB HBM3)
    # =========================================================================
    elif sm < 10:
        if prefer_qwen3:
            config = {
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.95,
                "max_model_len": 32768,
                "enforce_eager": False,
                "max_num_seqs": 64,
                "limit_mm_per_prompt": {"image": 16, "video": 8},
                "mm_processor_kwargs": {
                    "min_pixels": 784,
                    "max_pixels": 2073600,  # 1920×1080
                    "video_pruning_rate": 0.3,
                },
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
            }
        else:
            config = {
                "model": "Qwen/Qwen2-VL-7B-Instruct",
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.95,
                "max_model_len": 32768,
                "enforce_eager": False,
                "max_num_seqs": 64,
                "limit_mm_per_prompt": {"image": 16, "video": 4},
                "mm_processor_kwargs": {
                    "min_pixels": 784,
                    "max_pixels": 2073600,
                },
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
            }
    
    # =========================================================================
    # B200 / BLACKWELL (SM 10.0, 192GB HBM3e)
    # =========================================================================
    else:
        if prefer_qwen3:
            config = {
                "model": "Qwen/Qwen3-VL-8B-Instruct",
                "trust_remote_code": True,
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.95,
                "max_model_len": 131072,  # 128K context
                "enforce_eager": False,
                "max_num_seqs": 128,
                "limit_mm_per_prompt": {"image": 32, "video": 16},
                "mm_processor_kwargs": {
                    "min_pixels": 784,
                    "max_pixels": 4147200,  # 4K
                    "video_pruning_rate": 0.3,
                },
                "enable_prefix_caching": True,
                "enable_chunked_prefill": True,
            }
        else:
            # Can run 72B on single B200!
            config = {
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
    
    # Clean up None values
    config = {k: v for k, v in config.items() if v is not None}
    
    return config


def create_engine(
    model_preference: str = "auto",
    force_quantization: str | None = None,
    **override_kwargs,
):
    """
    Create vLLM engine with optimal GPU configuration.
    
    Args:
        model_preference: "qwen2", "qwen3", or "auto"
        force_quantization: Override quantization method
        **override_kwargs: Override any config parameter
    
    Returns:
        vLLM LLM instance
    """
    from vllm import LLM
    import torch
    
    gpu_info = setup_environment()
    config = get_engine_config(gpu_info, model_preference, force_quantization)
    
    # Apply overrides
    config.update(override_kwargs)
    
    print("\n" + "=" * 80)
    print("⚡ CREATING vLLM ENGINE")
    print("=" * 80)
    print(f"Model: {config['model']}")
    print(f"GPU: {gpu_info['name']} ({gpu_info['architecture']})")
    print(f"Config:")
    for k, v in config.items():
        if k != "model":
            print(f"  {k}: {v}")
    print("=" * 80)
    
    start = time.time()
    llm = LLM(**config)
    init_time = time.time() - start
    
    print(f"\n✅ Engine initialized in {init_time:.1f}s")
    
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    print(f"📊 GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    return llm


# =============================================================================
# CELL 3: PROMPT FORMATTING
# =============================================================================

# Qwen2-VL prompt format
QWEN2_VL_SYSTEM_PROMPT = """You are a helpful visual assistant. Analyze images and videos carefully and provide accurate, detailed responses."""

def format_qwen2_vl_prompt(instruction: str, system_prompt: str = None) -> str:
    """Format prompt for Qwen2-VL models."""
    if system_prompt is None:
        system_prompt = QWEN2_VL_SYSTEM_PROMPT
    
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# Qwen3-VL prompt format (same structure, but model handles timestamps internally for video)
QWEN3_VL_SYSTEM_PROMPT = """You are a helpful visual assistant with advanced image and video understanding capabilities. Analyze visual content carefully and provide accurate, detailed responses."""

def format_qwen3_vl_prompt(instruction: str, system_prompt: str = None) -> str:
    """Format prompt for Qwen3-VL models."""
    if system_prompt is None:
        system_prompt = QWEN3_VL_SYSTEM_PROMPT
    
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_prompt(instruction: str, model_name: str, system_prompt: str = None) -> str:
    """
    Auto-detect model family and format prompt appropriately.
    
    Args:
        instruction: User instruction text
        model_name: Model name (used to detect Qwen2 vs Qwen3)
        system_prompt: Optional custom system prompt
    
    Returns:
        Formatted prompt string
    """
    if "qwen3" in model_name.lower():
        return format_qwen3_vl_prompt(instruction, system_prompt)
    return format_qwen2_vl_prompt(instruction, system_prompt)


# =============================================================================
# CELL 4: INFERENCE FUNCTIONS
# =============================================================================

@dataclass
class InferenceResult:
    """Result from image/video inference."""
    text: str
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    
    def __repr__(self):
        return (
            f"InferenceResult(\n"
            f"  text={self.text[:100]}{'...' if len(self.text) > 100 else ''},\n"
            f"  latency_ms={self.latency_ms:.1f},\n"
            f"  tokens={self.tokens_generated},\n"
            f"  tok/s={self.tokens_per_second:.1f}\n"
            f")"
        )


def run_image_inference(
    llm,
    image,  # PIL.Image, path, or URL
    instruction: str,
    system_prompt: str = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> InferenceResult:
    """
    Run inference on a single image.
    
    Args:
        llm: vLLM LLM instance
        image: PIL.Image, file path, or URL
        instruction: User instruction
        system_prompt: Optional custom system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
    
    Returns:
        InferenceResult with text, timing, and stats
    """
    from vllm import SamplingParams
    from PIL import Image
    
    # Load image if needed
    if isinstance(image, (str, Path)):
        if str(image).startswith(("http://", "https://")):
            import requests
            from io import BytesIO
            response = requests.get(str(image))
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image)
    
    # Get model name for prompt formatting
    model_name = llm.llm_engine.model_config.model
    prompt = format_prompt(instruction, model_name, system_prompt)
    
    # Prepare inputs
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )
    
    # Run inference
    start = time.time()
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    latency = (time.time() - start) * 1000
    
    output = outputs[0].outputs[0]
    text = output.text
    tokens = len(output.token_ids)
    
    return InferenceResult(
        text=text,
        latency_ms=latency,
        tokens_generated=tokens,
        tokens_per_second=tokens / (latency / 1000) if latency > 0 else 0,
    )


def run_video_inference(
    llm,
    video,  # numpy array (T, H, W, C), path, or list of frames
    instruction: str,
    system_prompt: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    fps: float = 2.0,
) -> InferenceResult:
    """
    Run inference on a video.
    
    Note: Qwen3-VL with EVS enabled will automatically prune video tokens
    for faster inference with minimal quality loss.
    
    Args:
        llm: vLLM LLM instance
        video: Video as numpy array (T, H, W, C), file path, or list of PIL Images
        instruction: User instruction
        system_prompt: Optional custom system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        fps: Frames per second for video metadata
    
    Returns:
        InferenceResult with text, timing, and stats
    """
    from vllm import SamplingParams
    import numpy as np
    
    # Convert video to expected format
    if isinstance(video, (str, Path)):
        # Load video from file
        try:
            import cv2
            cap = cv2.VideoCapture(str(video))
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            video = np.stack(frames)
        except ImportError:
            raise ImportError("cv2 (opencv-python) required for video file loading")
    elif isinstance(video, list):
        # List of PIL Images
        video = np.stack([np.array(f) for f in video])
    
    # Get model name for prompt formatting
    model_name = llm.llm_engine.model_config.model
    
    # Use video-specific prompt format
    if "qwen3" in model_name.lower():
        prompt = (
            f"<|im_start|>system\n{system_prompt or QWEN3_VL_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    else:
        prompt = (
            f"<|im_start|>system\n{system_prompt or QWEN2_VL_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    
    # Video metadata for Qwen3-VL
    num_frames = len(video)
    video_metadata = {
        "fps": fps,
        "duration": num_frames / fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": False,
    }
    
    # Prepare inputs
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"video": [(video, video_metadata)]},
    }
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )
    
    # Run inference
    start = time.time()
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    latency = (time.time() - start) * 1000
    
    output = outputs[0].outputs[0]
    text = output.text
    tokens = len(output.token_ids)
    
    return InferenceResult(
        text=text,
        latency_ms=latency,
        tokens_generated=tokens,
        tokens_per_second=tokens / (latency / 1000) if latency > 0 else 0,
    )


def run_batch_inference(
    llm,
    images: list,
    instructions: list[str],
    system_prompt: str = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> list[InferenceResult]:
    """
    Run batch inference on multiple images.
    
    This leverages vLLM's continuous batching for higher throughput.
    
    Args:
        llm: vLLM LLM instance
        images: List of images (PIL.Image, path, or URL)
        instructions: List of instructions (one per image, or single for all)
        system_prompt: Optional custom system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        List of InferenceResult
    """
    from vllm import SamplingParams
    from PIL import Image
    
    # Normalize instructions to list
    if isinstance(instructions, str):
        instructions = [instructions] * len(images)
    
    assert len(images) == len(instructions), "Images and instructions must match"
    
    # Load images
    loaded_images = []
    for img in images:
        if isinstance(img, (str, Path)):
            if str(img).startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                response = requests.get(str(img))
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(img)
        loaded_images.append(img)
    
    # Get model name
    model_name = llm.llm_engine.model_config.model
    
    # Prepare batch inputs
    batch_inputs = []
    for img, inst in zip(loaded_images, instructions):
        prompt = format_prompt(inst, model_name, system_prompt)
        batch_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img},
        })
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<|im_end|>"],
    )
    
    # Run batch inference
    start = time.time()
    outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
    total_latency = (time.time() - start) * 1000
    
    # Process results
    results = []
    for output in outputs:
        out = output.outputs[0]
        tokens = len(out.token_ids)
        # Per-request latency is approximate (batch is processed together)
        per_request_latency = total_latency / len(outputs)
        
        results.append(InferenceResult(
            text=out.text,
            latency_ms=per_request_latency,
            tokens_generated=tokens,
            tokens_per_second=tokens / (per_request_latency / 1000) if per_request_latency > 0 else 0,
        ))
    
    print(f"\n📊 Batch Statistics:")
    print(f"   Total requests: {len(outputs)}")
    print(f"   Total time: {total_latency:.0f}ms")
    print(f"   Throughput: {len(outputs) / (total_latency / 1000):.1f} req/s")
    print(f"   Avg latency: {total_latency / len(outputs):.0f}ms per request")
    
    return results


# =============================================================================
# CELL 5: TEST IMAGE GENERATION
# =============================================================================

def create_test_image(width: int = 1024, height: int = 1024) -> "Image":
    """Create a test image with UI elements for testing."""
    from PIL import Image, ImageDraw
    
    img = Image.new("RGB", (width, height), "#0f1420")
    draw = ImageDraw.Draw(img)
    
    # Top bar
    draw.rectangle([0, 0, width, 60], fill="#1a2535")
    draw.text((20, 20), "Dashboard - Test Application", fill="#ffffff")
    
    # Sidebar
    draw.rectangle([0, 60, 200, height], fill="#141d2b")
    menu_items = ["Dashboard", "Analytics", "Reports", "Settings", "Users", "Admin"]
    for i, item in enumerate(menu_items):
        y = 100 + i * 50
        draw.rectangle([10, y, 190, y + 40], fill="#1f2d42")
        draw.text((30, y + 10), f"📊 {item}", fill="#b0b0b0")
    
    # Main content area with cards
    for row in range(2):
        for col in range(3):
            x = 220 + col * 260
            y = 80 + row * 200
            draw.rectangle([x, y, x + 240, y + 180], fill="#1a2535", outline="#2a3a4f")
            draw.text((x + 20, y + 20), f"Card {row * 3 + col + 1}", fill="#808080")
            draw.text((x + 20, y + 60), f"${(row * 3 + col + 1) * 1234:,}", fill="#00d084")
    
    # Action buttons
    buttons = [("Export", "#3a7bd5"), ("Refresh", "#00a86b"), ("Configure", "#e94560")]
    for i, (text, color) in enumerate(buttons):
        x = 220 + i * 150
        draw.rectangle([x, height - 80, x + 130, height - 40], fill=color)
        draw.text((x + 20, height - 70), text, fill="#ffffff")
    
    return img


def create_test_video(num_frames: int = 30, width: int = 640, height: int = 480):
    """Create a test video (sequence of frames) for testing."""
    import numpy as np
    
    frames = []
    for i in range(num_frames):
        # Create frame with moving element
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        
        # Moving circle
        cx = int(width * (i / num_frames))
        cy = height // 2
        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= 50 ** 2
        frame[mask] = [255, 100, 100]
        
        frames.append(frame)
    
    return np.stack(frames)


# =============================================================================
# CELL 6: BENCHMARK UTILITIES
# =============================================================================

def benchmark_model(
    llm,
    num_runs: int = 5,
    image_size: tuple[int, int] = (1024, 1024),
) -> dict:
    """
    Run benchmark on the model.
    
    Returns dict with timing statistics.
    """
    from PIL import Image
    import numpy as np
    
    # Create test image
    test_image = create_test_image(*image_size)
    instruction = "Describe this dashboard interface in detail."
    
    print(f"\n🔬 Running benchmark ({num_runs} runs)...")
    latencies = []
    tokens_list = []
    
    for i in range(num_runs):
        result = run_image_inference(llm, test_image, instruction)
        latencies.append(result.latency_ms)
        tokens_list.append(result.tokens_generated)
        print(f"   Run {i+1}: {result.latency_ms:.0f}ms, {result.tokens_generated} tokens")
    
    stats = {
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "mean_tokens": np.mean(tokens_list),
        "mean_tok_per_sec": np.mean(tokens_list) / (np.mean(latencies) / 1000),
    }
    
    print(f"\n📊 Benchmark Results:")
    print(f"   Mean latency: {stats['mean_latency_ms']:.0f}ms ± {stats['std_latency_ms']:.0f}ms")
    print(f"   Range: {stats['min_latency_ms']:.0f}ms - {stats['max_latency_ms']:.0f}ms")
    print(f"   Mean tokens: {stats['mean_tokens']:.0f}")
    print(f"   Throughput: {stats['mean_tok_per_sec']:.1f} tok/s")
    
    return stats


# =============================================================================
# CELL 7: ARCHITECTURE COMPARISON DISPLAY
# =============================================================================

def print_architecture_comparison():
    """Print architecture comparison between Qwen2-VL and Qwen3-VL."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                    QWEN2-VL vs QWEN3-VL ARCHITECTURE COMPARISON                     ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  ┌──────────────────────┬─────────────────────────┬────────────────────────────┐   ║
║  │ Component            │ Qwen2-VL                │ Qwen3-VL                   │   ║
║  ├──────────────────────┼─────────────────────────┼────────────────────────────┤   ║
║  │ Patch Embedding      │ Conv3D (no bias)        │ Conv3D (WITH bias)         │   ║
║  │ Position Encoding    │ 3D RoPE only            │ Learned + RoPE + Interp    │   ║
║  │ MLP Activation       │ QuickGELU               │ SiLU                       │   ║
║  │ Multi-Scale Features │ ❌                      │ ✅ DeepStack               │   ║
║  │ Video Token Pruning  │ ❌                      │ ✅ EVS                     │   ║
║  │ Max Video Frames     │ 14                      │ 24,576                     │   ║
║  │ Speculative Decode   │ Basic                   │ ✅ Eagle3                  │   ║
║  │ MoE Variants         │ ❌                      │ ✅ Qwen3-VL-30B-A3B        │   ║
║  └──────────────────────┴─────────────────────────┴────────────────────────────┘   ║
║                                                                                     ║
║  RECOMMENDATION:                                                                    ║
║  • Use Qwen3-VL for new projects (better quality, EVS for videos)                   ║
║  • Use Qwen2-VL for T4/older GPUs or if you have existing fine-tunes               ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
""")


def print_gpu_recommendations():
    """Print GPU-specific recommendations."""
    print("""
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                          GPU-SPECIFIC RECOMMENDATIONS                               ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  T4 (16GB, Turing SM 7.5):                                                          ║
║  ├── Best: Qwen2-VL-2B (FP16) or Qwen3-VL-4B (4-bit)                                ║
║  ├── Constraints: No FlashAttn v2, No BF16, enforce_eager=True                      ║
║  ├── Max resolution: ~720×720                                                       ║
║  └── Expected latency: ~800-1200ms                                                  ║
║                                                                                     ║
║  A100 (40/80GB, Ampere SM 8.0):                                                     ║
║  ├── Best: Qwen2-VL-7B or Qwen3-VL-8B (full precision)                              ║
║  ├── Features: FlashAttn v2, BF16                                                   ║
║  ├── Max resolution: ~1280×1280                                                     ║
║  └── Expected latency: ~200-400ms                                                   ║
║                                                                                     ║
║  H100 (80GB, Hopper SM 9.0):                                                        ║
║  ├── Best: Qwen3-VL-8B or Qwen3-VL-30B-A3B (MoE)                                    ║
║  ├── Features: FlashAttn v3, FP8, 3.35 TB/s                                         ║
║  ├── Max resolution: 1920×1080                                                      ║
║  └── Expected latency: ~100-200ms                                                   ║
║                                                                                     ║
║  B200 (192GB, Blackwell SM 10.0):                                                   ║
║  ├── Best: Qwen3-VL-8B (128K context) or Qwen2-VL-72B (single GPU!)                 ║
║  ├── Features: FP4 (future), 8 TB/s                                                 ║
║  ├── Max resolution: 4K (3840×2160)                                                 ║
║  └── Expected latency: ~50-100ms                                                    ║
║                                                                                     ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print_architecture_comparison()
    print_gpu_recommendations()
    
    print("\n" + "=" * 80)
    print("To use in a notebook:")
    print("=" * 80)
    print("""
# Cell 1: Setup
from qwen_vl_notebook import setup_environment, create_engine
gpu_info = setup_environment()
print(f"Detected: {gpu_info}")

# Cell 2: Create engine (auto-detects GPU)
llm = create_engine()  # Or create_engine(model_preference="qwen2")

# Cell 3: Run inference
from qwen_vl_notebook import run_image_inference, create_test_image
test_image = create_test_image()
result = run_image_inference(llm, test_image, "Describe this interface")
print(result)

# Cell 4: Benchmark
from qwen_vl_notebook import benchmark_model
stats = benchmark_model(llm, num_runs=5)
""")

