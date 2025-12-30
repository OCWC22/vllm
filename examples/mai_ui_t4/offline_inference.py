# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MAI-UI Offline Inference on T4 GPUs

This script provides optimized batch inference for MAI-UI models
on NVIDIA T4 GPUs using vLLM's PagedAttention and continuous batching.

Usage:
    # Single image inference
    python offline_inference.py --image screenshot.png --instruction "Click submit"
    
    # Batch inference from JSON
    python offline_inference.py --batch-file requests.json
    
    # Use 8B model with quantization
    python offline_inference.py --model-variant 8b --image screenshot.png --instruction "Click submit"

Example requests.json:
    [
        {"image": "screen1.png", "instruction": "Click the login button"},
        {"image": "screen2.png", "instruction": "Type 'hello' in the search box"}
    ]
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from vllm import LLM, SamplingParams

# Import T4 optimized config
from config import (
    T4_PROFILES,
    get_mai_ui_prompt,
    get_t4_engine_args,
    validate_t4_compatibility,
)


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load and validate an image for MAI-UI inference.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        PIL Image in RGB format
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    image = Image.open(path)
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    return image


def run_single_inference(
    llm: LLM,
    image: Image.Image,
    instruction: str,
    sampling_params: SamplingParams,
) -> str:
    """
    Run inference on a single image.
    
    Args:
        llm: Initialized vLLM LLM instance
        image: PIL Image (screenshot)
        instruction: GUI action instruction
        sampling_params: vLLM sampling parameters
    
    Returns:
        Generated action string (e.g., "pyautogui.click(500, 300)")
    """
    prompt = get_mai_ui_prompt(instruction)
    
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {"image": image},
    }
    
    outputs = llm.generate([inputs], sampling_params=sampling_params)
    
    return outputs[0].outputs[0].text.strip()


def run_batch_inference(
    llm: LLM,
    requests: list[dict],
    sampling_params: SamplingParams,
) -> list[dict]:
    """
    Run inference on a batch of requests.
    
    Args:
        llm: Initialized vLLM LLM instance
        requests: List of {"image": path, "instruction": str}
        sampling_params: vLLM sampling parameters
    
    Returns:
        List of {"image": path, "instruction": str, "action": str, "latency_ms": float}
    """
    # Prepare all inputs
    inputs = []
    images = []
    
    for req in requests:
        image = load_image(req["image"])
        images.append(image)
        prompt = get_mai_ui_prompt(req["instruction"])
        
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        })
    
    # Run batch inference
    start_time = time.perf_counter()
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    total_time = time.perf_counter() - start_time
    
    # Collect results
    results = []
    for i, (req, output) in enumerate(zip(requests, outputs)):
        results.append({
            "image": req["image"],
            "instruction": req["instruction"],
            "action": output.outputs[0].text.strip(),
            "tokens_generated": len(output.outputs[0].token_ids),
        })
    
    # Add timing info
    avg_latency_ms = (total_time / len(requests)) * 1000
    for result in results:
        result["avg_latency_ms"] = avg_latency_ms
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MAI-UI inference on T4 GPU with vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single inference
  python offline_inference.py --image screenshot.png --instruction "Click login"
  
  # Batch inference
  python offline_inference.py --batch-file requests.json --output results.json
  
  # Use 8B model with 4-bit quantization
  python offline_inference.py --profile 8b_quality --image screen.png --instruction "Click submit"
        """,
    )
    
    # Model configuration
    parser.add_argument(
        "--profile",
        type=str,
        default="2b_balanced",
        choices=list(T4_PROFILES.keys()),
        help="T4 optimization profile (default: 2b_balanced)",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        choices=["2b", "8b"],
        help="Override model variant (2b or 8b)",
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image",
        type=str,
        help="Path to single screenshot image",
    )
    input_group.add_argument(
        "--batch-file",
        type=str,
        help="Path to JSON file with batch requests",
    )
    
    parser.add_argument(
        "--instruction",
        type=str,
        help="GUI action instruction (required with --image)",
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for batch results",
    )
    
    # Sampling parameters
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)",
    )
    
    # Debugging
    parser.add_argument(
        "--check-compatibility",
        action="store_true",
        help="Check T4 compatibility and exit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed timing information",
    )
    
    args = parser.parse_args()
    
    # Check compatibility if requested
    if args.check_compatibility:
        print("\n🔍 Checking T4 Compatibility...\n")
        compat = validate_t4_compatibility()
        for key, value in compat.items():
            if isinstance(value, bool):
                status = "✅" if value else "❌"
            else:
                status = "📊"
            print(f"  {status} {key}: {value}")
        
        if compat.get("is_t4") or compat.get("is_turing"):
            print("\n✅ GPU is compatible with T4 optimizations!")
        else:
            print("\n⚠️  GPU is not a T4. Some optimizations may differ.")
        
        sys.exit(0)
    
    # Validate input arguments
    if args.image and not args.instruction:
        parser.error("--instruction is required when using --image")
    
    # Print configuration
    print("\n" + "=" * 60)
    print("MAI-UI T4-OPTIMIZED INFERENCE")
    print("=" * 60)
    print(f"  Profile: {args.profile}")
    if args.model_variant:
        print(f"  Model Override: {args.model_variant}")
    print(f"  Max Tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print("=" * 60 + "\n")
    
    # Get engine args
    engine_args = get_t4_engine_args(args.profile)
    
    # Override model variant if specified
    if args.model_variant:
        model_map = {
            "2b": "osunlp/MAI-UI-2B",
            "8b": "osunlp/MAI-UI-8B",
        }
        engine_args.model = model_map[args.model_variant]
        
        # Add quantization for 8B model
        if args.model_variant == "8b" and args.profile not in ["8b_quality"]:
            print("⚠️  8B model requires quantization on T4. Enabling BitsAndBytes 4-bit.")
            engine_args.quantization = "bitsandbytes"
            engine_args.load_format = "bitsandbytes"
    
    # Initialize LLM
    print("🚀 Initializing vLLM engine...")
    init_start = time.perf_counter()
    llm = LLM(**asdict(engine_args))
    init_time = time.perf_counter() - init_start
    print(f"✅ Engine initialized in {init_time:.2f}s\n")
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )
    
    # Run inference
    if args.image:
        # Single image inference
        print(f"📸 Processing: {args.image}")
        print(f"📝 Instruction: {args.instruction}\n")
        
        image = load_image(args.image)
        
        start_time = time.perf_counter()
        action = run_single_inference(llm, image, args.instruction, sampling_params)
        latency = time.perf_counter() - start_time
        
        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        print(f"  Action: {action}")
        print(f"  Latency: {latency * 1000:.1f}ms")
        print("=" * 60)
        
    else:
        # Batch inference
        print(f"📦 Loading batch from: {args.batch_file}")
        
        with open(args.batch_file) as f:
            requests = json.load(f)
        
        print(f"📊 Processing {len(requests)} requests...\n")
        
        start_time = time.perf_counter()
        results = run_batch_inference(llm, requests, sampling_params)
        total_time = time.perf_counter() - start_time
        
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['instruction']}")
            print(f"    → {result['action']}")
        
        print("\n" + "-" * 60)
        print(f"Total requests: {len(results)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg latency: {(total_time / len(results)) * 1000:.1f}ms")
        print(f"Throughput: {len(results) / total_time:.2f} req/s")
        print("=" * 60)
        
        # Save results if output specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")


if __name__ == "__main__":
    main()

