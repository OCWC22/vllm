#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MAI-UI T4-Optimized Server

Launches an OpenAI-compatible API server for MAI-UI inference on T4 GPUs.
Uses vLLM's built-in server with T4-optimized configurations.

Usage:
    # Start server with default configuration (2B model)
    python server.py --port 8000
    
    # Start with 8B model (4-bit quantization)
    python server.py --profile 8b_quality --port 8000
    
    # Custom configuration
    python server.py --max-model-len 2048 --max-num-seqs 2 --port 8000

The server exposes:
    - POST /v1/chat/completions  (OpenAI-compatible)
    - POST /v1/completions       (OpenAI-compatible)
    - GET  /health               (Health check)
    - GET  /v1/models            (List models)
"""

import argparse
import os
import subprocess
import sys

from config import T4_PROFILES, validate_t4_compatibility


def build_server_command(args: argparse.Namespace) -> list[str]:
    """
    Build the vLLM server command with T4-optimized arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        List of command-line arguments for subprocess
    """
    # Get the profile configuration
    profile = T4_PROFILES.get(args.profile)
    if profile is None:
        raise ValueError(f"Unknown profile: {args.profile}")
    
    # Model paths
    model_paths = {
        "2b": "osunlp/MAI-UI-2B",
        "8b": "osunlp/MAI-UI-8B",
    }
    
    model = args.model or model_paths.get(profile.model_variant.value)
    
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--dtype", profile.dtype,
        "--max-model-len", str(args.max_model_len or profile.max_model_len),
        "--gpu-memory-utilization", str(profile.gpu_memory_utilization),
        "--max-num-seqs", str(args.max_num_seqs or profile.max_num_seqs),
        "--host", args.host,
        "--port", str(args.port),
    ]
    
    # Add enforce-eager flag
    if profile.enforce_eager:
        cmd.append("--enforce-eager")
    
    # Add multimodal limits
    cmd.extend([
        "--limit-mm-per-prompt", f"image={profile.limit_mm_per_prompt}",
    ])
    
    # Add mm processor kwargs (for max_pixels)
    mm_kwargs = {
        "min_pixels": profile.min_pixels,
        "max_pixels": args.max_pixels or profile.max_pixels,
    }
    import json
    cmd.extend([
        "--mm-processor-kwargs", json.dumps(mm_kwargs),
    ])
    
    # Add quantization if specified
    if profile.quantization.value != "none":
        cmd.extend(["--quantization", profile.quantization.value])
        if profile.quantization.value == "bitsandbytes":
            cmd.extend(["--load-format", "bitsandbytes"])
    
    # Add optional flags
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    
    return cmd


def print_server_info(args: argparse.Namespace):
    """Print server configuration info."""
    profile = T4_PROFILES.get(args.profile)
    
    print("\n" + "=" * 70)
    print("🚀 MAI-UI T4-OPTIMIZED SERVER")
    print("=" * 70)
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  Configuration                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  Profile:          {args.profile:<48}│
│  Model:            {profile.model_variant.value:<48}│
│  Quantization:     {profile.quantization.value:<48}│
│  Max Context:      {profile.max_model_len:<48}│
│  Max Sequences:    {profile.max_num_seqs:<48}│
│  Max Pixels:       {profile.max_pixels:<48}│
├─────────────────────────────────────────────────────────────────────┤
│  Server                                                              │
├─────────────────────────────────────────────────────────────────────┤
│  Host:             {args.host:<48}│
│  Port:             {args.port:<48}│
│  API Endpoint:     http://{args.host}:{args.port}/v1/chat/completions          │
└─────────────────────────────────────────────────────────────────────┘
""")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Launch MAI-UI T4-optimized vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default 2B model
  python server.py
  
  # Start with 8B model (4-bit quantized)
  python server.py --profile 8b_quality
  
  # Custom host/port
  python server.py --host 0.0.0.0 --port 8080
  
  # With API key authentication
  python server.py --api-key your-secret-key

Client Example:
  curl http://localhost:8000/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -d '{
      "model": "osunlp/MAI-UI-2B",
      "messages": [{
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
          {"type": "text", "text": "Click the submit button"}
        ]
      }],
      "max_tokens": 128
    }'
        """,
    )
    
    # Profile selection
    parser.add_argument(
        "--profile",
        type=str,
        default="2b_balanced",
        choices=list(T4_PROFILES.keys()),
        help="T4 optimization profile (default: 2b_balanced)",
    )
    
    # Model override
    parser.add_argument(
        "--model",
        type=str,
        help="Override model path (default: from profile)",
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication",
    )
    
    # Override profile settings
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Override max model length",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        help="Override max concurrent sequences",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        help="Override max pixels for image processing",
    )
    
    # Additional flags
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in model",
    )
    parser.add_argument(
        "--check-compatibility",
        action="store_true",
        help="Check T4 compatibility and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
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
        sys.exit(0)
    
    # Build command
    cmd = build_server_command(args)
    
    # Print server info
    print_server_info(args)
    
    # Print or execute command
    if args.dry_run:
        print("Command that would be executed:")
        print(" \\\n    ".join(cmd))
        sys.exit(0)
    
    print("Starting vLLM server...")
    print("-" * 70)
    
    try:
        # Execute the server
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server exited with error code: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()

