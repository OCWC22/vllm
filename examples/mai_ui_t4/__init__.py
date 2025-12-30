# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
MAI-UI T4 Optimization Package

Provides production-ready configurations and utilities for running
MAI-UI (Qwen2-VL based GUI agent) on NVIDIA T4 GPUs with vLLM.

Quick Start:
    from examples.mai_ui_t4 import MAIUIClient, get_t4_engine_args
    
    # Option 1: Use the client for online inference
    client = MAIUIClient("http://localhost:8000")
    action = client.get_action("screenshot.png", "Click the submit button")
    
    # Option 2: Use engine args for offline inference
    from vllm import LLM
    from dataclasses import asdict
    
    engine_args = get_t4_engine_args("2b_balanced")
    llm = LLM(**asdict(engine_args))
"""

from .client import Action, MAIUIClient
from .config import (
    ModelVariant,
    QuantizationMethod,
    T4_PROFILES,
    T4OptimizationConfig,
    get_mai_ui_prompt,
    get_t4_engine_args,
    validate_t4_compatibility,
)

__all__ = [
    # Client
    "MAIUIClient",
    "Action",
    # Configuration
    "T4OptimizationConfig",
    "T4_PROFILES",
    "ModelVariant",
    "QuantizationMethod",
    "get_t4_engine_args",
    "get_mai_ui_prompt",
    "validate_t4_compatibility",
]

__version__ = "1.0.0"

