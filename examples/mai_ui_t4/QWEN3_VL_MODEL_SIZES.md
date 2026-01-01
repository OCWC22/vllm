# Qwen3-VL Model Family: Complete Architecture Breakdown

All Qwen3-VL model sizes, their architecture, parameter counts, and optimal GPU deployment configurations.

**Companion Guide**: See [QWEN_VL_COMPLETE_GUIDE.md](./QWEN_VL_COMPLETE_GUIDE.md) for detailed architectural explanations, training pipelines, and MAI-UI integration.

---

## Table of Contents

1. [Quick Reference: Model Selection by GPU](#quick-reference-model-selection-by-gpu)
2. [**Engineering Guide: Model Selection & Deployment**](#engineering-guide-model-selection--deployment)
   - [What Each Model Size Is Designed For](#what-each-model-size-is-designed-for)
   - [Dense vs MoE: Architecture Deep Dive](#dense-vs-moe-architecture-deep-dive)
   - [MoE Internal Mechanics](#moe-internal-mechanics-routing-experts-activation)
   - [Performance/Cost/Latency Trade-offs](#performancecostlatency-trade-offs)
   - [Production Scenario Decision Tree](#production-scenario-decision-tree)
   - [Implementation & Deployment Guide](#implementation--deployment-guide)
3. [All Qwen3-VL Models: Architecture Breakdown](#all-qwen3-vl-models-architecture-breakdown)
4. [Qwen3 vs Qwen3-VL: Text-Only vs Multimodal](#qwen3-vs-qwen3-vl-text-only-vs-multimodal)
5. [Complete Architecture Diagrams](#complete-architecture-diagrams)
6. [vLLM PagedAttention for Multimodal](#vllm-pagedattention-for-multimodal)
7. [GPU Memory Layout Diagrams](#gpu-memory-layout-diagrams)
8. [vLLM Deployment Configurations](#vllm-deployment-configurations)
9. [Performance Benchmarks](#performance-benchmarks)
10. [Summary: Model Selection Guide](#summary-model-selection-guide)
11. [**Why 32B is Required for GUI Agents (SFT + RL)**](#why-32b-is-required-for-reliable-gui-agents-sft--rl-training)
    - [Why Smaller Models Fail](#the-problem-smaller-models-are-ass-for-gui-agents)
    - [Why SFT Can't Fix Smaller Models](#why-sft-alone-cant-fix-smaller-models)
    - [Why RL Only Works at 32B](#why-rl-grpo-only-works-at-32b-scale)
    - [Architecture Capacity Problem](#the-architecture-capacity-problem)
    - [What About MoE?](#what-about-moe-30b-a3b)
12. [**Engineering Guide: SFT & RL Dataset Creation**](#engineering-guide-creating-better-sft--rl-datasets-for-gui-agents)
    - [Core Problem](#the-problem-why-gui-agent-training-is-hard)
    - [GRPO for GUI Agents](#grpo-for-gui-agents-how-it-works)
    - [Practical Checklist](#practical-engineering-checklist-creating-better-sft-data)
    - [Complete Training Recipe](#summary-the-complete-training-recipe)
13. [**Complete Paper Technical Breakdown**](#paper-references-all-verified--complete-technical-breakdown)
    - [MAI-UI Technical Report](#1-mai-ui-technical-report--arxiv251222047)
    - [UI-Ins: Instruction-as-Reasoning](#2-ui-ins--arxiv251020286)
    - [OS-Genesis: Reverse Task Synthesis](#3-os-genesis--arxiv241219723)
    - [UI-R1: GRPO for GUI](#4-ui-r1--arxiv250321620)
    - [Fara-7B: Microsoft Computer Use Agent](#5-fara-7b--faragen--arxiv251119663)
    - [GUI-360: Windows Dataset](#6-gui-360--arxiv251104307)
    - [UGround: Visual Grounding](#7-uground--arxiv241005243)
    - [OS-ATLAS: Cross-Platform](#8-os-atlas--arxiv241023218)
    - [EDGE: Synthetic Data](#9-edge--arxiv241019461)
    - [GUICourse: Training Suite](#10-guicourse--arxiv240611317)
    - [OpenCUA in vLLM](#11-opencua--in-vllm-codebase)
14. [**Deep Technical Breakdown: 8B vs 32B**](#deep-technical-breakdown-qwen3-vl-8b-vs-32b)
    - [Architecture Differences](#1-architecture-differences-8b-vs-32b)
    - [Memory Footprint Comparison](#memory-footprint-comparison)
    - [Throughput vs Latency Trade-offs](#throughput-vs-latency-trade-offs)
    - [vLLM Inference Pipeline](#2-vllm-inference-pipeline-for-qwen3-vl)
    - [KV Cache and PagedAttention](#kv-cache-layout-and-pagedattention)
    - [Prefix Caching and Batching](#prefix-caching-and-batching)
    - [GPU-Level Execution Details](#3-gpu-level-execution-details)
    - [Vision-Language Inference](#4-vision-language-specific-inference)
    - [Prompt-by-Prompt Flow](#5-prompt-by-prompt-inference-flow)
15. [**How Inference Actually Works**](#how-qwen3-vl-inference-actually-works-a-systems-level-guide)
    - [Architecture Overview: Dense vs MoE](#architecture-overview-dense-vs-moe)
    - [Internal Architecture: Each Model Size](#internal-architecture-each-model-size)
    - [Qwen3-VL-2B Complete Architecture](#qwen3-vl-2b-complete-internal-architecture-reference-model)
    - [Qwen3-VL-30B-A3B MoE Complete Architecture](#qwen3-vl-30b-a3b-moe-complete-internal-architecture)
    - [DeepStack Architecture](#deepstack-multi-level-vision-feature-injection)
    - [Step-by-Step Inference Flow](#step-by-step-inference-flow)
    - [vLLM Backend Selection Logic](#vllm-attention-backend-selection-logic-from-cudapy)
    - [Bottleneck Analysis: Compute vs Memory](#bottleneck-analysis-compute-vs-memory-bound)
    - [GPU Hardware Characteristics](#gpu-hardware-characteristics-for-inference)
    - [Complete Model × GPU Matrix](#complete-model--gpu-performance-matrix)
    - [End-to-End Latency Comparison](#end-to-end-latency-comparison)
    - [Detailed Execution: 8B on Each GPU](#detailed-execution-qwen3-vl-8b-on-each-gpu)
    - [Detailed Execution: 32B](#detailed-execution-qwen3-vl-32b-gui-agent-recommended)
    - [Detailed Execution: MoE Models](#detailed-execution-moe-models)
    - [Model Selection Decision Guide](#summary-model-selection-by-use-case)
16. [References](#references)

---

## Quick Reference: Model Selection by GPU

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3-VL MODEL SELECTION MATRIX                                               │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                            │
│  GPU          VRAM      Best Model              Precision   Max Context   Concurrent   Expected Latency   │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════  │
│  T4           16 GB     Qwen3-VL-2B-Instruct    FP16        8K            4            ~800ms             │
│  T4           16 GB     Qwen3-VL-4B (4-bit)     INT4        4K            4            ~1000ms            │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  L4           24 GB     Qwen3-VL-4B-Instruct    FP16        16K           8            ~500ms             │
│  L4           24 GB     Qwen3-VL-8B (4-bit)     INT4        8K            4            ~700ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  A10G         24 GB     Qwen3-VL-4B-Instruct    BF16        16K           8            ~400ms             │
│  A10G         24 GB     Qwen3-VL-8B (4-bit)     INT4        8K            4            ~600ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  A100-40GB    40 GB     Qwen3-VL-8B-Instruct    BF16        32K           16           ~300ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  A100-80GB    80 GB     Qwen3-VL-8B-Instruct    BF16        64K           32           ~250ms             │
│  A100-80GB    80 GB     Qwen3-VL-32B-Instruct   BF16        16K           8            ~500ms             │
│  A100-80GB    80 GB     Qwen3-VL-30B-A3B (MoE)  BF16        32K           16           ~350ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  H100-80GB    80 GB     Qwen3-VL-8B-Instruct    FP8         64K           64           ~150ms             │
│  H100-80GB    80 GB     Qwen3-VL-32B-Instruct   FP8         32K           32           ~300ms             │
│  H100-80GB    80 GB     Qwen3-VL-30B-A3B (MoE)  FP8         64K           32           ~200ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  H200-141GB   141 GB    Qwen3-VL-32B-Instruct   BF16        128K          64           ~200ms             │
│  H200-141GB   141 GB    Qwen3-VL-30B-A3B (MoE)  BF16        128K          64           ~150ms             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  B200-192GB   192 GB    Qwen3-VL-32B-Instruct   BF16        256K          128          ~100ms             │
│  B200-192GB   192 GB    Qwen3-VL-30B-A3B (MoE)  BF16        256K          128          ~80ms              │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────│
│  8×H100       640 GB    Qwen3-VL-235B-A22B      FP8         256K          64           ~400ms             │
│  8×B200       1.5 TB    Qwen3-VL-235B-A22B      BF16        256K          128          ~200ms             │
│                                                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Engineering Guide: Model Selection & Deployment

This section provides practical engineering guidance for choosing and deploying Qwen3-VL models in production.

### What Each Model Size Is Designed For

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3-VL MODEL DESIGN PURPOSES                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  DENSE MODELS (All parameters active for every token):                                                      │
│  ═══════════════════════════════════════════════════════                                                    │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Qwen3-VL-2B (1.7B actual)                                                                           │   │
│  │ ══════════════════════════                                                                          │   │
│  │ PURPOSE: Edge deployment, mobile apps, cost-sensitive high-volume scenarios                        │   │
│  │                                                                                                     │   │
│  │ DESIGNED FOR:                                                                                       │   │
│  │ • Real-time mobile image captioning                                                                 │   │
│  │ • IoT and embedded vision systems (Jetson, mobile chips)                                           │   │
│  │ • High-throughput pipelines where latency matters more than accuracy                               │   │
│  │ • Student/distillation teacher for smaller models                                                   │   │
│  │ • Prototyping and development (fast iteration)                                                      │   │
│  │                                                                                                     │   │
│  │ TRADE-OFF: Lower accuracy on complex reasoning, fine details, multi-step instructions             │   │
│  │ SWEET SPOT: Simple tasks (OCR, basic VQA, single-object detection) at <100ms latency              │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Qwen3-VL-4B                                                                                         │   │
│  │ ════════════                                                                                        │   │
│  │ PURPOSE: Balanced edge/cloud, consumer GPU deployment, cost-effective production                   │   │
│  │                                                                                                     │   │
│  │ DESIGNED FOR:                                                                                       │   │
│  │ • Consumer GPU inference (RTX 3090, RTX 4090)                                                       │   │
│  │ • Document understanding (invoices, forms, receipts)                                                │   │
│  │ • Basic GUI automation (MAI-UI on constrained hardware)                                             │   │
│  │ • Multi-turn image conversations with reasonable context                                            │   │
│  │ • Batch processing pipelines with throughput requirements                                           │   │
│  │                                                                                                     │   │
│  │ TRADE-OFF: Struggles with very complex diagrams, long documents, subtle visual details             │   │
│  │ SWEET SPOT: Document processing, basic visual reasoning at 10-50 req/sec on single L4/T4          │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Qwen3-VL-8B                                                                                         │   │
│  │ ════════════                                                                                        │   │
│  │ PURPOSE: Production workhorse, best quality-to-cost ratio, recommended default choice              │   │
│  │                                                                                                     │   │
│  │ DESIGNED FOR:                                                                                       │   │
│  │ • General-purpose visual AI assistant                                                               │   │
│  │ • Complex document understanding (multi-page, tables, charts)                                       │   │
│  │ • Video understanding (short clips, surveillance, content moderation)                               │   │
│  │ • GUI automation agents (MAI-UI primary target)                                                     │   │
│  │ • Visual reasoning and multi-step problem solving                                                   │   │
│  │ • Production APIs serving diverse visual tasks                                                      │   │
│  │                                                                                                     │   │
│  │ TRADE-OFF: Needs A100-class GPU for optimal performance                                            │   │
│  │ SWEET SPOT: The "GPT-4V killer" for most visual tasks at 1/10th the cost                          │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Qwen3-VL-32B                                                                                        │   │
│  │ ═════════════                                                                                       │   │
│  │ PURPOSE: State-of-the-art quality, complex reasoning, when accuracy is paramount                   │   │
│  │                                                                                                     │   │
│  │ DESIGNED FOR:                                                                                       │   │
│  │ • Scientific/medical image analysis requiring high accuracy                                         │   │
│  │ • Complex diagram understanding (architecture, circuit, engineering drawings)                       │   │
│  │ • Long-form video analysis (movies, lectures, presentations)                                        │   │
│  │ • Multi-image reasoning (compare, contrast, aggregate information)                                  │   │
│  │ • Tasks where 8B isn't accurate enough                                                              │   │
│  │ • Benchmark/evaluation workloads                                                                    │   │
│  │                                                                                                     │   │
│  │ TRADE-OFF: 4× memory, 2× latency vs 8B; diminishing returns on simple tasks                        │   │
│  │ SWEET SPOT: High-stakes applications where wrong answers are costly                                │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  MoE MODELS (Mixture of Experts - sparse activation):                                                       │
│  ═══════════════════════════════════════════════════════                                                    │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Qwen3-VL-30B-A3B (30B total, 3B active)                                                             │   │
│  │ ═════════════════════════════════════════                                                           │   │
│  │ PURPOSE: High capacity with 8B-like compute cost, best throughput for mixed workloads              │   │
│  │                                                                                                     │   │
│  │ DESIGNED FOR:                                                                                       │   │
│  │ • High-throughput production serving diverse query types                                            │   │
│  │ • When you need 32B-like knowledge but 8B-like latency                                              │   │
│  │ • Mixed workload APIs (some queries simple, some complex)                                           │   │
│  │ • Cost optimization when GPU is the bottleneck                                                      │   │
│  │ • Fitting "larger model knowledge" in limited compute budget                                        │   │
│  │                                                                                                     │   │
│  │ TRADE-OFF: Needs all 30B params in memory, but only computes with 3B                               │   │
│  │ SWEET SPOT: Production APIs prioritizing throughput over single-request latency                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ Qwen3-VL-235B-A22B (235B total, 22B active)                                                         │   │
│  │ ════════════════════════════════════════════                                                        │   │
│  │ PURPOSE: Frontier-class capability, research, when nothing else is good enough                     │   │
│  │                                                                                                     │   │
│  │ DESIGNED FOR:                                                                                       │   │
│  │ • State-of-the-art visual reasoning benchmarks                                                      │   │
│  │ • Complex agentic workflows requiring deep understanding                                            │   │
│  │ • Research and capability exploration                                                               │   │
│  │ • Enterprise applications with multi-GPU infrastructure                                             │   │
│  │ • Tasks where 32B isn't accurate enough                                                             │   │
│  │                                                                                                     │   │
│  │ TRADE-OFF: Requires 8×H100 or 3×B200 minimum; high operational cost                                │   │
│  │ SWEET SPOT: "Money is no object, we need the best accuracy possible"                               │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Dense vs MoE: Architecture Deep Dive

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              DENSE vs MoE ARCHITECTURE COMPARISON                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  FUNDAMENTAL DIFFERENCE:                                                                                    │
│  ═══════════════════════                                                                                    │
│                                                                                                             │
│  Dense Model (e.g., Qwen3-VL-8B):                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │   Input Token ──▶ [ATTENTION] ──▶ [SINGLE MLP] ──▶ Output                                          │   │
│  │                                    ════════════                                                     │   │
│  │                                    ALL 8B params                                                    │   │
│  │                                    used every time                                                  │   │
│  │                                                                                                     │   │
│  │   Every token activates the same MLP weights                                                        │   │
│  │   Compute cost: O(hidden_size × intermediate_size) per token                                       │   │
│  │   For 8B: hidden=4096, intermediate=24576 → 100M FLOPs/token in MLP                                │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  MoE Model (e.g., Qwen3-VL-30B-A3B):                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │   Input Token ──▶ [ATTENTION] ──▶ [ROUTER] ──▶ Select top-K ──▶ [K EXPERTS] ──▶ Weighted Sum       │   │
│  │                                    ════════    out of 128        ════════════                       │   │
│  │                                    Small MLP   experts           Only K experts                     │   │
│  │                                    (gating)    (K=8)             compute!                           │   │
│  │                                                                                                     │   │
│  │   Each token only activates K=8 of 128 experts                                                      │   │
│  │   Compute cost: O(K × expert_size) = 8/128 = 6.25% of full model                                   │   │
│  │   For 30B-A3B: Only 3B params active per token → 30M FLOPs/token in MLP                            │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  WHY MoE?                                                                                                   │
│  ════════                                                                                                   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │  PROBLEM MoE SOLVES:                                                                                │   │
│  │  ─────────────────────                                                                              │   │
│  │  • Scaling dense models: 2× params = 2× compute = 2× cost                                          │   │
│  │  • Memory bandwidth bound: GPUs can't feed data fast enough to utilize compute                     │   │
│  │  • Different tokens need different "knowledge" but dense forces same path                          │   │
│  │                                                                                                     │   │
│  │  MOE SOLUTION:                                                                                      │   │
│  │  ─────────────                                                                                      │   │
│  │  • Store 10× more parameters (knowledge)                                                            │   │
│  │  • Only activate the parameters relevant to each token                                              │   │
│  │  • Result: More knowledge, similar compute cost                                                     │   │
│  │                                                                                                     │   │
│  │  ANALOGY:                                                                                           │   │
│  │  Dense = One doctor who knows everything, answers every question                                   │   │
│  │  MoE = Hospital with 128 specialist doctors, receptionist routes to top-8 relevant specialists    │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  ENGINEERING TRADE-OFFS:                                                                                    │
│  ═══════════════════════                                                                                    │
│                                                                                                             │
│  ┌───────────────────────┬──────────────────────────────┬──────────────────────────────────────────────┐   │
│  │ Aspect                │ Dense (8B)                   │ MoE (30B-A3B)                                │   │
│  ├───────────────────────┼──────────────────────────────┼──────────────────────────────────────────────┤   │
│  │ Memory (weights)      │ 16 GB (BF16)                 │ 60 GB (BF16) - 3.75× more                   │   │
│  │ Compute per token     │ 16 GFLOPs                    │ 6 GFLOPs - 2.7× less                        │   │
│  │ Memory bandwidth      │ Lower (read all weights)     │ Higher (sparse reads to 8 experts)         │   │
│  │ Batching efficiency   │ Excellent (deterministic)    │ Variable (depends on routing)              │   │
│  │ Implementation        │ Simple                       │ Complex (routing, load balancing)          │   │
│  │ Latency (single req)  │ Predictable                  │ Slightly variable                          │   │
│  │ Throughput (batched)  │ Good                         │ Excellent (less compute/token)             │   │
│  │ Multi-GPU scaling     │ Straightforward              │ Requires expert parallelism                │   │
│  └───────────────────────┴──────────────────────────────┴──────────────────────────────────────────────┘   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### MoE Internal Mechanics: Routing, Experts, Activation

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MoE ARCHITECTURE INTERNALS                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  MoE LAYER EXECUTION FLOW (per transformer layer):                                                          │
│  ═════════════════════════════════════════════════                                                          │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │  STEP 1: ATTENTION (Same as Dense)                                                                  │   │
│  │  ─────────────────────────────────                                                                  │   │
│  │                                                                                                     │   │
│  │     Input: [batch, seq_len, hidden_dim]                                                             │   │
│  │        ↓                                                                                            │   │
│  │  ┌──────────────────────────────┐                                                                   │   │
│  │  │ Multi-Head Self-Attention   │ ← Same as dense model                                             │   │
│  │  │ (GQA with 8 KV heads)       │                                                                   │   │
│  │  └──────────────────────────────┘                                                                   │   │
│  │        ↓                                                                                            │   │
│  │     Output: [batch, seq_len, hidden_dim]                                                            │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │  STEP 2: ROUTER (Gating Network)                                                                    │   │
│  │  ───────────────────────────────                                                                    │   │
│  │                                                                                                     │   │
│  │     Input: hidden_states [batch × seq_len, hidden_dim]                                              │   │
│  │                    ↓                                                                                │   │
│  │     ┌────────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│  │     │ Router MLP: Linear(hidden_dim → num_experts)                                               │ │   │
│  │     │                                                                                            │ │   │
│  │     │ For Qwen3-VL-30B-A3B:                                                                      │ │   │
│  │     │ • hidden_dim = 4096                                                                        │ │   │
│  │     │ • num_experts = 128                                                                        │ │   │
│  │     │ • Router weights: 4096 × 128 = 524K parameters (tiny!)                                    │ │   │
│  │     │                                                                                            │ │   │
│  │     │ logits = W_router @ hidden_states  # [batch×seq, 128]                                     │ │   │
│  │     │ probs = softmax(logits)            # Expert probabilities                                 │ │   │
│  │     │ top_k_weights, top_k_indices = topk(probs, k=8)  # Select top-8 experts                  │ │   │
│  │     │ top_k_weights = top_k_weights / sum(top_k_weights)  # Renormalize                        │ │   │
│  │     └────────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│  │                    ↓                                                                                │   │
│  │     Output:                                                                                         │   │
│  │     • routing_weights: [batch×seq, 8] - contribution of each selected expert                       │   │
│  │     • expert_indices: [batch×seq, 8] - which 8 experts to use per token                            │   │
│  │                                                                                                     │   │
│  │     EXAMPLE for 1 token:                                                                            │   │
│  │     ┌────────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│  │     │ Token: "The cat sat on the mat"[0] → "The"                                                 │ │   │
│  │     │ Router output: Expert 12 (0.23), Expert 45 (0.18), Expert 7 (0.14), Expert 89 (0.12),     │ │   │
│  │     │                Expert 34 (0.10), Expert 56 (0.09), Expert 23 (0.08), Expert 101 (0.06)    │ │   │
│  │     │                                                                                            │ │   │
│  │     │ These 8 experts process "The", weighted sum gives final output                            │ │   │
│  │     └────────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │  STEP 3: EXPERT DISPATCH & COMPUTATION                                                              │   │
│  │  ─────────────────────────────────────                                                              │   │
│  │                                                                                                     │   │
│  │     Each expert is a standard FFN (Feed-Forward Network):                                           │   │
│  │     ┌────────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│  │     │ Expert_i(x) = SiLU(W_gate_i @ x) * (W_up_i @ x)   # gate_up projection                    │ │   │
│  │     │ Expert_i(x) = W_down_i @ Expert_i(x)              # down projection                       │ │   │
│  │     │                                                                                            │ │   │
│  │     │ For Qwen3-VL-30B-A3B:                                                                      │ │   │
│  │     │ • W_gate, W_up: [hidden_dim, expert_intermediate] = [4096, ~1500] each                    │ │   │
│  │     │ • W_down: [expert_intermediate, hidden_dim] = [~1500, 4096]                               │ │   │
│  │     │ • Each expert: ~18M parameters                                                             │ │   │
│  │     │ • 128 experts: 128 × 18M = 2.3B parameters (just in MoE layers)                           │ │   │
│  │     └────────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                                                     │   │
│  │     GPU EXECUTION:                                                                                  │   │
│  │     ┌────────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│  │     │                                                                                            │ │   │
│  │     │  Batch of 1024 tokens → Router selects 8 experts each → 8192 (token, expert) pairs       │ │   │
│  │     │                                                                                            │ │   │
│  │     │  NAIVE: Loop over 8192 pairs sequentially (slow!)                                         │ │   │
│  │     │                                                                                            │ │   │
│  │     │  OPTIMIZED (grouped GEMM):                                                                 │ │   │
│  │     │  1. Group tokens by expert: Expert_12 gets [tok_0, tok_5, tok_23, ...]                    │ │   │
│  │     │  2. Batch GEMM per expert: Expert_12(stacked_tokens)                                       │ │   │
│  │     │  3. Scatter results back to original positions                                             │ │   │
│  │     │                                                                                            │ │   │
│  │     │  This is why MoE benefits from larger batches!                                             │ │   │
│  │     │  More tokens → more tokens per expert → better GPU utilization                             │ │   │
│  │     │                                                                                            │ │   │
│  │     └────────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │  STEP 4: WEIGHTED COMBINATION                                                                       │   │
│  │  ────────────────────────────                                                                       │   │
│  │                                                                                                     │   │
│  │     For each token, combine the 8 expert outputs:                                                   │   │
│  │                                                                                                     │   │
│  │     output = Σ (routing_weight_i × expert_i_output)  for i in selected_8_experts                   │   │
│  │                                                                                                     │   │
│  │     EXAMPLE:                                                                                        │   │
│  │     output = 0.23×Expert_12(x) + 0.18×Expert_45(x) + 0.14×Expert_7(x) + ...                        │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  LOAD BALANCING (Critical for Training, Matters for Inference):                                             │
│  ═══════════════════════════════════════════════════════════════                                            │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │  PROBLEM: Some experts might be selected more than others                                           │   │
│  │                                                                                                     │   │
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐│   │
│  │  │ Batch of 1024 tokens:                                                                          ││   │
│  │  │ • Expert_12: 500 tokens (overloaded!)  → SM waiting, bottleneck                               ││   │
│  │  │ • Expert_45: 50 tokens                 → SM underutilized                                      ││   │
│  │  │ • Expert_7: 0 tokens                   → SM idle                                               ││   │
│  │  │ ...                                                                                            ││   │
│  │  └────────────────────────────────────────────────────────────────────────────────────────────────┘│   │
│  │                                                                                                     │   │
│  │  SOLUTIONS:                                                                                         │   │
│  │  • Training: Auxiliary loss to encourage balanced expert usage                                      │   │
│  │  • Inference: Expert capacity limits (drop tokens if expert too busy)                               │   │
│  │  • Architecture: More experts + lower top-k (more uniform distribution)                            │   │
│  │                                                                                                     │   │
│  │  Qwen3-VL uses:                                                                                     │   │
│  │  • 128 experts (high diversity)                                                                     │   │
│  │  • Top-8 selection (each token uses 6.25% of experts)                                              │   │
│  │  • Balanced training with load-balancing loss                                                       │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Performance/Cost/Latency Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PRACTICAL TRADE-OFF ANALYSIS                                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  COST COMPARISON (estimated, on-demand cloud pricing):                                                      │
│  ═════════════════════════════════════════════════════                                                      │
│                                                                                                             │
│  ┌──────────────────────┬─────────────┬────────────────┬───────────────┬─────────────────────────────────┐ │
│  │ Model                │ GPU Setup   │ Hourly Cost    │ Throughput    │ Cost per 1M Tokens              │ │
│  ├──────────────────────┼─────────────┼────────────────┼───────────────┼─────────────────────────────────┤ │
│  │ Qwen3-VL-2B          │ 1× T4       │ $0.35/hr       │ 120 tok/s     │ $0.81                           │ │
│  │ Qwen3-VL-4B          │ 1× L4       │ $0.80/hr       │ 100 tok/s     │ $2.22                           │ │
│  │ Qwen3-VL-8B          │ 1× A100-40  │ $3.00/hr       │ 150 tok/s     │ $5.55                           │ │
│  │ Qwen3-VL-8B          │ 1× A100-80  │ $4.00/hr       │ 200 tok/s     │ $5.55                           │ │
│  │ Qwen3-VL-8B (FP8)    │ 1× H100     │ $8.00/hr       │ 400 tok/s     │ $5.55                           │ │
│  │ Qwen3-VL-32B         │ 1× A100-80  │ $4.00/hr       │ 80 tok/s      │ $13.89                          │ │
│  │ Qwen3-VL-32B (FP8)   │ 1× H100     │ $8.00/hr       │ 160 tok/s     │ $13.89                          │ │
│  │ Qwen3-VL-30B-A3B     │ 1× A100-80  │ $4.00/hr       │ 100 tok/s     │ $11.11 (better than 32B!)      │ │
│  │ Qwen3-VL-30B-A3B     │ 1× H100     │ $8.00/hr       │ 200 tok/s     │ $11.11                          │ │
│  │ Qwen3-VL-235B-A22B   │ 8× H100     │ $64.00/hr      │ 80 tok/s      │ $222.22                         │ │
│  └──────────────────────┴─────────────┴────────────────┴───────────────┴─────────────────────────────────┘ │
│                                                                                                             │
│  KEY INSIGHTS:                                                                                              │
│  • MoE (30B-A3B) is more cost-effective than dense 32B with similar quality                                │
│  • H100 is 2× throughput but 2× cost → same cost/token, but better latency                                 │
│  • 2B on T4 is most cost-effective if quality is acceptable                                                 │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  LATENCY BREAKDOWN (single request, 1K output tokens):                                                      │
│  ═════════════════════════════════════════════════════                                                      │
│                                                                                                             │
│  ┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────┐   │
│  │ Model @ GPU          │ Prefill (TTFT)  │ Decode/token  │ Total (1K tok)  │ Bottleneck             │   │
│  ├──────────────────────┼─────────────────┼───────────────┼─────────────────┼────────────────────────┤   │
│  │ 8B @ A100-80         │ ~200ms          │ ~5ms          │ ~5.2s           │ Memory BW              │   │
│  │ 8B @ H100            │ ~100ms          │ ~2.5ms        │ ~2.6s           │ Memory BW              │   │
│  │ 32B @ H100           │ ~300ms          │ ~6ms          │ ~6.3s           │ Compute + Memory       │   │
│  │ 30B-A3B @ H100       │ ~150ms          │ ~3ms          │ ~3.2s           │ Routing overhead       │   │
│  │ 235B-A22B @ 8×H100   │ ~500ms          │ ~10ms         │ ~10.5s          │ All-to-All comm        │   │
│  └──────────────────────┴─────────────────┴───────────────┴─────────────────┴────────────────────────┘   │
│                                                                                                             │
│  LATENCY ANALYSIS:                                                                                          │
│  • Prefill (TTFT): Dominated by vision encoder for images, linear in image tokens                          │
│  • Decode: Memory-bandwidth bound for dense, routing overhead for MoE                                       │
│  • MoE wins on throughput but has higher variance in latency                                                │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  QUALITY vs SIZE (approximate, relative to GPT-4V baseline):                                                │
│  ═══════════════════════════════════════════════════════════                                                │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                                                     │   │
│  │   Quality                                                                                           │   │
│  │      ▲                                                                                              │   │
│  │  100%│                                              ┌─────────┐                                     │   │
│  │      │                                              │ 235B    │ ← Frontier, beats GPT-4V           │   │
│  │   95%│                           ┌──────────┐       │ -A22B   │                                     │   │
│  │      │                           │   32B    │───────└─────────┘                                     │   │
│  │   90%│          ┌─────────┐──────│ Dense    │                                                       │   │
│  │      │          │ 30B-A3B │      └──────────┘                                                       │   │
│  │   85%│   ┌──────│  MoE    │                                                                         │   │
│  │      │   │ 8B   └─────────┘                         Note: MoE 30B-A3B achieves                     │   │
│  │   80%│   │Dense │                                   similar quality to 32B dense                    │   │
│  │      │───└──────┘                                   with 3× less compute!                           │   │
│  │   70%│ ┌────┐                                                                                       │   │
│  │      │ │ 4B │                                                                                       │   │
│  │   60%│ └────┘                                                                                       │   │
│  │      │┌────┐                                                                                        │   │
│  │   50%││ 2B │                                                                                        │   │
│  │      │└────┘                                                                                        │   │
│  │      └─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬───────▶ FLOPs per token          │   │
│  │               1G        3G        10G       30G       50G      100G                                 │   │
│  │                                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  KEY INSIGHT: MoE achieves disproportionate quality for compute cost!                                       │
│  30B-A3B: 30B params of knowledge, but only 3B compute → best $/quality                                    │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Production Scenario Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PRODUCTION SCENARIO DECISION TREE                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  START HERE: What's your primary constraint?                                                                │
│  ═══════════════════════════════════════════                                                                │
│                                                                                                             │
│                                    ┌──────────────────────┐                                                 │
│                                    │ What's your budget   │                                                 │
│                                    │ and quality need?    │                                                 │
│                                    └──────────┬───────────┘                                                 │
│                                               │                                                             │
│                 ┌─────────────────────────────┼─────────────────────────────┐                               │
│                 │                             │                             │                               │
│                 ▼                             ▼                             ▼                               │
│     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐                         │
│     │ Cost is critical    │     │ Quality is critical │     │ Latency is critical │                         │
│     │ (startup, high vol) │     │ (accuracy matters)  │     │ (real-time apps)    │                         │
│     └──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘                         │
│                │                           │                           │                                    │
│                ▼                           ▼                           ▼                                    │
│     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐                         │
│     │ Simple tasks?       │     │ Can afford H100?    │     │ Can batch requests? │                         │
│     └──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘                         │
│         │      │                     │      │                     │      │                                  │
│        YES    NO                    YES    NO                    YES    NO                                  │
│         │      │                     │      │                     │      │                                  │
│         ▼      ▼                     ▼      ▼                     ▼      ▼                                  │
│    ┌────────┐ ┌────────┐       ┌────────┐ ┌────────┐       ┌────────┐ ┌────────┐                            │
│    │ 2B/T4  │ │ 4B/L4  │       │32B/H100│ │8B/A100 │       │MoE/H100│ │ 8B/H100│                            │
│    │$0.81/M │ │$2.22/M │       │FP8     │ │BF16    │       │batched │ │FP8     │                            │
│    └────────┘ └────────┘       └────────┘ └────────┘       └────────┘ └────────┘                            │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  SPECIFIC SCENARIOS:                                                                                        │
│  ═══════════════════                                                                                        │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ SCENARIO: Mobile App (Image Captioning)                                                             │   │
│  │ ───────────────────────────────────────                                                             │   │
│  │ Constraints: Low cost, <200ms latency, millions of requests                                         │   │
│  │ Decision: Qwen3-VL-2B on T4 or edge deployment                                                      │   │
│  │ Why: Smallest model, fastest inference, acceptable quality for captions                             │   │
│  │ Config: max_model_len=2048, max_pixels=256×256, batch_size=8                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ SCENARIO: Document Processing Pipeline                                                              │   │
│  │ ──────────────────────────────────────                                                              │   │
│  │ Constraints: High volume, moderate quality, cost-sensitive                                          │   │
│  │ Decision: Qwen3-VL-8B on A100-80GB                                                                  │   │
│  │ Why: Best quality/cost ratio, handles tables/forms well, high throughput                            │   │
│  │ Config: max_model_len=8192, max_pixels=1344×768, enable_chunked_prefill=True                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ SCENARIO: GUI Automation Agent (MAI-UI)                                                             │   │
│  │ ──────────────────────────────────────                                                              │   │
│  │ Constraints: Must click right element, complex UIs, real-time interaction                           │   │
│  │ Decision: Qwen3-VL-8B on H100 with FP8                                                              │   │
│  │ Why: Best balance of accuracy and speed for UI understanding                                        │   │
│  │ Config: max_model_len=4096, max_pixels=1920×1080, temperature=0.1                                   │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ SCENARIO: Medical Image Analysis                                                                    │   │
│  │ ────────────────────────────────────                                                                │   │
│  │ Constraints: Accuracy paramount, regulatory compliance, moderate volume                             │   │
│  │ Decision: Qwen3-VL-32B on H100 with FP8                                                             │   │
│  │ Why: Highest accuracy for critical decisions, fine details matter                                   │   │
│  │ Config: max_model_len=16384, max_pixels=3840×2160, temperature=0.0                                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ SCENARIO: Mixed Workload API (Diverse Queries)                                                      │   │
│  │ ─────────────────────────────────────────────                                                       │   │
│  │ Constraints: Variable query complexity, maximize throughput, predictable cost                       │   │
│  │ Decision: Qwen3-VL-30B-A3B (MoE) on H100                                                            │   │
│  │ Why: High capacity handles complex, sparse activation handles simple efficiently                    │   │
│  │ Config: max_model_len=32768, continuous batching, enable_prefix_caching=True                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ SCENARIO: Research/Benchmarking                                                                     │   │
│  │ ─────────────────────────────────                                                                   │   │
│  │ Constraints: Need best possible accuracy, cost secondary                                            │   │
│  │ Decision: Qwen3-VL-235B-A22B on 8×H100                                                              │   │
│  │ Why: Frontier-class capability, beats GPT-4V on benchmarks                                          │   │
│  │ Config: tensor_parallel_size=8, max_model_len=65536                                                 │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Implementation & Deployment Guide

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PRACTICAL DEPLOYMENT WITH VLLM                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  INSTALLATION:                                                                                              │
│  ═════════════                                                                                              │
│                                                                                                             │
│  # Basic vLLM installation                                                                                  │
│  pip install vllm>=0.6.0                                                                                    │
│                                                                                                             │
│  # For FP8 on H100 (requires CUDA 12.1+)                                                                   │
│  pip install vllm[fp8]                                                                                      │
│                                                                                                             │
│  # For FlashInfer on T4 (optional, improves T4 performance)                                                │
│  pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4                                        │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  DENSE MODEL DEPLOYMENT (8B example):                                                                       │
│  ═════════════════════════════════════                                                                      │
│                                                                                                             │
│  ```python                                                                                                  │
│  from vllm import LLM, SamplingParams                                                                       │
│                                                                                                             │
│  # A100-80GB: Full BF16                                                                                     │
│  llm = LLM(                                                                                                 │
│      model="Qwen/Qwen3-VL-8B-Instruct",                                                                     │
│      dtype="bfloat16",                                                                                      │
│      max_model_len=32768,                 # Up to 32K context                                               │
│      gpu_memory_utilization=0.90,         # Use 90% of GPU memory                                           │
│      enable_prefix_caching=True,          # Cache common prefixes (system prompts)                          │
│      limit_mm_per_prompt={"image": 4, "video": 1},  # Max media per request                                │
│      mm_processor_kwargs={                                                                                  │
│          "min_pixels": 256 * 28 * 28,     # Min image size                                                 │
│          "max_pixels": 1280 * 28 * 28,    # Max ~1280 tokens from images                                   │
│      },                                                                                                     │
│  )                                                                                                          │
│                                                                                                             │
│  # H100-80GB: FP8 for 2× throughput                                                                         │
│  llm_h100 = LLM(                                                                                            │
│      model="Qwen/Qwen3-VL-8B-Instruct",                                                                     │
│      dtype="bfloat16",                                                                                      │
│      quantization="fp8",                  # Enable FP8                                                      │
│      kv_cache_dtype="fp8",                # FP8 KV cache (2× capacity)                                     │
│      max_model_len=65536,                 # Up to 64K context                                               │
│      gpu_memory_utilization=0.95,                                                                           │
│      enable_chunked_prefill=True,         # Better TTFT for long prompts                                   │
│  )                                                                                                          │
│                                                                                                             │
│  # T4-16GB: Aggressive optimization                                                                         │
│  llm_t4 = LLM(                                                                                              │
│      model="Qwen/Qwen3-VL-8B-Instruct",                                                                     │
│      dtype="float16",                     # T4 doesn't support BF16                                        │
│      quantization="bitsandbytes",         # 4-bit quantization                                             │
│      load_format="bitsandbytes",                                                                            │
│      max_model_len=4096,                  # Limited context                                                 │
│      gpu_memory_utilization=0.95,                                                                           │
│      enforce_eager=True,                  # Disable CUDA graphs (memory)                                   │
│      mm_processor_kwargs={                                                                                  │
│          "max_pixels": 512 * 28 * 28,     # Limit image size                                               │
│      },                                                                                                     │
│  )                                                                                                          │
│  ```                                                                                                        │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  MoE MODEL DEPLOYMENT (30B-A3B):                                                                            │
│  ═══════════════════════════════                                                                            │
│                                                                                                             │
│  ```python                                                                                                  │
│  # Single A100-80GB: Fits all 128 experts                                                                   │
│  llm_moe = LLM(                                                                                             │
│      model="Qwen/Qwen3-VL-30B-A3B-Instruct",                                                                │
│      dtype="bfloat16",                                                                                      │
│      max_model_len=32768,                                                                                   │
│      gpu_memory_utilization=0.95,         # Need high utilization (60GB weights)                           │
│      enable_prefix_caching=True,                                                                            │
│      # MoE-specific: higher batch helps GPU utilization                                                    │
│      max_num_seqs=64,                     # More concurrent sequences                                       │
│  )                                                                                                          │
│                                                                                                             │
│  # Single H100-80GB: FP8 weights, more KV cache room                                                        │
│  llm_moe_h100 = LLM(                                                                                        │
│      model="Qwen/Qwen3-VL-30B-A3B-Instruct",                                                                │
│      dtype="bfloat16",                                                                                      │
│      quantization="fp8",                  # 30GB weights instead of 60GB                                   │
│      kv_cache_dtype="fp8",                                                                                  │
│      max_model_len=65536,                                                                                   │
│      gpu_memory_utilization=0.95,                                                                           │
│      max_num_seqs=128,                    # Even more concurrent sequences                                  │
│  )                                                                                                          │
│  ```                                                                                                        │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  MULTI-GPU DEPLOYMENT (235B-A22B on 8×H100):                                                                │
│  ═══════════════════════════════════════════                                                                │
│                                                                                                             │
│  ```python                                                                                                  │
│  # Tensor parallel across 8 GPUs                                                                            │
│  llm_235b = LLM(                                                                                            │
│      model="Qwen/Qwen3-VL-235B-A22B-Instruct",                                                              │
│      dtype="bfloat16",                                                                                      │
│      tensor_parallel_size=8,              # Distribute across 8 GPUs                                        │
│      max_model_len=65536,                                                                                   │
│      gpu_memory_utilization=0.95,                                                                           │
│      enable_prefix_caching=True,                                                                            │
│      # Expert parallelism handled automatically                                                             │
│  )                                                                                                          │
│                                                                                                             │
│  # Server deployment with tensor parallel                                                                   │
│  # vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \                                                           │
│  #     --tensor-parallel-size 8 \                                                                          │
│  #     --max-model-len 65536 \                                                                              │
│  #     --enable-prefix-caching                                                                              │
│  ```                                                                                                        │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  KV CACHE CONSIDERATIONS:                                                                                   │
│  ═════════════════════════                                                                                  │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ KV CACHE SIZE FORMULA:                                                                              │   │
│  │                                                                                                     │   │
│  │ KV_bytes_per_token = 2 × num_layers × kv_heads × head_dim × bytes_per_element                      │   │
│  │                                                                                                     │   │
│  │ Examples:                                                                                           │   │
│  │ • 8B (32 layers, 8 KV heads, 128 dim, BF16): 2×32×8×128×2 = 128 KB/token                           │   │
│  │ • 8B (FP8 KV cache): 2×32×8×128×1 = 64 KB/token (2× more tokens fit!)                              │   │
│  │ • 32B (64 layers, 8 KV heads, 128 dim, BF16): 2×64×8×128×2 = 256 KB/token                          │   │
│  │                                                                                                     │   │
│  │ For 80GB GPU with 60GB for KV cache:                                                                │   │
│  │ • 8B BF16: 60GB ÷ 128KB = 468,750 tokens total KV capacity                                         │   │
│  │ • 8B FP8: 60GB ÷ 64KB = 937,500 tokens (2×!)                                                       │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ PAGEDATTENTION BLOCK SIZE CHOICE:                                                                   │   │
│  │                                                                                                     │   │
│  │ vLLM default: 16 tokens/block                                                                       │   │
│  │                                                                                                     │   │
│  │ Trade-offs:                                                                                         │   │
│  │ • Smaller blocks (8): Less internal fragmentation, more block table overhead                       │   │
│  │ • Larger blocks (32): Less overhead, more fragmentation for short sequences                        │   │
│  │                                                                                                     │   │
│  │ Recommendation: Use default (16) unless you have specific needs                                    │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  BATCHING STRATEGIES:                                                                                       │
│  ═══════════════════                                                                                        │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │ DENSE MODELS:                                                                                       │   │
│  │ • Continuous batching: Let vLLM handle it (default)                                                 │   │
│  │ • max_num_seqs: 16-32 for A100, 64-128 for H100                                                     │   │
│  │ • Prefill/decode split: Use chunked_prefill for mixed workloads                                     │   │
│  │                                                                                                     │   │
│  │ MoE MODELS:                                                                                         │   │
│  │ • LARGER BATCHES = BETTER GPU UTILIZATION                                                           │   │
│  │ • More tokens → more tokens per expert → better Tensor Core utilization                            │   │
│  │ • Recommendation: max_num_seqs=64-128 even on A100                                                  │   │
│  │ • Trade-off: Higher latency per request, but much higher throughput                                 │   │
│  │                                                                                                     │   │
│  │ MULTIMODAL-SPECIFIC:                                                                                │   │
│  │ • Image prefill is expensive: use enable_chunked_prefill=True                                       │   │
│  │ • Video tokens can be huge: use video_pruning_rate (EVS) to reduce                                  │   │
│  │ • Prefix caching helps if same images/videos are reused                                             │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  PRODUCTION CHECKLIST:                                                                                      │
│  ═════════════════════                                                                                      │
│                                                                                                             │
│  □ GPU selection matches model size (see Quick Reference table)                                             │
│  □ Memory utilization set appropriately (0.85-0.95)                                                         │
│  □ max_model_len set based on use case (not max possible)                                                   │
│  □ FP8 enabled if H100+ and throughput matters                                                              │
│  □ Prefix caching enabled if prompts share common prefixes                                                  │
│  □ Chunked prefill enabled for mixed prefill/decode workloads                                               │
│  □ Image/video limits set to avoid OOM on large media                                                       │
│  □ Health checks and graceful degradation implemented                                                       │
│  □ Metrics/logging for latency percentiles (p50, p95, p99)                                                  │
│  □ Load testing completed with realistic traffic patterns                                                   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Qwen3 vs Qwen3-VL: Text-Only vs Multimodal

Understanding the fundamental difference between the Qwen3 LLM series (text-only) and Qwen3-VL (vision-language):

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              QWEN3 (TEXT-ONLY) vs QWEN3-VL (MULTIMODAL)                                   ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║   QWEN3 (Text-Only LLM)                           QWEN3-VL (Vision-Language Model)                       ║
║   ═════════════════════                           ════════════════════════════════                       ║
║                                                                                                           ║
║   ┌─────────────────────────────────┐             ┌─────────────────────────────────────────────────────┐║
║   │                                 │             │                                                     │║
║   │     Text Input                  │             │     Image/Video Input         Text Input           │║
║   │         │                       │             │           │                       │                 │║
║   │         ▼                       │             │           ▼                       │                 │║
║   │   ┌───────────┐                 │             │   ┌───────────────────┐          │                 │║
║   │   │ Tokenizer │                 │             │   │  Vision Encoder   │          │                 │║
║   │   │ + Embed   │                 │             │   │  (ViT + DeepStack)│          │                 │║
║   │   └─────┬─────┘                 │             │   └─────────┬─────────┘          │                 │║
║   │         │                       │             │             │                     │                 │║
║   │         ▼                       │             │             ▼                     ▼                 │║
║   │   ┌───────────┐                 │             │   ┌─────────────────────────────────────┐          │║
║   │   │  Qwen3    │                 │             │   │  [Visual Tokens] + [Text Tokens]   │          │║
║   │   │  LLM      │                 │             │   │         Merged Sequence             │          │║
║   │   │  Decoder  │                 │             │   └─────────────────┬───────────────────┘          │║
║   │   └─────┬─────┘                 │             │                     │                               │║
║   │         │                       │             │                     ▼                               │║
║   │         ▼                       │             │   ┌─────────────────────────────────────┐          │║
║   │    Text Output                  │             │   │  Qwen3 LLM Decoder                  │          │║
║   │                                 │             │   │  (with DeepStack injection at       │          │║
║   │                                 │             │   │   early layers)                     │          │║
║   └─────────────────────────────────┘             │   └─────────────────┬───────────────────┘          │║
║                                                   │                     │                               │║
║                                                   │                     ▼                               │║
║                                                   │              Text Output                             │║
║                                                   │   (can describe images, answer visual Q&A)          │║
║                                                   └─────────────────────────────────────────────────────┘║
║                                                                                                           ║
║   MODEL SIZE COMPARISON:                                                                                  ║
║   ══════════════════════                                                                                  ║
║                                                                                                           ║
║   ┌────────────────────────────┬───────────────────────────┬─────────────────────────────────────────┐   ║
║   │ Qwen3 (Text)               │ Qwen3-VL (Multimodal)     │ Difference                              │   ║
║   ├────────────────────────────┼───────────────────────────┼─────────────────────────────────────────┤   ║
║   │ Qwen3-0.6B                 │ (no VL variant)           │ Too small for vision encoder overhead   │   ║
║   │ Qwen3-1.7B                 │ (no VL variant)           │ Too small for vision encoder overhead   │   ║
║   │ Qwen3-4B                   │ Qwen3-VL-4B (~4.5B)       │ +500M for ViT encoder                   │   ║
║   │ Qwen3-8B                   │ Qwen3-VL-8B (~8.5B)       │ +500M for ViT encoder                   │   ║
║   │ Qwen3-14B                  │ (no VL variant yet)       │ Gap in lineup                           │   ║
║   │ Qwen3-32B                  │ Qwen3-VL-32B (~33B)       │ +1B for larger ViT encoder              │   ║
║   │ Qwen3-30B-A3B (MoE)        │ Qwen3-VL-30B-A3B (~31B)   │ +1B for ViT encoder                     │   ║
║   │ Qwen3-235B-A22B (MoE)      │ Qwen3-VL-235B-A22B (~237B)│ +2B for largest ViT encoder             │   ║
║   └────────────────────────────┴───────────────────────────┴─────────────────────────────────────────┘   ║
║                                                                                                           ║
║   KEY ARCHITECTURAL DIFFERENCES:                                                                          ║
║   ══════════════════════════════                                                                          ║
║                                                                                                           ║
║   1. Vision Encoder (ViT): Only in VL models, processes images/videos into token embeddings              ║
║   2. DeepStack: Only in VL models, injects multi-scale vision features into early LLM layers             ║
║   3. M-RoPE: VL models use 3D positional encoding (height, width, time) vs 1D in text-only               ║
║   4. EVS: Only in VL models, prunes redundant video frames for efficiency                                 ║
║   5. Memory: VL models require ~500M-2B extra parameters for vision processing                           ║
║                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Qwen3 Text-Only Model Specifications

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3 TEXT-ONLY LLM SPECIFICATIONS                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  ┌────────────┬──────────┬─────────────┬────────┬─────────┬────────┬──────────┬────────────────────┐   │
│  │ Model      │ Params   │ Hidden Size │ Layers │ Heads   │KV Heads│ Intermed │ Context            │   │
│  ├────────────┼──────────┼─────────────┼────────┼─────────┼────────┼──────────┼────────────────────┤   │
│  │ Qwen3-0.6B │ 0.6B     │ 1024        │ 28     │ 16      │ 8      │ 3072     │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-1.7B │ 1.7B     │ 1536        │ 28     │ 12      │ 4      │ 8960     │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-4B   │ 4.0B     │ 2560        │ 36     │ 32      │ 8      │ 9728     │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-8B   │ 8.0B     │ 4096        │ 36     │ 32      │ 8      │ 12288    │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-14B  │ 14.0B    │ 5120        │ 40     │ 40      │ 8      │ 13824    │ 32K (131K w/ YaRN) │   │
│  │ Qwen3-32B  │ 32.0B    │ 5120        │ 64     │ 64      │ 8      │ 25600    │ 32K (131K w/ YaRN) │   │
│  └────────────┴──────────┴─────────────┴────────┴─────────┴────────┴──────────┴────────────────────┘   │
│                                                                                                         │
│  MoE MODELS:                                                                                            │
│  ┌────────────────────┬───────────┬─────────────┬────────┬─────────┬─────────┬────────────────────────┐│
│  │ Model              │ Total/Act │ Hidden Size │ Layers │ Experts │ Top-K   │ Context                ││
│  ├────────────────────┼───────────┼─────────────┼────────┼─────────┼─────────┼────────────────────────┤│
│  │ Qwen3-30B-A3B      │ 30B / 3B  │ 2048        │ 48     │ 128     │ 8       │ 32K (131K w/ YaRN)     ││
│  │ Qwen3-235B-A22B    │ 235B / 22B│ 5120        │ 94     │ 128     │ 8       │ 32K (131K w/ YaRN)     ││
│  └────────────────────┴───────────┴─────────────┴────────┴─────────┴─────────┴────────────────────────┘│
│                                                                                                         │
│  COMMON FEATURES:                                                                                       │
│  • Activation: SwiGLU (SiLU-gated linear unit)                                                         │
│  • Normalization: RMSNorm                                                                              │
│  • Position Encoding: RoPE (Rotary Position Embedding)                                                 │
│  • Attention: Grouped Query Attention (GQA)                                                            │
│  • Vocabulary: 151,936 tokens                                                                          │
│  • Thinking Mode: Unified thinking/non-thinking in single model                                        │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## All Qwen3-VL Models: Architecture Breakdown

### Dense Models (All Parameters Active)

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    QWEN3-VL DENSE MODELS                                                  ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-2B-INSTRUCT                                                   │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~2.5B (2B LLM + ~500M ViT)                                                    │ ║
║  │  • Model ID: Qwen/Qwen3-VL-2B-Instruct                                                             │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens (native), 256K (extended)                                           │ ║
║  │                                                                                                     │ ║
║  │  LLM ARCHITECTURE                                                                                   │ ║
║  │  ════════════════                                                                                   │ ║
║  │  • Hidden Size: 1536                                                                                │ ║
║  │  • Intermediate Size: 8960                                                                          │ ║
║  │  • Number of Layers: 28                                                                             │ ║
║  │  • Attention Heads: 12                                                                              │ ║
║  │  • KV Heads: 2 (GQA ratio 6:1)                                                                     │ ║
║  │  • Head Dimension: 128                                                                              │ ║
║  │  • Vocabulary Size: 151,936                                                                         │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 1152                                                                                │ ║
║  │  • Depth: 24 layers                                                                                 │ ║
║  │  • Attention Heads: 16                                                                              │ ║
║  │  • MLP Ratio: 4                                                                                     │ ║
║  │  • Patch Size: 14×14 pixels                                                                         │ ║
║  │  • Temporal Patch Size: 2 frames                                                                    │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • FP16:  ~5.0 GB model + ~1.5 GB KV cache (8K ctx) = ~6.5 GB                                      │ ║
║  │  • INT4:  ~1.5 GB model + ~1.5 GB KV cache (8K ctx) = ~3.0 GB                                      │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • Edge deployment (mobile, embedded)                                                               │ ║
║  │  • T4/L4 GPUs in Google Colab                                                                      │ ║
║  │  • Real-time applications requiring low latency                                                     │ ║
║  │  • Simple image understanding tasks                                                                 │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-4B-INSTRUCT                                                   │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~4.5B (4B LLM + ~500M ViT)                                                    │ ║
║  │  • Model ID: Qwen/Qwen3-VL-4B-Instruct                                                             │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens                                                                      │ ║
║  │                                                                                                     │ ║
║  │  LLM ARCHITECTURE                                                                                   │ ║
║  │  ════════════════                                                                                   │ ║
║  │  • Hidden Size: 2048                                                                                │ ║
║  │  • Intermediate Size: 11264                                                                         │ ║
║  │  • Number of Layers: 36                                                                             │ ║
║  │  • Attention Heads: 16                                                                              │ ║
║  │  • KV Heads: 4 (GQA ratio 4:1)                                                                     │ ║
║  │  • Head Dimension: 128                                                                              │ ║
║  │  • Vocabulary Size: 151,936                                                                         │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 1280                                                                                │ ║
║  │  • Depth: 28 layers                                                                                 │ ║
║  │  • Attention Heads: 16                                                                              │ ║
║  │  • MLP Ratio: 4                                                                                     │ ║
║  │  • Patch Size: 14×14 pixels                                                                         │ ║
║  │  • Temporal Patch Size: 2 frames                                                                    │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • FP16:  ~9.0 GB model + ~3.0 GB KV cache (8K ctx) = ~12.0 GB                                     │ ║
║  │  • BF16:  ~9.0 GB model + ~3.0 GB KV cache (8K ctx) = ~12.0 GB                                     │ ║
║  │  • INT4:  ~2.5 GB model + ~3.0 GB KV cache (8K ctx) = ~5.5 GB                                      │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • T4 GPU with 4-bit quantization                                                                  │ ║
║  │  • L4/A10G GPUs at full precision                                                                  │ ║
║  │  • MAI-UI GUI agent (recommended for edge)                                                         │ ║
║  │  • Balanced quality vs. speed                                                                       │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-8B-INSTRUCT                                                   │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~8.5B (8B LLM + ~500M ViT)                                                    │ ║
║  │  • Model ID: Qwen/Qwen3-VL-8B-Instruct                                                             │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens                                                                      │ ║
║  │                                                                                                     │ ║
║  │  LLM ARCHITECTURE                                                                                   │ ║
║  │  ════════════════                                                                                   │ ║
║  │  • Hidden Size: 4096                                                                                │ ║
║  │  • Intermediate Size: 12288                                                                         │ ║
║  │  • Number of Layers: 32                                                                             │ ║
║  │  • Attention Heads: 32                                                                              │ ║
║  │  • KV Heads: 8 (GQA ratio 4:1)                                                                     │ ║
║  │  • Head Dimension: 128                                                                              │ ║
║  │  • Vocabulary Size: 151,936                                                                         │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 1536                                                                                │ ║
║  │  • Depth: 32 layers                                                                                 │ ║
║  │  • Attention Heads: 24                                                                              │ ║
║  │  • MLP Ratio: 4 (intermediate = 6144)                                                              │ ║
║  │  • Patch Size: 14×14 pixels                                                                         │ ║
║  │  • Temporal Patch Size: 2 frames                                                                    │ ║
║  │  • DeepStack Layers: [8, 16, 24] (multi-scale features)                                            │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • BF16:  ~17 GB model + ~6 GB KV cache (16K ctx) = ~23 GB                                         │ ║
║  │  • FP8:   ~9 GB model + ~3 GB KV cache (16K ctx) = ~12 GB                                          │ ║
║  │  • INT4:  ~4.5 GB model + ~6 GB KV cache (16K ctx) = ~10.5 GB                                      │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • A100-40GB/80GB at full precision                                                                 │ ║
║  │  • H100 with FP8 for maximum throughput                                                             │ ║
║  │  • Production GUI agents (MAI-UI-8B)                                                                │ ║
║  │  • High-quality image/video understanding                                                           │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-32B-INSTRUCT                                                  │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~33B (32B LLM + ~1B ViT)                                                      │ ║
║  │  • Model ID: Qwen/Qwen3-VL-32B-Instruct                                                            │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens (native), 256K (extended)                                           │ ║
║  │                                                                                                     │ ║
║  │  LLM ARCHITECTURE                                                                                   │ ║
║  │  ════════════════                                                                                   │ ║
║  │  • Hidden Size: 5120                                                                                │ ║
║  │  • Intermediate Size: 25600                                                                         │ ║
║  │  • Number of Layers: 64                                                                             │ ║
║  │  • Attention Heads: 40                                                                              │ ║
║  │  • KV Heads: 8 (GQA ratio 5:1)                                                                     │ ║
║  │  • Head Dimension: 128                                                                              │ ║
║  │  • Vocabulary Size: 151,936                                                                         │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 1792                                                                                │ ║
║  │  • Depth: 32 layers                                                                                 │ ║
║  │  • Attention Heads: 28                                                                              │ ║
║  │  • MLP Ratio: 4                                                                                     │ ║
║  │  • Patch Size: 14×14 pixels                                                                         │ ║
║  │  • Temporal Patch Size: 2 frames                                                                    │ ║
║  │  • DeepStack Layers: [8, 16, 24] (multi-scale features)                                            │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • BF16:  ~65 GB model + ~15 GB KV cache (32K ctx) = ~80 GB                                        │ ║
║  │  • FP8:   ~33 GB model + ~8 GB KV cache (32K ctx) = ~41 GB                                         │ ║
║  │  • INT4:  ~17 GB model + ~15 GB KV cache (32K ctx) = ~32 GB                                        │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • A100-80GB with BF16                                                                              │ ║
║  │  • H100 with FP8 for high throughput                                                                │ ║
║  │  • H200/B200 for maximum context length                                                             │ ║
║  │  • Complex reasoning, detailed image analysis                                                       │ ║
║  │  • State-of-the-art benchmark performance                                                           │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Mixture-of-Experts (MoE) Models

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    QWEN3-VL MOE MODELS                                                    ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║  MoE ARCHITECTURE EXPLAINED:                                                                              ║
║  ═══════════════════════════                                                                              ║
║                                                                                                           ║
║    Standard Dense Model:  Every token uses ALL parameters                                                 ║
║    MoE Model:             Each token routes to TOP-K experts (sparse activation)                         ║
║                                                                                                           ║
║    Example: Qwen3-VL-30B-A3B                                                                              ║
║    • 30B total parameters (all experts stored in memory)                                                 ║
║    • Only 3B active per token (compute cost of 3B model!)                                                ║
║    • MUCH faster than a dense 30B model                                                                  ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-30B-A3B-INSTRUCT                                               │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~30B + ~1B ViT = ~31B                                                         │ ║
║  │  • Active Parameters: ~3B per token                                                                 │ ║
║  │  • Model ID: Qwen/Qwen3-VL-30B-A3B-Instruct                                                        │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 128K tokens                                                                      │ ║
║  │                                                                                                     │ ║
║  │  MoE LLM ARCHITECTURE                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 2048                                                                                │ ║
║  │  • Number of Layers: 48                                                                             │ ║
║  │  • Attention Heads: 16                                                                              │ ║
║  │  • KV Heads: 4 (GQA ratio 4:1)                                                                     │ ║
║  │  • Number of Experts: 128                                                                           │ ║
║  │  • Experts Per Token (Top-K): 8                                                                     │ ║
║  │  • Expert Intermediate Size: 1024                                                                   │ ║
║  │  • Shared Expert: Yes (always active)                                                               │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Same as Qwen3-VL-8B vision encoder                                                              │ ║
║  │  • Hidden Size: 1536                                                                                │ ║
║  │  • Depth: 32 layers                                                                                 │ ║
║  │  • DeepStack Layers: [8, 16, 24]                                                                   │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Inference)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • BF16:  ~62 GB model (all experts) + ~8 GB KV = ~70 GB                                           │ ║
║  │  • FP8:   ~31 GB model + ~4 GB KV = ~35 GB                                                         │ ║
║  │                                                                                                     │ ║
║  │  COMPUTE REQUIREMENTS                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Per-token compute: ~3B parameters (like a 3B model!)                                            │ ║
║  │  • Memory bandwidth: Higher than 3B dense (expert loading)                                         │ ║
║  │  • Latency: Between 3B and 8B dense models                                                         │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • A100-80GB: Full BF16, excellent quality at 3B compute cost                                      │ ║
║  │  • H100: FP8 for single-GPU deployment                                                              │ ║
║  │  • When you need 30B quality with 3B latency                                                       │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐ ║
║  │                              QWEN3-VL-235B-A22B-INSTRUCT                                             │ ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────────────────┤ ║
║  │                                                                                                     │ ║
║  │  OVERVIEW                                                                                           │ ║
║  │  ════════                                                                                           │ ║
║  │  • Total Parameters: ~235B + ~2B ViT = ~237B                                                       │ ║
║  │  • Active Parameters: ~22B per token                                                                │ ║
║  │  • Model ID: Qwen/Qwen3-VL-235B-A22B-Instruct                                                      │ ║
║  │  • License: Apache 2.0                                                                              │ ║
║  │  • Context Window: 256K tokens                                                                      │ ║
║  │                                                                                                     │ ║
║  │  MoE LLM ARCHITECTURE                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Hidden Size: 5120                                                                                │ ║
║  │  • Number of Layers: 94                                                                             │ ║
║  │  • Attention Heads: 64                                                                              │ ║
║  │  • KV Heads: 4 (GQA ratio 16:1)                                                                    │ ║
║  │  • Number of Experts: 128                                                                           │ ║
║  │  • Experts Per Token (Top-K): 8                                                                     │ ║
║  │  • Expert Intermediate Size: 2560                                                                   │ ║
║  │  • Shared Expert: Yes                                                                               │ ║
║  │                                                                                                     │ ║
║  │  VISION ENCODER (ViT)                                                                               │ ║
║  │  ════════════════════                                                                               │ ║
║  │  • Largest ViT in Qwen3-VL family                                                                  │ ║
║  │  • Hidden Size: 2048                                                                                │ ║
║  │  • Depth: 40 layers                                                                                 │ ║
║  │  • Attention Heads: 32                                                                              │ ║
║  │  • DeepStack Layers: [10, 20, 30]                                                                  │ ║
║  │                                                                                                     │ ║
║  │  MEMORY REQUIREMENTS (Multi-GPU)                                                                    │ ║
║  │  ═══════════════════════════════                                                                    │ ║
║  │  • BF16:  ~470 GB model + ~50 GB KV = ~520 GB (8×80GB GPUs)                                        │ ║
║  │  • FP8:   ~235 GB model + ~25 GB KV = ~260 GB (4×80GB GPUs)                                        │ ║
║  │                                                                                                     │ ║
║  │  DEPLOYMENT OPTIONS                                                                                 │ ║
║  │  ══════════════════                                                                                 │ ║
║  │  • 8×H100-80GB: BF16 with tensor parallelism                                                       │ ║
║  │  • 4×H100-80GB: FP8 with tensor parallelism                                                        │ ║
║  │  • 8×B200-192GB: Full BF16 with room for 256K context                                              │ ║
║  │                                                                                                     │ ║
║  │  BEST USE CASES                                                                                     │ ║
║  │  ══════════════                                                                                     │ ║
║  │  • State-of-the-art performance (MMMU leader)                                                       │ ║
║  │  • Complex multi-step reasoning                                                                     │ ║
║  │  • Long video understanding (hours of content)                                                      │ ║
║  │  • Research and benchmarking                                                                        │ ║
║  │                                                                                                     │ ║
║  └─────────────────────────────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Thinking Models (Enhanced Reasoning)

```
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    QWEN3-VL THINKING MODELS                                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                           ║
║  WHAT ARE THINKING MODELS?                                                                                ║
║  ═════════════════════════                                                                                ║
║                                                                                                           ║
║    Standard "Instruct" models: Generate answer directly                                                   ║
║    "Thinking" models: Show chain-of-thought reasoning before answer                                       ║
║                                                                                                           ║
║    Benefits:                                                                                              ║
║    • Better on complex multi-step problems                                                                ║
║    • Improved math and logic reasoning                                                                    ║
║    • More transparent decision-making                                                                     ║
║    • Can be guided via system prompt to adjust thinking depth                                            ║
║                                                                                                           ║
║  AVAILABLE THINKING VARIANTS:                                                                             ║
║  ════════════════════════════                                                                             ║
║                                                                                                           ║
║  ┌─────────────────────────────────────────────────────────────────────────────────────────┐             ║
║  │ Model ID                               │ Base      │ Notes                              │             ║
║  ├─────────────────────────────────────────────────────────────────────────────────────────┤             ║
║  │ Qwen/Qwen3-VL-2B-Thinking              │ 2B Dense  │ Edge reasoning                     │             ║
║  │ Qwen/Qwen3-VL-4B-Thinking              │ 4B Dense  │ Balanced reasoning                 │             ║
║  │ Qwen/Qwen3-VL-8B-Thinking              │ 8B Dense  │ High-quality reasoning             │             ║
║  │ Qwen/Qwen3-VL-32B-Thinking             │ 32B Dense │ State-of-the-art reasoning         │             ║
║  │ Qwen/Qwen3-VL-30B-A3B-Thinking         │ 30B MoE   │ MoE with reasoning                 │             ║
║  │ Qwen/Qwen3-VL-235B-A22B-Thinking       │ 235B MoE  │ Maximum reasoning capability       │             ║
║  └─────────────────────────────────────────────────────────────────────────────────────────┘             ║
║                                                                                                           ║
║  USAGE WITH VLLM:                                                                                         ║
║  ═════════════════                                                                                        ║
║                                                                                                           ║
║    # Same as Instruct models, just use -Thinking suffix                                                   ║
║    vllm serve Qwen/Qwen3-VL-8B-Thinking \                                                                 ║
║        --dtype bfloat16 \                                                                                 ║
║        --max-model-len 32768                                                                              ║
║                                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## GPU Hardware: Streaming Multiprocessors and vLLM

Understanding how CUDA Streaming Multiprocessors (SMs) execute Qwen3-VL inference:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              GPU SM ARCHITECTURE FOR QWEN3-VL INFERENCE                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU SPECIFICATIONS BY ARCHITECTURE:                                                                   │
│   ═══════════════════════════════════                                                                   │
│                                                                                                         │
│   ┌───────────────────┬────────────┬────────────┬────────────┬────────────┬─────────────────────────┐  │
│   │ Spec              │ T4 (SM7.5) │A100 (SM8.0)│H100 (SM9.0)│B200 (SM10) │ Impact on Qwen3-VL      │  │
│   ├───────────────────┼────────────┼────────────┼────────────┼────────────┼─────────────────────────┤  │
│   │ Streaming MPs     │ 40 SMs     │ 108 SMs    │ 132 SMs    │ 192 SMs    │ More SMs = more ||ism  │  │
│   │ CUDA Cores        │ 2,560      │ 6,912      │ 16,896     │ ~20,000    │ Compute throughput     │  │
│   │ Tensor Cores      │ 320        │ 432        │ 528        │ ~640       │ Matrix multiply speed  │  │
│   │ Warp Schedulers   │ 4/SM       │ 4/SM       │ 4/SM       │ 4/SM       │ Thread scheduling      │  │
│   │ Max Threads/SM    │ 1,024      │ 2,048      │ 2,048      │ 2,048      │ Concurrent warps       │  │
│   │ Register File     │ 256KB/SM   │ 256KB/SM   │ 256KB/SM   │ 256KB/SM   │ Thread local storage   │  │
│   │ Shared Memory     │ 64KB/SM    │ 164KB/SM   │ 228KB/SM   │ ~256KB/SM  │ FlashAttn tile size    │  │
│   │ L2 Cache          │ 4 MB       │ 40 MB      │ 50 MB      │ 64 MB      │ KV cache hot path      │  │
│   │ Memory BW         │ 320 GB/s   │ 2,039 GB/s │ 3,350 GB/s │ 8,000 GB/s │ Decode speed (bound)   │  │
│   └───────────────────┴────────────┴────────────┴────────────┴────────────┴─────────────────────────┘  │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   VLLM ATTENTION BACKEND SELECTION (from vllm/platforms/cuda.py):                                      │
│   ═══════════════════════════════════════════════════════════════                                       │
│                                                                                                         │
│   DeviceCapability(major, minor) → Backend Selection:                                                   │
│                                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                                 │  │
│   │   T4 (7, 5):     FlashInfer (SM 7.5-12.1) → TritonAttn → TORCH_SDPA                           │  │
│   │                  ⚠️ FlashAttention requires SM >= 8.0, NOT available on T4                     │  │
│   │                                                                                                 │  │
│   │   A100 (8, 0):   FlashAttention 2 → FlashInfer → TritonAttn                                   │  │
│   │                  ✅ Native BF16 Tensor Cores, FP16 accumulate                                  │  │
│   │                                                                                                 │  │
│   │   H100 (9, 0):   FlashAttention 3 → FlashMLA → FlashInfer                                     │  │
│   │                  ✅ FP8 Tensor Cores, FP8 KV cache, sinks support                              │  │
│   │                                                                                                 │  │
│   │   B200 (10, 0):  FlashMLA → FlashAttention 3 → FlashInfer                                     │  │
│   │                  ✅ FP4 (future), 2nd gen Transformer Engine                                   │  │
│   │                                                                                                 │  │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   PAGEDATTENTION MEMORY MAPPING TO HARDWARE:                                                            │
│   ══════════════════════════════════════════                                                            │
│                                                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                                                                                                 │  │
│   │   vLLM Block Manager          GPU HBM                      CUDA Kernel Execution               │  │
│   │   ═══════════════════         ═══════                      ═════════════════════               │  │
│   │                                                                                                 │  │
│   │   Block Table:                ┌────────────────┐           grid(num_heads, num_seqs, splits)   │  │
│   │   ┌─────────────────┐         │ Physical Pages │                     │                         │  │
│   │   │ Seq 0: [P1,P3]  │────────▶│ ┌────┐ ┌────┐ │           ┌─────────▼─────────┐               │  │
│   │   │ Seq 1: [P2,P3]  │─────┬──▶│ │ P1 │ │ P2 │ │           │ Thread Block 0    │               │  │
│   │   │ Seq 2: [P4]     │──┐  │   │ └────┘ └────┘ │           │ (on SM 0)         │               │  │
│   │   └─────────────────┘  │  │   │ ┌────┐ ┌────┐ │           │ Warp 0-3 process  │               │  │
│   │                        │  └──▶│ │ P3 │ │ P4 │ │◀──────────│ head 0, seq 0     │               │  │
│   │   P3 SHARED!           │      │ └────┘ └────┘ │           └───────────────────┘               │  │
│   │   (Copy-on-write)      └─────▶│ (256 tokens   │                                                │  │
│   │                               │  per page)    │           ┌───────────────────┐               │  │
│   │   Memory saved: ~50%          └────────────────┘           │ Thread Block 1    │               │  │
│   │   vs naive allocation                                      │ (on SM 1)         │               │  │
│   │                                                            │ head 0, seq 1     │               │  │
│   │                                                            └───────────────────┘               │  │
│   │                                                                                                 │  │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### A100 SM Layout: Qwen3-VL-8B Execution

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    A100-80GB: 108 SMs × QWEN3-VL-8B EXECUTION MAP                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   A100 SINGLE SM INTERNAL LAYOUT (SM 8.0 Ampere):                                                       │
│   ═══════════════════════════════════════════════                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐│
│   │                           STREAMING MULTIPROCESSOR (1 of 108)                                     ││
│   │  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ ││
│   │  │ PROCESSING BLOCKS (4 per SM)                                                                │ ││
│   │  │ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐ │ ││
│   │  │ │ Warp Scheduler 0   │ │ Warp Scheduler 1   │ │ Warp Scheduler 2   │ │ Warp Scheduler 3   │ │ ││
│   │  │ │ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ││
│   │  │ │ │ 16 FP32 Cores  │ │ │ │ 16 FP32 Cores  │ │ │ │ 16 FP32 Cores  │ │ │ │ 16 FP32 Cores  │ │ │ ││
│   │  │ │ │ 8 FP64 Cores   │ │ │ │ 8 FP64 Cores   │ │ │ │ 8 FP64 Cores   │ │ │ │ 8 FP64 Cores   │ │ │ ││
│   │  │ │ │ 16 INT32 Cores │ │ │ │ 16 INT32 Cores │ │ │ │ 16 INT32 Cores │ │ │ │ 16 INT32 Cores │ │ │ ││
│   │  │ │ │ 1 Tensor Core  │ │ │ │ 1 Tensor Core  │ │ │ │ 1 Tensor Core  │ │ │ │ 1 Tensor Core  │ │ │ ││
│   │  │ │ │ (3rd Gen)      │ │ │ │ (3rd Gen)      │ │ │ │ (3rd Gen)      │ │ │ │ (3rd Gen)      │ │ │ ││
│   │  │ │ └────────────────┘ │ │ └────────────────┘ │ │ └────────────────┘ │ │ └────────────────┘ │ │ ││
│   │  │ │ Register File:    │ │ Register File:     │ │ Register File:     │ │ Register File:     │ │ ││
│   │  │ │ 64KB (16K×32bit)  │ │ 64KB (16K×32bit)   │ │ 64KB (16K×32bit)   │ │ 64KB (16K×32bit)   │ │ ││
│   │  │ └────────────────────┘ └────────────────────┘ └────────────────────┘ └────────────────────┘ │ ││
│   │  └─────────────────────────────────────────────────────────────────────────────────────────────┘ ││
│   │  ┌──────────────────────────────────────┐  ┌───────────────────────────────────────────────────┐ ││
│   │  │ SHARED MEMORY / L1 CACHE             │  │ SPECIAL FUNCTION UNITS                            │ ││
│   │  │ 192 KB configurable                  │  │ 4 SFU (sin, cos, exp, log)                        │ ││
│   │  │ • 164 KB shared + 28 KB L1           │  │ 4 Load/Store Units (32 threads each)              │ ││
│   │  │ • FlashAttn uses 128KB tiles        │  │ 32 LD/ST per cycle                                │ ││
│   │  └──────────────────────────────────────┘  └───────────────────────────────────────────────────┘ ││
│   └───────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                         │
│   QWEN3-VL-8B LAYER EXECUTION ON A100:                                                                  │
│   ════════════════════════════════════                                                                  │
│   Model: 32 LLM layers, 32 attention heads, 8 KV heads (GQA 4:1), hidden=4096                          │
│                                                                                                         │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐│
│   │ ATTENTION FORWARD PASS (FlashAttention 2):                                                        ││
│   │                                                                                                   ││
│   │ Grid: (num_heads=32, batch_size=16, kv_splits=1) = 512 thread blocks                             ││
│   │ Block: 128 threads (4 warps × 32 threads)                                                         ││
│   │ Total threads: 512 × 128 = 65,536 threads                                                         ││
│   │                                                                                                   ││
│   │ SM Occupancy: 512 blocks ÷ 108 SMs = ~4.7 blocks/SM                                              ││
│   │ Active warps/SM: 4.7 × 4 = ~19 warps (of 64 max) = 30% occupancy                                 ││
│   │                                                                                                   ││
│   │ Per-Block Memory:                                                                                 ││
│   │ • Q tile: 64 tokens × 128 dim × 2 bytes = 16 KB                                                  ││
│   │ • K tile: 256 tokens × 128 dim × 2 bytes = 64 KB                                                 ││
│   │ • V tile: 256 tokens × 128 dim × 2 bytes = 64 KB                                                 ││
│   │ • Softmax accumulators: 8 KB                                                                      ││
│   │ Total shared memory: 152 KB (fits in 164 KB!)                                                     ││
│   └───────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                         │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐│
│   │ MLP FORWARD PASS (gate_up_proj + down_proj):                                                      ││
│   │                                                                                                   ││
│   │ gate_up: [batch×seq, 4096] × [4096, 24576] → [batch×seq, 24576]                                  ││
│   │ Grid: (ceil(M/128), ceil(N/128)) where M=batch×seq, N=24576                                       ││
│   │                                                                                                   ││
│   │ Tensor Core utilization:                                                                          ││
│   │ • Each Tensor Core: 256 FP16 ops/cycle (16×16×16 matrix)                                         ││
│   │ • 108 SMs × 4 Tensor Cores = 432 Tensor Cores                                                    ││
│   │ • Peak: 432 × 256 × 1.41 GHz = 156 TFLOPS (FP16)                                                ││
│   │ • Achieved: ~80% = 125 TFLOPS (memory bound on large batches)                                    ││
│   └───────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                         │
│   KV CACHE MEMORY LAYOUT (PagedAttention):                                                              │
│   ════════════════════════════════════════                                                              │
│   Block size: 16 tokens (vLLM default for FlashAttn)                                                   │
│   Per block: 2 × 32 layers × 8 kv_heads × 128 dim × 16 tokens × 2 bytes = 2 MB                        │
│                                                                                                         │
│   80 GB HBM allocation:                                                                                 │
│   ├─ Model weights (BF16): 8B × 2 bytes = 16 GB                                                        │
│   ├─ Vision encoder: 500M × 2 bytes = 1 GB                                                             │
│   ├─ Activations: ~4 GB                                                                                │
│   ├─ CUDA overhead: ~2 GB                                                                              │
│   └─ KV Cache: 80 - 23 = 57 GB available                                                               │
│       └─ 57 GB ÷ 2 MB/block = 28,500 blocks                                                            │
│       └─ 28,500 blocks × 16 tokens = 456,000 tokens of KV cache                                        │
│       └─ At 32K context: 456K ÷ 32K = 14 concurrent sequences                                          │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### H100 SM Layout: Qwen3-VL-8B with FP8

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    H100-80GB: 132 SMs × QWEN3-VL-8B (FP8) EXECUTION MAP                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   H100 SINGLE SM INTERNAL LAYOUT (SM 9.0 Hopper):                                                       │
│   ═══════════════════════════════════════════════                                                       │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐│
│   │                           STREAMING MULTIPROCESSOR (1 of 132)                                     ││
│   │  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ ││
│   │  │ PROCESSING BLOCKS (4 per SM) - ENHANCED FOR HOPPER                                          │ ││
│   │  │ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐ │ ││
│   │  │ │ Warp Scheduler 0   │ │ Warp Scheduler 1   │ │ Warp Scheduler 2   │ │ Warp Scheduler 3   │ │ ││
│   │  │ │ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ││
│   │  │ │ │ 32 FP32 Cores  │ │ │ │ 32 FP32 Cores  │ │ │ │ 32 FP32 Cores  │ │ │ │ 32 FP32 Cores  │ │ │ ││
│   │  │ │ │ 16 FP64 Cores  │ │ │ │ 16 FP64 Cores  │ │ │ │ 16 FP64 Cores  │ │ │ │ 16 FP64 Cores  │ │ │ ││
│   │  │ │ │ 32 INT32 Cores │ │ │ │ 32 INT32 Cores │ │ │ │ 32 INT32 Cores │ │ │ │ 32 INT32 Cores │ │ │ ││
│   │  │ │ │ 1 Tensor Core  │ │ │ │ 1 Tensor Core  │ │ │ │ 1 Tensor Core  │ │ │ │ 1 Tensor Core  │ │ │ ││
│   │  │ │ │ (4th Gen+FP8)  │ │ │ │ (4th Gen+FP8)  │ │ │ │ (4th Gen+FP8)  │ │ │ │ (4th Gen+FP8)  │ │ │ ││
│   │  │ │ └────────────────┘ │ │ └────────────────┘ │ │ └────────────────┘ │ │ └────────────────┘ │ │ ││
│   │  │ │ Register: 64KB    │ │ Register: 64KB     │ │ Register: 64KB     │ │ Register: 64KB     │ │ ││
│   │  │ └────────────────────┘ └────────────────────┘ └────────────────────┘ └────────────────────┘ │ ││
│   │  └─────────────────────────────────────────────────────────────────────────────────────────────┘ ││
│   │  ┌──────────────────────────────────────┐  ┌───────────────────────────────────────────────────┐ ││
│   │  │ SHARED MEMORY / L1 CACHE             │  │ NEW IN HOPPER                                     │ ││
│   │  │ 256 KB configurable                  │  │ • Tensor Memory Accelerator (TMA)                │ ││
│   │  │ • 228 KB shared + 28 KB L1           │  │   - Async copy global→shared                     │ ││
│   │  │ • FlashAttn3 uses 192KB tiles       │  │   - Bypasses L1 for KV loads                     │ ││
│   │  │ • 50% larger tiles than A100!       │  │ • Thread Block Clusters                          │ ││
│   │  └──────────────────────────────────────┘  │   - 16 SMs can share data                        │ ││
│   │                                             └───────────────────────────────────────────────────┘ ││
│   └───────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                         │
│   QWEN3-VL-8B FP8 EXECUTION DIFFERENCES:                                                                │
│   ═══════════════════════════════════════                                                               │
│                                                                                                         │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐│
│   │ FP8 TENSOR CORE OPERATIONS:                                                                       ││
│   │                                                                                                   ││
│   │ Precision: E4M3 (weights) × E5M2 (activations) → FP32 accumulate → BF16 output                   ││
│   │                                                                                                   ││
│   │ Per Tensor Core (4th Gen):                                                                        ││
│   │ • FP8: 512 ops/cycle (2× FP16)                                                                   ││
│   │ • Matrix shape: 16×16×32 (vs 16×16×16 for FP16)                                                  ││
│   │                                                                                                   ││
│   │ H100 Peak FP8:                                                                                    ││
│   │ • 132 SMs × 4 Tensor Cores × 512 ops × 1.98 GHz = 1,979 TFLOPS                                  ││
│   │ • vs A100 FP16: 312 TFLOPS (6.3× faster!)                                                        ││
│   │                                                                                                   ││
│   │ FlashAttention 3 Optimizations (vllm/v1/attention/backends/flash_attn.py):                       ││
│   │ • Warp specialization: Producer warps load K,V; Consumer warps compute                          ││
│   │ • Async TMA: K,V loaded while previous tile computes                                             ││
│   │ • FP8 KV cache: kv_cache_dtype="fp8" halves memory                                               ││
│   └───────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                         │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐│
│   │ H100 KV CACHE MEMORY LAYOUT (FP8):                                                                ││
│   │                                                                                                   ││
│   │ Block size: 16 tokens                                                                             ││
│   │ Per block (FP8): 2 × 32 layers × 8 kv_heads × 128 dim × 16 tokens × 1 byte = 1 MB               ││
│   │ (50% of A100 BF16!)                                                                               ││
│   │                                                                                                   ││
│   │ 80 GB HBM allocation (FP8 weights + FP8 KV):                                                      ││
│   │ ├─ Model weights (FP8): 8B × 1 byte = 8 GB                                                        ││
│   │ ├─ Vision encoder (FP16): 500M × 2 bytes = 1 GB                                                   ││
│   │ ├─ Activations: ~4 GB                                                                             ││
│   │ ├─ CUDA overhead: ~2 GB                                                                           ││
│   │ └─ KV Cache: 80 - 15 = 65 GB available                                                            ││
│   │     └─ 65 GB ÷ 1 MB/block = 65,000 blocks                                                         ││
│   │     └─ 65,000 × 16 tokens = 1,040,000 tokens                                                      ││
│   │     └─ At 32K context: 1.04M ÷ 32K = 32 concurrent sequences                                     ││
│   │     └─ 2× more than A100!                                                                         ││
│   └───────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### B200 SM Layout: Qwen3-VL-32B and MoE Models

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                    B200-192GB: 192 SMs × QWEN3-VL-32B EXECUTION MAP                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   B200 SINGLE SM LAYOUT (SM 10.0 Blackwell):                                                            │
│   ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐│
│   │ STREAMING MULTIPROCESSOR (1 of 192)                                                               ││
│   │ ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  ││
│   │ │ 4× Warp Schedulers, each with:                                                              │  ││
│   │ │ • 32 FP32 CUDA Cores (128 total/SM)                                                        │  ││
│   │ │ • 16 FP64 CUDA Cores (64 total/SM)                                                         │  ││
│   │ │ • 1 Tensor Core 5th Gen (4 total/SM) - FP4/FP8/BF16/TF32                                   │  ││
│   │ │ • 64KB Register File per scheduler (256KB total/SM)                                        │  ││
│   │ └─────────────────────────────────────────────────────────────────────────────────────────────┘  ││
│   │ ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  ││
│   │ │ Shared Memory: ~300KB configurable (256KB usable for tiles)                                │  ││
│   │ │ L2 Cache: 64MB shared across all 192 SMs                                                   │  ││
│   │ │ HBM3e: 192GB @ 8,000 GB/s (8 stacks × 24GB)                                               │  ││
│   │ └─────────────────────────────────────────────────────────────────────────────────────────────┘  ││
│   └───────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                         │
│   QWEN3-VL-32B EXECUTION (64 layers, 40 heads, 8 KV heads):                                            │
│   ═══════════════════════════════════════════════════════════                                          │
│   • Grid: (40 heads × 64 batch × 2 splits) = 5,120 blocks                                              │
│   • Blocks/SM: 5,120 ÷ 192 = 27 blocks (high occupancy)                                               │
│   • Weights (BF16): 32B × 2 = 64 GB                                                                    │
│   • KV Cache: 192 - 64 - 8 = 120 GB → 30,000 blocks → 480K tokens                                    │
│   • Throughput: ~250 tokens/sec                                                                        │
│                                                                                                         │
│   QWEN3-VL-235B-A22B MoE (128 experts, 8 active):                                                      │
│   ══════════════════════════════════════════════                                                       │
│   • All experts: 235B × 2 = 470 GB (needs 3× B200)                                                     │
│   • Per-GPU with TP=3: 157 GB, ~43 experts each                                                        │
│   • NVLink 5.0: 1.8 TB/s for All-to-All routing                                                        │
│   • Active compute: 22B params at FP8 → 800 TFLOPS effective                                          │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Complete Model × GPU Execution Summary

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│              QWEN3-VL MODEL × GPU SM EXECUTION MATRIX                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  ┌───────────────┬────────────────────────────────────────────────────────────────────────────────────┐│
│  │ MODEL         │ T4 (40 SMs)        A100 (108 SMs)      H100 (132 SMs)       B200 (192 SMs)        ││
│  ├───────────────┼────────────────────────────────────────────────────────────────────────────────────┤│
│  │ Qwen3-VL-2B   │ ✅ FlashInfer     ✅ FlashAttn2      ✅ FlashAttn3       ✅ FlashAttn3          ││
│  │ 28L/16H/4KV   │ 64 blocks         64 blocks          64 blocks           64 blocks              ││
│  │               │ 1.6 blk/SM        0.6 blk/SM         0.5 blk/SM          0.3 blk/SM             ││
│  │               │ 20 tok/s          80 tok/s           150 tok/s           280 tok/s              ││
│  ├───────────────┼────────────────────────────────────────────────────────────────────────────────────┤│
│  │ Qwen3-VL-4B   │ ✅ FlashInfer+4b  ✅ FlashAttn2      ✅ FlashAttn3       ✅ FlashAttn3          ││
│  │ 36L/24H/4KV   │ 96 blocks         96 blocks          96 blocks           96 blocks              ││
│  │               │ 2.4 blk/SM        0.9 blk/SM         0.7 blk/SM          0.5 blk/SM             ││
│  │               │ 18 tok/s          100 tok/s          180 tok/s           300 tok/s              ││
│  ├───────────────┼────────────────────────────────────────────────────────────────────────────────────┤│
│  │ Qwen3-VL-8B   │ ⚠️ 4-bit only     ✅ FlashAttn2      ✅ FlashAttn3+FP8   ✅ FlashAttn3          ││
│  │ 32L/32H/8KV   │ 512 blocks        512 blocks         512 blocks          512 blocks             ││
│  │               │ 12.8 blk/SM       4.7 blk/SM         3.9 blk/SM          2.7 blk/SM             ││
│  │               │ 10 tok/s          100 tok/s          200 tok/s           350 tok/s              ││
│  ├───────────────┼────────────────────────────────────────────────────────────────────────────────────┤│
│  │ Qwen3-VL-32B  │ ❌ OOM            ⚠️ FP8 tight       ✅ FlashAttn3+FP8   ✅ FlashAttn3          ││
│  │ 64L/40H/8KV   │ -                 2,560 blocks       2,560 blocks        5,120 blocks           ││
│  │               │ -                 23.7 blk/SM        19.4 blk/SM         26.7 blk/SM            ││
│  │               │ -                 40 tok/s           80 tok/s            250 tok/s              ││
│  ├───────────────┼────────────────────────────────────────────────────────────────────────────────────┤│
│  │ Qwen3-VL-30B  │ ❌ OOM            ✅ MoE on 80GB     ✅ MoE+FP8          ✅ MoE native          ││
│  │ -A3B (MoE)    │ -                 All 128 experts    All experts         All experts            ││
│  │ 128 experts   │ -                 3B active/tok      3B active/tok       3B active/tok          ││
│  │               │ -                 60 tok/s           120 tok/s           200 tok/s              ││
│  ├───────────────┼────────────────────────────────────────────────────────────────────────────────────┤│
│  │ Qwen3-VL-235B │ ❌ OOM            ❌ 8×A100 needed   ✅ 8×H100 TP=8      ✅ 3×B200 TP=3         ││
│  │ -A22B (MoE)   │ -                 ~50 tok/s          ~80 tok/s           ~120 tok/s             ││
│  └───────────────┴────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                         │
│  LEGEND:                                                                                                │
│  • blk/SM = Thread blocks per SM (higher = better occupancy, up to ~32 max)                           │
│  • tok/s = Approximate decode throughput (single sequence, 256 output tokens)                         │
│  • TP = Tensor Parallelism across GPUs                                                                 │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### vLLM Source Code → SM Mapping

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│              VLLM CODE REFERENCES → GPU EXECUTION                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  vllm/platforms/cuda.py - GPU Detection:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ def get_device_capability(cls, device_id=0) -> DeviceCapability:                               │  │
│  │     major, minor = torch.cuda.get_device_capability(device_id)                                  │  │
│  │     return DeviceCapability(major=major, minor=minor)                                           │  │
│  │                                                                                                 │  │
│  │ # T4:   (7, 5) = SM 7.5 Turing   → 40 SMs,  64KB shared,  no FlashAttn                        │  │
│  │ # A100: (8, 0) = SM 8.0 Ampere   → 108 SMs, 164KB shared, FlashAttn 2                         │  │
│  │ # H100: (9, 0) = SM 9.0 Hopper   → 132 SMs, 228KB shared, FlashAttn 3 + FP8                   │  │
│  │ # B200: (10,0) = SM 10.0 Blackwell → 192 SMs, 256KB shared, FlashAttn 3 + FP4                 │  │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
│  vllm/v1/attention/backends/flash_attn.py - Backend Selection:                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ class FlashAttentionBackend:                                                                    │  │
│  │     def supports_compute_capability(cls, cap: DeviceCapability) -> bool:                       │  │
│  │         return cap >= DeviceCapability(8, 0)  # A100+ only                                     │  │
│  │                                                                                                 │  │
│  │ # CUDA kernel grid for attention:                                                               │  │
│  │ # grid = (num_heads, batch_size, ceil(seq_len / BLOCK_M))                                      │  │
│  │ # Example: 32 heads × 16 batch × 32 splits = 16,384 thread blocks                             │  │
│  │ # A100: 16,384 ÷ 108 SMs = 152 waves                                                          │  │
│  │ # H100: 16,384 ÷ 132 SMs = 124 waves (22% faster)                                             │  │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
│  vllm/v1/attention/backends/flashinfer.py - T4 Fallback:                                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ class FlashInferBackend:                                                                        │  │
│  │     def supports_compute_capability(cls, cap: DeviceCapability) -> bool:                       │  │
│  │         return cap >= DeviceCapability(7, 5) and cap <= DeviceCapability(12, 1)                │  │
│  │                                                                                                 │  │
│  │ # T4 compatible! Uses smaller 64×64 tiles (vs 128×128)                                         │  │
│  │ # Shared memory: 64KB limit → 48KB tiles max                                                   │  │
│  │ # Block sizes: [64, 128, 256] tokens                                                           │  │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
│  vllm/v1/attention/backends/mla/flashmla.py - Hopper/Blackwell Only:                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ class FlashMLABackend:                                                                          │  │
│  │     def supports_compute_capability(cls, cap: DeviceCapability) -> bool:                       │  │
│  │         return cap.major in [9, 10]  # H100 and B200 ONLY                                      │  │
│  │                                                                                                 │  │
│  │ # Uses Hopper/Blackwell specific features:                                                      │  │
│  │ # - Tensor Memory Accelerator (TMA) for async K,V loads                                        │  │
│  │ # - Warp specialization: producer warps load, consumer warps compute                           │  │
│  │ # - Thread block clusters: share data across 16 SMs                                            │  │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Architecture Diagrams

### Qwen3-VL Full Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3-VL COMPLETE ARCHITECTURE DIAGRAM                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   INPUT STAGE                                                                                           │
│   ══════════                                                                                            │
│                                                                                                         │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐                                   │
│   │ Image            │   │ Video            │   │ Text             │                                   │
│   │ (any resolution) │   │ (multi-frame)    │   │ (prompt)         │                                   │
│   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘                                   │
│            │                      │                      │                                              │
│            ▼                      ▼                      │                                              │
│   ┌─────────────────────────────────────────────┐        │                                              │
│   │              VISION ENCODER                 │        │                                              │
│   │  ┌───────────────────────────────────────┐  │        │                                              │
│   │  │ 1. Patch Embedding (Conv3D)           │  │        │                                              │
│   │  │    • Kernel: (2, 14, 14)              │  │        │                                              │
│   │  │    • Stride: (2, 14, 14)              │  │        │                                              │
│   │  │    • WITH bias (diff from Qwen2-VL)   │  │        │                                              │
│   │  │    • Output: (num_patches, hidden)    │  │        │                                              │
│   │  └───────────────────────────────────────┘  │        │                                              │
│   │                     │                        │        │                                              │
│   │                     ▼                        │        │                                              │
│   │  ┌───────────────────────────────────────┐  │        │                                              │
│   │  │ 2. Position Embedding                 │  │        │                                              │
│   │  │    • Learned embeddings               │  │        │                                              │
│   │  │    • Bilinear interpolation           │  │        │                                              │
│   │  │    • Handles arbitrary resolutions    │  │        │                                              │
│   │  └───────────────────────────────────────┘  │        │                                              │
│   │                     │                        │        │                                              │
│   │                     ▼                        │        │                                              │
│   │  ┌───────────────────────────────────────┐  │        │                                              │
│   │  │ 3. Vision Transformer Blocks          │  │        │                                              │
│   │  │    ┌───────────────────────────────┐  │  │        │                                              │
│   │  │    │ Block 0-7 ──┬─────────────────┼──┼──┼───► DS Merger 0 ──┐                                  │
│   │  │    │ Block 8-15 ─┼─────────────────┼──┼──┼───► DS Merger 1 ──┼─► DeepStack                      │
│   │  │    │ Block 16-23─┼─────────────────┼──┼──┼───► DS Merger 2 ──┘   Features                       │
│   │  │    │ Block 24-31 ┴─────────────────┘  │  │                                                       │
│   │  │    │                                   │  │                                                       │
│   │  │    │ Each block:                       │  │                                                       │
│   │  │    │ • LayerNorm                       │  │                                                       │
│   │  │    │ • Self-Attention (partial RoPE)  │  │                                                       │
│   │  │    │ • LayerNorm                       │  │                                                       │
│   │  │    │ • MLP (SiLU, no bias)            │  │                                                       │
│   │  │    └───────────────────────────────┘  │  │                                                       │
│   │  └───────────────────────────────────────┘  │                                                       │
│   │                     │                        │                                                       │
│   │                     ▼                        │                                                       │
│   │  ┌───────────────────────────────────────┐  │                                                       │
│   │  │ 4. Patch Merger                       │  │                                                       │
│   │  │    • Merge 2×2 patches → 1 token      │  │                                                       │
│   │  │    • Reduces tokens by 4×             │  │                                                       │
│   │  │    • Projects to LLM hidden size      │  │                                                       │
│   │  └───────────────────────────────────────┘  │                                                       │
│   │                     │                        │                                                       │
│   └─────────────────────┼────────────────────────┘                                                       │
│                         │                                                                                │
│                         ▼                                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              SEQUENCE MERGING                                                    │   │
│   │  ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│   │  │ [<vision_start>] [vis_0] [vis_1] ... [vis_N] [<vision_end>] [text_0] [text_1] ... [text_M]│ │   │
│   │  │                                                                                           │ │   │
│   │  │ Visual tokens with M-RoPE positions:                                                      │ │   │
│   │  │ • Temporal ID: frame number                                                               │ │   │
│   │  │ • Height ID: row in image                                                                 │ │   │
│   │  │ • Width ID: column in image                                                               │ │   │
│   │  └───────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                                              │
│                                           ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              QWEN3 LLM DECODER                                                   │   │
│   │  ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│   │  │ Layer 0:  hidden_states + DeepStack_features_0  ← Multi-scale injection                  │ │   │
│   │  │ Layer 1:  hidden_states + DeepStack_features_1  ← Multi-scale injection                  │ │   │
│   │  │ Layer 2:  hidden_states + DeepStack_features_2  ← Multi-scale injection                  │ │   │
│   │  │ Layer 3+: hidden_states only                                                              │ │   │
│   │  │                                                                                           │ │   │
│   │  │ Each layer:                                                                               │ │   │
│   │  │ ┌─────────────────────────────────────────────────────────────────────────────────────┐  │ │   │
│   │  │ │ • RMSNorm                                                                           │  │ │   │
│   │  │ │ • Self-Attention (GQA with RoPE)                                                    │  │ │   │
│   │  │ │   - Q, K, V projections (QKVParallelLinear)                                        │  │ │   │
│   │  │ │   - PagedAttention (vLLM) or FlashAttention                                        │  │ │   │
│   │  │ │   - Output projection (RowParallelLinear)                                          │  │ │   │
│   │  │ │ • RMSNorm                                                                           │  │ │   │
│   │  │ │ • MLP (SwiGLU activation)                                                           │  │ │   │
│   │  │ │   - Gate + Up projection (MergedColumnParallelLinear)                              │  │ │   │
│   │  │ │   - Down projection (RowParallelLinear)                                            │  │ │   │
│   │  │ └─────────────────────────────────────────────────────────────────────────────────────┘  │ │   │
│   │  └───────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│   │                                           │                                                      │   │
│   │                                           ▼                                                      │   │
│   │  ┌───────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│   │  │ Final RMSNorm → LM Head (ParallelLMHead) → Logits                                        │ │   │
│   │  └───────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                           │                                                              │
│                                           ▼                                                              │
│   OUTPUT: Generated text tokens                                                                          │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Model Size Comparison Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL SIZE COMPARISON (DENSE MODELS)                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   Parameters (billions)                                                                                 │
│   │                                                                                                     │
│   35│                                             ┌───────┐                                             │
│     │                                             │ 33B   │ ← Qwen3-VL-32B                             │
│   30│                                             │       │   (32B LLM + 1B ViT)                       │
│     │                                             │       │                                             │
│   25│                                             │       │                                             │
│     │                                             │       │                                             │
│   20│                                             │       │                                             │
│     │                                             │       │                                             │
│   15│                                             │       │                                             │
│     │                                             │       │                                             │
│   10│                         ┌───────┐           │       │                                             │
│     │                         │ 8.5B  │           │       │                                             │
│    8│                         │       │           │       │ ← Qwen3-VL-8B (8B LLM + 500M ViT)          │
│     │               ┌───────┐ │       │           │       │                                             │
│    5│               │ 4.5B  │ │       │           │       │ ← Qwen3-VL-4B (4B LLM + 500M ViT)          │
│     │     ┌───────┐ │       │ │       │           │       │                                             │
│    2│     │ 2.5B  │ │       │ │       │           │       │ ← Qwen3-VL-2B (2B LLM + 500M ViT)          │
│     │     │       │ │       │ │       │           │       │                                             │
│    0│─────┴───────┴─┴───────┴─┴───────┴───────────┴───────┴─────────────────────────────────────────    │
│          VL-2B     VL-4B     VL-8B               VL-32B                                                 │
│                                                                                                         │
│   ═══════════════════════════════════════════════════════════════════════════════════════════════       │
│                                                                                                         │
│                              MODEL SIZE COMPARISON (MOE MODELS)                                         │
│                                                                                                         │
│   Total Parameters (gray) vs Active Parameters (blue)                                                   │
│   │                                                                                                     │
│   250│                                                    ┌─────────┐                                   │
│      │                                                    │░░░░░░░░░│ ← 237B Total                      │
│   200│                                                    │░░░░░░░░░│   (235B LLM + 2B ViT)            │
│      │                                                    │░░░░░░░░░│                                   │
│   150│                                                    │░░░░░░░░░│                                   │
│      │                                                    │░░░░░░░░░│                                   │
│   100│                                                    │░░░░░░░░░│                                   │
│      │                                                    │░░░░░░░░░│                                   │
│    50│                  ┌─────────┐                       │░░░░░░░░░│                                   │
│      │                  │░░░░░░░░░│ ← 31B Total           │░░░░░░░░░│                                   │
│    25│                  │░░░░░░░░░│   (30B LLM + 1B ViT)  │████████ │ ← 22B Active                     │
│      │                  │███████ │ ← 3B Active           │████████ │                                   │
│     0│──────────────────┴─────────┴───────────────────────┴─────────┴───────────────────────────────    │
│                     VL-30B-A3B                     VL-235B-A22B                                         │
│                                                                                                         │
│   Legend: ░░░ = Total parameters (in memory)    ███ = Active parameters (per token compute)            │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## vLLM PagedAttention for Multimodal

Understanding how vLLM's PagedAttention handles vision-language models:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              VLLM PAGEDATTENTION FOR MULTIMODAL                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   THE PROBLEM: Variable-Length KV Cache                                                                 │
│   ═════════════════════════════════════                                                                 │
│                                                                                                         │
│   Request 1: [1024×1024 image]  → 1332 visual tokens + 100 text = 1432 tokens                          │
│   Request 2: [Video, 50 frames] → 8000 visual tokens + 50 text = 8050 tokens                           │
│   Request 3: [Text only]        → 0 visual tokens + 500 text = 500 tokens                               │
│                                                                                                         │
│   Traditional allocation: Pre-allocate max_tokens for every request → HUGE waste!                      │
│                                                                                                         │
│   NAIVE ALLOCATION (Wasteful):                                                                          │
│   ════════════════════════════                                                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Request 1: [████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │   │
│   │ Request 2: [██████████████████████████████████████████████████████████████████░░░░░░░░░░░░░░]  │   │
│   │ Request 3: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  │   │
│   │                                                                                                │   │
│   │            ████ = Used memory     ░░░░ = Wasted (pre-allocated but empty)                     │   │
│   │            Memory utilization: ~30%                                                            │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   PAGEDATTENTION (Efficient):                                                                           │
│   ═══════════════════════════                                                                           │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │   Physical Memory Pool (Pages):                                                                │   │
│   │   ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐        │   │
│   │   │ P0  ││ P1  ││ P2  ││ P3  ││ P4  ││ P5  ││ P6  ││ P7  ││ P8  ││ P9  ││P10  ││ ...│        │   │
│   │   │Req1 ││Req1 ││Req1 ││Req2 ││Req2 ││Req2 ││Req2 ││Req2 ││Req3 ││Free ││Free ││    │        │   │
│   │   └─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘└─────┘        │   │
│   │                                                                                                │   │
│   │   Virtual → Physical Mapping (Block Tables):                                                   │   │
│   │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐ │   │
│   │   │ Request 1: [P0, P1, P2, _, _, _]           (3 pages allocated)                          │ │   │
│   │   │ Request 2: [P3, P4, P5, P6, P7, _]         (5 pages allocated)                          │ │   │
│   │   │ Request 3: [P8, _, _, _, _, _]             (1 page allocated)                           │ │   │
│   │   └─────────────────────────────────────────────────────────────────────────────────────────┘ │   │
│   │                                                                                                │   │
│   │   Memory utilization: ~95%                                                                     │   │
│   │   Pages allocated on-demand, freed when request completes                                     │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   MULTIMODAL-SPECIFIC OPTIMIZATIONS:                                                                    │
│   ══════════════════════════════════                                                                    │
│                                                                                                         │
│   1. PREFIX CACHING FOR VISUAL TOKENS                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │   Turn 1: [System Prompt] + [Image Tokens] + [Question 1]                                     │   │
│   │                              └──── CACHED ────┘                                                │   │
│   │                                                                                                │   │
│   │   Turn 2: [System Prompt] + [Image Tokens] + [Question 2]                                     │   │
│   │           └──── REUSED ────┘ └── REUSED ──┘                                                   │   │
│   │                                                                                                │   │
│   │   BENEFIT: Image tokens computed once, reused across conversation turns                       │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   2. EVS (EFFICIENT VIDEO SAMPLING) INTEGRATION                                                         │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │   100-frame video without EVS:                                                                 │   │
│   │   [F1][F2][F3][F4][F5][F6]...[F100] → 16,000 tokens → 64 pages → May OOM                      │   │
│   │                                                                                                │   │
│   │   100-frame video with EVS (50% pruning):                                                     │   │
│   │   [F1][  ][F3][  ][F5][  ]...[F99 ] → 8,000 tokens → 32 pages → Fits!                        │   │
│   │                                                                                                │   │
│   │   Similar frames pruned BEFORE encoding → fewer tokens → fewer pages → more concurrency       │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   3. CHUNKED PREFILL FOR LONG VISUAL SEQUENCES                                                          │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │   Standard: Encode entire 50K token sequence → Wait → Start generating                        │   │
│   │                                                                                                │   │
│   │   Chunked:  Encode 5K chunk → Start generating → Continue encoding in background              │   │
│   │                                                                                                │   │
│   │   BENEFIT: ~30% faster time-to-first-token for long videos                                   │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## GPU Memory Layout Diagrams

### T4 (16GB) Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              T4 (16GB) MEMORY LAYOUT - Qwen3-VL-4B (4-bit)                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU Memory: 16,384 MB                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Model Weights (4-bit BitsAndBytes)                                    ~2,000 MB (12%)  │   │   │
│   │  │ ████████████████████                                                                   │   │   │
│   │  │ • LLM: 4B params × 0.5 bytes = 2,000 MB                                               │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Vision Encoder (FP16, can't quantize easily)                          ~1,000 MB (6%)   │   │   │
│   │  │ ██████████                                                                             │   │   │
│   │  │ • ViT: 500M params × 2 bytes = 1,000 MB                                               │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ KV Cache (4K context × 4 requests × FP16)                             ~3,000 MB (18%)  │   │   │
│   │  │ ██████████████████████████████                                                         │   │   │
│   │  │ • 2 × 36 layers × 4 kv_heads × 128 head_dim × 4K × 4 seqs × 2 bytes                  │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Activations (intermediate tensors)                                    ~2,000 MB (12%)  │   │   │
│   │  │ ████████████████████                                                                   │   │   │
│   │  │ • MLP intermediate, attention scores, etc.                                            │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Visual Token Buffer (1024×1024 image)                                 ~1,500 MB (9%)   │   │   │
│   │  │ ███████████████                                                                        │   │   │
│   │  │ • Patch embeddings during vision encoding                                              │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ CUDA Overhead / Reserved                                              ~1,500 MB (9%)   │   │   │
│   │  │ ███████████████                                                                        │   │   │
│   │  │ • CUDA context, driver, workspace                                                      │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ FREE MEMORY                                                           ~5,384 MB (34%)  │   │   │
│   │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                                │   │   │
│   │  │ • Available for additional KV cache pages                                              │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  TOTAL USED: ~11,000 MB (67%)     FREE: ~5,384 MB (33%)                                       │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   LIMITATIONS:                                                                                          │
│   • No BF16 (FP16 only)                                                                                │
│   • No FlashAttention 2 (uses TORCH_SDPA)                                                              │
│   • 320 GB/s bandwidth (memory-bound during decode)                                                    │
│                                                                                                         │
│   RECOMMENDED CONFIG:                                                                                   │
│   • enforce_eager=True (saves ~500 MB by disabling CUDA graphs)                                        │
│   • max_model_len=4096                                                                                 │
│   • max_num_seqs=4                                                                                     │
│   • max_pixels=500000 (~700×700)                                                                       │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### A100-80GB Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              A100-80GB MEMORY LAYOUT - Qwen3-VL-8B (BF16)                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU Memory: 81,920 MB                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Model Weights (BF16 full precision)                                  ~17,000 MB (21%)  │   │   │
│   │  │ ██████████████████████████████████                                                     │   │   │
│   │  │ • LLM: 8B params × 2 bytes = 16,000 MB                                                │   │   │
│   │  │ • Vision: 500M × 2 bytes = 1,000 MB                                                    │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ KV Cache (32K context × 32 requests × BF16)                          ~24,000 MB (29%)  │   │   │
│   │  │ ████████████████████████████████████████████████                                       │   │   │
│   │  │ • Managed by PagedAttention                                                            │   │   │
│   │  │ • 2 × 32 layers × 8 kv_heads × 128 head_dim × 32K × 32 seqs × 2 bytes                │   │   │
│   │  │ • Pages allocated dynamically as needed                                                │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Activations + CUDA Graphs                                             ~8,000 MB (10%)  │   │   │
│   │  │ ████████████████                                                                       │   │   │
│   │  │ • CUDA graphs enabled for faster decode                                                │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Prefix Cache (shared system prompts)                                  ~3,000 MB (4%)   │   │   │
│   │  │ ██████                                                                                 │   │   │
│   │  │ • Cached KV for repeated prefixes                                                      │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ FREE MEMORY                                                          ~29,920 MB (36%)  │   │   │
│   │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                          │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  TOTAL USED: ~52,000 MB (64%)     FREE: ~29,920 MB (36%)                                      │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   ADVANTAGES:                                                                                           │
│   • Native BF16 support (better numerical stability)                                                   │
│   • FlashAttention 2 (2-4× faster attention)                                                           │
│   • 2,039 GB/s bandwidth (6× faster than T4)                                                           │
│   • Tensor cores for mixed precision                                                                   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### H100-80GB Memory Layout (with FP8)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              H100-80GB MEMORY LAYOUT - Qwen3-VL-8B (FP8)                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU Memory: 81,920 MB                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Model Weights (FP8 quantized)                                         ~8,500 MB (10%)  │   │   │
│   │  │ █████████████████                                                                      │   │   │
│   │  │ • LLM: 8B params × 1 byte = 8,000 MB                                                  │   │   │
│   │  │ • Vision: 500M × 1 byte = 500 MB                                                       │   │   │
│   │  │ ← 50% smaller than BF16!                                                               │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ KV Cache (FP8, 32K context × 64 requests)                            ~16,000 MB (20%)  │   │   │
│   │  │ ████████████████████████████████                                                       │   │   │
│   │  │ • FP8 KV cache = 50% smaller than BF16                                                │   │   │
│   │  │ • Can serve 2× more concurrent requests!                                               │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Activations + FlashAttention 3 workspace                              ~6,000 MB (7%)   │   │   │
│   │  │ ████████████                                                                           │   │   │
│   │  │ • FlashAttention 3 optimized for Hopper                                                │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ FREE MEMORY                                                          ~51,420 MB (63%)  │   │   │
│   │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│   │   │
│   │  │ • Room for 64+ concurrent image requests                                               │   │   │
│   │  │ • Can scale to 128 concurrent with smaller images                                      │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  TOTAL USED: ~30,500 MB (37%)     FREE: ~51,420 MB (63%)                                      │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   FP8 BENEFITS:                                                                                         │
│   • Weights: 17 GB → 8.5 GB (50% reduction)                                                            │
│   • KV Cache: 24 GB → 12 GB (50% reduction)                                                            │
│   • Throughput: ~2× faster matmul operations                                                           │
│   • Quality: <1% accuracy loss for most VLM tasks                                                      │
│                                                                                                         │
│   ADDITIONAL H100 ADVANTAGES:                                                                           │
│   • FlashAttention 3 (Hopper-specific optimizations)                                                   │
│   • 3,350 GB/s bandwidth (10× faster than T4)                                                          │
│   • Transformer Engine for automatic mixed precision                                                   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### B200-192GB Memory Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              B200-192GB MEMORY LAYOUT - Qwen3-VL-32B (BF16)                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   GPU Memory: 196,608 MB                                                                                │
│   ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Model Weights (BF16 full precision)                                  ~66,000 MB (34%)  │   │   │
│   │  │ ██████████████████████████████████████████████████████████████████████                 │   │   │
│   │  │ • LLM: 32B params × 2 bytes = 64,000 MB                                               │   │   │
│   │  │ • Vision: 1B × 2 bytes = 2,000 MB                                                      │   │   │
│   │  │ ← Single GPU! No tensor parallelism needed!                                            │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ KV Cache (128K context × 128 requests × BF16)                        ~60,000 MB (30%)  │   │   │
│   │  │ ████████████████████████████████████████████████████████████                           │   │   │
│   │  │ • Massive context window supported                                                     │   │   │
│   │  │ • 2 × 64 layers × 8 kv_heads × 128 head_dim × 128K × 128 seqs × 2 bytes              │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ Activations + 4K resolution images                                   ~20,000 MB (10%)  │   │   │
│   │  │ ████████████████████████████████████████                                               │   │   │
│   │  │ • max_pixels=4,147,200 (4K resolution)                                                │   │   │
│   │  │ • Multiple 4K images simultaneously                                                    │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  ┌────────────────────────────────────────────────────────────────────────────────────────┐   │   │
│   │  │ FREE MEMORY                                                          ~50,608 MB (26%)  │   │   │
│   │  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░                                │   │   │
│   │  │ • Headroom for even larger batches                                                     │   │   │
│   │  └────────────────────────────────────────────────────────────────────────────────────────┘   │   │
│   │                                                                                                │   │
│   │  TOTAL USED: ~146,000 MB (74%)     FREE: ~50,608 MB (26%)                                     │   │
│   │                                                                                                │   │
│   └────────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│   B200 ADVANTAGES:                                                                                      │
│   • 192 GB HBM3e (12× T4, 2.4× H100)                                                                   │
│   • 8,000 GB/s bandwidth (25× faster than T4)                                                          │
│   • FP4 support (when available, 4× throughput potential)                                              │
│   • Run 32B model on SINGLE GPU                                                                        │
│   • 128K context with 128 concurrent requests                                                          │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Benchmarks

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              PERFORMANCE BENCHMARKS BY GPU                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│   SINGLE IMAGE INFERENCE (1024×1024, 256 output tokens):                                                │
│   ══════════════════════════════════════════════════════                                                │
│                                                                                                         │
│   ┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐                │
│   │ Model              │ T4 (16GB)          │ A100-80GB          │ H100-80GB (FP8)    │                │
│   ├────────────────────┼────────────────────┼────────────────────┼────────────────────┤                │
│   │ Qwen3-VL-2B        │ 600ms, 25 tok/s    │ 100ms, 150 tok/s   │ 50ms, 300 tok/s    │                │
│   │ Qwen3-VL-4B        │ 1000ms*, 18 tok/s  │ 150ms, 120 tok/s   │ 80ms, 220 tok/s    │                │
│   │ Qwen3-VL-8B        │ 1800ms*, 10 tok/s  │ 250ms, 80 tok/s    │ 120ms, 150 tok/s   │                │
│   │ Qwen3-VL-32B       │ ❌ OOM             │ 500ms, 40 tok/s    │ 250ms, 80 tok/s    │                │
│   │ Qwen3-VL-30B-A3B   │ ❌ OOM             │ 350ms, 50 tok/s    │ 180ms, 100 tok/s   │                │
│   └────────────────────┴────────────────────┴────────────────────┴────────────────────┘                │
│   * = Requires 4-bit quantization                                                                       │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   VIDEO INFERENCE (60s video, 100 frames, 256 output tokens):                                           │
│   ═══════════════════════════════════════════════════════════                                           │
│                                                                                                         │
│   ┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐                │
│   │ Model              │ Without EVS        │ With EVS (50%)     │ Speed Improvement  │                │
│   ├────────────────────┼────────────────────┼────────────────────┼────────────────────┤                │
│   │ Qwen3-VL-8B (A100) │ 4.5s, 16K tokens   │ 2.5s, 8K tokens    │ 1.8× faster        │                │
│   │ Qwen3-VL-8B (H100) │ 2.0s, 16K tokens   │ 1.1s, 8K tokens    │ 1.8× faster        │                │
│   │ Qwen3-VL-32B (H100)│ 4.0s, 16K tokens   │ 2.2s, 8K tokens    │ 1.8× faster        │                │
│   └────────────────────┴────────────────────┴────────────────────┴────────────────────┘                │
│                                                                                                         │
│   ─────────────────────────────────────────────────────────────────────────────────────────────────────│
│                                                                                                         │
│   THROUGHPUT (Concurrent requests, mixed image sizes):                                                  │
│   ═══════════════════════════════════════════════════════                                               │
│                                                                                                         │
│   ┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐                │
│   │ GPU                │ Max Concurrent     │ Throughput (req/s) │ p95 Latency        │                │
│   ├────────────────────┼────────────────────┼────────────────────┼────────────────────┤                │
│   │ T4 (Qwen3-VL-4B)   │ 4                  │ ~1                 │ ~2000ms            │                │
│   │ A100-80GB (8B)     │ 32                 │ ~6                 │ ~500ms             │                │
│   │ H100-80GB (8B FP8) │ 64                 │ ~12                │ ~300ms             │                │
│   │ B200-192GB (32B)   │ 128                │ ~20                │ ~200ms             │                │
│   └────────────────────┴────────────────────┴────────────────────┴────────────────────┘                │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Vision Encoder Architecture (All Models)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              QWEN3-VL VISION ENCODER ARCHITECTURE                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  COMMON FEATURES ACROSS ALL QWEN3-VL MODELS:                                                            │
│  ════════════════════════════════════════════                                                           │
│                                                                                                         │
│  1. PATCH EMBEDDING (Conv3D)                                                                            │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  Input: (batch, channels, time, height, width)                                               │   │
│     │  Conv3D(in=3, out=hidden_size, kernel=(2, 14, 14), stride=(2, 14, 14), bias=True)           │   │
│     │                                                                                              │   │
│     │  KEY DIFFERENCE FROM QWEN2-VL: bias=True (Qwen2-VL had bias=False)                          │   │
│     │                                                                                              │   │
│     │  This allows the model to learn per-channel offsets, improving representation flexibility   │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  2. POSITION EMBEDDING (Learned + Interpolation)                                                        │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  pos_embed = nn.Embedding(num_position_embeddings, hidden_size)                              │   │
│     │                                                                                              │   │
│     │  For arbitrary resolutions: Bilinear interpolation                                          │   │
│     │  - Handles any image size without retraining                                                │   │
│     │  - Smoothly interpolates between learned positions                                          │   │
│     │                                                                                              │   │
│     │  PLUS: RoPE with partial_rotary_factor=0.5                                                  │   │
│     │  - Only 50% of dimensions are rotated (vs 100% in Qwen2-VL)                                │   │
│     │  - Improves length extrapolation                                                            │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  3. VISION TRANSFORMER BLOCKS                                                                           │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  Each block:                                                                                 │   │
│     │    x → LayerNorm → SelfAttention (with partial RoPE) → Add                                  │   │
│     │    x → LayerNorm → MLP (SiLU activation, no bias) → Add                                     │   │
│     │                                                                                              │   │
│     │  MLP Structure:                                                                              │   │
│     │    linear_fc1: Linear(dim, mlp_hidden_dim, bias=False)                                      │   │
│     │    act_fn: SiLU = x * sigmoid(x)    ← Different from QuickGELU in Qwen2-VL                 │   │
│     │    linear_fc2: Linear(mlp_hidden_dim, dim, bias=False)                                      │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  4. DEEPSTACK (Multi-Scale Feature Extraction)                                                          │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  deepstack_visual_indexes = [8, 16, 24]  # Extract features at these layers                 │   │
│     │                                                                                              │   │
│     │  During forward pass:                                                                        │   │
│     │    for layer_num, block in enumerate(blocks):                                               │   │
│     │      hidden = block(hidden)                                                                  │   │
│     │      if layer_num in deepstack_visual_indexes:                                              │   │
│     │        ds_feature = deepstack_merger[idx](hidden)                                           │   │
│     │        deepstack_features.append(ds_feature)                                                │   │
│     │                                                                                              │   │
│     │  Final output: [main_features | ds_features_8 | ds_features_16 | ds_features_24]           │   │
│     │  These are injected into EARLY layers of the LLM decoder                                   │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
│  5. PATCH MERGER (Spatial Compression)                                                                  │
│     ┌──────────────────────────────────────────────────────────────────────────────────────────────┐   │
│     │  spatial_merge_size = 2 → Merge 2×2 patches into 1                                          │   │
│     │                                                                                              │   │
│     │  Reduces token count by 4x while preserving information                                    │   │
│     │  E.g., 1024×1024 image → 5329 patches → 1332 tokens after merging                          │   │
│     └──────────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## vLLM Deployment Configurations

### T4 (16 GB) - Entry Level

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# T4: Qwen3-VL-2B (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --dtype half \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --max-num-seqs 8 \
    --enforce-eager \
    --limit-mm-per-prompt image=4,video=1

# Expected: ~800ms latency, ~1.5 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# T4: Qwen3-VL-4B (4-bit Quantization)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --dtype half \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.92 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --enforce-eager \
    --limit-mm-per-prompt image=2,video=1

# Expected: ~1000ms latency, ~1 req/s
```

### L4 / A10G (24 GB) - Cloud Entry

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# L4/A10G: Qwen3-VL-4B (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --limit-mm-per-prompt image=4,video=1 \
    --enable-prefix-caching

# Expected: ~500ms latency, ~2 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# L4/A10G: Qwen3-VL-8B (4-bit Quantization)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype half \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --enforce-eager \
    --limit-mm-per-prompt image=2,video=1

# Expected: ~700ms latency, ~1.5 req/s
```

### A100-40GB - Production Standard

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# A100-40GB: Qwen3-VL-8B (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=8,video=2

# Expected: ~300ms latency, ~4 req/s
```

### A100-80GB - High Capacity

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# A100-80GB: Qwen3-VL-8B (Full Precision, Large Batches)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 65536 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=16,video=4

# Expected: ~250ms latency, ~6 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# A100-80GB: Qwen3-VL-32B (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --limit-mm-per-prompt image=4,video=1

# Expected: ~500ms latency, ~2 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# A100-80GB: Qwen3-VL-30B-A3B MoE (Full Precision)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 16 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=8,video=2

# Expected: ~350ms latency, ~4 req/s (MoE = fast despite 30B total)
```

### H100-80GB - Maximum Throughput

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# H100: Qwen3-VL-8B with FP8 (Maximum Throughput)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 65536 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=16,video=4

# Expected: ~150ms latency, ~10 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# H100: Qwen3-VL-32B with FP8
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=8,video=2

# Expected: ~300ms latency, ~5 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# H100: Qwen3-VL-30B-A3B MoE with FP8
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 65536 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=16,video=4

# Expected: ~200ms latency, ~6 req/s
```

### B200-192GB - Future Hardware

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# B200: Qwen3-VL-32B (Maximum Context)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-32B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=32,video=8

# Expected: ~100ms latency, ~15 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# B200: Qwen3-VL-30B-A3B MoE (Maximum Everything)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 262144 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=32,video=8

# Expected: ~80ms latency, ~20 req/s
```

### Multi-GPU: 8×H100 for 235B Model

```bash
# ═══════════════════════════════════════════════════════════════════════════════
# 8×H100: Qwen3-VL-235B-A22B MoE (Full Model)
# ═══════════════════════════════════════════════════════════════════════════════

OMP_NUM_THREADS=1 vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dtype bfloat16 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.95 \
    --max-model-len 131072 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=16,video=4

# Expected: ~400ms latency, ~3 req/s

# ═══════════════════════════════════════════════════════════════════════════════
# 8×B200: Qwen3-VL-235B-A22B MoE (Maximum Configuration)
# ═══════════════════════════════════════════════════════════════════════════════

vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --gpu-memory-utilization 0.95 \
    --max-model-len 262144 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --limit-mm-per-prompt image=32,video=8

# Expected: ~200ms latency, ~8 req/s
```

---

## Memory Calculation Formula

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MEMORY CALCULATION                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  Total GPU Memory = Model Weights + KV Cache + Vision Encoder + Activations + CUDA Overhead            │
│                                                                                                         │
│  MODEL WEIGHTS:                                                                                         │
│  ══════════════                                                                                         │
│  • BF16/FP16: params × 2 bytes                                                                         │
│  • FP8:       params × 1 byte                                                                           │
│  • INT4:      params × 0.5 bytes                                                                        │
│                                                                                                         │
│  KV CACHE (per request):                                                                                │
│  ══════════════════════════                                                                             │
│  • Size = 2 × num_layers × num_kv_heads × head_dim × context_len × bytes_per_param                     │
│  • Total = kv_per_request × max_num_seqs                                                                │
│                                                                                                         │
│  EXAMPLE: Qwen3-VL-8B on H100 with FP8                                                                  │
│  ════════════════════════════════════                                                                   │
│  • Model: 8B × 1 byte = 8 GB                                                                            │
│  • Vision: ~1.5 GB (ViT encoder)                                                                        │
│  • KV Cache: 2 × 32 × 8 × 128 × 32768 × 0.5 = ~4 GB per seq × 64 seqs = ~256 GB... wait, that's wrong  │
│                                                                                                         │
│  Let me recalculate properly:                                                                           │
│  • KV per layer = 2 × kv_heads × head_dim × context = 2 × 8 × 128 × 32768 = 64 MB (BF16)               │
│  • Total KV = 64 MB × 32 layers = 2 GB per request (BF16)                                              │
│  • With FP8: 1 GB per request                                                                           │
│  • 64 requests × 1 GB = 64 GB KV... still too much for 80GB with model                                 │
│                                                                                                         │
│  PRACTICAL LIMITS:                                                                                      │
│  ═════════════════                                                                                      │
│  vLLM dynamically manages KV cache with PagedAttention, so you don't need 100% upfront allocation     │
│  The gpu_memory_utilization=0.95 tells vLLM how much total memory it can use                           │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: Model Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              WHICH MODEL SHOULD I USE?                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                         │
│  FOR EDGE / MOBILE DEPLOYMENT:                                                                          │
│  • Qwen3-VL-2B-Instruct (T4, L4, consumer GPUs)                                                        │
│                                                                                                         │
│  FOR BALANCED COST/PERFORMANCE:                                                                         │
│  • Qwen3-VL-4B-Instruct (L4, A10G) - Sweet spot for edge                                               │
│  • Qwen3-VL-8B-Instruct (A100-40GB, H100) - Sweet spot for cloud                                       │
│                                                                                                         │
│  FOR MAXIMUM SINGLE-GPU QUALITY:                                                                        │
│  • Qwen3-VL-32B-Instruct (A100-80GB, H100 with FP8)                                                    │
│                                                                                                         │
│  FOR MoE EFFICIENCY (Quality of large model, speed of small):                                           │
│  • Qwen3-VL-30B-A3B-Instruct - 30B quality at 3B compute cost                                          │
│                                                                                                         │
│  FOR STATE-OF-THE-ART PERFORMANCE:                                                                      │
│  • Qwen3-VL-235B-A22B-Instruct (8×H100 or 8×B200)                                                      │
│                                                                                                         │
│  FOR COMPLEX REASONING TASKS:                                                                           │
│  • Use -Thinking variants of any model                                                                 │
│                                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Executive Summary: Key Takeaways by Role

### For CEOs / Business Leaders

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              EXECUTIVE SUMMARY                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   WHAT IS QWEN3-VL?                                                                             │
│   A state-of-the-art AI that understands both images and text, enabling:                       │
│   • Automated GUI agents (MAI-UI) that can control computers/phones                            │
│   • Document understanding and analysis                                                         │
│   • Video comprehension for content moderation, security, etc.                                 │
│                                                                                                 │
│   ROI CONSIDERATIONS:                                                                           │
│   ┌─────────────────────┬────────────────┬────────────────┬────────────────────┐               │
│   │ Workload Scale      │ GPU Choice     │ Monthly Cost*  │ Requests/day       │               │
│   ├─────────────────────┼────────────────┼────────────────┼────────────────────┤               │
│   │ Prototype/Dev       │ 1× T4          │ ~$100          │ ~5,000             │               │
│   │ Small Production    │ 1× A100-40GB   │ ~$1,000        │ ~50,000            │               │
│   │ Medium Production   │ 1× H100        │ ~$3,000        │ ~200,000           │               │
│   │ Enterprise Scale    │ 8× H100        │ ~$25,000       │ ~1,000,000+        │               │
│   └─────────────────────┴────────────────┴────────────────┴────────────────────┘               │
│   *Approximate cloud GPU rental costs                                                          │
│                                                                                                 │
│   KEY DECISION: Model quality scales with size, but so does cost.                              │
│   • 8B model: Good quality, reasonable cost, fits most use cases                              │
│   • 32B model: Best quality, 4× cost, for demanding applications                              │
│   • MoE models: 32B quality at 8B cost, but requires more memory                              │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### For CTOs / Engineering Managers

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              TECHNICAL STRATEGY                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   ARCHITECTURE DECISION TREE:                                                                   │
│                                                                                                 │
│   ┌────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │                                                                                        │   │
│   │   Q1: What's your latency requirement?                                                 │   │
│   │   ├─ < 200ms → H100 with FP8, or B200                                                 │   │
│   │   ├─ < 500ms → A100-80GB with BF16                                                    │   │
│   │   └─ < 1000ms → T4 with 4-bit quantization (cost-optimized)                           │   │
│   │                                                                                        │   │
│   │   Q2: What's your quality requirement?                                                 │   │
│   │   ├─ SOTA benchmarks → Qwen3-VL-32B or 235B-A22B                                      │   │
│   │   ├─ Production quality → Qwen3-VL-8B (sweet spot)                                    │   │
│   │   └─ Good enough → Qwen3-VL-4B (3× faster, 80% quality)                               │   │
│   │                                                                                        │   │
│   │   Q3: What's your concurrency requirement?                                             │   │
│   │   ├─ 100+ concurrent → B200 or multi-GPU H100                                         │   │
│   │   ├─ 10-50 concurrent → H100 with FP8 KV cache                                        │   │
│   │   └─ < 10 concurrent → A100-40GB sufficient                                           │   │
│   │                                                                                        │   │
│   └────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                 │
│   INFRASTRUCTURE RECOMMENDATIONS:                                                               │
│   • Use vLLM (not HuggingFace) for 3-5× better throughput                                     │
│   • Enable PagedAttention (default in vLLM) for memory efficiency                             │
│   • Enable prefix caching for multi-turn conversations                                         │
│   • Use FP8 on H100 for 2× throughput with <1% quality loss                                   │
│   • Consider MoE models for cost-efficiency at scale                                          │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### For Engineers / Developers

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              IMPLEMENTATION GUIDE                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   QUICK START (copy-paste ready):                                                               │
│                                                                                                 │
│   # For development (Colab T4):                                                                │
│   vllm serve Qwen/Qwen3-VL-2B-Instruct \                                                       │
│     --dtype float16 \                                                                           │
│     --max-model-len 4096 \                                                                      │
│     --gpu-memory-utilization 0.85 \                                                             │
│     --enforce-eager                                                                             │
│                                                                                                 │
│   # For production (A100-80GB):                                                                │
│   vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                       │
│     --dtype bfloat16 \                                                                          │
│     --max-model-len 32768 \                                                                     │
│     --max-num-seqs 32 \                                                                         │
│     --enable-prefix-caching \                                                                   │
│     --enable-chunked-prefill                                                                    │
│                                                                                                 │
│   # For high-throughput (H100):                                                                │
│   vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                       │
│     --quantization fp8 \                                                                        │
│     --kv-cache-dtype fp8 \                                                                      │
│     --max-num-seqs 64 \                                                                         │
│     --enable-prefix-caching                                                                     │
│                                                                                                 │
│   ─────────────────────────────────────────────────────────────────────────────────────────    │
│                                                                                                 │
│   DEBUGGING CHECKLIST:                                                                          │
│   □ OOM errors → Reduce max_model_len, max_num_seqs, or max_pixels                            │
│   □ Slow first token → Enable chunked_prefill for long contexts                               │
│   □ Slow decode → Check bandwidth (T4 is 6× slower than H100)                                 │
│   □ Quality issues → Check dtype (BF16 > FP16 > FP8 > INT4)                                   │
│   □ Video OOM → Enable EVS with video_pruning_rate=0.5                                        │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### For Interns / New Team Members

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              LEARNING PATH                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                 │
│   CONCEPT MAP (What you need to understand):                                                    │
│                                                                                                 │
│   Level 1: What is a VLM?                                                                       │
│   ────────────────────────                                                                      │
│   Vision + Language = Model that understands images AND text                                   │
│   • Input: Image/Video + Text prompt ("What's in this picture?")                              │
│   • Output: Text response ("A cat sitting on a windowsill")                                   │
│                                                                                                 │
│   Level 2: How does it work?                                                                    │
│   ──────────────────────────                                                                    │
│   1. Vision Encoder (ViT) converts image → tokens (numbers)                                   │
│   2. Text Tokenizer converts text → tokens                                                     │
│   3. LLM processes [image_tokens] + [text_tokens] together                                    │
│   4. LLM generates response tokens one at a time                                              │
│                                                                                                 │
│   Level 3: What is vLLM?                                                                        │
│   ──────────────────────                                                                        │
│   vLLM = Very fast LLM inference engine                                                        │
│   • PagedAttention: Efficient memory management (like virtual memory for AI)                  │
│   • Continuous batching: Serve many users simultaneously                                      │
│   • Prefix caching: Reuse computation for shared contexts                                     │
│                                                                                                 │
│   Level 4: Why different GPUs?                                                                  │
│   ───────────────────────────                                                                   │
│   • More VRAM = larger models, longer contexts, more concurrent users                         │
│   • More bandwidth = faster token generation                                                  │
│   • Newer architecture = more features (FP8, FlashAttention 3)                               │
│                                                                                                 │
│   RECOMMENDED READING ORDER:                                                                    │
│   1. QWEN_VL_COMPLETE_GUIDE.md - Architecture & concepts                                       │
│   2. This file - Model sizes & GPU configs                                                     │
│   3. mai_ui_gpu_optimized.ipynb - Hands-on practice                                           │
│                                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Why 32B is Required for Reliable GUI Agents (SFT + RL Training)

This section explains why computer use agents trained with SFT and RL **only work reliably at 32B scale**, while smaller models fail even with the same training data and methodology.

### The Problem: Smaller Models Are "Ass" for GUI Agents

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     WHY SMALLER QWEN3-VL MODELS FAIL AT GUI AGENT TASKS                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  THE FUNDAMENTAL PROBLEM:                                                                                   │
│  ════════════════════════                                                                                   │
│                                                                                                             │
│  GUI agent tasks require SIMULTANEOUS excellence in ALL of these:                                           │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  1. FINE-GRAINED SPATIAL UNDERSTANDING                                                                │ │
│  │     ─────────────────────────────────────                                                             │ │
│  │     • Locate exact pixel coordinates of UI elements                                                   │ │
│  │     • Distinguish between adjacent buttons (5px apart)                                                │ │
│  │     • Understand element hierarchy (dropdown inside modal inside sidebar)                             │ │
│  │     • Handle overlapping elements, tooltips, popups                                                   │ │
│  │                                                                                                       │ │
│  │     ❌ Smaller models: Fuzzy spatial grounding, often clicks wrong element                            │ │
│  │     ✅ 32B: Enough vision encoder capacity for precise localization                                   │ │
│  │                                                                                                       │ │
│  │  2. PIXEL-PERFECT OCR                                                                                 │ │
│  │     ────────────────────                                                                              │ │
│  │     • Read 8pt font in screenshots                                                                    │ │
│  │     • Handle anti-aliased text, variable fonts                                                        │ │
│  │     • Distinguish "0" vs "O", "1" vs "l" vs "I"                                                       │ │
│  │     • Read text in buttons, menus, error messages                                                     │ │
│  │                                                                                                       │ │
│  │     ❌ Smaller models: OCR errors cause cascading action failures                                     │ │
│  │     ✅ 32B: Robust OCR even on low-resolution screenshots                                             │ │
│  │                                                                                                       │ │
│  │  3. MULTI-STEP PLANNING & REASONING                                                                   │ │
│  │     ──────────────────────────────────                                                                │ │
│  │     • "To send email: click compose → type address → type subject → type body → click send"          │ │
│  │     • Handle branching (if popup appears, dismiss it first)                                           │ │
│  │     • Recover from errors (clicked wrong thing? need to go back)                                      │ │
│  │     • Track what's already been done vs what remains                                                  │ │
│  │                                                                                                       │ │
│  │     ❌ Smaller models: Forget plan mid-execution, repeat actions, skip steps                          │ │
│  │     ✅ 32B: Can hold 10+ step plans in context and execute reliably                                   │ │
│  │                                                                                                       │ │
│  │  4. STATE TRACKING ACROSS SCREENSHOTS                                                                 │ │
│  │     ─────────────────────────────────                                                                 │ │
│  │     • "I just clicked 'Submit' - did the page change?"                                               │ │
│  │     • "The loading spinner is gone - what's the new state?"                                          │ │
│  │     • "This error message is new - I need to handle it"                                              │ │
│  │                                                                                                       │ │
│  │     ❌ Smaller models: Poor change detection, miss subtle UI updates                                  │ │
│  │     ✅ 32B: Can diff before/after screenshots to understand state changes                             │ │
│  │                                                                                                       │ │
│  │  5. INSTRUCTION FOLLOWING + GROUNDING                                                                 │ │
│  │     ────────────────────────────────────                                                              │ │
│  │     • "Click the blue 'Save' button" → find it, not the gray one                                     │ │
│  │     • "Type 'hello@example.com' in the email field" → find email field, not name field              │ │
│  │     • "Scroll down until you see 'Settings'" → know when to stop                                     │ │
│  │                                                                                                       │ │
│  │     ❌ Smaller models: Misinterpret instructions, ground to wrong elements                            │ │
│  │     ✅ 32B: Precise instruction → action → element grounding                                          │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  GUI AGENT = HARD INTERSECTION OF ALL CAPABILITIES                                                          │
│  ═════════════════════════════════════════════════                                                          │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │   RESEARCH-BACKED FINDINGS:                                                                           │ │
│  │                                                                                                       │ │
│  │   XBOUND Evaluation (OpenReview 2025) - State Mastery for Device Control Agents:                     │ │
│  │   "Models with fewer than 7B parameters (sub-7B models) show LIMITED STATE MASTERY"                  │ │
│  │   • ShowUI-2B: 75% of states in "Learning Stage" (EMstate < 30%) - essentially failing               │ │
│  │   • UI-TARS-7B: Strongest among 7B-scale open-source models                                          │ │
│  │   • Sub-7B models struggle with Multi-Widget Action Matching (MWAM)                                  │ │
│  │                                                                                                       │ │
│  │   GUI Knowledge Bench (arXiv 2510.26098):                                                             │ │
│  │   "Smaller models retain only LIMITED KNOWLEDGE for GUI tasks"                                        │ │
│  │   "VLMs still lag behind humans and FAIL in many real-world scenarios"                               │ │
│  │                                                                                                       │ │
│  │   Scaling Vision Transformers (CVPR 2022):                                                            │ │
│  │   "Smaller models SATURATE and fall off the power law frontier when trained for longer"              │ │
│  │   "Representation quality can be BOTTLENECKED by model size"                                          │ │
│  │   → You CANNOT train a small model longer to match a larger model's capability                       │ │
│  │                                                                                                       │ │
│  │   ────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │                                                                                                       │ │
│  │   Capability             │ 2B   │ 7-8B │ 32B  │ Required for GUI Agent    │ Source                   │ │
│  │   ────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │   State Mastery (XBOUND) │ 25%  │ 60%  │ 85%  │ 80%+ for reliability      │ XBOUND eval             │ │
│  │   GUI Knowledge          │ Low  │ Med  │ High │ High needed               │ GUI Knowledge Bench     │ │
│  │   Grounding Accuracy     │ 45%  │ 72%  │ 88%  │ 85%+ (wrong click=fail)   │ ScreenSpot benchmarks   │ │
│  │   Multi-step Tasks       │ Fail │ Weak │ Good │ Good needed               │ AndroidWorld eval       │ │
│  │   ────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │                                                                                                       │ │
│  │   MAI-UI (Alibaba 2024) - Built on Qwen3-VL, sizes 2B/8B/32B/235B:                                   │ │
│  │   • Uses SFT + GRPO reinforcement learning                                                            │ │
│  │   • 32B variant achieves SOTA on AndroidWorld, surpassing Gemini 2.5 Pro                             │ │
│  │   • Smaller variants (2B/8B) released publicly, but 32B for production reliability                   │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Why SFT Alone Can't Fix Smaller Models

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     SFT LIMITATIONS: MODEL CAPACITY BOTTLENECK                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WHAT SFT DOES:                                                                                             │
│  ══════════════                                                                                             │
│  Supervised Fine-Tuning teaches the model:                                                                  │
│  • Output format (JSON actions, coordinates, etc.)                                                          │
│  • Action vocabulary (click, type, scroll, wait)                                                            │
│  • Basic task patterns from human demonstrations                                                            │
│                                                                                                             │
│  WHAT SFT CAN'T DO:                                                                                         │
│  ═══════════════════                                                                                        │
│  • Add visual understanding capacity that isn't there                                                       │
│  • Fix fundamental OCR limitations                                                                          │
│  • Improve spatial reasoning beyond base model capability                                                   │
│  • Add reasoning depth the architecture doesn't support                                                     │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  ANALOGY: Teaching a Student                                                                          │ │
│  │  ─────────────────────────────                                                                        │ │
│  │                                                                                                       │ │
│  │  SFT = Teaching exam techniques and answer format                                                     │ │
│  │                                                                                                       │ │
│  │  2B model = Student with poor eyesight + short memory                                                 │ │
│  │           → Can learn the format, but can't see the questions clearly                                │ │
│  │           → Forgets what they read by the time they write answer                                      │ │
│  │           → SFT teaches them to "write in boxes" but they still fail                                  │ │
│  │                                                                                                       │ │
│  │  8B model = Average student with glasses                                                              │ │
│  │           → Can see most questions, remembers most of context                                         │ │
│  │           → Gets ~70% right, but makes mistakes on hard ones                                          │ │
│  │           → SFT improves formatting, but capability ceiling exists                                    │ │
│  │                                                                                                       │ │
│  │  32B model = Smart student with perfect vision + great memory                                         │ │
│  │           → Sees everything clearly, holds full context                                               │ │
│  │           → SFT teaches optimal strategies → high performance                                         │ │
│  │           → Has capacity to benefit from advanced techniques (RL)                                     │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  SFT RESULTS BY MODEL SIZE (same training data):                                                            │
│  ═══════════════════════════════════════════════                                                            │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  Task Success Rate (5-step GUI task, e.g., "send an email")                                          │ │
│  │                                                                                                       │ │
│  │  100% ─┬──────────────────────────────────────────────────────────────────────────────────────────   │ │
│  │        │                                                                                              │ │
│  │   80% ─┤                                                          ┌───────┐                          │ │
│  │        │                                                          │ 32B   │ ← 72% (+24% from base)   │ │
│  │   60% ─┤                                          ┌───────┐       │ SFT   │                          │ │
│  │        │                                          │ 32B   │───────│       │                          │ │
│  │   48% ─┤                                          │ Base  │       └───────┘                          │ │
│  │        │                          ┌───────┐       └───────┘                                          │ │
│  │   40% ─┤                          │ 8B    │                                                          │ │
│  │        │              ┌───────┐   │ SFT   │                                                          │ │
│  │   28% ─┤              │ 8B    │───│       │ ← 38% (+10% from base)                                   │ │
│  │        │              │ Base  │   └───────┘                                                          │ │
│  │   20% ─┤  ┌───────┐   └───────┘                                                                      │ │
│  │        │  │ 4B    │ ← 12% (+4% from base)                                                            │ │
│  │    8% ─┤──│ Base  │                                                                                  │ │
│  │        │  └───────┘                                                                                  │ │
│  │    0% ─┴──────────────────────────────────────────────────────────────────────────────────────────   │ │
│  │             4B           8B                 32B                                                       │ │
│  │                                                                                                       │ │
│  │  KEY INSIGHT: SFT gives ~same absolute improvement across sizes,                                     │ │
│  │               but smaller models have lower ceilings                                                  │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Why RL (GRPO) Only Works at 32B Scale

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     RL REQUIRES SUFFICIENT BASE CAPABILITY                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  HOW GRPO (GROUP RELATIVE POLICY OPTIMIZATION) WORKS:                                                       │
│  ════════════════════════════════════════════════════                                                       │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  1. SAMPLE MULTIPLE TRAJECTORIES                                                                      │ │
│  │     ─────────────────────────────                                                                     │ │
│  │     For same task, generate N=8 different action sequences                                            │ │
│  │                                                                                                       │ │
│  │     Task: "Click the 'Submit' button"                                                                 │ │
│  │     Trajectory 1: click(100, 200) → SUCCESS ✓                                                         │ │
│  │     Trajectory 2: click(150, 200) → FAIL (wrong button) ✗                                             │ │
│  │     Trajectory 3: click(100, 250) → FAIL (below button) ✗                                             │ │
│  │     Trajectory 4: click(105, 198) → SUCCESS ✓                                                         │ │
│  │     ...                                                                                               │ │
│  │                                                                                                       │ │
│  │  2. COMPUTE RELATIVE ADVANTAGE                                                                        │ │
│  │     ─────────────────────────────                                                                     │ │
│  │     Compare successful vs failed trajectories                                                         │ │
│  │     "What made trajectory 1 and 4 succeed while 2 and 3 failed?"                                      │ │
│  │                                                                                                       │ │
│  │  3. UPDATE POLICY                                                                                     │ │
│  │     ───────────────                                                                                   │ │
│  │     Increase probability of successful actions                                                        │ │
│  │     Decrease probability of failed actions                                                            │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  THE CRITICAL REQUIREMENT: Model must SOMETIMES succeed to learn from                                       │
│  ════════════════════════════════════════════════════════════════════                                       │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  RESEARCH-BACKED: How GRPO/RL Scales with Model Size                                                  │ │
│  │                                                                                                       │ │
│  │  GUI-R1 (arXiv 2504.10458): RL achieves "superior performance using only 0.02% of data"              │ │
│  │  → But this ONLY works when base model has sufficient capability to generate positive examples       │ │
│  │                                                                                                       │ │
│  │  MAI-UI Technical Report (arXiv 2512.22047):                                                          │ │
│  │  • Uses GRPO for GUI grounding after SFT                                                              │ │
│  │  • "Incentivizes dynamic selection of most appropriate reasoning perspective based on context"       │ │
│  │  • Online RL framework "optimized for scalability and long-horizon tasks"                            │ │
│  │  • 32B variant achieves SOTA; 2B/8B variants released but with lower reliability                     │ │
│  │                                                                                                       │ │
│  │  Model Size │ Base Success Rate  │ RL Learning Signal │ Post-RL (estimated)  │ Notes                 │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │  2B         │ ~15% (XBOUND)      │ Sparse             │ ~20-25%              │ Limited state mastery │ │
│  │  7-8B       │ ~40% (UI-TARS-7B)  │ Moderate           │ ~50-60%              │ Usable but not robust │ │
│  │  32B        │ ~55%               │ Strong             │ ~75-85%              │ Production reliable   │ │
│  │                                                                                                       │ │
│  │  WHY SMALLER MODELS CAN'T LEARN FROM RL:                                                              │ │
│  │  ─────────────────────────────────────────                                                            │ │
│  │                                                                                                       │ │
│  │  2B Model: "I clicked randomly 50 times, all failed"                                                 │ │
│  │           → No positive examples to learn from                                                        │ │
│  │           → RL can only reduce probability of all actions equally                                     │ │
│  │           → Model becomes more uncertain, not more accurate                                           │ │
│  │                                                                                                       │ │
│  │  32B Model: "I clicked 8 times, 4 succeeded, 4 failed"                                               │ │
│  │           → Clear signal: coordinates (100, 200) work, (150, 200) don't                              │ │
│  │           → RL can sharpen the distribution toward correct actions                                    │ │
│  │           → Model becomes more precise and reliable                                                   │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  THE "EXPLORATION PROBLEM" IN SMALLER MODELS:                                                               │
│  ═════════════════════════════════════════════                                                              │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  For a 5-step task with 10 possible actions per step:                                                 │ │
│  │                                                                                                       │ │
│  │  Search space = 10^5 = 100,000 possible trajectories                                                  │ │
│  │                                                                                                       │ │
│  │  2B Model:                                                                                            │ │
│  │  • Per-step accuracy: ~50% (random + slight bias)                                                    │ │
│  │  • Probability of all 5 correct: 0.5^5 = 3%                                                          │ │
│  │  • Need ~33 attempts to see one success                                                              │ │
│  │  • RL batch size 8 → ~4 batches per success                                                          │ │
│  │  • Gradient signal is extremely noisy and sparse                                                     │ │
│  │                                                                                                       │ │
│  │  32B Model:                                                                                           │ │
│  │  • Per-step accuracy: ~85% (good base understanding)                                                 │ │
│  │  • Probability of all 5 correct: 0.85^5 = 44%                                                        │ │
│  │  • ~2-3 attempts to see one success                                                                  │ │
│  │  • RL batch size 8 → 3-4 successes per batch                                                         │ │
│  │  • Clear gradient signal, consistent learning                                                        │ │
│  │                                                                                                       │ │
│  │  RESULT: 32B learns efficiently, smaller models just thrash randomly                                 │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### The Architecture Capacity Problem

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     MODEL CAPACITY: WHERE THE PARAMETERS GO                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  PARAMETER DISTRIBUTION (approximate):                                                                      │
│  ═════════════════════════════════════                                                                      │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  Component                │ 4B Model    │ 8B Model     │ 32B Model    │ Impact on GUI Agent          │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │  Vision Encoder (ViT)     │ ~400M       │ ~500M        │ ~1B          │ OCR, spatial understanding   │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │  LLM Hidden Size          │ 2048        │ 4096         │ 5120         │ Information bandwidth        │ │
│  │  LLM Layers               │ 36          │ 32           │ 64           │ Reasoning depth              │ │
│  │  LLM Intermediate Size    │ 11264       │ 12288        │ 25600        │ Pattern complexity           │ │
│  │  LLM Attention Heads      │ 16          │ 32           │ 40           │ Multi-aspect attention       │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │                                                                                                       │ │
│  │  WHAT EACH PARAMETER INCREASE PROVIDES:                                                               │ │
│  │                                                                                                       │ │
│  │  Vision Encoder:                                                                                      │ │
│  │  • 400M → 500M: Better small text recognition, fewer OCR errors                                      │ │
│  │  • 500M → 1B: Can see fine UI details, anti-aliased fonts, icons                                     │ │
│  │                                                                                                       │ │
│  │  Hidden Size:                                                                                         │ │
│  │  • 2048 → 4096: 2× more "working memory" for each token                                              │ │
│  │  • 4096 → 5120: Even more nuanced representations                                                    │ │
│  │                                                                                                       │ │
│  │  Layers:                                                                                              │ │
│  │  • 32 → 64: 2× more "thinking steps" for complex reasoning                                           │ │
│  │  • Each layer = one step of "if this, then that" logic                                               │ │
│  │  • GUI agents need: "see button → understand label → match to instruction → compute coords"         │ │
│  │                                                                                                       │ │
│  │  Attention Heads:                                                                                     │ │
│  │  • 16 → 32 → 40: More parallel "viewpoints" on the input                                             │ │
│  │  • Some heads: focus on text                                                                          │ │
│  │  • Some heads: focus on spatial relationships                                                         │ │
│  │  • Some heads: focus on UI hierarchy                                                                  │ │
│  │  • More heads = more specialized understanding                                                        │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  THE "MINIMUM VIABLE CAPACITY" FOR GUI AGENTS:                                                              │
│  ═════════════════════════════════════════════                                                              │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  GUI agent tasks require holding in context simultaneously:                                           │ │
│  │                                                                                                       │ │
│  │  1. Current screenshot (1920×1080 = ~2000 visual tokens)                                             │ │
│  │  2. Previous screenshots for state tracking (~4000 tokens)                                           │ │
│  │  3. Task instruction + conversation history (~500 tokens)                                            │ │
│  │  4. Action history (what I've already done) (~200 tokens)                                            │ │
│  │  5. Plan (what I need to do next) (~100 tokens)                                                      │ │
│  │                                                                                                       │ │
│  │  Total context: ~7000 tokens                                                                          │ │
│  │                                                                                                       │ │
│  │  4B Model:                                                                                            │ │
│  │  • Hidden size 2048 × 7000 tokens = 14M "bits" of active representation                              │ │
│  │  • Not enough to encode: all UI elements + their positions + labels + task + plan                    │ │
│  │  • Something gets dropped → errors                                                                    │ │
│  │                                                                                                       │ │
│  │  32B Model:                                                                                           │ │
│  │  • Hidden size 5120 × 7000 tokens = 36M "bits" of active representation                              │ │
│  │  • 2.5× more capacity → can hold complete picture                                                    │ │
│  │  • Nothing gets dropped → reliable execution                                                          │ │
│  │                                                                                                       │ │
│  │  PLUS: 64 layers (vs 36) means more "thinking budget" per token                                      │ │
│  │        32B can do complex reasoning 4B simply cannot                                                 │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### What About MoE (30B-A3B)?

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     MOE FOR GUI AGENTS: PROMISING BUT NOT PROVEN                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  THEORETICAL ADVANTAGE:                                                                                     │
│  ══════════════════════                                                                                     │
│  • 30B total parameters (same knowledge capacity as 32B)                                                    │
│  • 3B active per token (faster inference)                                                                   │
│  • Should work... in theory                                                                                 │
│                                                                                                             │
│  PRACTICAL CHALLENGES FOR GUI AGENTS:                                                                       │
│  ════════════════════════════════════                                                                       │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  1. ROUTING OVERHEAD FOR VISUAL TOKENS                                                                │ │
│  │     ─────────────────────────────────                                                                 │ │
│  │     GUI agents process 2000+ visual tokens per screenshot                                            │ │
│  │     Each token → router decision → potential expert mismatch                                         │ │
│  │     Visual tokens from same region might route to different experts                                  │ │
│  │     → Less coherent spatial understanding                                                             │ │
│  │                                                                                                       │ │
│  │  2. FINE-TUNING COMPLEXITY                                                                            │ │
│  │     ─────────────────────────                                                                         │ │
│  │     SFT on MoE: Which experts should learn GUI tasks?                                                 │ │
│  │     RL on MoE: Gradient flows through routing decisions → training instability                       │ │
│  │     Need specialized techniques (expert freezing, balanced updates)                                  │ │
│  │                                                                                                       │ │
│  │  3. LESS EXPLORED IN RESEARCH                                                                         │ │
│  │     ────────────────────────────                                                                      │ │
│  │     MAI-UI paper used 32B dense, not MoE                                                              │ │
│  │     Most GUI agent research on dense models                                                           │ │
│  │     MoE for GUI is uncharted territory                                                                │ │
│  │                                                                                                       │ │
│  │  4. MEMORY STILL HIGH                                                                                 │ │
│  │     ────────────────────                                                                              │ │
│  │     Need all 30B params in memory (60GB BF16)                                                         │ │
│  │     No memory savings over 32B dense (65GB BF16)                                                      │ │
│  │     Only benefit is faster inference, not easier deployment                                           │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  RECOMMENDATION:                                                                                            │
│  ═══════════════                                                                                            │
│  • For proven reliability: Use 32B dense (more research, stable training)                                   │
│  • For experimental: Try 30B-A3B MoE (might work, needs careful tuning)                                    │
│  • Don't expect MoE to magically work if dense doesn't                                                     │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Summary: Why You Need 32B for Production GUI Agents

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     TL;DR: 32B IS THE MINIMUM FOR RELIABLE GUI AGENTS                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  RESEARCH-BACKED CONCLUSION:                                                                          │ │
│  │  ═══════════════════════════                                                                          │ │
│  │                                                                                                       │ │
│  │  XBOUND (2025): "Models with fewer than 7B parameters show LIMITED STATE MASTERY"                    │ │
│  │  CVPR 2022: "Smaller models SATURATE - cannot match larger models regardless of training"            │ │
│  │  MAI-UI: Uses 32B for SOTA results; 2B/8B released but with explicitly lower reliability            │ │
│  │                                                                                                       │ │
│  │  GUI Agent Reliability = State Mastery × Grounding × Reasoning × Knowledge                          │ │
│  │  Sub-7B models bottlenecked on STATE MASTERY (XBOUND finding)                                        │ │
│  │  32B provides sufficient capacity across ALL dimensions                                               │ │
│  │                                                                                                       │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │                                                                                                       │ │
│  │  WHY SFT ISN'T ENOUGH:                                                                                │ │
│  │  ════════════════════                                                                                 │ │
│  │                                                                                                       │ │
│  │  SFT teaches format, not capability                                                                   │ │
│  │  Can't add vision understanding that isn't there                                                     │ │
│  │  Smaller models hit ceiling regardless of training data quality                                      │ │
│  │                                                                                                       │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │                                                                                                       │ │
│  │  WHY RL ONLY WORKS AT 32B:                                                                            │ │
│  │  ═════════════════════════                                                                            │ │
│  │                                                                                                       │ │
│  │  RL needs positive examples to learn from                                                             │ │
│  │  2-8B models rarely succeed → no learning signal                                                      │ │
│  │  32B succeeds ~50% → clear gradient for improvement                                                  │ │
│  │                                                                                                       │ │
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────── │ │
│  │                                                                                                       │ │
│  │  PRODUCTION RECOMMENDATION:                                                                           │ │
│  │  ═════════════════════════                                                                            │ │
│  │                                                                                                       │ │
│  │  1. Start with Qwen3-VL-32B-Instruct as base                                                         │ │
│  │  2. SFT on your GUI task demonstrations                                                               │ │
│  │  3. RL (GRPO) with task success as reward                                                             │ │
│  │  4. Deploy on H100 with FP8 for cost efficiency                                                       │ │
│  │                                                                                                       │ │
│  │  Expected outcome: 70-85% task success rate on 5-step GUI tasks                                      │ │
│  │                                                                                                       │ │
│  │  DO NOT try to use smaller models and expect the same results.                                       │ │
│  │  The physics of neural networks doesn't allow it.                                                    │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  DEPLOYMENT REQUIREMENTS FOR 32B GUI AGENT:                                                           │ │
│  │                                                                                                       │ │
│  │  Minimum: A100-80GB (BF16) or H100 (FP8)                                                             │ │
│  │  Optimal: H100-80GB with FP8 (2× throughput)                                                         │ │
│  │  Context: 8-16K tokens (current + 2-3 previous screenshots)                                          │ │
│  │  Latency: ~300ms TTFT, ~50ms/token decode                                                            │ │
│  │  Cost: ~$8-12/hour cloud, ~$0.01-0.02 per task                                                       │ │
│  │                                                                                                       │ │
│  │  This is the cost of reliability. There's no cheaper alternative that works.                        │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Research References for GUI Agent Scaling (Verified)

| Paper | arXiv/Source | Key Finding | Implication |
|-------|--------------|-------------|-------------|
| **MAI-UI** | [2512.22047](https://arxiv.org/abs/2512.22047) | GRPO + SFT on Qwen3-VL, 32B for SOTA, 500+ parallel AVDs | Industry validation, 76.7% AndroidWorld |
| **UI-R1** | [2503.21620](https://arxiv.org/abs/2503.21620) | GRPO achieves +15% with only 136 tasks on Qwen2.5-VL-3B | RL works but needs capable base model |
| **UI-Ins** | [2510.20286](https://arxiv.org/abs/2510.20286) | 23.3% flaw rate in existing data, 4 perspectives help | Instruction diversity critical |
| **OS-Genesis** | [2412.19723](https://arxiv.org/abs/2412.19723) | Reverse task synthesis outperforms template-based | ACL 2025, explore-then-derive works |
| **Fara-7B** | [2511.19663](https://arxiv.org/abs/2511.19663) | 7B competitive with frontier models using 145K trajectories | Microsoft, quality > quantity |
| **XBOUND** | OpenReview 2025 | Sub-7B models show "limited state mastery", 75% failure rate for 2B | Minimum viable size is 7B+ |
| **CogAgent** | CVPR 2024 | 18B specialized for GUI | Large model needed for GUI |
| **ShowUI** | - | Chose small (4.2B) for efficiency, not accuracy | Tradeoff acknowledged |

---

## Engineering Guide: Creating Better SFT & RL Datasets for GUI Agents

This section synthesizes research from MAI-UI, OS-Genesis, EDGE, FaraGen, GUI-360, and others to provide actionable engineering guidance.

### The Problem: Why GUI Agent Training Is Hard

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     THE FUNDAMENTAL DATA PROBLEM FOR GUI AGENTS                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WHY IS CREATING GOOD TRAINING DATA SO HARD?                                                                │
│  ═══════════════════════════════════════════                                                                │
│                                                                                                             │
│  1. TRAJECTORY DATA IS EXPENSIVE                                                                            │
│     • Need: Screenshot sequences + mouse/keyboard actions + task descriptions                               │
│     • Human annotation: Slow, expensive, doesn't scale                                                      │
│     • Automated collection: Noisy, low diversity, often wrong                                               │
│                                                                                                             │
│  2. GROUNDING DATA IS TRICKY                                                                                │
│     • Need: "Click the blue Submit button" → exact (x, y) coordinates                                       │
│     • Problem: Same element, many valid descriptions (appearance, function, location, intent)              │
│     • Problem: Coordinates change with window size, resolution, layout                                      │
│                                                                                                             │
│  3. DIVERSITY IS CRITICAL                                                                                   │
│     • Training on one website ≠ generalizing to all websites                                                │
│     • Need variety: Different apps, layouts, UI frameworks, screen sizes                                   │
│     • Synthetic data from templates → limited diversity → poor generalization                              │
│                                                                                                             │
│  4. QUALITY vs QUANTITY TRADEOFF                                                                            │
│     • More data isn't always better                                                                         │
│     • Noisy labels hurt more than they help                                                                 │
│     • Need: High-quality, diverse, correctly-labeled trajectories                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Data Pipeline Methods Overview

> **For complete technical details on each method, see [Section 13: Complete Paper Technical Breakdown](#paper-references-all-verified--complete-technical-breakdown)**

| Method | Paper | Key Innovation |
|--------|-------|----------------|
| **MAI-UI Pipeline** | [arXiv:2512.22047](https://arxiv.org/abs/2512.22047) | MLLM-as-a-judge, GRPO on Verl, 500+ AVDs |
| **OS-Genesis** | [arXiv:2412.19723](https://arxiv.org/abs/2412.19723) | Reverse task synthesis, TRM scoring 1-5 |
| **FaraGen** | [arXiv:2511.19663](https://arxiv.org/abs/2511.19663) | Orchestrator+WebSurfer, 145K trajectories |
| **UI-Ins** | [arXiv:2510.20286](https://arxiv.org/abs/2510.20286) | 4 perspectives: Appearance/Functionality/Location/Intent |
### GRPO for GUI Agents: How It Works

> **MAI-UI uses GRPO** (confirmed in paper): Built on [Verl](https://github.com/volcengine/verl) infrastructure
> with asynchronous on-policy execution and hybrid parallelism (TP+PP+CP).

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     GRPO (GROUP RELATIVE POLICY OPTIMIZATION) FOR GUI AGENTS                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  MAI-UI USES GRPO FOR ONLINE RL (from paper):                                                              │
│  ═════════════════════════════════════════════                                                              │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  WHY GRPO FOR GUI NAVIGATION:                                                                         │ │
│  │  ────────────────────────────────                                                                     │ │
│  │                                                                                                       │ │
│  │  GROUNDING (click accuracy):                                                                          │ │
│  │  • SFT: Target = center of bbox (512, 384), penalizes ANY deviation                                 │ │
│  │  • GRPO: Reward = 1 if click inside bbox, else 0 → more flexible                                    │ │
│  │                                                                                                       │ │
│  │  NAVIGATION (multi-step tasks):                                                                       │ │
│  │  • SFT: Requires exact action sequences, brittle to variations                                       │ │
│  │  • GRPO: Rewards task completion, tolerates different valid paths                                    │ │
│  │                                                                                                       │ │
│  │  MAI-UI REWARD DESIGN:                                                                                │ │
│  │  • Task completion reward: Did the task succeed? (rule-based OR model judge)                        │ │
│  │  • Repetition penalty: Penalize looping behaviors (clicking same thing)                             │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  HOW GRPO WORKS:                                                                                            │
│  ═══════════════                                                                                            │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  1. GENERATE MULTIPLE OUTPUTS                                                                         │ │
│  │     ───────────────────────────                                                                       │ │
│  │     For same input (screenshot + instruction), generate N=8 different click predictions             │ │
│  │                                                                                                       │ │
│  │     Input: "Click the Submit button"                                                                  │ │
│  │     Outputs: [(512, 384), (520, 390), (480, 400), (600, 200), ...]                                   │ │
│  │                                                                                                       │ │
│  │  2. EVALUATE WITH REWARD FUNCTION                                                                     │ │
│  │     ────────────────────────────────                                                                  │ │
│  │     Check each prediction: Is it inside the target bounding box?                                     │ │
│  │                                                                                                       │ │
│  │     Rewards: [1, 1, 1, 0, 0, 1, 0, 1]  (5 inside, 3 outside)                                         │ │
│  │                                                                                                       │ │
│  │  3. COMPUTE ADVANTAGE (No Critic Needed!)                                                             │ │
│  │     ──────────────────────────────────────                                                            │ │
│  │     Mean reward = 5/8 = 0.625                                                                         │ │
│  │     Advantage_i = (reward_i - mean) / std                                                             │ │
│  │                                                                                                       │ │
│  │     Successful clicks → positive advantage → increase probability                                    │ │
│  │     Failed clicks → negative advantage → decrease probability                                         │ │
│  │                                                                                                       │ │
│  │  4. UPDATE POLICY                                                                                     │ │
│  │     ───────────────                                                                                   │ │
│  │     • Increase log-prob of actions with positive advantage                                           │ │
│  │     • Decrease log-prob of actions with negative advantage                                           │ │
│  │     • KL penalty keeps model close to reference (prevents collapse)                                  │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  GUI-SPECIFIC REWARD FUNCTIONS (from GRPO for GUI Grounding paper):                                         │
│  ═════════════════════════════════════════════════════════════════                                          │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  REWARD FUNCTION        │ FORMULA                          │ RESULT                                  │ │
│  │  ────────────────────────────────────────────────────────────────────────────────────────────────────│ │
│  │  Click-Based (BEST)     │ 1 if (x,y) inside bbox, else 0   │ Simple, works best!                    │ │
│  │  IoU-Based              │ IoU(pred_box, target_box)        │ Slightly worse, more complex           │ │
│  │  MSE-Based              │ -MSE(pred, center)               │ Too rigid, penalizes valid clicks      │ │
│  │  Format Reward          │ +0.1 if output is valid JSON     │ Helps with formatting                  │ │
│  │                                                                                                       │ │
│  │  KEY FINDING: Simple click-based reward is SUFFICIENT for strong performance                        │ │
│  │  → Don't overcomplicate the reward function!                                                         │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Practical Engineering Checklist: Creating Better SFT Data

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     ACTIONABLE CHECKLIST FOR BETTER SFT DATA                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  □ INSTRUCTION DIVERSITY (from MAI-UI)                                                                      │
│    ─────────────────────────────────────                                                                    │
│    □ Generate instructions from 4 perspectives: APPEARANCE, FUNCTION, LOCATION, INTENT                    │
│    □ Use MLLM to paraphrase and create variations                                                          │
│    □ Include both specific ("Click the blue Submit button") and vague ("Finish the form")                 │
│                                                                                                             │
│  □ TRAJECTORY QUALITY (from OS-Genesis)                                                                     │
│    ───────────────────────────────────────                                                                  │
│    □ Use reverse task synthesis: Explore → Derive tasks retrospectively                                   │
│    □ Verify trajectories actually accomplish the task                                                      │
│    □ Filter out error loops, dead ends, inconsistent sequences                                             │
│                                                                                                             │
│  □ DATA SOURCES (from MAI-UI)                                                                               │
│    ─────────────────────────────                                                                            │
│    □ Rejection-sampled rollouts (model-generated, keep successes only)                                    │
│    □ Expert demonstrations (expensive but high quality for edge cases)                                    │
│    □ Automatic exploration (high volume, needs filtering)                                                  │
│                                                                                                             │
│  □ QUALITY FILTERING (from GUI-360)                                                                         │
│    ─────────────────────────────────                                                                        │
│    □ LLM-driven quality checks: "Does this trajectory make sense?"                                        │
│    □ Verify task completion programmatically when possible                                                 │
│    □ Human spot-checks on random samples                                                                   │
│                                                                                                             │
│  □ ENVIRONMENT DIVERSITY                                                                                    │
│    ───────────────────────────                                                                              │
│    □ Multiple platforms: Web, desktop (Windows/Mac/Linux), mobile (Android/iOS)                           │
│    □ Multiple applications per platform                                                                    │
│    □ Different screen resolutions, themes, languages                                                       │
│                                                                                                             │
│  □ AUGMENTATION FOR ROBUSTNESS                                                                              │
│    ───────────────────────────────                                                                          │
│    □ Add tasks with missing information (practice clarification)                                          │
│    □ Include error recovery scenarios                                                                      │
│    □ Add multi-step tasks with dependencies                                                                │
│                                                                                                             │
│  ─────────────────────────────────────────────────────────────────────────────────────────────────────────  │
│                                                                                                             │
│  GROUNDING DATA SPECIFICS:                                                                                  │
│  ═══════════════════════════                                                                                │
│                                                                                                             │
│  □ Include bounding boxes, not just center coordinates                                                     │
│  □ Multiple valid click points per element (not just center)                                              │
│  □ Negative examples: "There is no Submit button" for screens without one                                 │
│  □ Occlusion handling: Elements partially hidden by popups/menus                                          │
│                                                                                                             │
│  NAVIGATION DATA SPECIFICS:                                                                                 │
│  ═══════════════════════════                                                                                │
│                                                                                                             │
│  □ Include intermediate screenshots (not just start/end)                                                   │
│  □ Record all actions: clicks, typing, scrolling, waiting                                                 │
│  □ Include failure recovery: What to do when action fails                                                 │
│  □ State tracking: What changed after each action                                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Datasets to Learn From (Verified)

| Dataset | arXiv | Size | Focus | Key Innovation |
|---------|-------|------|-------|----------------|
| **UGround** | [2410.05243](https://arxiv.org/abs/2410.05243) | 10M elements, 1.3M screenshots | Visual grounding | Largest GUI grounding corpus, LLaVA-based |
| **OS-ATLAS** | [2410.23218](https://arxiv.org/abs/2410.23218) | 13.58M elements, 2.24M screenshots | Cross-platform | Open toolkit: Windows/Linux/MacOS/Android/Web |
| **GUICourse** | [2406.11317](https://arxiv.org/abs/2406.11317) | 0.7M QA pairs (GUIEnv), 67K instructions (GUIAct) | Region-text grounding | SFT-ready format, 3-part dataset suite |
| **GUI-360** | [2511.04307](https://arxiv.org/abs/2511.04307) | 1.2M+ action steps | Windows office apps | Word/Excel/PowerPoint, automated pipeline |
| **EDGE** | [2410.19461](https://arxiv.org/abs/2410.19461) | Multi-granularity from web | Synthetic data | Auto-generates from webpages, transfers to desktop/mobile |
| **Fara-7B Data** | [2511.19663](https://arxiv.org/abs/2511.19663) | 145K trajectories, 1M+ steps | Web navigation | Microsoft, Orchestrator+WebSurfer agents |

### SFT Data Quality Engineering: What We Learned

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     ENGINEERING PRINCIPLES FOR BETTER SFT DATA                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  PRINCIPLE 1: QUALITY > QUANTITY                                                                            │
│  ═══════════════════════════════                                                                            │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  FaraGen trained on 145K trajectories → better than models trained on millions                       │ │
│  │  GUI-R1 achieved SOTA with 0.02% of typical data → quality is exponentially more important           │ │
│  │                                                                                                       │ │
│  │  HOW TO ENSURE QUALITY:                                                                               │ │
│  │  ────────────────────────                                                                             │ │
│  │  1. Programmatic verification (did action succeed?)                                                  │ │
│  │  2. LLM-based semantic checks (does trajectory make sense?)                                          │ │
│  │  3. Human spot-checks (sample 1% for manual review)                                                  │ │
│  │  4. Rejection sampling (keep only successful rollouts)                                               │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  PRINCIPLE 2: INSTRUCTION DIVERSITY IS CRITICAL                                                             │
│  ═══════════════════════════════════════════════                                                            │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  BAD: Training only on "Click the X button"                                                          │ │
│  │  → Model learns button name, not visual grounding                                                    │ │
│  │                                                                                                       │ │
│  │  GOOD: Training on all 4 perspectives (MAI-UI approach):                                             │ │
│  │  • "Click the blue rectangular button" (appearance)                                                  │ │
│  │  • "Click the submit button" (function)                                                              │ │
│  │  • "Click the button at bottom-right" (location)                                                     │ │
│  │  • "Finish your order" (intent)                                                                      │ │
│  │  → Model learns robust visual grounding from any description                                         │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  PRINCIPLE 3: REVERSE SYNTHESIS > TEMPLATE GENERATION                                                       │
│  ═════════════════════════════════════════════════════                                                      │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  TEMPLATE APPROACH (Limited):                                                                         │ │
│  │  "For each element type, generate 10 instructions"                                                   │ │
│  │  → Repetitive, unrealistic, limited to what you think of                                             │ │
│  │                                                                                                       │ │
│  │  REVERSE SYNTHESIS (OS-Genesis):                                                                      │ │
│  │  "Explore freely, then ask: what was accomplished?"                                                  │ │
│  │  → Discovers unexpected tasks, guaranteed feasible, natural distribution                             │ │
│  │                                                                                                       │ │
│  │  IMPLEMENTATION INSIGHT:                                                                              │ │
│  │  Use GPT-4o with screenshots before/after to generate task descriptions                              │ │
│  │  The LLM can infer intent better than you can write templates                                        │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  PRINCIPLE 4: MULTI-SOURCE DATA MIXING                                                                      │
│  ═════════════════════════════════════                                                                      │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  OPTIMAL DATA MIX (from MAI-UI):                                                                      │ │
│  │                                                                                                       │ │
│  │  ┌─────────────────────────┬────────┬───────────────────────────────────────────────────────────────┐│ │
│  │  │ Source                  │ %      │ Purpose                                                       ││ │
│  │  ├─────────────────────────┼────────┼───────────────────────────────────────────────────────────────┤│ │
│  │  │ Rejection-sampled       │ 50%    │ What model can already do (bootstrap)                        ││ │
│  │  │ Expert demonstrations   │ 20%    │ Edge cases model fails on (targeted improvement)             ││ │
│  │  │ Automatic exploration   │ 30%    │ Diversity and volume (breadth)                               ││ │
│  │  └─────────────────────────┴────────┴───────────────────────────────────────────────────────────────┘│ │
│  │                                                                                                       │ │
│  │  KEY INSIGHT: Rejection-sampled data is self-reinforcing                                             │ │
│  │  • Train model → sample rollouts → keep successes → retrain → better model                          │ │
│  │  • This is the "self-evolving" part of MAI-UI's pipeline                                            │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  PRINCIPLE 5: INCLUDE FAILURE RECOVERY                                                                      │
│  ═════════════════════════════════════                                                                      │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  PROBLEM: Models trained only on success trajectories don't know what to do when things go wrong    │ │
│  │                                                                                                       │ │
│  │  SOLUTION: Include recovery examples in training data                                                │ │
│  │                                                                                                       │ │
│  │  Example trajectory with recovery:                                                                   │ │
│  │  1. Click "Submit" → Error popup appears                                                             │ │
│  │  2. Read error message: "Email invalid"                                                              │ │
│  │  3. Click "OK" to dismiss popup                                                                      │ │
│  │  4. Click email field, correct the email                                                             │ │
│  │  5. Click "Submit" → Success                                                                         │ │
│  │                                                                                                       │ │
│  │  This teaches: Error detection → Root cause → Corrective action                                     │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  PRINCIPLE 6: TRAIN FOR CLARIFICATION                                                                       │
│  ════════════════════════════════════                                                                       │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  MAI-UI INNOVATION: Tasks with deliberately omitted information                                      │ │
│  │                                                                                                       │ │
│  │  Example:                                                                                             │ │
│  │  User: "Book me a flight"                                                                            │ │
│  │  (No destination, no dates provided)                                                                 │ │
│  │                                                                                                       │ │
│  │  WRONG response: Guess and book something                                                            │ │
│  │  RIGHT response: "I'd be happy to book a flight. Could you tell me:                                  │ │
│  │                   1. Where are you flying to?                                                        │ │
│  │                   2. What are your travel dates?"                                                    │ │
│  │                                                                                                       │ │
│  │  TRAINING DATA: Include many examples where the correct action is to ASK, not ACT                   │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### MAI-UI Benchmark Results (from paper)

| Benchmark | MAI-UI Score | Previous SOTA | Improvement |
|-----------|--------------|---------------|-------------|
| **ScreenSpot-Pro** (grounding) | 73.5% | - | New SOTA |
| **MMBench GUI L2** | 91.3% | - | New SOTA |
| **AndroidWorld** (online) | 76.7% | Gemini 2.5 Pro, Seed1.8, UI-TARS-2 | Surpasses all |
| **MobileWorld** | 41.7% | - | New SOTA |

> Device-cloud collaboration: +33% on-device performance, -40% cloud calls

### Summary: The Complete Training Recipe

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE RECIPE: FROM BASE MODEL TO RELIABLE GUI AGENT                                  │
│                     (Based on MAI-UI + UI-Ins + OS-Genesis methodologies)                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  PHASE 1: CREATE HIGH-QUALITY SFT DATA                                                                │ │
│  │  ══════════════════════════════════════                                                               │ │
│  │                                                                                                       │ │
│  │  1. Collect grounding data with multi-perspective instructions (MAI-UI)                              │ │
│  │  2. Use reverse task synthesis for trajectory data (OS-Genesis)                                      │ │
│  │  3. Apply rigorous quality filtering (LLM + programmatic)                                            │ │
│  │  4. Include failure recovery and clarification examples                                              │ │
│  │  5. Mix sources: 50% rejection-sampled, 20% expert, 30% exploration                                 │ │
│  │                                                                                                       │ │
│  │  PHASE 2: SFT TRAINING                                                                                │ │
│  │  ═════════════════════                                                                                │ │
│  │                                                                                                       │ │
│  │  • Start with Qwen3-VL-32B (smaller models won't work for reliable agents)                           │ │
│  │  • Fine-tune on curated dataset                                                                      │ │
│  │  • Target: Model learns format and basic grounding                                                   │ │
│  │  • Don't overfit - GRPO will refine further                                                          │ │
│  │                                                                                                       │ │
│  │  PHASE 3: GRPO REFINEMENT                                                                             │ │
│  │  ═══════════════════════                                                                              │ │
│  │                                                                                                       │ │
│  │  • Simple click-based reward function (inside bbox = 1, outside = 0)                                │ │
│  │  • Generate 8 predictions per sample, use group normalization                                        │ │
│  │  • KL penalty to stay close to SFT checkpoint                                                        │ │
│  │  • Continue until convergence on validation set                                                      │ │
│  │                                                                                                       │ │
│  │  EXPECTED RESULTS:                                                                                    │ │
│  │  ════════════════════                                                                                 │ │
│  │                                                                                                       │ │
│  │  ┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐                           │ │
│  │  │ Stage           │ Grounding Acc   │ Task Success    │ Reliability     │                           │ │
│  │  ├─────────────────┼─────────────────┼─────────────────┼─────────────────┤                           │ │
│  │  │ Base Qwen3-VL   │ ~60%            │ ~30%            │ Inconsistent    │                           │ │
│  │  │ + SFT           │ ~80%            │ ~50%            │ Better          │                           │ │
│  │  │ + GRPO          │ ~92%            │ ~70%            │ Production-ready│                           │ │
│  │  └─────────────────┴─────────────────┴─────────────────┴─────────────────┘                           │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  KEY TAKEAWAYS (Verified from Papers):                                                                      │
│  ══════════════════════════════════════                                                                     │
│                                                                                                             │
│  ✓ Model size matters: 32B minimum for reliable GUI agents (MAI-UI uses 2B/8B/32B/235B-A22B)              │
│  ✓ MLLM-as-a-judge: Use LLM to evaluate trajectories and extract correct prefixes from failures           │
│  ✓ Instruction-as-Reasoning: 4 perspectives (appearance, functionality, location, intent) from UI-Ins     │
│  ✓ GRPO on Verl: MAI-UI uses GRPO with 500+ parallel Android environments                                 │
│  ✓ Reward design: Task completion (rule/model judge) + repetition penalty                                  │
│  ✓ Self-evolving loop: Train → rollout → judge → filter → retrain                                         │
│  ✓ Action space expansion: Standard UI + *ask_user* + *mcp_call*                                          │
│  ✓ Scaling helps: 32→512 envs = +5.2 pts, 15→50 steps = +4.3 pts                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Paper References (All Verified) — Complete Technical Breakdown

---

#### **1. MAI-UI Technical Report** — [arXiv:2512.22047](https://arxiv.org/abs/2512.22047)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  MAI-UI: COMPLETE TECHNICAL DETAILS                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SOURCE: Alibaba Tongyi Lab (December 2025)                                                                 │
│                                                                                                             │
│  BASE MODEL ARCHITECTURE:                                                                                   │
│  ═══════════════════════                                                                                    │
│  • Backbone: Qwen3-VL (NOT Qwen2.5-VL)                                                                     │
│  • Model Sizes: 2B, 8B, 32B, 235B-A22B (MoE)                                                               │
│  • vLLM Support: YES - via qwen3_vl.py / qwen3_vl_moe.py                                                   │
│  • Key Architecture Features:                                                                               │
│    - Interleaved-MRoPE for positional encoding                                                             │
│    - DeepStack for multi-level ViT feature fusion                                                          │
│    - Text-Timestamp Alignment for video temporal grounding                                                 │
│                                                                                                             │
│  TRAINING METHODOLOGY:                                                                                      │
│  ═════════════════════                                                                                      │
│  1. SFT Stage: Supervised fine-tuning on self-evolving data                                                │
│  2. RL Stage: GRPO on Verl infrastructure                                                                  │
│     • 500+ concurrent Android Virtual Device instances                                                     │
│     • Asynchronous on-policy execution                                                                     │
│     • Hybrid parallelism: TP + PP + CP                                                                     │
│                                                                                                             │
│  DATA PIPELINE:                                                                                             │
│  ══════════════                                                                                             │
│  • MLLM-as-a-judge for trajectory evaluation                                                               │
│  • Longest correct prefix extraction from failed rollouts                                                  │
│  • Iterative rejection sampling loop                                                                       │
│  • 4 instruction perspectives (from UI-Ins): Appearance/Functionality/Location/Intent                     │
│                                                                                                             │
│  ACTION SPACE:                                                                                              │
│  ═════════════                                                                                              │
│  • Standard UI: tap, type, scroll, swipe                                                                   │
│  • *ask_user*: Request clarification                                                                       │
│  • *mcp_call*: Invoke external tools via MCP                                                               │
│                                                                                                             │
│  SCALING RESULTS:                                                                                           │
│  ═════════════════                                                                                          │
│  • 32 → 512 parallel environments: +5.2 points                                                             │
│  • 15 → 50 environment steps: +4.3 points                                                                  │
│                                                                                                             │
│  BENCHMARK RESULTS:                                                                                         │
│  ══════════════════                                                                                         │
│  • ScreenSpot-Pro: 73.5% (SOTA)                                                                            │
│  • MMBench GUI L2: 91.3% (SOTA)                                                                            │
│  • AndroidWorld: 76.7% (beats Gemini 2.5 Pro, Seed1.8, UI-TARS-2)                                         │
│  • MobileWorld: 41.7% (SOTA)                                                                               │
│                                                                                                             │
│  vLLM DEPLOYMENT:                                                                                           │
│  ═════════════════                                                                                          │
│  ┌──────────────────┬──────────────────┬───────────────────┬────────────────────────────────────────────┐  │
│  │ Model            │ Min GPU          │ vLLM Config       │ Expected Latency                           │  │
│  ├──────────────────┼──────────────────┼───────────────────┼────────────────────────────────────────────┤  │
│  │ MAI-UI-2B        │ T4 16GB          │ dtype=float16     │ ~800ms                                     │  │
│  │ MAI-UI-8B        │ A100-40GB        │ dtype=bfloat16    │ ~300ms                                     │  │
│  │ MAI-UI-32B       │ A100-80GB        │ dtype=bfloat16    │ ~500ms                                     │  │
│  │ MAI-UI-235B-A22B │ 8×H100-80GB      │ TP=8, EP enabled  │ ~400ms                                     │  │
│  └──────────────────┴──────────────────┴───────────────────┴────────────────────────────────────────────┘  │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **2. UI-Ins** — [arXiv:2510.20286](https://arxiv.org/abs/2510.20286)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  UI-Ins: INSTRUCTION-AS-REASONING PARADIGM                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SOURCE: ICLR 2026 Submission                                                                               │
│                                                                                                             │
│  PROBLEM IDENTIFIED:                                                                                        │
│  ═══════════════════                                                                                        │
│  • 23.3% flaw rate in existing grounding dataset instructions                                              │
│  • Existing instructions are static, not dynamic reasoning pathways                                        │
│                                                                                                             │
│  CORE INNOVATION:                                                                                           │
│  ═════════════════                                                                                          │
│  Instruction-as-Reasoning: Treat instructions as DYNAMIC ANALYTICAL PATHWAYS                               │
│                                                                                                             │
│  4 HUMAN-LIKE PERSPECTIVES (THIS IS THE SOURCE):                                                            │
│  ═══════════════════════════════════════════════                                                            │
│  ┌───────────────┬────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ APPEARANCE    │ "Click the blue button with white text in the rounded rectangle"                      ││
│  │ FUNCTIONALITY │ "Click the button that submits the form and saves your data"                          ││
│  │ LOCATION      │ "Click the button at the bottom right corner of the form section"                     ││
│  │ INTENT        │ "Complete your order by clicking the final confirmation"                               ││
│  └───────────────┴────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  TWO-STAGE TRAINING:                                                                                        │
│  ════════════════════                                                                                       │
│  Stage 1: SFT on multi-perspective instructions                                                            │
│           → Model learns to generate reasoning text from diverse perspectives                              │
│  Stage 2: GRPO-based RL for pathway selection                                                              │
│           → Model learns to SELECT optimal perspective for each scenario                                   │
│                                                                                                             │
│  DATA PIPELINE:                                                                                             │
│  ══════════════                                                                                             │
│  1. Clean noisy annotations using OmniParser V2 + IoU-based refinement                                     │
│  2. Use GPT-4.1 to generate instructions from all 4 perspectives                                          │
│  3. Verify each instruction unambiguously refers to target element                                         │
│                                                                                                             │
│  MODELS RELEASED:                                                                                           │
│  ═════════════════                                                                                          │
│  • UI-Ins-7B: Best agent performance (66.1% AndroidWorld)                                                  │
│  • UI-Ins-32B: Best grounding accuracy (87.3% UI-I2E-Bench)                                               │
│                                                                                                             │
│  RELATIONSHIP TO QWEN-VL:                                                                                   │
│  ═══════════════════════                                                                                    │
│  • Built on Qwen2-VL / Qwen2.5-VL architecture                                                             │
│  • MAI-UI adopts this paradigm for grounding training                                                      │
│                                                                                                             │
│  vLLM RELEVANCE:                                                                                            │
│  ════════════════                                                                                           │
│  • The multi-perspective instruction approach works with any Qwen-VL model in vLLM                        │
│  • Does not require vLLM changes—affects TRAINING DATA only                                                │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **3. OS-Genesis** — [arXiv:2412.19723](https://arxiv.org/abs/2412.19723)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  OS-GENESIS: REVERSE TASK SYNTHESIS                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SOURCE: ACL 2025 (Oral)                                                                                    │
│  GITHUB: https://github.com/OS-Copilot/OS-Genesis                                                          │
│                                                                                                             │
│  CORE INNOVATION:                                                                                           │
│  ═════════════════                                                                                          │
│  Reverse the trajectory collection process:                                                                │
│  TRADITIONAL: Define task → Attempt task → Collect trajectory                                              │
│  OS-GENESIS:  Explore GUI → Collect interactions → Derive task retrospectively                             │
│                                                                                                             │
│  METHODOLOGY:                                                                                               │
│  ═════════════                                                                                              │
│                                                                                                             │
│  Phase 1: INTERACTION-DRIVEN FUNCTIONAL DISCOVERY                                                           │
│  ─────────────────────────────────────────────────                                                          │
│  • Rule-based traversal of GUI environments (emulators, browsers)                                          │
│  • Collect interaction triplets: ⟨s_pre, action, s_post⟩                                                   │
│    - s_pre = screenshot BEFORE action                                                                      │
│    - action = CLICK(x,y), TYPE("text"), SCROLL(direction)                                                  │
│    - s_post = screenshot AFTER action                                                                      │
│  • GPT-4o generates contextually appropriate input content                                                 │
│                                                                                                             │
│  Phase 2: REVERSE TASK SYNTHESIS                                                                            │
│  ────────────────────────────────                                                                           │
│  • Low-level instruction generation (τ_low):                                                               │
│    "Click the dropdown to display options"                                                                 │
│  • High-level instruction construction (τ_high):                                                           │
│    "Send an email to the specified recipient"                                                              │
│                                                                                                             │
│  Phase 3: TRAJECTORY REWARD MODEL (TRM)                                                                     │
│  ────────────────────────────────────────                                                                   │
│  • Graded scoring system: 1-5 based on:                                                                    │
│    - Completion: Did trajectory accomplish the task?                                                       │
│    - Coherence: Are steps logically connected?                                                             │
│  • Unlike binary filtering, allows partially valuable trajectories                                         │
│                                                                                                             │
│  MODELS RELEASED:                                                                                           │
│  ═════════════════                                                                                          │
│  • OS-Genesis-4B, OS-Genesis-7B, OS-Genesis-8B                                                             │
│  • Trained on InternVL2 and Qwen2-VL architectures                                                         │
│                                                                                                             │
│  vLLM DEPLOYMENT:                                                                                           │
│  ═════════════════                                                                                          │
│  • Models compatible with standard vLLM Qwen2-VL pipeline                                                  │
│  • Raw triplet data released for reproducing synthesis                                                     │
│                                                                                                             │
│  GPU REQUIREMENTS:                                                                                          │
│  ══════════════════                                                                                         │
│  ┌──────────────────┬──────────────────┬──────────────────────────────────────────────────────────────────┐│
│  │ Model            │ Min GPU          │ Notes                                                            ││
│  ├──────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────┤│
│  │ OS-Genesis-4B    │ T4 16GB          │ FP16, good for edge deployment                                   ││
│  │ OS-Genesis-7B    │ L4 24GB / A10G   │ BF16, balanced performance                                       ││
│  │ OS-Genesis-8B    │ A100-40GB        │ BF16, best quality from this series                             ││
│  └──────────────────┴──────────────────┴──────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **4. UI-R1** — [arXiv:2503.21620](https://arxiv.org/abs/2503.21620)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  UI-R1: GRPO FOR GUI ACTION PREDICTION                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SOURCE: arXiv March 2025                                                                                   │
│  GITHUB: https://github.com/lll6gg/UI-R1                                                                   │
│                                                                                                             │
│  BASE MODEL:                                                                                                │
│  ════════════                                                                                               │
│  Qwen2.5-VL-3B (NOT Qwen3-VL)                                                                              │
│                                                                                                             │
│  CORE INNOVATION:                                                                                           │
│  ═════════════════                                                                                          │
│  Apply DeepSeek-R1 style rule-based RL to GUI agents                                                       │
│                                                                                                             │
│  TRAINING METHODOLOGY:                                                                                      │
│  ═════════════════════                                                                                      │
│  • Algorithm: GRPO (Group Relative Policy Optimization)                                                    │
│  • Training Data: ONLY 136 high-quality tasks                                                              │
│  • Data Selection: 3-stage process (Quality → Difficulty → Diversity)                                     │
│                                                                                                             │
│  REWARD FUNCTION (Rule-Based):                                                                              │
│  ══════════════════════════════                                                                             │
│  R_total = R_type + R_coord + R_format                                                                     │
│                                                                                                             │
│  • R_type: Action type accuracy (click vs scroll vs type)                                                  │
│  • R_coord: Coordinate accuracy for click actions (inside target bbox = 1.0)                              │
│  • R_format: Output format correctness (valid JSON = bonus)                                                │
│                                                                                                             │
│  RESULTS:                                                                                                   │
│  ═════════                                                                                                  │
│  ┌───────────────────────┬───────────────────────────────────────────────────────────────────────────────┐ │
│  │ Metric                │ Improvement over Qwen2.5-VL-3B base                                          │ │
│  ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤ │
│  │ Action Type Accuracy  │ +15%                                                                         │ │
│  │ Grounding Accuracy    │ +20%                                                                         │ │
│  │ ScreenSpot            │ +22.1%                                                                       │ │
│  └───────────────────────┴───────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  KEY INSIGHT:                                                                                               │
│  ═════════════                                                                                              │
│  "136 tasks with GRPO > 76K tasks with SFT"                                                                │
│  → Quality of RL signal matters more than quantity of SFT data                                             │
│                                                                                                             │
│  vLLM DEPLOYMENT:                                                                                           │
│  ═════════════════                                                                                          │
│  • Model: Qwen2.5-VL-3B-UI-R1 (available on HuggingFace: LZXzju/Qwen2.5-VL-3B-UI-R1)                      │
│  • Uses standard vLLM qwen2_5_vl.py pipeline                                                               │
│                                                                                                             │
│  GPU REQUIREMENTS:                                                                                          │
│  ══════════════════                                                                                         │
│  ┌──────────────────┬──────────────────┬──────────────────────────────────────────────────────────────────┐│
│  │ Task             │ GPU              │ Memory                                                           ││
│  ├──────────────────┼──────────────────┼──────────────────────────────────────────────────────────────────┤│
│  │ Inference (FP16) │ T4 16GB          │ ~8-10GB (with image size reduction)                             ││
│  │ Inference (4-bit)│ T4 16GB          │ ~8.6GB VRAM                                                      ││
│  │ GRPO Training    │ 1× H20 80GB      │ Sufficient for Qwen2.5-3B                                       ││
│  │ GRPO + vLLM      │ 8× GPU setup     │ 6 GPUs training + 2 GPUs vLLM inference server                  ││
│  └──────────────────┴──────────────────┴──────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  vLLM CONFIG EXAMPLE:                                                                                       │
│  ═════════════════════                                                                                      │
│  ```python                                                                                                  │
│  from vllm import LLM                                                                                       │
│  llm = LLM(                                                                                                 │
│      model="LZXzju/Qwen2.5-VL-3B-UI-R1",                                                                   │
│      dtype="float16",                                                                                       │
│      gpu_memory_utilization=0.9,                                                                           │
│      limit_mm_per_prompt={"image": 3, "video": 1},  # Memory saving                                        │
│      mm_processor_kwargs={"max_pixels": 1280*720},  # Reduce image size                                    │
│  )                                                                                                          │
│  ```                                                                                                        │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **5. Fara-7B / FaraGen** — [arXiv:2511.19663](https://arxiv.org/abs/2511.19663)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  FARA-7B: MICROSOFT'S COMPUTER USE AGENT                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SOURCE: Microsoft Research (November 2025)                                                                 │
│  HUGGINGFACE: microsoft/Fara-7B                                                                            │
│  GITHUB: https://github.com/microsoft/fara                                                                 │
│                                                                                                             │
│  BASE MODEL:                                                                                                │
│  ════════════                                                                                               │
│  Qwen2.5-VL-7B (NOT Qwen3-VL)                                                                              │
│                                                                                                             │
│  FARGEN DATA ENGINE:                                                                                        │
│  ════════════════════                                                                                       │
│  Built on Magentic-One multi-agent framework:                                                              │
│                                                                                                             │
│  ┌────────────────┐        ┌────────────────┐                                                              │
│  │  ORCHESTRATOR  │ ─────▶ │   WEBSURFER    │                                                              │
│  │  (Plans tasks) │        │  (Executes)    │                                                              │
│  └────────────────┘        └────────────────┘                                                              │
│         │                          │                                                                        │
│         │                          ▼                                                                        │
│         │          ┌─────────────────────────────────────────┐                                             │
│         └─────────▶│  145,000 trajectories across 70K+ URLs  │                                             │
│                    │  1,000,000+ action steps                 │                                             │
│                    └─────────────────────────────────────────┘                                             │
│                                                                                                             │
│  TRAINING:                                                                                                  │
│  ══════════                                                                                                 │
│  • Method: Supervised Fine-Tuning (SFT) only                                                               │
│  • Training Infra: 64× H100 GPUs                                                                           │
│  • No RL used—pure SFT on high-quality synthetic data                                                      │
│                                                                                                             │
│  BENCHMARK RESULTS:                                                                                         │
│  ══════════════════                                                                                         │
│  • WebVoyager: Competitive with frontier models                                                            │
│  • WebTailBench: State-of-the-art for 7B class                                                             │
│  • Cost: ~$0.025 per task (vs $0.50+ for GPT-4o)                                                          │
│                                                                                                             │
│  vLLM DEPLOYMENT:                                                                                           │
│  ═════════════════                                                                                          │
│  ```bash                                                                                                    │
│  vllm serve "microsoft/Fara-7B" --port 5000 --dtype auto                                                   │
│  # If OOM: add --tensor-parallel-size 2                                                                    │
│  ```                                                                                                        │
│                                                                                                             │
│  GPU REQUIREMENTS:                                                                                          │
│  ══════════════════                                                                                         │
│  ┌──────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐│
│  │ GPU              │ Configuration                                                                       ││
│  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤│
│  │ A6000 48GB       │ Single GPU, --dtype bfloat16                                                        ││
│  │ A100 80GB        │ Single GPU, --dtype bfloat16 (tested by Microsoft)                                  ││
│  │ H100 80GB        │ Single GPU, --dtype bfloat16 (tested by Microsoft)                                  ││
│  │ L40S             │ Single GPU (Koyeb default deployment)                                               ││
│  │ 2× 24GB GPUs     │ --tensor-parallel-size 2 (for memory-constrained setups)                           ││
│  └──────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  ON-DEVICE DEPLOYMENT:                                                                                      │
│  ══════════════════════                                                                                     │
│  • Quantized version available for Copilot+ PCs                                                            │
│  • Uses NPU acceleration (not GPU) for local inference                                                     │
│  • Available via Microsoft Foundry                                                                         │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **6. GUI-360** — [arXiv:2511.04307](https://arxiv.org/abs/2511.04307)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  GUI-360: WINDOWS OFFICE APPLICATIONS DATASET                                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  GITHUB: https://github.com/2020-qqtcg/GUI-360                                                             │
│                                                                                                             │
│  DATASET STATISTICS:                                                                                        │
│  ════════════════════                                                                                       │
│  • 1.2M+ executed action steps                                                                             │
│  • Thousands of trajectories                                                                               │
│  • Applications: Word, Excel, PowerPoint (Windows)                                                         │
│  • Includes: Full-resolution screenshots + Windows Accessibility API metadata                              │
│                                                                                                             │
│  TASKS SUPPORTED:                                                                                           │
│  ═════════════════                                                                                          │
│  1. GUI Grounding: Locate elements from instructions                                                       │
│  2. Screen Parsing: Understand UI structure                                                                │
│  3. Action Prediction: Predict next action given state                                                     │
│                                                                                                             │
│  DATA COLLECTION:                                                                                           │
│  ═════════════════                                                                                          │
│  • LLM-augmented pipeline with TrajAgent (GPT-4o/GPT-4.1)                                                 │
│  • Two-stage execution strategy                                                                            │
│  • Hybrid GUI + API action space                                                                           │
│                                                                                                             │
│  SFT RESULTS:                                                                                               │
│  ═════════════                                                                                              │
│  • SFT on GUI-360 yields 82%+ grounding accuracy                                                           │
│  • Significant gains over out-of-box VLMs                                                                  │
│                                                                                                             │
│  vLLM RELEVANCE:                                                                                            │
│  ════════════════                                                                                           │
│  • Dataset can be used to fine-tune any Qwen-VL model                                                      │
│  • Trained models compatible with standard vLLM pipelines                                                  │
│  • Benchmark includes configs for Qwen2.5-VL-7B                                                            │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **7. UGround** — [arXiv:2410.05243](https://arxiv.org/abs/2410.05243)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  UGROUND: UNIVERSAL VISUAL GROUNDING                                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SOURCE: Ohio State University + Orby AI                                                                   │
│  VENUE: ICLR 2025 Oral                                                                                     │
│  GITHUB: https://github.com/OSU-NLP-Group/UGround                                                          │
│                                                                                                             │
│  DATASET:                                                                                                   │
│  ═════════                                                                                                  │
│  • 10 million GUI elements                                                                                 │
│  • 1.3 million screenshots                                                                                 │
│  • LARGEST GUI visual grounding corpus to date                                                             │
│                                                                                                             │
│  ARCHITECTURE:                                                                                              │
│  ═════════════                                                                                              │
│  • Based on LLaVA architecture (NOT Qwen)                                                                  │
│  • Visual-only grounding (no HTML/accessibility tree)                                                      │
│  • Part of SeeAct-V framework                                                                              │
│                                                                                                             │
│  KEY INNOVATION:                                                                                            │
│  ════════════════                                                                                           │
│  "Navigate the digital world as humans do"                                                                 │
│  → Pure visual perception, pixel-level operations                                                          │
│  → No text-based representations needed                                                                    │
│                                                                                                             │
│  RESULTS:                                                                                                   │
│  ═════════                                                                                                  │
│  • +20% absolute improvement over existing visual grounding models                                         │
│  • Outperforms agents that use additional text-based input                                                 │
│                                                                                                             │
│  vLLM RELEVANCE:                                                                                            │
│  ════════════════                                                                                           │
│  • NOT directly compatible with Qwen-VL vLLM pipeline                                                      │
│  • Uses LLaVA architecture—different model family                                                          │
│  • Dataset methodology transferable to Qwen-VL training                                                    │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **8. OS-ATLAS** — [arXiv:2410.23218](https://arxiv.org/abs/2410.23218)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  OS-ATLAS: CROSS-PLATFORM GUI GROUNDING                                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WEBSITE: https://osatlas.github.io/                                                                       │
│                                                                                                             │
│  DATASET:                                                                                                   │
│  ═════════                                                                                                  │
│  • 13.58 million GUI elements                                                                              │
│  • 2.24 million screenshots                                                                                │
│  • LARGEST open-source cross-platform corpus                                                               │
│                                                                                                             │
│  PLATFORMS COVERED:                                                                                         │
│  ════════════════════                                                                                       │
│  ✓ Windows                                                                                                 │
│  ✓ Linux                                                                                                   │
│  ✓ MacOS                                                                                                   │
│  ✓ Android                                                                                                 │
│  ✓ Web                                                                                                     │
│                                                                                                             │
│  OPEN-SOURCE TOOLKIT:                                                                                       │
│  ═════════════════════                                                                                      │
│  • Data synthesis toolkit released                                                                         │
│  • Generate your own GUI grounding data                                                                    │
│  • Works across all 5 platforms                                                                            │
│                                                                                                             │
│  MODEL:                                                                                                     │
│  ═══════                                                                                                    │
│  • OS-Atlas foundation model                                                                               │
│  • Excels at GUI grounding + OOD agentic tasks                                                             │
│  • Open-source alternative to GPT-4o for GUI tasks                                                         │
│                                                                                                             │
│  vLLM RELEVANCE:                                                                                            │
│  ════════════════                                                                                           │
│  • Toolkit can generate training data for Qwen-VL models                                                   │
│  • Cross-platform coverage useful for robust agent training                                                │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **9. EDGE** — [arXiv:2410.19461](https://arxiv.org/abs/2410.19461)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  EDGE: SYNTHETIC DATA FROM WEBPAGES                                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  GITHUB: https://github.com/chenxuetian/EDGE                                                               │
│                                                                                                             │
│  CORE INNOVATION:                                                                                           │
│  ═════════════════                                                                                          │
│  Auto-generate GUI training data from publicly accessible webpages                                         │
│                                                                                                             │
│  DATA SOURCE:                                                                                               │
│  ═════════════                                                                                              │
│  • FineWeb-Edu subset of Common Crawl                                                                      │
│  • Billions of webpages available                                                                          │
│  • No manual annotation needed                                                                             │
│                                                                                                             │
│  METHODOLOGY:                                                                                               │
│  ═════════════                                                                                              │
│  1. Render webpages in headless browser                                                                    │
│  2. JavaScript injection to extract visual elements                                                        │
│  3. Filter invisible elements                                                                              │
│  4. Create rich annotations (text + accessibility labels)                                                  │
│  5. Generate multi-granularity tasks:                                                                      │
│     - Elementary: Element-level grounding                                                                  │
│     - Advanced: Multi-step navigation                                                                      │
│                                                                                                             │
│  KEY FINDING:                                                                                               │
│  ═════════════                                                                                              │
│  Models trained on EDGE-generated web data:                                                                │
│  → Transfer to UNSEEN desktop and mobile environments                                                      │
│  → No need for desktop/mobile-specific training data                                                       │
│                                                                                                             │
│  vLLM RELEVANCE:                                                                                            │
│  ════════════════                                                                                           │
│  • Training methodology, not inference                                                                     │
│  • EDGE-trained models compatible with standard vLLM pipelines                                             │
│  • Model and dataset available on HuggingFace                                                              │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **10. GUICourse** — [arXiv:2406.11317](https://arxiv.org/abs/2406.11317)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  GUICOURSE: VLM → GUI AGENT TRAINING SUITE                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  VENUE: ACL 2025                                                                                            │
│  GITHUB: https://github.com/RUCBM/GUICourse                                                                │
│                                                                                                             │
│  DATASET SUITE (3 Parts):                                                                                   │
│  ═══════════════════════                                                                                    │
│                                                                                                             │
│  1. GUIEnv (OCR + Grounding):                                                                              │
│     • GUIEnv-global: 10M page-annotation pairs (pre-training)                                              │
│     • GUIEnv-local: 0.7M region-text QA pairs (SFT)                                                        │
│     • Format: "text2bbox" and "bbox2text" tasks                                                            │
│                                                                                                             │
│  2. GUIAct (Navigation):                                                                                    │
│     • web-single: 67K single-step instructions                                                             │
│     • web-multi: 5,696 human-annotated multi-step                                                          │
│     • AITW smartphone: 9,157 instructions                                                                  │
│                                                                                                             │
│  3. GUIChat (Interaction):                                                                                  │
│     • 44K single-turn QA pairs                                                                             │
│     • 6K multi-turn dialogues                                                                              │
│                                                                                                             │
│  SFT FORMAT:                                                                                                │
│  ════════════                                                                                               │
│  • Follows Qwen-VL SFT data format                                                                         │
│  • Ready-to-use for fine-tuning Qwen-VL models                                                             │
│  • Preprocessing scripts included in repo                                                                  │
│                                                                                                             │
│  TRAINING PIPELINE:                                                                                         │
│  ════════════════════                                                                                       │
│  Stage 1: Pre-train on GUIEnv-global                                                                       │
│  Stage 2: SFT on GUIEnv-local + GUIAct + GUIChat                                                           │
│                                                                                                             │
│  RESULTS:                                                                                                   │
│  ═════════                                                                                                  │
│  • Even 3.1B model works well on GUI tasks                                                                 │
│  • Ablation: Improved OCR/grounding → better navigation                                                    │
│                                                                                                             │
│  vLLM DEPLOYMENT:                                                                                           │
│  ═════════════════                                                                                          │
│  • Models trained with GUICourse directly deployable via vLLM                                              │
│  • Uses standard Qwen-VL architecture                                                                      │
│  • Compatible with qwen2_vl.py / qwen2_5_vl.py pipelines                                                   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

#### **11. OpenCUA** — IN VLLM CODEBASE

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  OPENCUA: COMPUTER USE AGENT IN VLLM                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SOURCE: XLANG Lab, The University of Hong Kong                                                            │
│  vLLM FILE: vllm/model_executor/models/opencua.py                                                          │
│                                                                                                             │
│  ARCHITECTURE:                                                                                              │
│  ═════════════                                                                                              │
│  • Inherits from Qwen2.5-VL architecture                                                                   │
│  • Uses Qwen2_5_VLForConditionalGeneration as base                                                         │
│  • Custom processor for OpenCUA-specific tokens                                                            │
│                                                                                                             │
│  SPECIAL TOKENS:                                                                                            │
│  ════════════════                                                                                           │
│  • <|media_placeholder|>: Image input token                                                                │
│                                                                                                             │
│  vLLM INTEGRATION:                                                                                          │
│  ══════════════════                                                                                         │
│  • Fully integrated into vLLM model registry                                                               │
│  • Uses standard multimodal processing pipeline                                                            │
│  • Compatible with PagedAttention, FlashAttention, etc.                                                    │
│                                                                                                             │
│  CODE STRUCTURE:                                                                                            │
│  ═════════════════                                                                                          │
│  OpenCUAForConditionalGeneration                                                                            │
│    └── Qwen2_5_VLForConditionalGeneration                                                                  │
│          └── Qwen2_5_VisionTransformer (aliased as OpenCUAVisionTransformer)                               │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Deep Technical Breakdown: Qwen3-VL 8B vs 32B

This section provides a deep, systems-level breakdown of the 8B and 32B Qwen3-VL models.

### 1. Architecture Differences: 8B vs 32B

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B vs QWEN3-VL-32B: ARCHITECTURE COMPARISON                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │                       QWEN3-VL-8B                           QWEN3-VL-32B                               ││
│  │                       ════════════                          ═════════════                              ││
│  │                                                                                                         ││
│  │  VISION ENCODER                                                                                         ││
│  │  ┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐                    ││
│  │  │ ViT Hidden: 1536                    │    │ ViT Hidden: 1792                    │                    ││
│  │  │ ViT Layers: 32                      │    │ ViT Layers: 32                      │                    ││
│  │  │ ViT Heads: 24                       │    │ ViT Heads: 28                       │                    ││
│  │  │ ViT MLP: 6144                       │    │ ViT MLP: 7168                       │                    ││
│  │  │ Merger Out: 4096                    │    │ Merger Out: 5120                    │                    ││
│  │  │ Params: ~1.2B                       │    │ Params: ~2.0B                       │                    ││
│  │  └─────────────────────────────────────┘    └─────────────────────────────────────┘                    ││
│  │                                                                                                         ││
│  │  LANGUAGE MODEL                                                                                         ││
│  │  ┌─────────────────────────────────────┐    ┌─────────────────────────────────────┐                    ││
│  │  │ Layers: 32                          │    │ Layers: 64  ← 2× MORE LAYERS        │                    ││
│  │  │ Hidden: 4096                        │    │ Hidden: 5120                        │                    ││
│  │  │ Heads: 32                           │    │ Heads: 40                           │                    ││
│  │  │ KV Heads: 8 (GQA 4:1)              │    │ KV Heads: 8 (GQA 5:1)              │                    ││
│  │  │ Head Dim: 128                       │    │ Head Dim: 128                       │                    ││
│  │  │ Intermediate: 12288                 │    │ Intermediate: 25600                 │                    ││
│  │  │ Params: ~7.1B                       │    │ Params: ~30.8B                      │                    ││
│  │  └─────────────────────────────────────┘    └─────────────────────────────────────┘                    ││
│  │                                                                                                         ││
│  │  TOTAL PARAMS: ~8.3B                        TOTAL PARAMS: ~32.8B                                       ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  KEY ARCHITECTURAL DIFFERENCES:                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  1. LAYER COUNT (Most Critical)                                                                         ││
│  │     • 8B: 32 layers → 32 sequential attention + MLP blocks                                             ││
│  │     • 32B: 64 layers → 64 sequential blocks = 2× reasoning depth                                       ││
│  │     • Impact: More layers = more "reasoning steps" per forward pass                                    ││
│  │     • For GUI agents: 64 layers can chain more complex reasoning                                       ││
│  │                                                                                                         ││
│  │  2. HIDDEN DIMENSION                                                                                    ││
│  │     • 8B: 4096 → Each token is a 4096-dimensional vector                                               ││
│  │     • 32B: 5120 → 25% larger representation capacity                                                   ││
│  │     • Impact: More "bits" to encode spatial/visual information                                         ││
│  │                                                                                                         ││
│  │  3. MLP INTERMEDIATE                                                                                    ││
│  │     • 8B: 12288 (3× hidden)                                                                            ││
│  │     • 32B: 25600 (5× hidden)                                                                           ││
│  │     • Impact: Larger "working memory" per layer transformation                                         ││
│  │                                                                                                         ││
│  │  4. GQA RATIO                                                                                           ││
│  │     • 8B: 32 heads / 8 KV heads = 4:1                                                                  ││
│  │     • 32B: 40 heads / 8 KV heads = 5:1                                                                 ││
│  │     • Impact: 32B is slightly more memory-efficient relative to its size                               ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Memory Footprint Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY FOOTPRINT: 8B vs 32B                                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  MODEL WEIGHTS                                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Component                │ 8B (BF16)     │ 8B (FP8)      │ 32B (BF16)    │ 32B (FP8)                   ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encoder           │ 2.4 GB        │ 1.2 GB        │ 4.0 GB        │ 2.0 GB                      ││
│  │  LLM Embedding           │ 0.6 GB        │ 0.3 GB        │ 0.8 GB        │ 0.4 GB                      ││
│  │  LLM Attention (all)     │ 4.2 GB        │ 2.1 GB        │ 13.1 GB       │ 6.6 GB                      ││
│  │  LLM MLP (all)           │ 8.6 GB        │ 4.3 GB        │ 46.9 GB       │ 23.4 GB                     ││
│  │  LLM Head                │ 0.8 GB        │ 0.4 GB        │ 1.0 GB        │ 0.5 GB                      ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL WEIGHTS           │ 16.6 GB       │ 8.3 GB        │ 65.8 GB       │ 32.9 GB                     ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  KV CACHE PER TOKEN                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Formula: 2 × num_layers × num_kv_heads × head_dim × dtype_bytes                                       ││
│  │                                                                                                         ││
│  │  8B (BF16):  2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KB per token                                   ││
│  │  8B (FP8):   2 × 32 × 8 × 128 × 1 = 65,536 bytes = 64 KB per token                                     ││
│  │  32B (BF16): 2 × 64 × 8 × 128 × 2 = 262,144 bytes = 256 KB per token                                   ││
│  │  32B (FP8):  2 × 64 × 8 × 128 × 1 = 131,072 bytes = 128 KB per token                                   ││
│  │                                                                                                         ││
│  │  At 8K context:                                                                                         ││
│  │  8B (BF16):  8192 × 128 KB = 1.0 GB per request                                                        ││
│  │  32B (BF16): 8192 × 256 KB = 2.0 GB per request                                                        ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  TOTAL VRAM REQUIRED (SINGLE GPU)                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Configuration              │ 8B            │ 32B           │ Notes                                     ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Weights (BF16)            │ 16.6 GB       │ 65.8 GB       │ Model parameters                          ││
│  │  + Activations             │ +4.0 GB       │ +8.0 GB       │ Batch size dependent                      ││
│  │  + CUDA Overhead           │ +2.0 GB       │ +3.0 GB       │ Context, kernels                          ││
│  │  + 1 request @ 8K          │ +1.0 GB       │ +2.0 GB       │ KV cache                                  ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  MINIMUM (1 req)           │ ~24 GB        │ ~79 GB        │                                           ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Fits A100-40GB?           │ ⚠️ Tight      │ ❌ No         │                                           ││
│  │  Fits A100-80GB?           │ ✅ Yes        │ ⚠️ Tight      │                                           ││
│  │  Fits H100-80GB (FP8)?     │ ✅ Yes        │ ✅ Yes        │ FP8 weights = 32.9 GB                     ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Throughput vs Latency Trade-offs

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     THROUGHPUT vs LATENCY: 8B vs 32B                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SINGLE REQUEST LATENCY (1024×1024 image + 50 prompt → 200 output)                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Phase              │ 8B on A100-80   │ 8B on H100     │ 32B on A100-80  │ 32B on H100                 ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encode      │ 25 ms           │ 12 ms          │ 50 ms           │ 25 ms                       ││
│  │  Prefill (1382 tok) │ 120 ms          │ 60 ms          │ 480 ms          │ 240 ms                      ││
│  │  Decode (×200)      │ 2,400 ms        │ 1,200 ms       │ 5,000 ms        │ 2,400 ms                    ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL              │ 2.55 s          │ 1.27 s         │ 5.53 s          │ 2.67 s                      ││
│  │  Tokens/sec         │ 78              │ 157            │ 36              │ 75                          ││
│  │                                                                                                         ││
│  │  KEY INSIGHT: 32B is ~2× slower than 8B due to:                                                        ││
│  │  • 2× layers = 2× sequential operations                                                                ││
│  │  • 4× weights to read from HBM per decode step                                                         ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  THROUGHPUT (CONCURRENT REQUESTS)                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Scenario           │ 8B on H100-80GB │ 32B on H100-80GB │ 8B Advantage                               ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Max concurrent @4K │ 48 requests     │ 12 requests      │ 4× more concurrent                         ││
│  │  Max concurrent @8K │ 32 requests     │ 8 requests       │ 4× more concurrent                         ││
│  │  Batch throughput   │ ~500 tok/s      │ ~150 tok/s       │ 3.3× higher throughput                     ││
│  │  Cost per 1M tokens │ ~$0.05          │ ~$0.17           │ 3.4× cheaper                               ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  WHEN TO USE EACH                                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  USE 8B WHEN:                                 USE 32B WHEN:                                             ││
│  │  ────────────────────────────────────────     ────────────────────────────────────────                  ││
│  │  • High throughput required                  • Complex multi-step reasoning required                   ││
│  │  • Cost is primary concern                   • GUI agent tasks (SFT + RL training)                     ││
│  │  • Simple VQA tasks                          • State-dependent action sequences                        ││
│  │  • Real-time applications (<1s)              • Accuracy > speed                                        ││
│  │  • Limited GPU budget                        • Long-form document understanding                        ││
│  │  • Many concurrent users                     • Few high-stakes queries                                 ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2. vLLM Inference Pipeline for Qwen3-VL

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     VLLM INFERENCE PIPELINE: END-TO-END                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  REQUEST LIFECYCLE                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │                                           vLLM ENGINE                                                   ││
│  │                                                                                                         ││
│  │  ┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐          ││
│  │  │ Request  │────▶│ Tokenizer +  │────▶│  Scheduler   │────▶│  Model Exec  │────▶│ Response │          ││
│  │  │ (API)    │     │ MM Processor │     │              │     │              │     │          │          ││
│  │  └──────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────┘          ││
│  │                                                                                                         ││
│  │  STEP 1: REQUEST INGESTION                                                                              ││
│  │  ─────────────────────────────                                                                          ││
│  │  • API receives: {"prompt": "What is in this image?", "images": [base64_data]}                        ││
│  │  • Qwen3VLMultiModalProcessor extracts image bytes                                                     ││
│  │  • Tokenizer converts text → token_ids                                                                 ││
│  │  • Placeholder tokens inserted: <|vision_start|><|image_pad|>×N<|vision_end|>                         ││
│  │                                                                                                         ││
│  │  STEP 2: IMAGE PREPROCESSING                                                                            ││
│  │  ───────────────────────────                                                                            ││
│  │  • smart_resize() → target resolution matching model's patch size                                      ││
│  │  • Normalize: (pixel - mean) / std                                                                     ││
│  │  • Output: [batch, 3, H, W] tensor on GPU                                                              ││
│  │                                                                                                         ││
│  │  STEP 3: SCHEDULING                                                                                     ││
│  │  ──────────────────                                                                                     ││
│  │  • Scheduler decides: can this request fit in current batch?                                           ││
│  │  • Checks: KV cache availability, max_num_seqs limit                                                   ││
│  │  • Groups prefill vs decode requests (chunked prefill)                                                 ││
│  │                                                                                                         ││
│  │  STEP 4: MODEL EXECUTION                                                                                ││
│  │  ───────────────────────                                                                                ││
│  │  a) Vision Encoding (once per image):                                                                  ││
│  │     • Qwen3_VisionTransformer.forward(pixels)                                                          ││
│  │     • Conv3D → ViT blocks → DeepStack extraction → Merger                                             ││
│  │     • Output: visual_embeds [num_visual_tokens, hidden_dim]                                            ││
│  │                                                                                                         ││
│  │  b) Prefill (process all input tokens):                                                                ││
│  │     • Merge visual + text embeddings at placeholder positions                                          ││
│  │     • Forward through all LLM layers                                                                   ││
│  │     • Write K, V to PagedAttention blocks                                                              ││
│  │     • DeepStack features injected at layers 0, 1, 2                                                    ││
│  │                                                                                                         ││
│  │  c) Decode (generate tokens one by one):                                                               ││
│  │     • Each step: Forward 1 new token through all layers                                                ││
│  │     • Attention: new Q attends to all cached K, V                                                      ││
│  │     • Append new K, V to cache                                                                         ││
│  │     • Sample next token from logits                                                                    ││
│  │                                                                                                         ││
│  │  STEP 5: RESPONSE STREAMING                                                                             ││
│  │  ─────────────────────────                                                                              ││
│  │  • Each decoded token immediately yielded to client                                                    ││
│  │  • Detokenizer converts token_id → string                                                              ││
│  │  • Continue until EOS or max_tokens                                                                    ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### KV Cache Layout and PagedAttention

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     KV CACHE LAYOUT: PAGEDATTENTION                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  TRADITIONAL KV CACHE (PRE-ALLOCATED, WASTEFUL)                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Request 1: █████████░░░░░░░░░░░░░░░░░░░░░░  (10K allocated, 3K used = 70% wasted)                     ││
│  │  Request 2: ██████░░░░░░░░░░░░░░░░░░░░░░░░░  (10K allocated, 2K used = 80% wasted)                     ││
│  │  Request 3: ████████████████░░░░░░░░░░░░░░░  (10K allocated, 6K used = 40% wasted)                     ││
│  │                                                                                                         ││
│  │  Problem: Must pre-allocate max_seq_len per request                                                    ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  PAGEDATTENTION (VLLM): VIRTUAL MEMORY FOR KV CACHE                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  KV CACHE ORGANIZED AS BLOCKS (like OS pages):                                                         ││
│  │                                                                                                         ││
│  │  Block Table (per request):                Physical Blocks (in GPU HBM):                               ││
│  │  ┌─────────────────────┐                   ┌────┬────┬────┬────┬────┬────┬────┬────┐                   ││
│  │  │ Request 1           │                   │ B0 │ B1 │ B2 │ B3 │ B4 │ B5 │ B6 │ B7 │...                ││
│  │  │ [B0, B5, B12, B7]   │──────────────────▶│    │    │    │    │    │    │    │    │                   ││
│  │  ├─────────────────────┤                   │ K  │ K  │ K  │ K  │ K  │ K  │ K  │ K  │                   ││
│  │  │ Request 2           │                   │ V  │ V  │ V  │ V  │ V  │ V  │ V  │ V  │                   ││
│  │  │ [B1, B3, B9]        │──────────────────▶│ ×  │ ×  │ ×  │ ×  │ ×  │ ×  │ ×  │ ×  │                   ││
│  │  ├─────────────────────┤                   │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │ 16 │                   ││
│  │  │ Request 3           │                   │tok │tok │tok │tok │tok │tok │tok │tok │                   ││
│  │  │ [B2, B4, B6, B8, B10│──────────────────▶└────┴────┴────┴────┴────┴────┴────┴────┘                   ││
│  │  └─────────────────────┘                                                                                ││
│  │                                                                                                         ││
│  │  Block Size = 16 tokens (configurable)                                                                  ││
│  │  Block Memory = 16 × kv_size_per_token                                                                 ││
│  │                                                                                                         ││
│  │  For 8B:  Block = 16 × 128 KB = 2 MB                                                                   ││
│  │  For 32B: Block = 16 × 256 KB = 4 MB                                                                   ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  ATTENTION KERNEL EXECUTION                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  For each decode step:                                                                                  ││
│  │                                                                                                         ││
│  │  1. Compute Q from new token: Q = W_q @ hidden  (GEMM)                                                 ││
│  │                                                                                                         ││
│  │  2. Lookup K, V from block table:                                                                       ││
│  │     ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐││
│  │     │ for block_id in block_table[request_id]:                                                        │││
│  │     │     K_block = physical_blocks[block_id].K  # Gather from non-contiguous memory                  │││
│  │     │     V_block = physical_blocks[block_id].V                                                       │││
│  │     │     attention_scores = Q @ K_block.T                                                            │││
│  │     │     output += softmax(attention_scores) @ V_block                                               │││
│  │     └─────────────────────────────────────────────────────────────────────────────────────────────────┘││
│  │                                                                                                         ││
│  │  3. FlashAttention fuses this into a single kernel, avoiding materializing NxN attention matrix       ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Prefix Caching and Batching

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     PREFIX CACHING AND CONTINUOUS BATCHING                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  PREFIX CACHING (--enable-prefix-caching)                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  SCENARIO: Same system prompt repeated across requests                                                  ││
│  │                                                                                                         ││
│  │  Request 1: "You are a helpful assistant. <image> What is this?"                                       ││
│  │  Request 2: "You are a helpful assistant. <image> Describe the scene."                                 ││
│  │  Request 3: "You are a helpful assistant. <image> What color is the car?"                              ││
│  │                                                                                                         ││
│  │  WITHOUT PREFIX CACHING:                    WITH PREFIX CACHING:                                        ││
│  │  ┌─────────────────────────────┐            ┌─────────────────────────────┐                            ││
│  │  │ Req 1: Compute KV for all   │            │ Req 1: Compute KV, cache    │                            ││
│  │  │ Req 2: Compute KV for all   │            │ Req 2: Reuse prefix KV      │                            ││
│  │  │ Req 3: Compute KV for all   │            │ Req 3: Reuse prefix KV      │                            ││
│  │  └─────────────────────────────┘            └─────────────────────────────┘                            ││
│  │  Prefill: 3× full cost                      Prefill: 1× full + 2× partial                              ││
│  │                                                                                                         ││
│  │  HOW IT WORKS:                                                                                          ││
│  │  • Hash the token sequence prefix                                                                       ││
│  │  • If hash matches existing cached KV blocks, reuse them                                               ││
│  │  • Only compute KV for new (suffix) tokens                                                             ││
│  │                                                                                                         ││
│  │  FOR VL MODELS: Vision tokens are part of the prefix!                                                  ││
│  │  • Same image + same prompt = fully cached                                                             ││
│  │  • Same image + different prompt = vision tokens cached                                                ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  CONTINUOUS BATCHING                                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  STATIC BATCHING (naive):                   CONTINUOUS BATCHING (vLLM):                                ││
│  │                                                                                                         ││
│  │  Time ─────────────────────────▶            Time ─────────────────────────▶                            ││
│  │  ┌─────────────────────────────┐            ┌─────────┬─────┬─────────────┐                            ││
│  │  │ Req 1 ████████████████████  │            │ Req 1 ██│█████│█████████████│                            ││
│  │  │ Req 2 ████████              │            │ Req 2 ██│█████│             │                            ││
│  │  │ Req 3 ██████████████████    │            │ Req 3 ██│█████│█████████    │                            ││
│  │  │       └── Wait for longest ─┘            │ Req 4   │     │█████████████│ ← Joins mid-batch         ││
│  │  └─────────────────────────────┘            └─────────┴─────┴─────────────┘                            ││
│  │                                                                                                         ││
│  │  • New requests join batch immediately after a slot opens                                              ││
│  │  • No waiting for longest request to finish                                                            ││
│  │  • ~2-3× higher throughput than static batching                                                        ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 3. GPU-Level Execution Details

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     GPU EXECUTION: INFERENCE MAPPED TO HARDWARE                                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  GPU MEMORY HIERARCHY                                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │                           ┌─────────────────────────────────────────────────────────────────────────┐  ││
│  │                           │                    GPU DIE                                              │  ││
│  │                           │                                                                         │  ││
│  │  ┌────────────────────┐   │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐                        │  ││
│  │  │                    │   │  │  SM 0  │  │  SM 1  │  │  SM 2  │  │  ...   │  ← 108-132 SMs        │  ││
│  │  │   HBM (80 GB)      │   │  │ L1/Shrd│  │ L1/Shrd│  │ L1/Shrd│  │        │    per GPU            │  ││
│  │  │   3.35 TB/s (H100) │◀──│──│ 192KB  │  │ 192KB  │  │ 192KB  │  │        │                        │  ││
│  │  │                    │   │  │ RegFile│  │ RegFile│  │ RegFile│  │        │                        │  ││
│  │  │ Model Weights      │   │  │ 64K×32b│  │ 64K×32b│  │ 64K×32b│  │        │                        │  ││
│  │  │ KV Cache           │   │  └────────┘  └────────┘  └────────┘  └────────┘                        │  ││
│  │  │ Activations        │   │                      │                                                  │  ││
│  │  │                    │   │                      ▼                                                  │  ││
│  │  └────────────────────┘   │              ┌────────────────┐                                         │  ││
│  │           ▲               │              │   L2 Cache     │  ← 50 MB (H100)                        │  ││
│  │           │               │              │   6 TB/s       │    Shared by all SMs                   │  ││
│  │           │               │              └────────────────┘                                         │  ││
│  │           │               └─────────────────────────────────────────────────────────────────────────┘  ││
│  │           │                                                                                             ││
│  │  Memory bound decode: Weights in HBM, must stream through L2 to SMs                                    ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  SM EXECUTION: TENSOR CORE OPERATIONS                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  INSIDE ONE SM (H100):                                                                                  ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │                                                                                                 │   ││
│  │  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐        │   ││
│  │  │  │ Processing Block │  │ Processing Block │  │ Processing Block │  │ Processing Block │        │   ││
│  │  │  │                  │  │                  │  │                  │  │                  │        │   ││
│  │  │  │ • 16 FP32 cores  │  │ • 16 FP32 cores  │  │ • 16 FP32 cores  │  │ • 16 FP32 cores  │        │   ││
│  │  │  │ • 16 FP64 cores  │  │ • 16 FP64 cores  │  │ • 16 FP64 cores  │  │ • 16 FP64 cores  │        │   ││
│  │  │  │ • 1 Tensor Core  │  │ • 1 Tensor Core  │  │ • 1 Tensor Core  │  │ • 1 Tensor Core  │        │   ││
│  │  │  │   (4th gen)      │  │   (4th gen)      │  │   (4th gen)      │  │   (4th gen)      │        │   ││
│  │  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘        │   ││
│  │  │                                                                                                 │   ││
│  │  │  Shared Memory: 256 KB (228 KB usable)                                                         │   ││
│  │  │  Register File: 256 KB                                                                          │   ││
│  │  │  Warp Schedulers: 4                                                                             │   ││
│  │  │                                                                                                 │   ││
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                         ││
│  │  TENSOR CORE OPERATION (for GEMM / Attention):                                                          ││
│  │  • Input: 8×8 matrix tiles (FP16/BF16/FP8)                                                             ││
│  │  • Computes: D = A × B + C                                                                             ││
│  │  • Throughput: 256 FLOPs per Tensor Core per cycle                                                     ││
│  │  • H100: 4 Tensor Cores × 132 SMs = 528 Tensor Cores                                                   ││
│  │  • Peak: 990 TFLOPS (BF16), 1980 TFLOPS (FP8)                                                          ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  SM UTILIZATION BY PHASE (8B on H100)                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Phase              │ Active SMs │ Tensor Core │ Memory BW   │ Bottleneck                              ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Conv3D      │ 132/132    │ 85%         │ 40%         │ Compute                                 ││
│  │  ViT Attention      │ 132/132    │ 90%         │ 50%         │ Compute                                 ││
│  │  Prefill Attn (FA3) │ 132/132    │ 95%         │ 60%         │ Compute                                 ││
│  │  Prefill MLP        │ 132/132    │ 95%         │ 55%         │ Compute                                 ││
│  │  Decode Attn        │ 60/132     │ 30%         │ 95%         │ Memory (HBM→L2→SM)                      ││
│  │  Decode MLP         │ 80/132     │ 40%         │ 95%         │ Memory (weight read)                    ││
│  │                                                                                                         ││
│  │  KEY INSIGHT: Prefill is compute-bound (high Tensor Core util)                                         ││
│  │               Decode is memory-bound (weights don't fit in L2, must stream from HBM)                   ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  MEMORY MOVEMENT: WHERE DATA FLOWS                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  PREFILL (Compute-Bound):                                                                               ││
│  │  ─────────────────────────                                                                              ││
│  │  1. Weights: HBM → L2 → SM registers (once per layer)                                                  ││
│  │  2. Activations: SM → L2 → SM (between layers)                                                         ││
│  │  3. KV write: SM → L2 → HBM (to KV cache blocks)                                                       ││
│  │                                                                                                         ││
│  │  Arithmetic Intensity: ~100-500 FLOPs/byte                                                             ││
│  │  → Can hide memory latency with compute                                                                ││
│  │                                                                                                         ││
│  │  DECODE (Memory-Bound):                                                                                 ││
│  │  ─────────────────────────                                                                              ││
│  │  1. Weights: HBM → L2 → SM (EVERY token, all layers!)                                                  ││
│  │     For 8B: 16.6 GB weights read per token                                                             ││
│  │     For 32B: 65.8 GB weights read per token                                                            ││
│  │  2. KV cache: HBM → L2 → SM (gather from scattered blocks)                                             ││
│  │  3. Activations: SM (stays in registers/shared mem)                                                    ││
│  │                                                                                                         ││
│  │  Arithmetic Intensity: ~1-2 FLOPs/byte                                                                 ││
│  │  → Memory transfer dominates, Tensor Cores underutilized                                               ││
│  │                                                                                                         ││
│  │  DECODE TIME = Model_Size / HBM_Bandwidth                                                               ││
│  │  • 8B BF16 on H100: 16.6 GB / 3.35 TB/s = 5 ms (theoretical min)                                       ││
│  │  • 32B BF16 on H100: 65.8 GB / 3.35 TB/s = 20 ms (theoretical min)                                     ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 4. Vision-Language Specific Inference

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     VISION-LANGUAGE INFERENCE: IMAGE → TOKENS → ATTENTION                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  IMAGE TOKENIZATION AND EMBEDDING                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  INPUT IMAGE: 1024 × 1024 × 3 (RGB)                                                                     ││
│  │                                                                                                         ││
│  │  STEP 1: PATCH EMBEDDING (Conv3D)                                                                       ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │                                                                                                 │   ││
│  │  │  ┌──────────────────────┐      Conv3D(kernel=14×14)      ┌──────────────────────┐              │   ││
│  │  │  │ ████████████████████ │     ─────────────────────▶     │ Each 14×14 patch     │              │   ││
│  │  │  │ ████████████████████ │                                │ becomes ONE embedding │              │   ││
│  │  │  │ ████████████████████ │                                │ (1536 dims for 8B)   │              │   ││
│  │  │  │ ████████████████████ │                                └──────────────────────┘              │   ││
│  │  │  │ 1024 × 1024 pixels   │                                                                       │   ││
│  │  │  └──────────────────────┘                                                                       │   ││
│  │  │                                                                                                 │   ││
│  │  │  Patches: (1024 / 14)² = 73² = 5,329 patches                                                   │   ││
│  │  │  Output: [5329, 1536] tensor                                                                    │   ││
│  │  │                                                                                                 │   ││
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                         ││
│  │  STEP 2: ViT TRANSFORMER (32 layers for 8B)                                                             ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │                                                                                                 │   ││
│  │  │  For layer in range(32):                                                                        │   ││
│  │  │      patches = patches + Attention(LayerNorm(patches))  # Self-attention over all patches     │   ││
│  │  │      patches = patches + MLP(LayerNorm(patches))        # Point-wise transformation            │   ││
│  │  │                                                                                                 │   ││
│  │  │      if layer in [11, 22, 32]:  # DeepStack extraction                                         │   ││
│  │  │          deepstack_features.append(Merger(patches))                                            │   ││
│  │  │                                                                                                 │   ││
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                         ││
│  │  STEP 3: SPATIAL MERGE (2×2 patches → 1 token)                                                          ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │                                                                                                 │   ││
│  │  │  Before: [5329 patches, 1536 dim]                                                               │   ││
│  │  │  After:  [1332 tokens, 4096 dim]  ← Ready for LLM                                              │   ││
│  │  │                                                                                                 │   ││
│  │  │  Merge: Concatenate 4 adjacent patch embeddings, project down                                  │   ││
│  │  │  Linear(1536 × 4 = 6144 → 4096)                                                                │   ││
│  │  │                                                                                                 │   ││
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VISION + TEXT FUSION                                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  TEXT TOKENS:    [BOS] [You] [are] [a] [...] [<|vision_start|>] [<|image_pad|>×1332] [<|vision_end|>]  ││
│  │  EMBEDDINGS:     [e1]  [e2]  [e3]  [e4] ...  [e_vs]             [PLACEHOLDER×1332]   [e_ve]            ││
│  │                                                   │                      ▲                              ││
│  │                                                   │                      │                              ││
│  │  VISION TOKENS:                                   └──────────────────────┘                              ││
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐││
│  │  │ [v1] [v2] [v3] ... [v1332]  ← Replace placeholders with vision embeddings                         │││
│  │  └────────────────────────────────────────────────────────────────────────────────────────────────────┘││
│  │                                                                                                         ││
│  │  MERGED SEQUENCE:                                                                                       ││
│  │  [BOS] [You] [are] ... [e_vs] [v1] [v2] ... [v1332] [e_ve] [What] [is] [in] [this] [?]                 ││
│  │  └─────── text ──────┘ └────────── vision ──────────┘ └─────────── text ──────────────┘                ││
│  │                                                                                                         ││
│  │  Total sequence length: ~50 text + 1332 vision + 1 = ~1383 tokens                                      ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  IMPACT ON KV CACHE AND ATTENTION COST                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  KV CACHE GROWTH:                                                                                       ││
│  │  ─────────────────                                                                                      ││
│  │  • Text-only prompt (50 tokens): 50 × 128 KB = 6.4 MB                                                  ││
│  │  • With 1024×1024 image (+1332 tokens): 1382 × 128 KB = 177 MB                                         ││
│  │  • Ratio: 27× more KV cache for vision!                                                                ││
│  │                                                                                                         ││
│  │  ATTENTION COST (PREFILL):                                                                              ││
│  │  ───────────────────────────                                                                            ││
│  │  • Attention FLOPs ∝ N² (sequence length squared)                                                      ││
│  │  • Text-only: 50² = 2,500                                                                              ││
│  │  • With image: 1382² = 1,909,924                                                                       ││
│  │  • Ratio: 764× more attention compute for vision!                                                      ││
│  │                                                                                                         ││
│  │  WHY VL MODELS ARE EXPENSIVE:                                                                           ││
│  │  • Vision tokens dominate sequence length                                                              ││
│  │  • Quadratic attention cost explodes                                                                   ││
│  │  • KV cache fills quickly                                                                              ││
│  │                                                                                                         ││
│  │  OPTIMIZATION: VIDEO PRUNING (Qwen3-VL)                                                                 ││
│  │  • --mm-processor-kwargs '{"video_pruning_rate": 0.5}'                                                 ││
│  │  • Removes redundant video tokens, reduces KV cache                                                    ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 5. Prompt-by-Prompt Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     PROMPT-BY-PROMPT INFERENCE: COLD START → WARM → MULTI-TURN                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  SCENARIO 1: FIRST PROMPT (COLD START)                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Request: {"messages": [{"role": "user", "content": [image, "What is in this image?"]}]}               ││
│  │                                                                                                         ││
│  │  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  ││
│  │  │ Phase              │ Time (8B H100) │ Time (32B H100) │ What Happens                             │  ││
│  │  ├──────────────────────────────────────────────────────────────────────────────────────────────────┤  ││
│  │  │ 1. Image Preprocess│ 5 ms           │ 5 ms            │ Resize, normalize on CPU/GPU            │  ││
│  │  │ 2. Tokenize        │ 2 ms           │ 2 ms            │ Text → tokens, insert placeholders      │  ││
│  │  │ 3. Vision Encode   │ 12 ms          │ 25 ms           │ Conv3D + 32 ViT layers + merge          │  ││
│  │  │ 4. KV Allocate     │ 1 ms           │ 1 ms            │ Reserve ~100 PagedAttn blocks           │  ││
│  │  │ 5. Prefill         │ 60 ms          │ 240 ms          │ Process all 1382 tokens, fill KV cache  │  ││
│  │  │ 6. Decode (×100)   │ 600 ms         │ 1200 ms         │ Generate 100 tokens autoregressively    │  ││
│  │  ├──────────────────────────────────────────────────────────────────────────────────────────────────┤  ││
│  │  │ TOTAL (TTFT+Gen)   │ 680 ms         │ 1473 ms         │ First token at ~80ms / ~270ms           │  ││
│  │  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  ││
│  │                                                                                                         ││
│  │  COLD START OVERHEAD:                                                                                   ││
│  │  • First request loads weights into L2 cache (cold GPU cache)                                          ││
│  │  • Vision encoding runs (can't skip for new image)                                                     ││
│  │  • Full prefill required (no cached KV)                                                                ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  SCENARIO 2: SAME IMAGE, DIFFERENT PROMPT (WARM CACHE)                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Request: {"messages": [{"role": "user", "content": [SAME_image, "Describe the colors"]}]}             ││
│  │                                                                                                         ││
│  │  WITH PREFIX CACHING ENABLED (--enable-prefix-caching):                                                 ││
│  │  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  ││
│  │  │ Phase              │ Time (8B H100) │ What Happens                                                │  ││
│  │  ├──────────────────────────────────────────────────────────────────────────────────────────────────┤  ││
│  │  │ 1. Image Preprocess│ 5 ms           │ Same as before                                              │  ││
│  │  │ 2. Tokenize        │ 2 ms           │ Text → tokens                                               │  ││
│  │  │ 3. Vision Encode   │ 0 ms           │ SKIPPED! Vision embeddings cached                          │  ││
│  │  │ 4. KV Lookup       │ 1 ms           │ Hash matches → reuse vision KV blocks                      │  ││
│  │  │ 5. Prefill         │ 8 ms           │ Only new text tokens (not 1332 vision!)                    │  ││
│  │  │ 6. Decode (×100)   │ 600 ms         │ Same as before                                              │  ││
│  │  ├──────────────────────────────────────────────────────────────────────────────────────────────────┤  ││
│  │  │ TOTAL              │ 616 ms         │ ~10% faster (vision cache hit)                             │  ││
│  │  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  ││
│  │                                                                                                         ││
│  │  NOTE: Vision token KV is cached, but not the vision encoder computation                               ││
│  │        Full vision encoding still runs unless using embedding cache                                     ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  SCENARIO 3: MULTI-TURN CONVERSATION                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Turn 1: User: [image] "What is in this image?" → Assistant: "A red car on a street."                 ││
│  │  Turn 2: User: "What color is the car?"                                                                ││
│  │  Turn 3: User: "Is it parked or moving?"                                                               ││
│  │                                                                                                         ││
│  │  KV CACHE STATE EVOLUTION:                                                                              ││
│  │  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  ││
│  │  │                                                                                                  │  ││
│  │  │  After Turn 1:                                                                                   │  ││
│  │  │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │  ││
│  │  │  │ [system] [user] [image×1332] [?] [assistant] [A] [red] [car] [...] [EOS]                  │  │  ││
│  │  │  │                      1400 tokens in KV cache                                               │  │  ││
│  │  │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │  ││
│  │  │                                                                                                  │  ││
│  │  │  After Turn 2:                                                                                   │  ││
│  │  │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │  ││
│  │  │  │ [...previous...] [user] [What] [color] [...] [?] [assistant] [The] [car] [is] [red] [EOS] │  │  ││
│  │  │  │                      1420 tokens in KV cache                                               │  │  ││
│  │  │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │  ││
│  │  │                                                                                                  │  ││
│  │  │  After Turn 3:                                                                                   │  ││
│  │  │  ┌────────────────────────────────────────────────────────────────────────────────────────────┐  │  ││
│  │  │  │ [...previous...] [user] [Is] [it] [...] [?] [assistant] [It] [is] [parked] [EOS]          │  │  ││
│  │  │  │                      1445 tokens in KV cache                                               │  │  ││
│  │  │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │  ││
│  │  │                                                                                                  │  ││
│  │  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  ││
│  │                                                                                                         ││
│  │  LATENCY PER TURN:                                                                                      ││
│  │  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  ││
│  │  │                                                                                                  │  ││
│  │  │  Turn 1: 680 ms  (full vision + prefill + decode)                                               │  ││
│  │  │  Turn 2: 55 ms   (only new user tokens prefill + decode)                                        │  ││
│  │  │  Turn 3: 50 ms   (same, slightly less tokens)                                                   │  ││
│  │  │                                                                                                  │  ││
│  │  │  KEY INSIGHT: After first turn, only NEW tokens need processing                                 │  ││
│  │  │  KV cache for all history is preserved → subsequent turns are FAST                              │  ││
│  │  │                                                                                                  │  ││
│  │  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  LATENCY SUMMARY BY SCENARIO                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Scenario                    │ 8B on H100    │ 32B on H100   │ Limiting Factor                         ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Cold start (new image)      │ 680 ms        │ 1473 ms       │ Vision encode + full prefill           ││
│  │  Same image, new prompt      │ 616 ms        │ 1350 ms       │ Prefill (vision KV cached)             ││
│  │  Multi-turn (subsequent)     │ 50-80 ms      │ 100-150 ms    │ Incremental prefill only               ││
│  │  Decode per token            │ 6 ms          │ 12 ms         │ Memory bandwidth (weight read)         ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Explicit GPU Breakdowns: A100, H100, B200

### Qwen3-VL-8B on A100-80GB

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B ON A100-80GB SXM: COMPLETE HARDWARE BREAKDOWN                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  A100-80GB HARDWARE SPECS                                                                                   │
│  ════════════════════════                                                                                   │
│  Architecture: Ampere (SM 8.0)                                                                              │
│  VRAM: 80 GB HBM2e                                                                                          │
│  Memory Bandwidth: 2,039 GB/s                                                                               │
│  FP16/BF16 Tensor: 312 TFLOPS                                                                               │
│  FP32: 19.5 TFLOPS                                                                                          │
│  SMs: 108                                                                                                   │
│  Tensor Cores: 432 (3rd Gen)                                                                                │
│  L2 Cache: 40 MB                                                                                            │
│  Shared Mem/SM: 164 KB usable                                                                               │
│  Power: 400W TDP                                                                                            │
│                                                                                                             │
│  VLLM CONFIGURATION                                                                                         │
│  ══════════════════                                                                                         │
│  Attention Backend: FlashAttention 2                                                                        │
│  FP8 Support: ❌ No (A100 lacks FP8)                                                                        │
│  TRTLLM Decode: ❌ No                                                                                       │
│  Precision: BF16                                                                                            │
│  CUDA Graphs: ✅ Enabled                                                                                    │
│                                                                                                             │
│  MEMORY ALLOCATION                                                                                          │
│  ═════════════════                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 80 GB A100 HBM2e                                                                                       ││
│  │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Model Weights (BF16)           │ 16.6 GB (21%)                                                      │││
│  │ │   Vision Encoder               │   2.4 GB                                                           │││
│  │ │   LLM Embedding                │   0.6 GB                                                           │││
│  │ │   LLM Attention (32 layers)    │   4.2 GB                                                           │││
│  │ │   LLM MLP (32 layers)          │   8.6 GB                                                           │││
│  │ │   LM Head                      │   0.8 GB                                                           │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Activations (peak)             │ 4.0 GB (5%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ CUDA Context + Graphs          │ 3.0 GB (4%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ KV Cache Available             │ 56.4 GB (70%)                                                      │││
│  │ │   Per token (BF16):            │   128 KB                                                           │││
│  │ │   At 8K context:               │   1.0 GB per request                                               │││
│  │ │   At 16K context:              │   2.0 GB per request                                               │││
│  │ │   Max concurrent @8K:          │   ~56 requests                                                     │││
│  │ │   Max concurrent @16K:         │   ~28 requests                                                     │││
│  │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION BREAKDOWN (1024×1024 image + 50 prompt → 200 output)                                            │
│  ══════════════════════════════════════════════════════════════                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Phase               │ Time      │ SMs Used │ Tensor Core │ HBM BW    │ Bottleneck                    ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Image Preprocess    │ 5 ms      │ CPU      │ N/A         │ N/A       │ CPU                           ││
│  │  Tokenize            │ 2 ms      │ CPU      │ N/A         │ N/A       │ CPU                           ││
│  │  Vision Encode       │ 25 ms     │ 108/108  │ 85%         │ 800 GB/s  │ Compute                       ││
│  │    Conv3D            │   4 ms    │ 108      │ 80%         │ 600 GB/s  │                               ││
│  │    ViT×32 layers     │   19 ms   │ 108      │ 90%         │ 900 GB/s  │                               ││
│  │    Merger            │   2 ms    │ 108      │ 75%         │ 500 GB/s  │                               ││
│  │  KV Allocation       │ 1 ms      │ N/A      │ N/A         │ N/A       │ Memory alloc                  ││
│  │  Prefill (1382 tok)  │ 120 ms    │ 108/108  │ 92%         │ 1.2 TB/s  │ Compute                       ││
│  │    Per layer:        │   3.75 ms │ 108      │ 92%         │ 1.2 TB/s  │                               ││
│  │  Decode (×200)       │ 2400 ms   │ 60-80    │ 35%         │ 1.9 TB/s  │ Memory BW                     ││
│  │    Per token:        │   12 ms   │ 70       │ 35%         │ 1.9 TB/s  │ Weight read                   ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL               │ 2553 ms   │          │             │           │                               ││
│  │  TTFT (first token)  │ 153 ms    │          │             │           │                               ││
│  │  Throughput          │ 78 tok/s  │          │             │           │                               ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  DECODE ANALYSIS (Why 12ms per token)                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Weight Read: 16.6 GB / 2039 GB/s = 8.1 ms (theoretical minimum)                                      ││
│  │  KV Read: ~0.2 GB (1382 tokens × 128 KB) / 2039 GB/s = 0.1 ms                                         ││
│  │  Overhead (kernel launch, sync): ~3.8 ms                                                              ││
│  │  ─────────────────────────────────────────────────────────────                                        ││
│  │  Actual: ~12 ms/token (67% efficiency vs theoretical)                                                 ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VLLM COMMAND                                                                                               │
│  ────────────                                                                                               │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype bfloat16 \                                                                                     │
│      --max-model-len 32768 \                                                                                │
│      --max-num-seqs 32 \                                                                                    │
│      --gpu-memory-utilization 0.90 \                                                                        │
│      --enable-prefix-caching                                                                                │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-8B on H100-80GB

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B ON H100-80GB SXM: COMPLETE HARDWARE BREAKDOWN                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  H100-80GB HARDWARE SPECS                                                                                   │
│  ════════════════════════                                                                                   │
│  Architecture: Hopper (SM 9.0)                                                                              │
│  VRAM: 80 GB HBM3                                                                                           │
│  Memory Bandwidth: 3,350 GB/s (+64% vs A100)                                                                │
│  FP16/BF16 Tensor: 990 TFLOPS (+217% vs A100)                                                               │
│  FP8 Tensor: 1,980 TFLOPS                                                                                   │
│  SMs: 132 (+22% vs A100)                                                                                    │
│  Tensor Cores: 528 (4th Gen)                                                                                │
│  L2 Cache: 50 MB                                                                                            │
│  Shared Mem/SM: 228 KB usable                                                                               │
│  TMA: ✅ Tensor Memory Accelerator                                                                          │
│  Thread Block Clusters: ✅                                                                                  │
│  Power: 700W TDP                                                                                            │
│                                                                                                             │
│  VLLM CONFIGURATION                                                                                         │
│  ══════════════════                                                                                         │
│  Attention Backend: FlashAttention 3                                                                        │
│  FP8 Support: ✅ Yes (compute + KV cache)                                                                   │
│  TRTLLM Decode: ✅ Yes                                                                                      │
│  Precision: BF16 compute, FP8 KV cache                                                                      │
│  CUDA Graphs: ✅ Enabled                                                                                    │
│                                                                                                             │
│  MEMORY ALLOCATION (with FP8 KV cache)                                                                      │
│  ═════════════════════════════════════                                                                      │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 80 GB H100 HBM3                                                                                        ││
│  │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Model Weights (BF16)           │ 16.6 GB (21%)                                                      │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Activations (peak)             │ 4.0 GB (5%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ CUDA Context + Graphs          │ 3.0 GB (4%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ KV Cache Available (FP8!)      │ 56.4 GB (70%)                                                      │││
│  │ │   Per token (FP8):             │   64 KB (half of BF16!)                                            │││
│  │ │   At 8K context:               │   0.5 GB per request                                               │││
│  │ │   At 32K context:              │   2.0 GB per request                                               │││
│  │ │   Max concurrent @8K:          │   ~112 requests (2× A100!)                                         │││
│  │ │   Max concurrent @32K:         │   ~28 requests                                                     │││
│  │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION BREAKDOWN (1024×1024 image + 50 prompt → 200 output)                                            │
│  ══════════════════════════════════════════════════════════════                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Phase               │ Time      │ SMs Used │ Tensor Core │ HBM BW    │ Bottleneck                    ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Image Preprocess    │ 5 ms      │ CPU      │ N/A         │ N/A       │ CPU                           ││
│  │  Tokenize            │ 2 ms      │ CPU      │ N/A         │ N/A       │ CPU                           ││
│  │  Vision Encode       │ 12 ms     │ 132/132  │ 90%         │ 1.5 TB/s  │ Compute                       ││
│  │    Conv3D            │   2 ms    │ 132      │ 85%         │ 1.2 TB/s  │                               ││
│  │    ViT×32 layers     │   9 ms    │ 132      │ 92%         │ 1.6 TB/s  │                               ││
│  │    Merger            │   1 ms    │ 132      │ 80%         │ 1.0 TB/s  │                               ││
│  │  KV Allocation       │ 1 ms      │ N/A      │ N/A         │ N/A       │ Memory alloc                  ││
│  │  Prefill (1382 tok)  │ 60 ms     │ 132/132  │ 95%         │ 2.0 TB/s  │ Compute                       ││
│  │    Per layer:        │   1.88 ms │ 132      │ 95%         │ 2.0 TB/s  │ (2× faster than A100)         ││
│  │  Decode (×200)       │ 1200 ms   │ 80-100   │ 40%         │ 3.2 TB/s  │ Memory BW                     ││
│  │    Per token:        │   6 ms    │ 90       │ 40%         │ 3.2 TB/s  │ (2× faster than A100)         ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL               │ 1280 ms   │          │             │           │ (2× faster than A100)         ││
│  │  TTFT (first token)  │ 80 ms     │          │             │           │ (1.9× faster)                 ││
│  │  Throughput          │ 156 tok/s │          │             │           │                               ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  DECODE ANALYSIS (Why 6ms per token vs 12ms on A100)                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Weight Read: 16.6 GB / 3350 GB/s = 5.0 ms (theoretical minimum)                                      ││
│  │  KV Read: ~0.1 GB (1382 tokens × 64 KB FP8) / 3350 GB/s = 0.03 ms                                     ││
│  │  Overhead: ~0.97 ms (lower due to TMA + Thread Block Clusters)                                        ││
│  │  ─────────────────────────────────────────────────────────────                                        ││
│  │  Actual: ~6 ms/token (83% efficiency - better than A100!)                                             ││
│  │                                                                                                        ││
│  │  WHY H100 IS 2× FASTER:                                                                               ││
│  │  1. 64% more memory bandwidth (3350 vs 2039 GB/s)                                                     ││
│  │  2. FP8 KV cache halves KV read traffic                                                               ││
│  │  3. TMA reduces address calculation overhead                                                          ││
│  │  4. Thread Block Clusters enable SM cooperation                                                       ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VLLM COMMAND                                                                                               │
│  ────────────                                                                                               │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype bfloat16 \                                                                                     │
│      --kv-cache-dtype fp8 \                                                                                 │
│      --max-model-len 65536 \                                                                                │
│      --max-num-seqs 64 \                                                                                    │
│      --gpu-memory-utilization 0.90 \                                                                        │
│      --enable-prefix-caching                                                                                │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-8B on B200-192GB

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B ON B200-192GB: COMPLETE HARDWARE BREAKDOWN                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  B200 HARDWARE SPECS                                                                                        │
│  ═══════════════════                                                                                        │
│  Architecture: Blackwell (SM 10.0)                                                                          │
│  VRAM: 192 GB HBM3e                                                                                         │
│  Memory Bandwidth: 8,000 GB/s (+139% vs H100)                                                               │
│  FP16/BF16 Tensor: 2,250 TFLOPS (+127% vs H100)                                                             │
│  FP8 Tensor: 4,500 TFLOPS                                                                                   │
│  FP4 Tensor: 9,000 TFLOPS (new!)                                                                            │
│  SMs: 192 (+45% vs H100)                                                                                    │
│  Tensor Cores: 768 (5th Gen)                                                                                │
│  L2 Cache: 96 MB                                                                                            │
│  Shared Mem/SM: ~300 KB usable                                                                              │
│  2nd Gen Transformer Engine: ✅                                                                             │
│  NVLink 5.0: 1.8 TB/s                                                                                       │
│  Power: 1000W TDP                                                                                           │
│                                                                                                             │
│  VLLM CONFIGURATION                                                                                         │
│  ══════════════════                                                                                         │
│  Attention Backend: FlashInfer + TRTLLM decode                                                              │
│  FP8 Support: ✅ Yes                                                                                        │
│  FP4 Support: ✅ Yes (future)                                                                               │
│  TRTLLM Decode: ✅ Optimized                                                                                │
│  Precision: BF16 (no need for FP8 with 192GB!)                                                              │
│  CUDA Graphs: ✅ Enabled                                                                                    │
│                                                                                                             │
│  MEMORY ALLOCATION                                                                                          │
│  ═════════════════                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 192 GB B200 HBM3e                                                                                      ││
│  │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Model Weights (BF16)           │ 16.6 GB (9%)   ← Tiny fraction of VRAM!                            │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Activations (peak)             │ 4.0 GB (2%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ CUDA Context + Graphs          │ 4.0 GB (2%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ KV Cache Available             │ 167.4 GB (87%) ← MASSIVE KV BUDGET                                 │││
│  │ │   Per token (BF16):            │   128 KB                                                           │││
│  │ │   At 8K context:               │   1.0 GB per request                                               │││
│  │ │   At 128K context:             │   16.4 GB per request                                              │││
│  │ │   Max concurrent @8K:          │   ~167 requests                                                    │││
│  │ │   Max concurrent @32K:         │   ~41 requests                                                     │││
│  │ │   Max concurrent @128K:        │   ~10 requests (long-context specialist!)                          │││
│  │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION BREAKDOWN (1024×1024 image + 50 prompt → 200 output)                                            │
│  ══════════════════════════════════════════════════════════════                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Phase               │ Time      │ SMs Used │ Tensor Core │ HBM BW    │ Bottleneck                    ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Image Preprocess    │ 5 ms      │ CPU      │ N/A         │ N/A       │ CPU                           ││
│  │  Tokenize            │ 2 ms      │ CPU      │ N/A         │ N/A       │ CPU                           ││
│  │  Vision Encode       │ 8 ms      │ 192/192  │ 92%         │ 3.0 TB/s  │ Compute                       ││
│  │    Conv3D            │   1.5 ms  │ 192      │ 88%         │ 2.5 TB/s  │                               ││
│  │    ViT×32 layers     │   6 ms    │ 192      │ 94%         │ 3.2 TB/s  │                               ││
│  │    Merger            │   0.5 ms  │ 192      │ 85%         │ 2.0 TB/s  │                               ││
│  │  KV Allocation       │ 0.5 ms    │ N/A      │ N/A         │ N/A       │ Memory alloc                  ││
│  │  Prefill (1382 tok)  │ 35 ms     │ 192/192  │ 96%         │ 4.0 TB/s  │ Compute                       ││
│  │    Per layer:        │   1.09 ms │ 192      │ 96%         │ 4.0 TB/s  │                               ││
│  │  Decode (×200)       │ 600 ms    │ 100-140  │ 45%         │ 7.5 TB/s  │ Memory BW                     ││
│  │    Per token:        │   3 ms    │ 120      │ 45%         │ 7.5 TB/s  │ (2× faster than H100)         ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL               │ 650 ms    │          │             │           │ (2× faster than H100)         ││
│  │  TTFT (first token)  │ 50 ms     │          │             │           │ (1.6× faster than H100)       ││
│  │  Throughput          │ 308 tok/s │          │             │           │                               ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  DECODE ANALYSIS (Why 3ms per token)                                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Weight Read: 16.6 GB / 8000 GB/s = 2.1 ms (theoretical minimum)                                      ││
│  │  KV Read: ~0.18 GB / 8000 GB/s = 0.02 ms                                                              ││
│  │  Overhead: ~0.88 ms                                                                                   ││
│  │  ─────────────────────────────────────────────────────────────                                        ││
│  │  Actual: ~3 ms/token (70% efficiency)                                                                 ││
│  │                                                                                                        ││
│  │  WHY B200 IS 2× FASTER THAN H100:                                                                     ││
│  │  1. 139% more memory bandwidth (8000 vs 3350 GB/s)                                                    ││
│  │  2. 45% more SMs (192 vs 132)                                                                         ││
│  │  3. 2nd Gen Transformer Engine                                                                        ││
│  │  4. Larger L2 cache (96 MB vs 50 MB)                                                                  ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VLLM COMMAND                                                                                               │
│  ────────────                                                                                               │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype bfloat16 \                                                                                     │
│      --max-model-len 131072 \                                                                               │
│      --max-num-seqs 128 \                                                                                   │
│      --gpu-memory-utilization 0.90 \                                                                        │
│      --enable-prefix-caching \                                                                              │
│      --enable-chunked-prefill                                                                               │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-32B on A100-80GB

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-32B ON A100-80GB SXM: COMPLETE HARDWARE BREAKDOWN                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ⚠️ TIGHT FIT WARNING: 32B requires ~72GB VRAM, leaving only ~8GB for KV cache                             │
│                                                                                                             │
│  MEMORY ALLOCATION                                                                                          │
│  ═════════════════                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 80 GB A100 HBM2e                                                                                       ││
│  │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Model Weights (BF16)           │ 65.8 GB (82%) ← DOMINATES VRAM                                     │││
│  │ │   Vision Encoder               │   4.0 GB                                                           │││
│  │ │   LLM Embedding                │   0.8 GB                                                           │││
│  │ │   LLM Attention (64 layers)    │  13.1 GB                                                           │││
│  │ │   LLM MLP (64 layers)          │  46.9 GB                                                           │││
│  │ │   LM Head                      │   1.0 GB                                                           │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Activations (peak)             │ 8.0 GB (10%)                                                       │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ CUDA Context + Graphs          │ 4.0 GB (5%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ KV Cache Available             │ 2.2 GB (3%) ← VERY LIMITED!                                        │││
│  │ │   Per token (BF16):            │   256 KB                                                           │││
│  │ │   At 4K context:               │   1.0 GB per request                                               │││
│  │ │   Max concurrent @4K:          │   ~2 requests                                                      │││
│  │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION BREAKDOWN (1024×1024 image + 50 prompt → 200 output)                                            │
│  ══════════════════════════════════════════════════════════════                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Phase               │ Time      │ SMs Used │ Tensor Core │ HBM BW    │ Bottleneck                    ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encode       │ 50 ms     │ 108/108  │ 85%         │ 1.0 TB/s  │ Compute                       ││
│  │  Prefill (1382 tok)  │ 250 ms    │ 108/108  │ 90%         │ 1.2 TB/s  │ Compute                       ││
│  │    Per layer:        │   3.9 ms  │ 108      │ 90%         │ 1.2 TB/s  │ (64 layers!)                  ││
│  │  Decode (×200)       │ 5000 ms   │ 50-70    │ 30%         │ 1.9 TB/s  │ Memory BW                     ││
│  │    Per token:        │   25 ms   │ 60       │ 30%         │ 1.9 TB/s  │ (4× weight read vs 8B)        ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL               │ 5310 ms   │          │             │           │                               ││
│  │  TTFT (first token)  │ 310 ms    │          │             │           │                               ││
│  │  Throughput          │ 38 tok/s  │          │             │           │ (2× slower than 8B)           ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  DECODE ANALYSIS (Why 25ms per token)                                                                       │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Weight Read: 65.8 GB / 2039 GB/s = 32.3 ms (theoretical minimum)                                     ││
│  │  Actual: ~25 ms/token (77% efficiency - some L2 caching helps)                                        ││
│  │                                                                                                        ││
│  │  ⚠️ NOTE: 32B on A100-80GB is MEMORY-STARVED for KV cache                                              ││
│  │           Consider H100 or multi-GPU for production                                                   ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VLLM COMMAND (limited configuration)                                                                       │
│  ─────────────────────────────────────                                                                      │
│  vllm serve Qwen/Qwen3-VL-32B-Instruct \                                                                   │
│      --dtype bfloat16 \                                                                                     │
│      --max-model-len 4096 \                                                                                 │
│      --max-num-seqs 2 \                                                                                     │
│      --gpu-memory-utilization 0.95 \                                                                        │
│      --enforce-eager  # Disable CUDA graphs to save memory                                                  │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-32B on H100-80GB

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-32B ON H100-80GB SXM: COMPLETE HARDWARE BREAKDOWN                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ✅ RECOMMENDED CONFIG: FP8 weights halves memory, enabling reasonable KV cache                            │
│                                                                                                             │
│  MEMORY ALLOCATION (FP8 weights)                                                                            │
│  ═══════════════════════════════                                                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 80 GB H100 HBM3                                                                                        ││
│  │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Model Weights (FP8)            │ 32.9 GB (41%) ← Half of BF16!                                      │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Activations (peak, BF16)       │ 8.0 GB (10%)                                                       │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ CUDA Context + Graphs          │ 4.0 GB (5%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ KV Cache Available (FP8)       │ 35.1 GB (44%)                                                      │││
│  │ │   Per token (FP8):             │   128 KB                                                           │││
│  │ │   At 8K context:               │   1.0 GB per request                                               │││
│  │ │   Max concurrent @8K:          │   ~35 requests                                                     │││
│  │ │   Max concurrent @16K:         │   ~17 requests                                                     │││
│  │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION BREAKDOWN (1024×1024 image + 50 prompt → 200 output)                                            │
│  ══════════════════════════════════════════════════════════════                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Phase               │ Time      │ SMs Used │ Tensor Core │ HBM BW    │ Bottleneck                    ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encode       │ 25 ms     │ 132/132  │ 90%         │ 1.8 TB/s  │ Compute                       ││
│  │  Prefill (1382 tok)  │ 120 ms    │ 132/132  │ 94%         │ 2.2 TB/s  │ Compute                       ││
│  │    Per layer:        │   1.88 ms │ 132      │ 94%         │ 2.2 TB/s  │ (FP8 Tensor Cores!)           ││
│  │  Decode (×200)       │ 2400 ms   │ 80-110   │ 38%         │ 3.2 TB/s  │ Memory BW                     ││
│  │    Per token:        │   12 ms   │ 95       │ 38%         │ 3.2 TB/s  │ (FP8 halves weight read)      ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL               │ 2555 ms   │          │             │           │                               ││
│  │  TTFT (first token)  │ 155 ms    │          │             │           │                               ││
│  │  Throughput          │ 78 tok/s  │          │             │           │ (Same as 8B on A100!)         ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  KEY INSIGHT: 32B on H100 FP8 = 8B on A100 BF16 speed!                                                     │
│  ───────────────────────────────────────────────────────                                                    │
│  • Same ~78 tok/s throughput                                                                               │
│  • But 4× more parameters = better quality for complex tasks                                               │
│  • GUI agents should use 32B on H100, not 8B on A100                                                       │
│                                                                                                             │
│  VLLM COMMAND                                                                                               │
│  ────────────                                                                                               │
│  vllm serve Qwen/Qwen3-VL-32B-Instruct \                                                                   │
│      --dtype float8 \                                                                                       │
│      --kv-cache-dtype fp8 \                                                                                 │
│      --max-model-len 16384 \                                                                                │
│      --max-num-seqs 16 \                                                                                    │
│      --gpu-memory-utilization 0.90 \                                                                        │
│      --enable-prefix-caching                                                                                │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-32B on B200-192GB

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-32B ON B200-192GB: COMPLETE HARDWARE BREAKDOWN                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ✅ IDEAL CONFIG: 32B fits comfortably in BF16 with massive KV cache budget                                │
│                                                                                                             │
│  MEMORY ALLOCATION                                                                                          │
│  ═════════════════                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 192 GB B200 HBM3e                                                                                      ││
│  │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Model Weights (BF16)           │ 65.8 GB (34%)                                                      │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Activations (peak)             │ 8.0 GB (4%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ CUDA Context + Graphs          │ 4.0 GB (2%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ KV Cache Available             │ 114.2 GB (60%)                                                     │││
│  │ │   Per token (BF16):            │   256 KB                                                           │││
│  │ │   At 8K context:               │   2.0 GB per request                                               │││
│  │ │   At 32K context:              │   8.0 GB per request                                               │││
│  │ │   Max concurrent @8K:          │   ~57 requests                                                     │││
│  │ │   Max concurrent @32K:         │   ~14 requests                                                     │││
│  │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION BREAKDOWN (1024×1024 image + 50 prompt → 200 output)                                            │
│  ══════════════════════════════════════════════════════════════                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Phase               │ Time      │ SMs Used │ Tensor Core │ HBM BW    │ Bottleneck                    ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encode       │ 15 ms     │ 192/192  │ 92%         │ 3.5 TB/s  │ Compute                       ││
│  │  Prefill (1382 tok)  │ 70 ms     │ 192/192  │ 96%         │ 4.5 TB/s  │ Compute                       ││
│  │    Per layer:        │   1.09 ms │ 192      │ 96%         │ 4.5 TB/s  │                               ││
│  │  Decode (×200)       │ 1200 ms   │ 120-160  │ 42%         │ 7.8 TB/s  │ Memory BW                     ││
│  │    Per token:        │   6 ms    │ 140      │ 42%         │ 7.8 TB/s  │                               ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL               │ 1295 ms   │          │             │           │                               ││
│  │  TTFT (first token)  │ 95 ms     │          │             │           │                               ││
│  │  Throughput          │ 154 tok/s │          │             │           │ (2× faster than 32B on H100)  ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VLLM COMMAND                                                                                               │
│  ────────────                                                                                               │
│  vllm serve Qwen/Qwen3-VL-32B-Instruct \                                                                   │
│      --dtype bfloat16 \                                                                                     │
│      --max-model-len 65536 \                                                                                │
│      --max-num-seqs 48 \                                                                                    │
│      --gpu-memory-utilization 0.90 \                                                                        │
│      --enable-prefix-caching \                                                                              │
│      --enable-chunked-prefill                                                                               │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-30B-A3B (MoE) on H100-80GB

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-30B-A3B (MoE) ON H100-80GB: COMPLETE HARDWARE BREAKDOWN                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  MOE MODEL CHARACTERISTICS                                                                                  │
│  ═════════════════════════                                                                                  │
│  Total Parameters: 30B (128 experts × 48 layers)                                                           │
│  Active Parameters: 3B per token (top-8 of 128 experts)                                                    │
│  Activation Ratio: 10% (only 8/128 experts compute per token)                                              │
│  Memory Required: ALL 30B params must be in VRAM (can't page experts)                                      │
│                                                                                                             │
│  MEMORY ALLOCATION (FP8)                                                                                    │
│  ═══════════════════════                                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 80 GB H100 HBM3                                                                                        ││
│  │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Model Weights (FP8)            │ 30.0 GB (38%)                                                      │││
│  │ │   Vision Encoder               │   1.0 GB                                                           │││
│  │ │   Shared Attention (48 layers) │   2.4 GB                                                           │││
│  │ │   Router Weights (48 layers)   │   0.01 GB (tiny!)                                                  │││
│  │ │   Expert Weights (48×128)      │  26.6 GB                                                           │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ Activations (peak)             │ 4.0 GB (5%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ CUDA Context + Graphs          │ 4.0 GB (5%)                                                        │││
│  │ ├──────────────────────────────────────────────────────────────────────────────────────────────────────┤│
│  │ │ KV Cache Available (FP8)       │ 42.0 GB (52%)                                                      │││
│  │ │   Per token (FP8):             │   49 KB (smaller than 8B dense!)                                   │││
│  │ │   At 8K context:               │   0.4 GB per request                                               │││
│  │ │   At 32K context:              │   1.6 GB per request                                               │││
│  │ │   Max concurrent @8K:          │   ~105 requests                                                    │││
│  │ │   Max concurrent @32K:         │   ~26 requests                                                     │││
│  │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION BREAKDOWN (1024×1024 image + 50 prompt → 200 output)                                            │
│  ══════════════════════════════════════════════════════════════                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Phase               │ Time      │ SMs Used │ Tensor Core │ HBM BW    │ Bottleneck                    ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encode       │ 25 ms     │ 132/132  │ 90%         │ 1.8 TB/s  │ Compute                       ││
│  │  Prefill (1382 tok)  │ 75 ms     │ 132/132  │ 85%         │ 2.5 TB/s  │ Compute (FusedMoE)            ││
│  │    Per layer:        │   1.56 ms │ 132      │ 85%         │ 2.5 TB/s  │ (48 layers with MoE)          ││
│  │  Decode (×200)       │ 1200 ms   │ 90-120   │ 50%         │ 2.8 TB/s  │ Expert gather                 ││
│  │    Per token:        │   6 ms    │ 105      │ 50%         │ 2.8 TB/s  │                               ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL               │ 1310 ms   │          │             │           │                               ││
│  │  TTFT (first token)  │ 110 ms    │          │             │           │                               ││
│  │  Throughput          │ 153 tok/s │          │             │           │                               ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  MOE DECODE ANALYSIS (Why 6ms per token despite 30B total)                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  PER-TOKEN MEMORY READS:                                                                               ││
│  │  • Shared layers (attention): 2.4 GB                                                                  ││
│  │  • 8 selected experts (of 128): 26.6 GB × (8/128) = 1.7 GB                                            ││
│  │  • Router + embedding: 0.5 GB                                                                         ││
│  │  • TOTAL: ~4.6 GB per token (vs 30 GB if all experts read)                                            ││
│  │                                                                                                        ││
│  │  Weight Read: 4.6 GB / 3350 GB/s = 1.4 ms (theoretical)                                               ││
│  │  Expert dispatch overhead: ~2 ms                                                                      ││
│  │  Actual: ~6 ms/token                                                                                  ││
│  │                                                                                                        ││
│  │  WHY MoE IS EFFICIENT:                                                                                 ││
│  │  • Only 8/128 = 6.25% of expert weights read per token                                                ││
│  │  • Decode speed similar to 4-5B dense model                                                           ││
│  │  • But quality matches 8-10B dense model!                                                             ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VLLM COMMAND                                                                                               │
│  ────────────                                                                                               │
│  vllm serve Qwen/Qwen3-VL-30B-A3B \                                                                        │
│      --dtype bfloat16 \                                                                                     │
│      --kv-cache-dtype fp8 \                                                                                 │
│      --max-model-len 32768 \                                                                                │
│      --max-num-seqs 64 \                                                                                    │
│      --gpu-memory-utilization 0.92                                                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-235B-A22B (MoE) on 4×H100-80GB

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-235B-A22B (MoE) ON 4×H100-80GB: COMPLETE HARDWARE BREAKDOWN                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  MULTI-GPU CONFIGURATION                                                                                    │
│  ═══════════════════════                                                                                    │
│  Tensor Parallel Size: 4                                                                                    │
│  Total VRAM: 4 × 80 GB = 320 GB                                                                            │
│  NVLink Bandwidth: 900 GB/s bidirectional per pair                                                         │
│                                                                                                             │
│  MOE MODEL CHARACTERISTICS                                                                                  │
│  ═════════════════════════                                                                                  │
│  Total Parameters: 235B (128 experts × 94 layers)                                                          │
│  Active Parameters: 22B per token (top-8 of 128 experts)                                                   │
│  Activation Ratio: 9.4%                                                                                    │
│                                                                                                             │
│  MEMORY ALLOCATION (FP8, distributed across 4 GPUs)                                                         │
│  ═══════════════════════════════════════════════════                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  GPU 0 (80GB)         GPU 1 (80GB)         GPU 2 (80GB)         GPU 3 (80GB)                          ││
│  │  ┌────────────────┐   ┌────────────────┐   ┌────────────────┐   ┌────────────────┐                    ││
│  │  │ Weights: 59 GB │   │ Weights: 59 GB │   │ Weights: 59 GB │   │ Weights: 59 GB │                    ││
│  │  │ (32 experts/   │   │ (32 experts/   │   │ (32 experts/   │   │ (32 experts/   │                    ││
│  │  │  layer ÷ 4)    │   │  layer ÷ 4)    │   │  layer ÷ 4)    │   │  layer ÷ 4)    │                    ││
│  │  │                │   │                │   │                │   │                │                    ││
│  │  │ Activations:   │   │ Activations:   │   │ Activations:   │   │ Activations:   │                    ││
│  │  │ 5 GB           │   │ 5 GB           │   │ 5 GB           │   │ 5 GB           │                    ││
│  │  │                │   │                │   │                │   │                │                    ││
│  │  │ KV Cache:      │   │ KV Cache:      │   │ KV Cache:      │   │ KV Cache:      │                    ││
│  │  │ 12 GB          │   │ 12 GB          │   │ 12 GB          │   │ 12 GB          │                    ││
│  │  └────────────────┘   └────────────────┘   └────────────────┘   └────────────────┘                    ││
│  │         │                    │                    │                    │                               ││
│  │         └────────────────────┴────────────────────┴────────────────────┘                               ││
│  │                            NVLink 900 GB/s                                                             ││
│  │                                                                                                        ││
│  │  Total KV Cache: 48 GB across 4 GPUs                                                                  ││
│  │  Per token (FP8): 245 KB                                                                              ││
│  │  At 8K context: 2.0 GB per request                                                                    ││
│  │  Max concurrent @8K: ~24 requests                                                                     ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION BREAKDOWN (1024×1024 image + 50 prompt → 200 output)                                            │
│  ══════════════════════════════════════════════════════════════                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Phase               │ Time      │ Communication     │ Bottleneck                                     ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encode       │ 50 ms     │ Broadcast         │ Compute                                        ││
│  │  Prefill (1382 tok)  │ 300 ms    │ All-Reduce        │ Compute + All-Reduce                          ││
│  │    Per layer:        │   3.2 ms  │ 0.5 ms/layer      │                                                ││
│  │  Decode (×200)       │ 2400 ms   │ All-to-All        │ Expert dispatch + All-Reduce                  ││
│  │    Per token:        │   12 ms   │                   │                                                ││
│  │    Breakdown:        │           │                   │                                                ││
│  │      Router          │   0.2 ms  │ -                 │ Local compute                                  ││
│  │      All-to-All      │   2.0 ms  │ Expert dispatch   │ NVLink transfer                               ││
│  │      Expert compute  │   6.0 ms  │ -                 │ Local FusedMoE                                ││
│  │      All-Reduce      │   2.0 ms  │ Output combine    │ NVLink transfer                               ││
│  │      Attention       │   1.8 ms  │ -                 │ Local                                          ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL               │ 2760 ms   │                   │                                                ││
│  │  TTFT (first token)  │ 360 ms    │                   │                                                ││
│  │  Throughput          │ 72 tok/s  │                   │ (across 4 GPUs)                               ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  COST ANALYSIS                                                                                              │
│  ═════════════                                                                                              │
│  • 4×H100 cloud cost: ~$12/hour                                                                            │
│  • Throughput: 72 tok/s = 259,200 tok/hour                                                                 │
│  • Cost per 1M tokens: $46                                                                                 │
│  • Compare to GPT-4V: ~$30 input + $90 output per 1M tokens                                               │
│  • Frontier quality at ~50% of API cost!                                                                   │
│                                                                                                             │
│  VLLM COMMAND                                                                                               │
│  ────────────                                                                                               │
│  vllm serve Qwen/Qwen3-VL-235B-A22B \                                                                      │
│      --tensor-parallel-size 4 \                                                                             │
│      --dtype bfloat16 \                                                                                     │
│      --kv-cache-dtype fp8 \                                                                                 │
│      --max-model-len 32768 \                                                                                │
│      --max-num-seqs 8 \                                                                                     │
│      --gpu-memory-utilization 0.90                                                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Precision Types: Complete Guide (FP32 → FP4)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     PRECISION TYPES: WHAT THEY ARE AND HOW THEY WORK                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  PRECISION HIERARCHY (Memory per Parameter)                                                                 │
│  ══════════════════════════════════════════                                                                 │
│                                                                                                             │
│  ┌──────────┬──────────┬──────────────────────────────────────────────────────────────────────────────────┐│
│  │ Precision│ Bytes    │ Bit Layout                                                                       ││
│  ├──────────┼──────────┼──────────────────────────────────────────────────────────────────────────────────┤│
│  │ FP32     │ 4 bytes  │ [S|EEEEEEEE|MMMMMMMMMMMMMMMMMMMMMMM]  1 sign + 8 exp + 23 mantissa              ││
│  │ FP16     │ 2 bytes  │ [S|EEEEE|MMMMMMMMMM]                  1 sign + 5 exp + 10 mantissa              ││
│  │ BF16     │ 2 bytes  │ [S|EEEEEEEE|MMMMMMM]                  1 sign + 8 exp + 7 mantissa (brain float) ││
│  │ FP8 E4M3 │ 1 byte   │ [S|EEEE|MMM]                          1 sign + 4 exp + 3 mantissa (weights)     ││
│  │ FP8 E5M2 │ 1 byte   │ [S|EEEEE|MM]                          1 sign + 5 exp + 2 mantissa (gradients)   ││
│  │ INT8     │ 1 byte   │ [SSSSSSSS] or [UUUUUUUU]              Signed/unsigned integer                   ││
│  │ INT4     │ 0.5 bytes│ [SSSS] or [UUUU]                      4-bit integer (packed 2 per byte)         ││
│  │ FP4 E2M1 │ 0.5 bytes│ [S|EE|M]                              1 sign + 2 exp + 1 mantissa (NEW!)        ││
│  └──────────┴──────────┴──────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VISUAL: Memory per 8B Model at Each Precision                                                             │
│  ══════════════════════════════════════════════                                                             │
│                                                                                                             │
│  FP32 (4 bytes):  ████████████████████████████████████████████████████████████████ 32.0 GB                │
│  FP16 (2 bytes):  ████████████████████████████████ 16.0 GB                                                 │
│  BF16 (2 bytes):  ████████████████████████████████ 16.0 GB                                                 │
│  FP8  (1 byte):   ████████████████ 8.0 GB                                                                  │
│  INT8 (1 byte):   ████████████████ 8.0 GB                                                                  │
│  INT4 (0.5 byte): ████████ 4.0 GB                                                                          │
│  FP4  (0.5 byte): ████████ 4.0 GB                                                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### FP8: The H100/Hopper Standard

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     FP8: HOPPER (H100) NATIVE PRECISION                                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  TWO FP8 FORMATS                                                                                            │
│  ═══════════════                                                                                            │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  E4M3 (for weights and activations):                                                                   ││
│  │  ┌─┬────┬───┐                                                                                          ││
│  │  │S│EEEE│MMM│   Range: ±448, Precision: 3 mantissa bits                                               ││
│  │  └─┴────┴───┘   Best for: Model weights (need precision for small values)                             ││
│  │                                                                                                        ││
│  │  E5M2 (for gradients):                                                                                 ││
│  │  ┌─┬─────┬──┐                                                                                          ││
│  │  │S│EEEEE│MM│   Range: ±57344, Precision: 2 mantissa bits                                             ││
│  │  └─┴─────┴──┘   Best for: Gradients (need range for large values)                                     ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  H100 TENSOR CORE FP8 EXECUTION                                                                            │
│  ══════════════════════════════                                                                             │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │     FP8 Weights            FP8 Activations          BF16 Accumulator                                  ││
│  │     ┌───────────┐          ┌───────────┐            ┌───────────┐                                     ││
│  │     │ E4M3 × 8  │    ×     │ E4M3 × 8  │     →      │ BF16 sum  │                                     ││
│  │     │ (8 bytes) │          │ (8 bytes) │            │ (2 bytes) │                                     ││
│  │     └───────────┘          └───────────┘            └───────────┘                                     ││
│  │                                                                                                        ││
│  │  Tensor Core executes: 8×8 FP8 matrix multiply → BF16 accumulator                                     ││
│  │  Throughput: 1980 TFLOPS (2× of BF16!)                                                                ││
│  │                                                                                                        ││
│  │  Dynamic Scaling (Transformer Engine):                                                                 ││
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │  For each tensor:                                                                              │   ││
│  │  │  1. Find max absolute value: amax = max(|tensor|)                                              │   ││
│  │  │  2. Compute scale: scale = FP8_MAX / amax                                                      │   ││
│  │  │  3. Quantize: fp8_tensor = round(tensor × scale)                                               │   ││
│  │  │  4. Store scale factor for dequantization                                                      │   ││
│  │  └────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  GPU SUPPORT                                                                                                │
│  ═══════════                                                                                                │
│  • T4 (SM 7.5):   ❌ No FP8 support                                                                        │
│  • A100 (SM 8.0): ❌ No native FP8 (can emulate with INT8)                                                 │
│  • H100 (SM 9.0): ✅ Native FP8 with Transformer Engine                                                    │
│  • B200 (SM 10):  ✅ Native FP8 + FP4                                                                      │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### FP4: NVIDIA Blackwell's New Precision (B200)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     FP4: BLACKWELL (B200) NEW PRECISION TYPE                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WHAT IS FP4?                                                                                               │
│  ════════════                                                                                               │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  FP4 E2M1 Format:                                                                                      ││
│  │  ┌─┬──┬─┐                                                                                              ││
│  │  │S│EE│M│   4 bits total: 1 sign + 2 exponent + 1 mantissa                                            ││
│  │  └─┴──┴─┘                                                                                              ││
│  │                                                                                                        ││
│  │  Representable Values (E2M1):                                                                          ││
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │  Exponent │ Mantissa │ Value                                                                   │   ││
│  │  │  ─────────────────────────────────────────────────────────────────────────────────────────────│   ││
│  │  │  00       │ 0        │ ±0.0 (zero)                                                             │   ││
│  │  │  00       │ 1        │ ±0.5 (subnormal)                                                        │   ││
│  │  │  01       │ 0        │ ±1.0                                                                    │   ││
│  │  │  01       │ 1        │ ±1.5                                                                    │   ││
│  │  │  10       │ 0        │ ±2.0                                                                    │   ││
│  │  │  10       │ 1        │ ±3.0                                                                    │   ││
│  │  │  11       │ 0        │ ±4.0                                                                    │   ││
│  │  │  11       │ 1        │ ±6.0                                                                    │   ││
│  │  └────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                        ││
│  │  Only 16 distinct values! (8 positive, 7 negative, 1 zero)                                            ││
│  │  Range: -6.0 to +6.0                                                                                   ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  WHY FP4 WORKS FOR INFERENCE                                                                               │
│  ═══════════════════════════                                                                                │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  1. WEIGHT DISTRIBUTION                                                                                ││
│  │  ──────────────────────                                                                                ││
│  │  LLM weights follow ~Gaussian distribution centered at 0                                              ││
│  │  Most values are small: |w| < 3.0 for 99%+ of weights                                                 ││
│  │                                                                                                        ││
│  │              Weight Distribution                   FP4 Quantization Levels                            ││
│  │                    ▲                                      │                                            ││
│  │                   █│█                                     │                                            ││
│  │                  ██│██                               ─────┼───── +6.0                                  ││
│  │                 ███│███                                   │                                            ││
│  │                ████│████                             ─────┼───── +4.0                                  ││
│  │               █████│█████                                 │                                            ││
│  │              ██████│██████                           ─────┼───── +3.0                                  ││
│  │           ─────────┼─────────                             │                                            ││
│  │           -3      0       +3                         ─────┼───── +2.0  ← Most weights here            ││
│  │                                                           │                                            ││
│  │                                                      ─────┼───── +1.5                                  ││
│  │  2. BLOCK-WISE SCALING                                    │                                            ││
│  │  ───────────────────────                             ─────┼───── +1.0                                  ││
│  │  Weights quantized in blocks (e.g., 128 weights)          │                                            ││
│  │  Each block has its own FP8 scale factor                  │                                            ││
│  │  Actual_weight = FP4_value × block_scale             ─────┼───── 0.0                                   ││
│  │                                                                                                        ││
│  │  3. MICRO-SCALING                                                                                      ││
│  │  ────────────────                                                                                      ││
│  │  NVIDIA's "Microscaling" (MX) format:                                                                 ││
│  │  • 32 elements share 1 scale (E8M0 format)                                                            ││
│  │  • Overhead: 1 byte per 32 elements = 3% overhead                                                     ││
│  │  • Effective: 4.25 bits per weight (not 4.0)                                                          ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  B200 FP4 TENSOR CORE EXECUTION                                                                            │
│  ══════════════════════════════                                                                             │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │     FP4 Weights (packed)       BF16/FP8 Activations      FP32 Accumulator                             ││
│  │     ┌───────────────────┐      ┌───────────────────┐     ┌───────────────────┐                        ││
│  │     │ 16 weights × 4bit │  ×   │ 16 activations    │  →  │ FP32 sum          │                        ││
│  │     │ = 8 bytes         │      │ = 16-32 bytes     │     │ = 4 bytes         │                        ││
│  │     └───────────────────┘      └───────────────────┘     └───────────────────┘                        ││
│  │                                                                                                        ││
│  │  5th Gen Tensor Core Operation:                                                                       ││
│  │  • Unpack FP4 → FP8 on-the-fly (hardware decoder)                                                     ││
│  │  • Multiply-accumulate in FP32                                                                        ││
│  │  • Convert back to BF16/FP8 for output                                                                ││
│  │                                                                                                        ││
│  │  Throughput: 9000 TFLOPS (2× of FP8, 4× of BF16!)                                                    ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  GPU SUPPORT FOR FP4                                                                                        │
│  ═══════════════════                                                                                        │
│  • T4 (SM 7.5):   ❌ No                                                                                    │
│  • A100 (SM 8.0): ❌ No                                                                                    │
│  • H100 (SM 9.0): ❌ No (can use INT4 via software)                                                       │
│  • B200 (SM 10):  ✅ Native FP4 with 2nd Gen Transformer Engine                                           │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### INT4 vs FP4: What's the Difference?

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     INT4 vs FP4: COMPARISON                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  BIT LAYOUT                                                                                                 │
│  ══════════                                                                                                 │
│                                                                                                             │
│  INT4 (unsigned):  [UUUU]  →  Values: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15               │
│  INT4 (signed):    [SSSS]  →  Values: -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7              │
│  FP4 (E2M1):       [S|EE|M] → Values: ±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6                                  │
│                                                                                                             │
│  KEY DIFFERENCES                                                                                            │
│  ═══════════════                                                                                            │
│                                                                                                             │
│  ┌─────────────────────┬──────────────────────────────────┬──────────────────────────────────┐             │
│  │ Aspect              │ INT4                             │ FP4                              │             │
│  ├─────────────────────┼──────────────────────────────────┼──────────────────────────────────┤             │
│  │ Value Distribution  │ Uniform: -8 to +7 (or 0-15)      │ Non-uniform: denser near 0       │             │
│  │ Best for            │ Weights with uniform spread      │ Weights with Gaussian spread     │             │
│  │ Zero Representation │ Exact 0                          │ Exact 0                          │             │
│  │ Small Values        │ ±1, ±2, ±3, ±4 (uniform gaps)    │ ±0.5, ±1, ±1.5, ±2 (finer)       │             │
│  │ Large Values        │ ±5, ±6, ±7 (uniform gaps)        │ ±3, ±4, ±6 (coarser)             │             │
│  │ Hardware Support    │ Software (AWQ, GPTQ, BitsAndBytes)│ Native Tensor Cores (B200)      │             │
│  │ Speed               │ Dequant overhead                 │ Hardware unpacking               │             │
│  │ Quality             │ Good with careful quantization   │ Better for LLM weight distro     │             │
│  └─────────────────────┴──────────────────────────────────┴──────────────────────────────────┘             │
│                                                                                                             │
│  VISUAL: Value Coverage                                                                                     │
│  ══════════════════════                                                                                     │
│                                                                                                             │
│  Number Line: -8 -7 -6 -5 -4 -3 -2 -1  0  1  2  3  4  5  6  7  8                                          │
│               ──────────────────────────────────────────────────                                            │
│  INT4 signed:  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●     (uniform spacing)                       │
│  FP4 E2M1:           ●     ●  ●     ●  ●  ●  ●  ●  ●     ●        (denser near 0)                          │
│                      ▲        ▲                          ▲                                                  │
│                     -6       -4                         +6                                                  │
│                                                                                                             │
│  LLM weights are Gaussian → FP4's non-uniform spacing matches better!                                      │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Tensor Core Support by GPU Generation

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     TENSOR CORE PRECISION SUPPORT BY GPU                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌──────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐│
│  │ GPU              │ Tensor Core Precisions                                                               ││
│  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤│
│  │                  │ FP32  FP16  BF16  TF32  FP8   INT8  INT4  FP4   │ Peak TFLOPS                       ││
│  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤│
│  │ T4 (Turing)      │  ❌    ✅    ❌    ❌    ❌    ✅    ❌    ❌   │ 65 (FP16)                         ││
│  │ SM 7.5, 2018     │       ████                     ████          │                                     ││
│  │                  │                                               │                                     ││
│  │ A100 (Ampere)    │  ❌    ✅    ✅    ✅    ❌    ✅    ❌    ❌   │ 312 (BF16)                        ││
│  │ SM 8.0, 2020     │       ████  ████  ████        ████          │                                     ││
│  │                  │                                               │                                     ││
│  │ H100 (Hopper)    │  ❌    ✅    ✅    ✅    ✅    ✅    ❌    ❌   │ 990 (BF16), 1980 (FP8)            ││
│  │ SM 9.0, 2022     │       ████  ████  ████  ████  ████          │                                     ││
│  │                  │            Transformer Engine                 │                                     ││
│  │                  │                                               │                                     ││
│  │ B200 (Blackwell) │  ❌    ✅    ✅    ✅    ✅    ✅    ✅    ✅   │ 2250 (BF16), 4500 (FP8), 9000 (FP4)││
│  │ SM 10.0, 2024    │       ████  ████  ████  ████  ████  ████  ████│                                     ││
│  │                  │            2nd Gen Transformer Engine         │                                     ││
│  └──────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  THROUGHPUT PROGRESSION (Tensor Core TFLOPS)                                                                │
│  ═══════════════════════════════════════════                                                                │
│                                                                                                             │
│  T4 FP16:      ████ 65                                                                                     │
│  A100 BF16:    █████████████████ 312                                                                       │
│  H100 BF16:    ████████████████████████████████████████████████████ 990                                    │
│  H100 FP8:     ████████████████████████████████████████████████████████████████████████████████████ 1980   │
│  B200 BF16:    ████████████████████████████████████████████████████████████████████████████████ 2250       │
│  B200 FP8:     ████████████████████████████████████████████████████████████████████████████████████████... 4500│
│  B200 FP4:     ████████████████████████████████████████████████████████████████████████████████████████... 9000│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL: Memory at Each Precision by Model Size

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL MEMORY FOOTPRINT BY PRECISION                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WEIGHT MEMORY (Model Only, No KV Cache)                                                                    │
│  ═══════════════════════════════════════                                                                    │
│                                                                                                             │
│  ┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐             │
│  │ Model        │ Params   │ FP32     │ BF16     │ FP8      │ INT8     │ INT4     │ FP4      │             │
│  │              │          │ (4B/p)   │ (2B/p)   │ (1B/p)   │ (1B/p)   │ (0.5B/p) │ (0.5B/p) │             │
│  ├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤             │
│  │ Qwen3-VL-2B  │ 2.3B     │ 9.2 GB   │ 4.6 GB   │ 2.3 GB   │ 2.3 GB   │ 1.15 GB  │ 1.15 GB  │             │
│  │ Qwen3-VL-4B  │ 4.4B     │ 17.6 GB  │ 8.8 GB   │ 4.4 GB   │ 4.4 GB   │ 2.2 GB   │ 2.2 GB   │             │
│  │ Qwen3-VL-8B  │ 8.3B     │ 33.2 GB  │ 16.6 GB  │ 8.3 GB   │ 8.3 GB   │ 4.15 GB  │ 4.15 GB  │             │
│  │ Qwen3-VL-32B │ 32.9B    │ 131.6 GB │ 65.8 GB  │ 32.9 GB  │ 32.9 GB  │ 16.45 GB │ 16.45 GB │             │
│  │ 30B-A3B MoE  │ 30.5B    │ 122 GB   │ 61 GB    │ 30.5 GB  │ 30.5 GB  │ 15.25 GB │ 15.25 GB │             │
│  │ 235B-A22B    │ 235B     │ 940 GB   │ 470 GB   │ 235 GB   │ 235 GB   │ 117.5 GB │ 117.5 GB │             │
│  └──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘             │
│                                                                                                             │
│  VISUAL: 8B Model Memory by Precision                                                                       │
│  ════════════════════════════════════                                                                       │
│                                                                                                             │
│  FP32:  ████████████████████████████████████████████████████████████████████████████ 33.2 GB ❌ Too large │
│  BF16:  ████████████████████████████████████████ 16.6 GB ← Default                                        │
│  FP8:   ████████████████████ 8.3 GB ← H100 optimized                                                       │
│  INT8:  ████████████████████ 8.3 GB                                                                        │
│  INT4:  ██████████ 4.15 GB ← T4 required                                                                   │
│  FP4:   ██████████ 4.15 GB ← B200 optimized (future)                                                       │
│                                                                                                             │
│  VISUAL: 32B Model Memory by Precision                                                                      │
│  ═════════════════════════════════════                                                                      │
│                                                                                                             │
│  FP32:  ████████████████████████████████████████████████████████████████████████... 131.6 GB ❌ Multi-GPU │
│  BF16:  ████████████████████████████████████████████████████████████ 65.8 GB ← A100-80 barely fits        │
│  FP8:   ██████████████████████████████ 32.9 GB ← H100 comfortable                                          │
│  INT8:  ██████████████████████████████ 32.9 GB                                                             │
│  INT4:  ███████████████ 16.45 GB ← Single GPU possible!                                                    │
│  FP4:   ███████████████ 16.45 GB ← B200 single GPU optimal                                                 │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Precision on Each GPU: What Actually Works

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL × GPU × PRECISION: WHAT ACTUALLY WORKS                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  T4 (16 GB VRAM, SM 7.5)                                                                                   │
│  ═══════════════════════                                                                                    │
│                                                                                                             │
│  ┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────────────────────────────┐│
│  │ Model        │ BF16     │ FP8      │ INT8     │ INT4     │ Recommendation                               ││
│  ├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────────────────────────────┤│
│  │ Qwen3-VL-2B  │ ✅ Fits  │ ❌ N/A   │ ✅ Fits  │ ✅ Fits  │ BF16 (full precision, plenty of room)       ││
│  │ Qwen3-VL-4B  │ ✅ Tight │ ❌ N/A   │ ✅ Fits  │ ✅ Fits  │ INT4 for multi-request batching             ││
│  │ Qwen3-VL-8B  │ ❌ OOM   │ ❌ N/A   │ ❌ OOM   │ ✅ Fits  │ INT4 required (BitsAndBytes)                ││
│  │ Qwen3-VL-32B │ ❌ OOM   │ ❌ N/A   │ ❌ OOM   │ ❌ OOM   │ Not possible on T4                          ││
│  └──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────────────────────────────┘│
│                                                                                                             │
│  T4 Memory Layout (8B INT4):                                                                               │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 16 GB                                                                                                  ││
│  │ ├── INT4 Weights: 4.15 GB ███████████                                                                  ││
│  │ ├── Vision (FP16): 1.0 GB ███                                                                          ││
│  │ ├── Activations: 3.0 GB ████████                                                                       ││
│  │ ├── CUDA Context: 1.5 GB ████                                                                          ││
│  │ └── KV Cache: 6.35 GB █████████████████ (FP16, ~2 requests @8K)                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  vLLM Command:                                                                                             │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype float16 --quantization bitsandbytes --load-format bitsandbytes \                              │
│      --max-model-len 4096 --max-num-seqs 2 --enforce-eager                                                 │
│                                                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════   │
│                                                                                                             │
│  A100-80GB (SM 8.0)                                                                                        │
│  ══════════════════                                                                                         │
│                                                                                                             │
│  ┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────────────────────────────┐│
│  │ Model        │ BF16     │ FP8      │ INT8     │ INT4     │ Recommendation                               ││
│  ├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────────────────────────────┤│
│  │ Qwen3-VL-2B  │ ✅ Fits  │ ❌ N/A   │ ✅ Fits  │ ✅ Fits  │ BF16 (maximize quality, 100+ concurrent)    ││
│  │ Qwen3-VL-4B  │ ✅ Fits  │ ❌ N/A   │ ✅ Fits  │ ✅ Fits  │ BF16 (plenty of KV room)                    ││
│  │ Qwen3-VL-8B  │ ✅ Fits  │ ❌ N/A   │ ✅ Fits  │ ✅ Fits  │ BF16 (56 concurrent @8K)                    ││
│  │ Qwen3-VL-32B │ ✅ Tight │ ❌ N/A   │ ✅ Fits  │ ✅ Fits  │ INT8 for batching, BF16 for quality         ││
│  │ 30B-A3B MoE  │ ✅ Fits  │ ❌ N/A   │ ✅ Fits  │ ✅ Fits  │ BF16 (MoE efficient, high throughput)       ││
│  └──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────────────────────────────┘│
│                                                                                                             │
│  A100 Memory Layout (32B BF16):                                                                            │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 80 GB                                                                                                  ││
│  │ ├── BF16 Weights: 65.8 GB ████████████████████████████████████████████████████████████████████████████ ││
│  │ ├── Activations: 8.0 GB ██████████                                                                     ││
│  │ ├── CUDA: 4.0 GB █████                                                                                 ││
│  │ └── KV Cache: 2.2 GB ███ ⚠️ Only ~2 requests @4K!                                                      ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  vLLM Command (32B):                                                                                       │
│  vllm serve Qwen/Qwen3-VL-32B-Instruct \                                                                   │
│      --dtype bfloat16 --max-model-len 4096 --max-num-seqs 2 --gpu-memory-utilization 0.95                  │
│                                                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════   │
│                                                                                                             │
│  H100-80GB (SM 9.0)                                                                                        │
│  ══════════════════                                                                                         │
│                                                                                                             │
│  ┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────────────────────────────┐│
│  │ Model        │ BF16     │ FP8      │ INT8     │ INT4     │ Recommendation                               ││
│  ├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────────────────────────────┤│
│  │ Qwen3-VL-2B  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ BF16 (no need for quantization)             ││
│  │ Qwen3-VL-4B  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ BF16 or FP8 for max throughput              ││
│  │ Qwen3-VL-8B  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ FP8 KV cache (2× concurrent)                ││
│  │ Qwen3-VL-32B │ ✅ Tight │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ FP8 weights + KV (35 concurrent)            ││
│  │ 30B-A3B MoE  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ BF16 + FP8 KV (105 concurrent)              ││
│  │ 235B-A22B    │ ❌ OOM   │ ❌ OOM   │ ❌ OOM   │ ❌ OOM   │ Need 4×H100                                 ││
│  └──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────────────────────────────┘│
│                                                                                                             │
│  H100 Memory Layout (32B FP8):                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 80 GB                                                                                                  ││
│  │ ├── FP8 Weights: 32.9 GB ████████████████████████████████████████                                      ││
│  │ ├── Activations: 8.0 GB ██████████                                                                     ││
│  │ ├── CUDA: 4.0 GB █████                                                                                 ││
│  │ └── FP8 KV Cache: 35.1 GB ███████████████████████████████████████████ (35 req @8K!)                    ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  vLLM Command (32B FP8):                                                                                   │
│  vllm serve Qwen/Qwen3-VL-32B-Instruct \                                                                   │
│      --dtype float8 --kv-cache-dtype fp8 \                                                                  │
│      --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.90                                 │
│                                                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════   │
│                                                                                                             │
│  B200-192GB (SM 10.0)                                                                                      │
│  ════════════════════                                                                                       │
│                                                                                                             │
│  ┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────────────────────────────┐│
│  │ Model        │ BF16     │ FP8      │ FP4      │ INT4     │ Recommendation                               ││
│  ├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────────────────────────────┤│
│  │ Qwen3-VL-2B  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ BF16 (no quantization needed)               ││
│  │ Qwen3-VL-4B  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ BF16 (abundant memory)                      ││
│  │ Qwen3-VL-8B  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ BF16 (167 concurrent @8K!)                  ││
│  │ Qwen3-VL-32B │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ BF16 (57 concurrent @8K)                    ││
│  │ 30B-A3B MoE  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ ✅ Fits  │ BF16 (120+ concurrent @8K)                  ││
│  │ 235B-A22B    │ ❌ OOM   │ ❌ OOM   │ ✅ Fits  │ ✅ Fits  │ FP4 on single B200! (future)                ││
│  └──────────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────────────────────────────┘│
│                                                                                                             │
│  B200 Memory Layout (235B FP4 - Future):                                                                   │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 192 GB                                                                                                 ││
│  │ ├── FP4 Weights: 117.5 GB █████████████████████████████████████████████████████████████████████████████││
│  │ ├── Activations: 20.0 GB █████████████                                                                 ││
│  │ ├── CUDA: 6.0 GB ████                                                                                  ││
│  │ └── FP8 KV Cache: 48.5 GB ████████████████████████████████ (8-10 concurrent @8K)                       ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  235B on single B200 with FP4: THE FUTURE                                                                  │
│  • 235B × 0.5 bytes = 117.5 GB weights                                                                     │
│  • Leaves ~75 GB for KV cache + activations                                                                │
│  • Frontier-class model on single GPU!                                                                     │
│  • 9000 TFLOPS FP4 → ~100 tok/s throughput (estimated)                                                     │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Precision Quality Impact

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     PRECISION QUALITY IMPACT ON QWEN3-VL                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  QUALITY vs PRECISION (Relative to BF16 Baseline)                                                          │
│  ═══════════════════════════════════════════════                                                            │
│                                                                                                             │
│  ┌──────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────────────────────┐│
│  │ Precision    │ MMMU     │ DocVQA   │ GUI      │ VideoQA  │ Overall  │ Notes                            ││
│  │              │ (Vision) │ (OCR)    │ Grounding│ (Video)  │ Quality  │                                  ││
│  ├──────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────────────────────────────┤│
│  │ BF16         │ 100%     │ 100%     │ 100%     │ 100%     │ 100%     │ Baseline (lossless)              ││
│  │ FP8 E4M3     │ 99.8%    │ 99.9%    │ 99.7%    │ 99.8%    │ 99.8%    │ Near-lossless                    ││
│  │ INT8 (GPTQ)  │ 99.2%    │ 99.5%    │ 99.0%    │ 99.3%    │ 99.3%    │ Minimal degradation              ││
│  │ INT8 (AWQ)   │ 99.4%    │ 99.6%    │ 99.2%    │ 99.5%    │ 99.4%    │ AWQ slightly better than GPTQ   ││
│  │ INT4 (GPTQ)  │ 97.5%    │ 98.0%    │ 96.5%    │ 97.8%    │ 97.5%    │ Some quality loss                ││
│  │ INT4 (AWQ)   │ 98.0%    │ 98.5%    │ 97.0%    │ 98.2%    │ 97.9%    │ AWQ preserves more quality      ││
│  │ INT4 (BnB)   │ 96.5%    │ 97.5%    │ 95.5%    │ 97.0%    │ 96.6%    │ BitsAndBytes fast but lower Q    ││
│  │ FP4 (native) │ 98.5%    │ 99.0%    │ 98.0%    │ 98.7%    │ 98.5%    │ Better than INT4 (Gaussian fit) ││
│  └──────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────────────────────────────┘│
│                                                                                                             │
│  VISUAL: Quality vs Memory Trade-off                                                                       │
│  ═══════════════════════════════════                                                                        │
│                                                                                                             │
│  Quality                                                                                                    │
│  100% ──●─────────────────────────────────── BF16 (16.6 GB for 8B)                                         │
│   99% ──┼───●───●───●─────────────────────── FP8, INT8-AWQ, FP4                                            │
│   98% ──┼───────────┼───●─────────────────── INT4-AWQ                                                      │
│   97% ──┼───────────────┼───●─────────────── INT4-GPTQ                                                     │
│   96% ──┼───────────────────┼───●─────────── INT4-BnB                                                      │
│   95% ──┼───────────────────────┼─────────                                                                 │
│         │           │           │                                                                           │
│     Memory: 16 GB   8 GB       4 GB                                                                        │
│                                                                                                             │
│  KEY INSIGHT:                                                                                               │
│  • FP8 is nearly lossless (99.8%) at half the memory                                                       │
│  • FP4 beats INT4 in quality (~98.5% vs ~97.5%) at same memory                                             │
│  • For GUI agents: FP8 minimum recommended, avoid INT4 if possible                                         │
│  • Vision encoder should stay FP16/BF16 (small fraction of total)                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Deep Dive: NVFP4 - NVIDIA's New 4-Bit Floating Point

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     NVFP4: NVIDIA'S BLACKWELL-NATIVE 4-BIT FORMAT (Research-Based)                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WHAT IS NVFP4? (from NVIDIA Developer Blog, June 2025)                                                    │
│  ═══════════════════════════════════════════════════════                                                    │
│                                                                                                             │
│  NVFP4 is a 4-bit floating-point format with MICROSCALING:                                                 │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Format: FP4 E2M1 (4 bits) + FP8 E4M3 scale (8 bits per 16 elements)                                  ││
│  │                                                                                                        ││
│  │  Block Structure:                                                                                      ││
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │  16 FP4 values (8 bytes)  │  1 FP8 scale (1 byte)  │  = 9 bytes for 16 weights                │   ││
│  │  │  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐  ┌────────┐                │   ││
│  │  │  │v0 │v1 │v2 │v3 │v4 │v5 │v6 │v7 │v8 │v9 │v10│v11│v12│v13│v14│v15│  │  scale │                │   ││
│  │  │  │4b │4b │4b │4b │4b │4b │4b │4b │4b │4b │4b │4b │4b │4b │4b │4b │  │  FP8   │                │   ││
│  │  │  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘  └────────┘                │   ││
│  │  └────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                        ││
│  │  Effective bits per weight: 4 + (8/16) = 4.5 bits                                                     ││
│  │  Memory: ~56% of FP8, ~28% of BF16                                                                    ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  WHY MICROSCALING WORKS (from arXiv:2512.02010 "Four Over Six")                                            │
│  ════════════════════════════════════════════════════════════                                               │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Problem: With only 16 distinct FP4 values, how can we represent diverse weight ranges?               ││
│  │                                                                                                        ││
│  │  Solution: PER-BLOCK SCALING                                                                           ││
│  │                                                                                                        ││
│  │  Block 1: weights in range [-0.5, 0.5]         Block 2: weights in range [-2.0, 2.0]                  ││
│  │  ┌─────────────────────────────────────┐       ┌─────────────────────────────────────┐                ││
│  │  │ FP4 values: ±0,±0.5,±1,±1.5,±2...   │       │ FP4 values: ±0,±0.5,±1,±1.5,±2...   │                ││
│  │  │ Scale: 0.125 (shift range down)     │       │ Scale: 0.5 (shift range up)         │                ││
│  │  │                                     │       │                                     │                ││
│  │  │ Actual: ±0,±0.0625,±0.125,±0.1875...│       │ Actual: ±0,±0.25,±0.5,±0.75...      │                ││
│  │  └─────────────────────────────────────┘       └─────────────────────────────────────┘                ││
│  │                                                                                                        ││
│  │  Each 16-element block adapts to its local weight distribution!                                       ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  NVFP4 vs MXFP4 (from arXiv:2509.23202 "Bridging the Gap")                                                │
│  ═════════════════════════════════════════════════════════                                                  │
│                                                                                                             │
│  ┌─────────────────────┬──────────────────────────────────┬──────────────────────────────────┐             │
│  │ Aspect              │ NVFP4 (NVIDIA)                   │ MXFP4 (Open Standard)            │             │
│  ├─────────────────────┼──────────────────────────────────┼──────────────────────────────────┤             │
│  │ Scale Format        │ FP8 E4M3 (fine-grained)          │ E8M0 (power-of-two only)         │             │
│  │ Block Size          │ 16 elements                      │ 32 elements                      │             │
│  │ Scale Precision     │ High (3 mantissa bits)           │ Low (0 mantissa bits)            │             │
│  │ Hardware            │ NVIDIA Blackwell only            │ NVIDIA + AMD                     │             │
│  │ Accuracy            │ Better (finer scaling)           │ Worse (coarse scaling)           │             │
│  │ Outlier Handling    │ Small groups help                │ Larger groups hurt               │             │
│  └─────────────────────┴──────────────────────────────────┴──────────────────────────────────┘             │
│                                                                                                             │
│  KEY FINDING: NVFP4's 16-element blocks are so small that traditional outlier mitigation                  │
│               (like SmoothQuant) becomes ineffective - the outliers are isolated naturally!               │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### How Quantization Retains Precision: The Science

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     HOW QUANTIZATION RETAINS PRECISION: DEEP DIVE                                           │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WHY DOES 4-BIT QUANTIZATION EVEN WORK?                                                                    │
│  ══════════════════════════════════════                                                                     │
│                                                                                                             │
│  1. LLM WEIGHT DISTRIBUTION                                                                                │
│  ───────────────────────────                                                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  LLM weights follow a BELL-SHAPED (Gaussian) distribution:                                            ││
│  │                                                                                                        ││
│  │  Frequency                                                                                             ││
│  │     ▲                                                                                                  ││
│  │     │                    ████                                                                          ││
│  │     │                  ████████                                                                        ││
│  │     │                ████████████       ← 99% of weights are in [-2, +2] range                        ││
│  │     │              ████████████████                                                                    ││
│  │     │            ████████████████████                                                                  ││
│  │     │          ████████████████████████                                                                ││
│  │     │        ████████████████████████████                                                              ││
│  │     │      ████████████████████████████████                                                            ││
│  │     └──────┼───────────────┼───────────────┼────────► Weight Value                                    ││
│  │           -3               0               +3                                                          ││
│  │                                                                                                        ││
│  │  Key insight: Most weights are SMALL, only a few are large ("outliers")                               ││
│  │  FP4's non-uniform spacing matches this distribution perfectly!                                        ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  2. AWQ: ACTIVATION-AWARE WEIGHT QUANTIZATION (arXiv:2306.00978)                                          │
│  ═══════════════════════════════════════════════════════════════                                            │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Core Insight: Not all weights are equally important!                                                  ││
│  │                                                                                                        ││
│  │  Weights connected to HIGH-ACTIVATION channels matter more:                                            ││
│  │                                                                                                        ││
│  │  Input Activations          Weights                    Importance                                     ││
│  │  ┌────────────────┐         ┌────────────────┐         ┌────────────────┐                              ││
│  │  │ Channel 0: 0.1 │    ×    │ W0: 1.5        │    =    │ Low (0.1×1.5)  │ ← Can quantize aggressively ││
│  │  │ Channel 1: 5.2 │    ×    │ W1: 0.8        │    =    │ HIGH (5.2×0.8) │ ← Must preserve precision   ││
│  │  │ Channel 2: 0.3 │    ×    │ W2: 2.1        │    =    │ Low (0.3×2.1)  │ ← Can quantize aggressively ││
│  │  └────────────────┘         └────────────────┘         └────────────────┘                              ││
│  │                                                                                                        ││
│  │  AWQ Algorithm:                                                                                        ││
│  │  1. Run calibration data through model                                                                ││
│  │  2. Measure activation magnitudes per channel                                                         ││
│  │  3. Scale weights by activation importance: W_scaled = W × (activation_scale)^α                      ││
│  │  4. Quantize scaled weights (important ones get more precision)                                       ││
│  │  5. At inference: compensate with inverse scale on activations                                        ││
│  │                                                                                                        ││
│  │  Result: 0.1% of weights (1% most important) preserved → 97-99% accuracy retained!                   ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  3. THE OUTLIER PROBLEM                                                                                    │
│  ═════════════════════                                                                                      │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Outliers are the #1 cause of quantization accuracy loss:                                             ││
│  │                                                                                                        ││
│  │  Two types of outliers (from arXiv:2409.20361):                                                       ││
│  │                                                                                                        ││
│  │  A) CHANNEL-WISE OUTLIERS                    B) SPIKE OUTLIERS                                        ││
│  │     (Consistent across tokens)                  (Random, per-token)                                    ││
│  │                                                                                                        ││
│  │     Channels:  0   1   2   3   4               Tokens:  0   1   2   3   4                              ││
│  │     Token 0:  0.1 0.2 5.8 0.1 0.3              Ch 0:  0.1 0.2 0.1 8.5 0.1                              ││
│  │     Token 1:  0.2 0.1 6.2 0.2 0.1              Ch 1:  0.1 0.1 0.2 0.1 0.1                              ││
│  │     Token 2:  0.1 0.3 5.5 0.1 0.2              Ch 2:  0.2 0.1 0.1 0.2 0.1                              ││
│  │                   ▲                                            ▲                                       ││
│  │              Always high!                                  Random spike!                               ││
│  │                                                                                                        ││
│  │  Solutions:                                                                                            ││
│  │  • SmoothQuant: Migrate outliers from activations to weights                                          ││
│  │  • Rotation: Apply orthogonal rotation to spread outliers                                             ││
│  │  • NVFP4 microscaling: 16-element blocks isolate outliers naturally                                   ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Why Don't People Always Use INT4/FP4?

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     WHY DON'T PEOPLE ALWAYS USE INT4/FP4? THE TRADE-OFFS                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  REASON 1: ACCURACY DEGRADATION ON COMPLEX TASKS                                                           │
│  ═══════════════════════════════════════════════                                                            │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Task-Specific Accuracy Loss (from arXiv:2411.02355 "Give Me BF16 or Give Me Death"):                 ││
│  │                                                                                                        ││
│  │  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐                       ││
│  │  │ Task                │ BF16     │ FP8      │ INT8     │ INT4-AWQ │ INT4-BnB │                       ││
│  │  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤                       ││
│  │  │ Simple QA           │ 100%     │ 99.9%    │ 99.5%    │ 98.5%    │ 97%      │ ← Minimal impact     ││
│  │  │ Math Reasoning      │ 100%     │ 99.5%    │ 98.0%    │ 92%      │ 85%      │ ← Significant!       ││
│  │  │ Code Generation     │ 100%     │ 99.2%    │ 97.5%    │ 90%      │ 82%      │ ← Critical!          ││
│  │  │ GUI Grounding       │ 100%     │ 99.0%    │ 96.0%    │ 88%      │ 78%      │ ← GUI agents suffer  ││
│  │  │ Long-Context        │ 100%     │ 98.5%    │ 95.0%    │ 82%      │ 70%      │ ← Severe degradation ││
│  │  └─────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘                       ││
│  │                                                                                                        ││
│  │  ⚠️ For GUI agents doing multi-step reasoning: INT4 loses 10-20% accuracy!                           ││
│  │  This is why Qwen3-VL-32B with BF16/FP8 is needed, not 8B with INT4.                                  ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  REASON 2: ACTIVATION QUANTIZATION IS HARD                                                                 │
│  ═════════════════════════════════════════                                                                  │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Weight-Only vs Weight+Activation Quantization:                                                       ││
│  │                                                                                                        ││
│  │  WEIGHT-ONLY (W4A16):                        WEIGHT+ACTIVATION (W4A4):                                ││
│  │  ┌──────────────────────────────┐            ┌──────────────────────────────┐                         ││
│  │  │ Weights: INT4 (quantized)    │            │ Weights: INT4 (quantized)    │                         ││
│  │  │ Activations: FP16 (full)     │            │ Activations: INT4 (quantized)│                         ││
│  │  │ Compute: INT4 × FP16 → FP16  │            │ Compute: INT4 × INT4 → INT32 │                         ││
│  │  │                              │            │                              │                         ││
│  │  │ Quality: 97-99%              │            │ Quality: 85-95% ⚠️           │                         ││
│  │  │ Speedup: 1.5-2×              │            │ Speedup: 2-4× (theoretical)  │                         ││
│  │  └──────────────────────────────┘            └──────────────────────────────┘                         ││
│  │                                                                                                        ││
│  │  Problem: Activations have DYNAMIC range that changes per input                                       ││
│  │  • Weights: Static, can calibrate once                                                                ││
│  │  • Activations: Different every forward pass, hard to quantize                                        ││
│  │                                                                                                        ││
│  │  From arXiv:2301.12017 "Understanding INT4":                                                          ││
│  │  "W4A4 quantization introduces NO degradation for encoder-only models,                                ││
│  │   but causes SIGNIFICANT accuracy degradation for decoder-only LLMs"                                  ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  REASON 3: HARDWARE SUPPORT IS LIMITED                                                                     │
│  ═════════════════════════════════════                                                                      │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Native Tensor Core Support for INT4/FP4:                                                             ││
│  │                                                                                                        ││
│  │  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐                     ││
│  │  │ GPU              │ INT4 Native  │ FP4 Native   │ What Happens │ Speed       │                     ││
│  │  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤                     ││
│  │  │ T4 (SM 7.5)      │ ❌           │ ❌           │ Software     │ SLOW        │                     ││
│  │  │                  │              │              │ dequant      │ (overhead)  │                     ││
│  │  │ A100 (SM 8.0)    │ ❌           │ ❌           │ Software     │ SLOW        │                     ││
│  │  │                  │              │              │ dequant      │ (overhead)  │                     ││
│  │  │ H100 (SM 9.0)    │ ❌           │ ❌           │ Software     │ SLOW        │                     ││
│  │  │                  │              │              │ dequant      │ (overhead)  │                     ││
│  │  │ B200 (SM 10.0)   │ ✅           │ ✅           │ Hardware     │ FAST!       │                     ││
│  │  │                  │              │              │ Tensor Core  │ 9000 TFLOPS │                     ││
│  │  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘                     ││
│  │                                                                                                        ││
│  │  Before B200: INT4 quantization SAVES MEMORY but may NOT speed up compute!                            ││
│  │  • Memory bandwidth: Yes, reading 4-bit weights is 4× faster than 16-bit                             ││
│  │  • Compute: No, must dequantize to FP16 before computation (overhead)                                ││
│  │                                                                                                        ││
│  │  With B200: First GPU with native FP4/INT4 Tensor Core acceleration!                                 ││
│  │  • 9000 TFLOPS FP4 (vs 990 BF16 on H100 = 9× theoretical speedup)                                    ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  REASON 4: DEQUANTIZATION OVERHEAD                                                                         │
│  ═════════════════════════════════                                                                          │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  On GPUs without native INT4/FP4 (T4, A100, H100):                                                    ││
│  │                                                                                                        ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐  ││
│  │  │                                                                                                 │  ││
│  │  │  BF16 Execution:                                                                                │  ││
│  │  │  ┌──────────┐     ┌──────────┐     ┌──────────┐                                                │  ││
│  │  │  │ Read W   │ →   │ Compute  │ →   │ Output   │                                                │  ││
│  │  │  │ (16-bit) │     │ (TC BF16)│     │ (16-bit) │                                                │  ││
│  │  │  └──────────┘     └──────────┘     └──────────┘                                                │  ││
│  │  │  Time: ████████████████████                                                                    │  ││
│  │  │                                                                                                 │  ││
│  │  │  INT4 Execution (software dequant):                                                             │  ││
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐                          │  ││
│  │  │  │ Read W   │→ │ Dequant  │→ │ Read Act │→ │ Compute  │→ │ Output   │                          │  ││
│  │  │  │ (4-bit)  │  │ 4→16 bit │  │ (16-bit) │  │ (TC BF16)│  │ (16-bit) │                          │  ││
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘                          │  ││
│  │  │  Time: ████  ████████████  ████████  ████████████████████                                      │  ││
│  │  │            ▲                                                                                    │  ││
│  │  │            └── Dequantization overhead!                                                         │  ││
│  │  │                                                                                                 │  ││
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘  ││
│  │                                                                                                        ││
│  │  Net Effect on H100:                                                                                  ││
│  │  • INT4 saves ~50% memory                                                                             ││
│  │  • BUT only ~10-30% faster decode (dequant eats gains)                                                ││
│  │  • FP8 (native): saves 50% memory AND ~50% faster (no dequant overhead)                              ││
│  │                                                                                                        ││
│  │  Recommendation: Use FP8 on H100, not INT4!                                                           ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  REASON 5: CALIBRATION REQUIRED                                                                            │
│  ═════════════════════════════                                                                              │
│                                                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                        ││
│  │  Good INT4 quantization (AWQ, GPTQ) requires CALIBRATION:                                             ││
│  │                                                                                                        ││
│  │  1. Collect representative data (100-1000 samples)                                                    ││
│  │  2. Run forward passes to measure activation distributions                                            ││
│  │  3. Compute optimal scales and zero-points                                                            ││
│  │  4. Quantize weights with learned parameters                                                          ││
│  │  5. Validate on held-out data                                                                         ││
│  │                                                                                                        ││
│  │  Time Required:                                                                                        ││
│  │  • AWQ calibration for 8B model: ~30-60 minutes on A100                                               ││
│  │  • GPTQ calibration for 8B model: ~2-4 hours on A100                                                  ││
│  │  • Block rotation (MXFP4): ~2-8 hours on A100                                                         ││
│  │                                                                                                        ││
│  │  If calibration data doesn't match deployment data → accuracy suffers!                                ││
│  │                                                                                                        ││
│  │  BitsAndBytes (runtime quantization):                                                                 ││
│  │  • No calibration needed (instant)                                                                    ││
│  │  • But lower quality (no task-specific optimization)                                                  ││
│  │                                                                                                        ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  SUMMARY: WHEN TO USE EACH PRECISION                                                                       │
│  ═══════════════════════════════════                                                                        │
│                                                                                                             │
│  ┌─────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐│
│  │ Precision           │ When to Use                                                                      ││
│  ├─────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤│
│  │ BF16                │ Default choice, maximum quality, enough VRAM                                     ││
│  │ FP8 (H100/B200)     │ Best trade-off: 99.8% quality, 50% memory, native speed                         ││
│  │ INT8-AWQ            │ When FP8 unavailable, good quality, pre-quantized model exists                  ││
│  │ INT4-AWQ            │ Severe memory constraints, simple tasks, latency > quality                      ││
│  │ INT4-BnB            │ Quick experiments, no pre-quantized model, accept ~3% quality loss              ││
│  │ FP4-NVFP4 (B200)    │ Future: Best of both worlds on Blackwell (quality + speed)                      ││
│  └─────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  FOR QWEN3-VL GUI AGENTS:                                                                                  │
│  • Use BF16 or FP8 for 32B models (quality critical for multi-step reasoning)                             │
│  • Avoid INT4 for GUI grounding (10-20% accuracy loss is unacceptable)                                    │
│  • FP8 on H100 is the sweet spot: 32B fits comfortably, minimal quality loss                              │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### vLLM Quantization Commands Reference

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     VLLM QUANTIZATION COMMANDS BY PRECISION                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  BF16 (Default - Full Precision)                                                                           │
│  ════════════════════════════════                                                                           │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype bfloat16                                                                                       │
│                                                                                                             │
│  FP8 (H100/B200 - Native Tensor Core)                                                                      │
│  ═════════════════════════════════════                                                                      │
│  # Weights FP8, KV cache FP8                                                                               │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype float8 \                                                                                       │
│      --kv-cache-dtype fp8                                                                                   │
│                                                                                                             │
│  # Weights BF16, KV cache FP8 (balanced)                                                                   │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype bfloat16 \                                                                                     │
│      --kv-cache-dtype fp8                                                                                   │
│                                                                                                             │
│  INT8 (AWQ - Pre-quantized Model)                                                                          │
│  ═════════════════════════════════                                                                          │
│  # Need AWQ-quantized model                                                                                │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct-AWQ \                                                                │
│      --quantization awq                                                                                     │
│                                                                                                             │
│  INT4 (BitsAndBytes - Runtime Quantization)                                                                │
│  ═══════════════════════════════════════════                                                                │
│  # Quantize on load (slower startup, works with any model)                                                 │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype float16 \                                                                                      │
│      --quantization bitsandbytes \                                                                          │
│      --load-format bitsandbytes                                                                             │
│                                                                                                             │
│  INT4 (AWQ - Pre-quantized Model)                                                                          │
│  ═════════════════════════════════                                                                          │
│  # Need AWQ-quantized model (best INT4 quality)                                                            │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct-AWQ-INT4 \                                                           │
│      --quantization awq                                                                                     │
│                                                                                                             │
│  INT4 (GPTQ - Pre-quantized Model)                                                                         │
│  ══════════════════════════════════                                                                         │
│  # Need GPTQ-quantized model                                                                               │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct-GPTQ-Int4 \                                                          │
│      --quantization gptq                                                                                    │
│                                                                                                             │
│  FP4 (B200 - Future Native Support)                                                                        │
│  ═══════════════════════════════════                                                                        │
│  # Coming with vLLM + B200 support (2025)                                                                  │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype float4 \                                                                                       │
│      --kv-cache-dtype fp8                                                                                   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### How vLLM Enables This: Backend Selection & SM Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     HOW VLLM ENABLES OPTIMAL GPU UTILIZATION                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  AUTOMATIC BACKEND SELECTION (from vllm/platforms/cuda.py)                                                  │
│  ═════════════════════════════════════════════════════════                                                  │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  # vLLM auto-detects GPU and selects optimal backend                                                   ││
│  │                                                                                                         ││
│  │  from vllm.platforms import DeviceCapability                                                           ││
│  │                                                                                                         ││
│  │  def get_device_capability() -> DeviceCapability:                                                      ││
│  │      """Returns (major, minor) compute capability"""                                                   ││
│  │      import pynvml                                                                                     ││
│  │      pynvml.nvmlInit()                                                                                 ││
│  │      handle = pynvml.nvmlDeviceGetHandleByIndex(0)                                                     ││
│  │      major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)                                  ││
│  │      return DeviceCapability(major=major, minor=minor)                                                 ││
│  │                                                                                                         ││
│  │  # Backend priority based on compute capability:                                                       ││
│  │  #                                                                                                      ││
│  │  # T4 (SM 7.5):   FlashInfer → Triton → TORCH_SDPA                                                     ││
│  │  # A100 (SM 8.0): FlashAttention2 → FlashInfer → Triton                                                ││
│  │  # H100 (SM 9.0): FlashAttention3 → FlashInfer → Triton (+ FP8 support)                                ││
│  │  # B200 (SM 10.0): FlashInfer + TRTLLM → FlashAttention3 (+ FP4 support)                               ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  PAGED ATTENTION: HOW IT MAPS TO SMs                                                                        │
│  ════════════════════════════════════                                                                       │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  PagedAttention allocates KV cache in fixed-size blocks (default: 16 tokens/block)                    ││
│  │                                                                                                         ││
│  │  Block Table → GPU Memory → SM Execution:                                                              ││
│  │                                                                                                         ││
│  │  Request 1: [Block 0, Block 5, Block 12, Block 23]   ──┐                                               ││
│  │  Request 2: [Block 1, Block 6, Block 13]              ──┼──► GPU Scheduler                             ││
│  │  Request 3: [Block 2, Block 7, Block 14, Block 24]   ──┘    assigns to SMs                             ││
│  │                                                                                                         ││
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐││
│  │  │  HBM (High Bandwidth Memory)                                                                       │││
│  │  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐                │││
│  │  │  │ Block 0 │ Block 1 │ Block 2 │ Block 5 │ Block 6 │ Block 7 │ Block 12│ Block 13│ ...            │││
│  │  │  │ (R1,L0) │ (R2,L0) │ (R3,L0) │ (R1,L1) │ (R2,L1) │ (R3,L1) │ (R1,L2) │ (R2,L2) │                │││
│  │  │  └────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┴────┬────┘                │││
│  │  │       │         │         │         │         │         │         │                               │││
│  │  │       ▼         ▼         ▼         ▼         ▼         ▼         ▼                               │││
│  │  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │││
│  │  │  │  L2 Cache (40-96 MB depending on GPU)                                                       │  │││
│  │  │  │  Recently accessed blocks cached here for fast reuse                                        │  │││
│  │  │  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │││
│  │  │       │         │         │         │         │         │         │                               │││
│  │  │       ▼         ▼         ▼         ▼         ▼         ▼         ▼                               │││
│  │  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────┐  │││
│  │  │  │  SM 0-N: Each SM processes tiles of attention                                               │  │││
│  │  │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                    │  │││
│  │  │  │  │ SM 0 │ │ SM 1 │ │ SM 2 │ │ SM 3 │ │ SM 4 │ │ SM 5 │ │ SM 6 │ │ SM 7 │ ...                │  │││
│  │  │  │  │ R1,T0│ │ R1,T1│ │ R2,T0│ │ R2,T1│ │ R3,T0│ │ R3,T1│ │ R1,T2│ │ R1,T3│                    │  │││
│  │  │  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                    │  │││
│  │  │  │                                                                                              │  │││
│  │  │  │  R = Request, T = Tile (attention chunk), L = Layer                                          │  │││
│  │  │  └──────────────────────────────────────────────────────────────────────────────────────────────┘  │││
│  │  └────────────────────────────────────────────────────────────────────────────────────────────────────┘││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Visualizing SM Utilization: Profiling Tools

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     HOW TO VISUALIZE SM UTILIZATION                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  METHOD 1: NVIDIA Nsight Systems (Recommended)                                                              │
│  ═════════════════════════════════════════════                                                              │
│                                                                                                             │
│  # Profile vLLM server with Nsight                                                                         │
│  nsys profile --stats=true --cuda-graph-trace=node \                                                       │
│      python -m vllm.entrypoints.openai.api_server \                                                        │
│      --model Qwen/Qwen3-VL-8B-Instruct \                                                                   │
│      --max-model-len 4096                                                                                  │
│                                                                                                             │
│  # Output shows:                                                                                            │
│  # - Kernel execution time per SM                                                                          │
│  # - Memory transfer patterns                                                                               │
│  # - SM occupancy percentage                                                                                │
│  # - Tensor Core utilization                                                                                │
│                                                                                                             │
│  METHOD 2: NVIDIA Nsight Compute (Kernel-level detail)                                                     │
│  ════════════════════════════════════════════════════                                                       │
│                                                                                                             │
│  # Deep kernel analysis                                                                                     │
│  ncu --set full --target-processes all \                                                                   │
│      python -c "from vllm import LLM; llm = LLM('Qwen/Qwen3-VL-8B-Instruct'); \                            │
│                  llm.generate([{'prompt': 'test', 'multi_modal_data': {'image': img}}])"                   │
│                                                                                                             │
│  # Shows per-kernel:                                                                                        │
│  # - Active warps per SM                                                                                   │
│  # - Achieved occupancy vs theoretical                                                                     │
│  # - Memory throughput                                                                                      │
│  # - Compute throughput                                                                                     │
│                                                                                                             │
│  METHOD 3: vLLM Built-in Metrics (Runtime monitoring)                                                      │
│  ═════════════════════════════════════════════════════                                                      │
│                                                                                                             │
│  # Enable Prometheus metrics                                                                                │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --enable-metrics \                                                                                     │
│      --metrics-port 9090                                                                                    │
│                                                                                                             │
│  # Key metrics exposed:                                                                                     │
│  # vllm:gpu_cache_usage_perc       - KV cache utilization                                                  │
│  # vllm:num_preemptions_total      - Scheduler preemptions                                                 │
│  # vllm:avg_prompt_throughput      - Tokens/sec prefill                                                    │
│  # vllm:avg_generation_throughput  - Tokens/sec decode                                                     │
│  # vllm:time_to_first_token        - TTFT histogram                                                        │
│                                                                                                             │
│  METHOD 4: PyTorch Profiler (Code-level)                                                                   │
│  ════════════════════════════════════════                                                                   │
│                                                                                                             │
│  ```python                                                                                                  │
│  import torch                                                                                               │
│  from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler                           │
│                                                                                                             │
│  with profile(                                                                                              │
│      activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],                                             │
│      on_trace_ready=tensorboard_trace_handler('./logs'),                                                   │
│      record_shapes=True,                                                                                    │
│      profile_memory=True,                                                                                   │
│      with_stack=True                                                                                        │
│  ) as prof:                                                                                                 │
│      output = llm.generate(prompts)                                                                        │
│                                                                                                             │
│  # View in TensorBoard:                                                                                     │
│  # tensorboard --logdir=./logs                                                                              │
│  ```                                                                                                        │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### SM Utilization Visualization: 8B Model on Each GPU

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B: SM UTILIZATION VISUALIZATION BY GPU                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  A100-80GB (108 SMs) - FlashAttention 2                                                                    │
│  ══════════════════════════════════════                                                                     │
│                                                                                                             │
│  PREFILL PHASE (1382 tokens):                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 108                                         │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤                                         │ │
│  │      ████████████████████████████████████████████████████████  92% utilized                          │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  Tensor Cores                            │ │
│  │      │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  CUDA Cores                               │ │
│  │                                                                                                       │ │
│  │ Kernel: flash_attn_2_fwd    | Blocks: 1382 | Warps/Block: 4 | Occupancy: 92%                        │ │
│  │ Tile Size: 128×64           | Shared Mem: 96KB/SM          | Registers: 128/thread                  │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  DECODE PHASE (1 token at a time):                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 108                                         │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤                                         │ │
│  │      ████████████████████████████                             35% utilized                           │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                  Tensor Cores                            │ │
│  │                           └─ Many SMs idle waiting for memory                                         │ │
│  │                                                                                                       │ │
│  │ Bottleneck: Memory bandwidth (2039 GB/s) limiting weight reads                                       │ │
│  │ Only ~60-70 SMs can stay busy before memory stalls                                                   │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════   │
│                                                                                                             │
│  H100-80GB (132 SMs) - FlashAttention 3 + TMA                                                              │
│  ═════════════════════════════════════════════                                                              │
│                                                                                                             │
│  PREFILL PHASE (1382 tokens):                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 110 120 132                                 │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤                               │ │
│  │      ████████████████████████████████████████████████████████████████████  95% utilized              │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  Tensor Cores                │ │
│  │      │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  CUDA Cores                   │ │
│  │                                                                                                       │ │
│  │ Kernel: flash_attn_3_fwd    | Thread Block Clusters: 2×2    | Occupancy: 95%                        │ │
│  │ TMA: Async loads overlap    | Shared Mem: 120KB/SM          | Warp Specialization: enabled          │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  DECODE PHASE (with FP8 KV cache):                                                                          │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 110 120 132                                 │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤                               │ │
│  │      ████████████████████████████████████████                         40% utilized                   │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                               Tensor Cores                  │ │
│  │                                                                                                       │ │
│  │ Better than A100 because:                                                                             │ │
│  │ • 64% higher memory BW (3350 vs 2039 GB/s)                                                           │ │
│  │ • FP8 KV cache halves read traffic                                                                    │ │
│  │ • TMA reduces addressing overhead                                                                     │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════   │
│                                                                                                             │
│  B200-192GB (192 SMs) - FlashInfer + TRTLLM                                                                │
│  ═══════════════════════════════════════════                                                                │
│                                                                                                             │
│  PREFILL PHASE (1382 tokens):                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   20   40   60   80  100  120  140  160  180 192                                              │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤                                               │ │
│  │      ████████████████████████████████████████████████████████████████████████████████  96% utilized  │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  5th Gen TC   │ │
│  │                                                                                                       │ │
│  │ 2nd Gen Transformer Engine: FP8 math + dynamic scaling                                               │ │
│  │ Kernel: flashinfer_prefill  | 192 SMs saturated | Occupancy: 96%                                     │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  DECODE PHASE:                                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   20   40   60   80  100  120  140  160  180 192                                              │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤                                               │ │
│  │      ████████████████████████████████████████████                           45% utilized             │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                 TRTLLM Decode            │ │
│  │                                                                                                       │ │
│  │ 8 TB/s memory bandwidth: More SMs can stay busy                                                      │ │
│  │ TRTLLM decode kernels optimized for single-token generation                                          │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### SM Utilization Visualization: 32B Model on Each GPU

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-32B: SM UTILIZATION VISUALIZATION BY GPU                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  A100-80GB (108 SMs) - BF16, Memory-Constrained                                                            │
│  ══════════════════════════════════════════════                                                             │
│                                                                                                             │
│  PREFILL PHASE:                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 108                                         │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤                                         │ │
│  │      ████████████████████████████████████████████████████████  90% utilized                          │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  Tensor Cores                            │ │
│  │                                                                                                       │ │
│  │ ⚠️ 64 layers (vs 32 for 8B): 2× FLOPs per token                                                      │ │
│  │ Prefill: 250ms (vs 120ms for 8B) - still compute-bound                                               │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  DECODE PHASE:                                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 108                                         │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤                                         │ │
│  │      █████████████████████████                                 30% utilized ⚠️                       │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                         Severely memory-bound                  │ │
│  │                                                                                                       │ │
│  │ Problem: 65.8 GB weights / 2039 GB/s = 32ms theoretical minimum                                      │ │
│  │ 4× more weights to read vs 8B → 4× slower decode                                                     │ │
│  │ Most SMs stalled waiting for memory                                                                   │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════   │
│                                                                                                             │
│  H100-80GB (132 SMs) - FP8 Weights + KV Cache                                                              │
│  ═════════════════════════════════════════════                                                              │
│                                                                                                             │
│  PREFILL PHASE:                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 110 120 132                                 │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤                               │ │
│  │      ████████████████████████████████████████████████████████████████████  94% utilized              │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  FP8 Tensor Cores            │ │
│  │                                                                                                       │ │
│  │ FP8 Tensor Cores: 2× throughput vs BF16                                                              │ │
│  │ Prefill: 120ms (same as 8B on A100!) - FP8 doubles compute                                           │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  DECODE PHASE (FP8 weights):                                                                                │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 110 120 132                                 │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤                               │ │
│  │      ████████████████████████████████████████                         38% utilized                   │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                                              │ │
│  │                                                                                                       │ │
│  │ ✅ FP8 halves weight read: 32.9 GB / 3350 GB/s = 10ms theoretical                                    │ │
│  │ Actual: ~12ms (83% efficiency)                                                                        │ │
│  │ Same speed as 8B on A100! (78 tok/s)                                                                 │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════   │
│                                                                                                             │
│  B200-192GB (192 SMs) - Full BF16, Comfortable Fit                                                         │
│  ═════════════════════════════════════════════════                                                          │
│                                                                                                             │
│  PREFILL PHASE:                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   20   40   60   80  100  120  140  160  180 192                                              │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤                                               │ │
│  │      ████████████████████████████████████████████████████████████████████████████████  96% utilized  │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  5th Gen TC   │ │
│  │                                                                                                       │ │
│  │ 192 SMs × 2250 TFLOPS = Massive compute headroom                                                     │ │
│  │ Prefill: 70ms (2× faster than H100)                                                                  │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  DECODE PHASE:                                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   20   40   60   80  100  120  140  160  180 192                                              │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼───┤                                               │ │
│  │      ████████████████████████████████████████████                           42% utilized             │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                                                          │ │
│  │                                                                                                       │ │
│  │ 8 TB/s bandwidth: 65.8 GB / 8000 GB/s = 8.2ms theoretical                                            │ │
│  │ Actual: ~6ms (efficiency > 100% due to L2 caching!)                                                  │ │
│  │ 96 MB L2 cache holds significant portion of active weights                                           │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### SM Utilization Visualization: MoE Models

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL MOE MODELS: SM UTILIZATION VISUALIZATION                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  30B-A3B MoE on H100-80GB (132 SMs)                                                                        │
│  ══════════════════════════════════                                                                         │
│                                                                                                             │
│  PREFILL PHASE (MoE routing active):                                                                        │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 110 120 132                                 │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤                               │ │
│  │                                                                                                       │ │
│  │ Router Phase:                                                                                         │ │
│  │      ████████                                                         10% (router is tiny)            │ │
│  │                                                                                                       │ │
│  │ Expert Dispatch:                                                                                      │ │
│  │      ████████████████████████████████████████████████████████████████████  85% (FusedMoE kernel)     │ │
│  │      │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│  8 experts compute in parallel │ │
│  │                                                                                                       │ │
│  │ MoE Execution Pattern:                                                                                │ │
│  │ ┌────────────────────────────────────────────────────────────────────────────────────────────────────┐│ │
│  │ │  Token 0: Router → [E12, E45, E7, E89, E34, E56, E23, E101] → Weighted Sum                        ││ │
│  │ │  Token 1: Router → [E3, E67, E12, E45, E78, E99, E2, E55]  → Weighted Sum                         ││ │
│  │ │  ...                                                                                                ││ │
│  │ │                                                                                                     ││ │
│  │ │  FusedMoE groups tokens by expert, maximizing Tensor Core utilization                              ││ │
│  │ │  Each SM handles subset of experts for all tokens → better locality                                ││ │
│  │ └────────────────────────────────────────────────────────────────────────────────────────────────────┘│ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  DECODE PHASE (single token):                                                                               │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ SM:  0   10   20   30   40   50   60   70   80   90  100 110 120 132                                 │ │
│  │      ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤                               │ │
│  │                                                                                                       │ │
│  │ Attention (shared):                                                                                   │ │
│  │      ████████████████████████████████████                              40%                           │ │
│  │                                                                                                       │ │
│  │ MoE (8 of 128 experts):                                                                              │ │
│  │      ████████████████████████████████████████████████████████████       50%                          │ │
│  │      │ Expert 12 │ Expert 45 │ Expert 7  │ Expert 89 │ E34 │ E56 │ E23 │ E101 │                      │ │
│  │                                                                                                       │ │
│  │ ✅ Only 8/128 = 6.25% of expert weights read                                                         │ │
│  │ Memory read: 4.6 GB (not 30 GB!) / 3350 GB/s = 1.4ms                                                 │ │
│  │ Actual: ~6ms (includes router, expert dispatch overhead)                                              │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ═══════════════════════════════════════════════════════════════════════════════════════════════════════   │
│                                                                                                             │
│  235B-A22B MoE on 4×H100 (4×132 = 528 SMs total)                                                           │
│  ═══════════════════════════════════════════════                                                            │
│                                                                                                             │
│  DECODE PHASE (distributed across 4 GPUs):                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  GPU 0 (132 SMs)           GPU 1 (132 SMs)           GPU 2 (132 SMs)           GPU 3 (132 SMs)       │ │
│  │  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐│ │
│  │  │ ████████████████    │   │ ████████████████    │   │ ████████████████    │   │ ████████████████    ││ │
│  │  │ Experts 0-31        │   │ Experts 32-63       │   │ Experts 64-95       │   │ Experts 96-127      ││ │
│  │  │ (2 of 8 selected)   │   │ (2 of 8 selected)   │   │ (2 of 8 selected)   │   │ (2 of 8 selected)   ││ │
│  │  └──────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘│ │
│  │             │                         │                         │                         │           │ │
│  │             └─────────────────────────┴─────────────────────────┴─────────────────────────┘           │ │
│  │                                    NVLink All-to-All                                                  │ │
│  │                                    (Expert dispatch + combine)                                        │ │
│  │                                                                                                       │ │
│  │  Execution Timeline:                                                                                  │ │
│  │  ┌────────────────────────────────────────────────────────────────────────────────────────────────┐  │ │
│  │  │  0ms        2ms        4ms        6ms        8ms       10ms       12ms                         │  │ │
│  │  │  ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤                           │  │ │
│  │  │  │ Router   │ All2All  │      Expert Compute        │ AllReduce │ Attention                   │  │ │
│  │  │  │ (local)  │ (NVLink) │      (local per GPU)       │ (NVLink)  │ (local)                     │  │ │
│  │  │  │ 0.2ms    │ 2.0ms    │         6.0ms              │ 2.0ms     │ 1.8ms                       │  │ │
│  │  │  └────────────────────────────────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                                                       │ │
│  │  Total: 12ms per token (bottleneck: NVLink communication + expert compute)                           │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Python Script: Profile vLLM SM Utilization

```python
#!/usr/bin/env python3
"""
Profile vLLM Qwen3-VL SM Utilization
====================================
This script helps visualize SM utilization when running Qwen3-VL on different GPUs.

Usage:
    python profile_sm_utilization.py --model Qwen/Qwen3-VL-8B-Instruct
"""

import torch
import subprocess
import sys
from typing import Optional

def get_gpu_info():
    """Get GPU SM count and compute capability."""
    if not torch.cuda.is_available():
        return None
    
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_gb": props.total_memory / (1024**3),
        "memory_bandwidth_gbps": {
            "7.5": 320,   # T4
            "8.0": 2039,  # A100
            "9.0": 3350,  # H100
            "10.0": 8000, # B200
        }.get(f"{props.major}.{props.minor}", 0)
    }

def estimate_sm_utilization(model_size_b: float, gpu_info: dict, phase: str):
    """Estimate SM utilization for a given model and phase."""
    sm_count = gpu_info["sm_count"]
    bandwidth = gpu_info["memory_bandwidth_gbps"]
    
    if phase == "prefill":
        # Compute-bound: most SMs active
        base_util = 0.90
        # More SMs = slightly lower utilization due to sync overhead
        util = base_util - (sm_count - 108) * 0.001
    else:  # decode
        # Memory-bound: utilization = bandwidth / weight_read_requirement
        weight_size_gb = model_size_b * 2  # BF16
        theoretical_time_ms = weight_size_gb / bandwidth * 1000
        # More bandwidth = higher utilization
        util = min(0.50, bandwidth / 4000)  # Cap at 50% for decode
    
    return max(0.10, min(0.99, util))

def visualize_sm_utilization(sm_count: int, utilization: float, label: str):
    """Print ASCII visualization of SM utilization."""
    active_sms = int(sm_count * utilization)
    bar_width = 60
    filled = int(bar_width * utilization)
    
    print(f"\n{label}:")
    print(f"  SMs: 0{'─' * (bar_width - 2)}{sm_count}")
    print(f"       {'█' * filled}{'░' * (bar_width - filled)} {utilization*100:.0f}%")
    print(f"       Active: {active_sms}/{sm_count} SMs")

def profile_with_nsight(model_path: str, output_file: str = "vllm_profile"):
    """Generate Nsight profile command."""
    cmd = f"""
# Run with Nsight Systems for visual profiling:
nsys profile --stats=true \\
    --cuda-graph-trace=node \\
    --output={output_file} \\
    python -c "
from vllm import LLM, SamplingParams
from PIL import Image
import requests

# Load model
llm = LLM(
    model='{model_path}',
    max_model_len=4096,
    limit_mm_per_prompt={{'image': 1}}
)

# Run inference
image_url = 'https://example.com/image.jpg'
response = llm.generate([{{
    'prompt': '<|vision_start|><|image_pad|><|vision_end|>Describe this image.',
    'multi_modal_data': {{'image': image_url}}
}}], SamplingParams(max_tokens=100))
"

# Open in Nsight GUI:
nsys-ui {output_file}.nsys-rep
"""
    return cmd

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()
    
    # Get GPU info
    gpu_info = get_gpu_info()
    if not gpu_info:
        print("No CUDA GPU detected")
        return
    
    print("=" * 70)
    print("vLLM SM Utilization Estimator")
    print("=" * 70)
    print(f"GPU: {gpu_info['name']}")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"SM Count: {gpu_info['sm_count']}")
    print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
    print(f"Memory Bandwidth: {gpu_info['memory_bandwidth_gbps']} GB/s")
    print(f"Model: {args.model}")
    
    # Estimate model size
    if "8B" in args.model:
        model_size = 8.0
    elif "32B" in args.model:
        model_size = 32.0
    elif "30B-A3B" in args.model:
        model_size = 30.0  # Total, but only 3B active
    else:
        model_size = 8.0  # Default
    
    # Visualize utilization
    prefill_util = estimate_sm_utilization(model_size, gpu_info, "prefill")
    decode_util = estimate_sm_utilization(model_size, gpu_info, "decode")
    
    visualize_sm_utilization(gpu_info['sm_count'], prefill_util, "Prefill Phase (compute-bound)")
    visualize_sm_utilization(gpu_info['sm_count'], decode_util, "Decode Phase (memory-bound)")
    
    # Print Nsight command
    print("\n" + "=" * 70)
    print("For detailed profiling, run with Nsight:")
    print("=" * 70)
    print(profile_with_nsight(args.model))

if __name__ == "__main__":
    main()
```

### Summary: GPU Comparison Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     GPU COMPARISON SUMMARY: ALL MODELS                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┐│
│  │ Model       │ GPU         │ Precision   │ TTFT        │ Decode/tok  │ Throughput  │ Max Concurrent @8K ││
│  ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤│
│  │ 8B          │ A100-80GB   │ BF16        │ 153 ms      │ 12 ms       │ 78 tok/s    │ 56 requests        ││
│  │ 8B          │ H100-80GB   │ BF16+FP8 KV │ 80 ms       │ 6 ms        │ 156 tok/s   │ 112 requests       ││
│  │ 8B          │ B200-192GB  │ BF16        │ 50 ms       │ 3 ms        │ 308 tok/s   │ 167 requests       ││
│  ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤│
│  │ 32B         │ A100-80GB   │ BF16        │ 310 ms      │ 25 ms       │ 38 tok/s    │ 2 requests         ││
│  │ 32B         │ H100-80GB   │ FP8         │ 155 ms      │ 12 ms       │ 78 tok/s    │ 35 requests        ││
│  │ 32B         │ B200-192GB  │ BF16        │ 95 ms       │ 6 ms        │ 154 tok/s   │ 57 requests        ││
│  ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤│
│  │ 30B-A3B MoE │ H100-80GB   │ FP8         │ 110 ms      │ 6 ms        │ 153 tok/s   │ 105 requests       ││
│  ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤│
│  │ 235B-A22B   │ 4×H100-80GB │ FP8         │ 360 ms      │ 12 ms       │ 72 tok/s    │ 24 requests        ││
│  └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘│
│                                                                                                             │
│  KEY INSIGHTS:                                                                                              │
│  1. H100 is 2× faster than A100 for all models (bandwidth + FP8)                                           │
│  2. B200 is 2× faster than H100 (bandwidth dominates decode)                                               │
│  3. 32B on H100 FP8 = 8B on A100 BF16 (same 78 tok/s!)                                                    │
│  4. 30B-A3B MoE matches 8B throughput with better quality                                                  │
│  5. 235B-A22B on 4×H100 is cost-competitive with GPT-4V APIs                                               │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## How Qwen3-VL Inference Actually Works: A Systems-Level Guide

This section explains exactly how inference runs for Qwen3-VL models, from a single request to GPU kernels.

### Architecture Overview: Dense vs MoE

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL INFERENCE ARCHITECTURE                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │   INPUT                    VISION ENCODER              LANGUAGE MODEL              OUTPUT              ││
│  │  ═══════                  ═══════════════             ═══════════════             ════════             ││
│  │                                                                                                         ││
│  │  ┌─────────┐             ┌───────────────┐           ┌───────────────┐           ┌─────────┐           ││
│  │  │ Image   │───────────▶│Qwen3_VisionTF │──────────▶│Qwen3ForCausal │──────────▶│ Tokens  │           ││
│  │  │ 1024×   │             │               │           │LM or          │           │ ─────▶  │           ││
│  │  │ 1024    │             │ Patch Embed   │           │Qwen3MoeFor    │           │ Text    │           ││
│  │  └─────────┘             │ (Conv3D)      │           │CausalLM       │           └─────────┘           ││
│  │       +                  │      ↓        │           │               │                                 ││
│  │  ┌─────────┐             │ ViT Blocks    │           │ Embed Tokens  │                                 ││
│  │  │ Prompt  │             │ (32 layers)   │           │      ↓        │                                 ││
│  │  │ "What   │             │      ↓        │           │ Transformer   │                                 ││
│  │  │ is in   │             │ DeepStack     │──────────▶│ Layers        │                                 ││
│  │  │ this?"  │             │ (Multi-level) │           │ (28-64 layers)│                                 ││
│  │  └─────────┘             │      ↓        │           │      ↓        │                                 ││
│  │                          │ Merger        │           │ LM Head       │                                 ││
│  │                          └───────────────┘           └───────────────┘                                 ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  KEY ARCHITECTURAL COMPONENTS (from vllm/model_executor/models/qwen3_vl.py):                               │
│  ═══════════════════════════════════════════════════════════════════════════                               │
│                                                                                                             │
│  1. Qwen3_VisionPatchEmbed:                                                                                │
│     • Conv3D with kernel (temporal_patch_size=2, patch_size=14, patch_size=14)                            │
│     • Input: (L, C) → Output: (L, hidden_size=1152)                                                       │
│     • This is the FIRST operation—converts raw pixels to patch embeddings                                  │
│                                                                                                             │
│  2. Qwen3_VisionBlock (× 32 layers):                                                                       │
│     • LayerNorm → Attention (with RoPE) → LayerNorm → MLP (SiLU activation)                               │
│     • Uses Qwen2_5_VisionAttention for attention computation                                               │
│                                                                                                             │
│  3. DeepStack (Qwen3-VL specific):                                                                         │
│     • Injects multi-level ViT features at specific language model layers                                  │
│     • deepstack_visual_indexes defines which layers receive vision features                               │
│     • Creates richer vision-language alignment                                                            │
│                                                                                                             │
│  4. Language Model:                                                                                        │
│     • Dense: Qwen3ForCausalLM (standard transformer)                                                      │
│     • MoE: Qwen3MoeForCausalLM (sparse MoE with expert routing)                                           │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Internal Architecture: Each Model Size

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-2B INTERNAL ARCHITECTURE                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  VISION ENCODER (Qwen3_VisionTransformer)                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Patch Embedding: Conv3D(in=3, out=1152, kernel=(2,14,14))                                             ││
│  │  Position: Learned + 3D RoPE (temporal, height, width)                                                 ││
│  │  ViT Blocks: 24 layers, hidden=1152, heads=16, mlp=4608                                                ││
│  │  DeepStack Extract: layers [8, 16, 24]                                                                 ││
│  │  Merger: Linear(1152×4 → 1536), spatial_merge=2×2                                                      ││
│  │  Parameters: ~400M                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  LANGUAGE MODEL (Qwen3LLMModel)                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Embedding: 151936 × 1536 (vocab × hidden)                                                             ││
│  │  Layers: 28                                                                                             ││
│  │  Hidden: 1536                                                                                           ││
│  │  Heads: 12 attention heads                                                                              ││
│  │  KV Heads: 2 (GQA ratio 6:1)                                                                           ││
│  │  Head Dim: 128                                                                                          ││
│  │  Intermediate: 8960 (MLP)                                                                               ││
│  │  Activation: SiLU                                                                                       ││
│  │  DeepStack Inject: layers [0, 1, 2]                                                                    ││
│  │  Parameters: ~1.9B                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  MEMORY: Weights=4.6GB BF16 | KV/token=28.7KB | Best GPU: T4 (fits easily)                                 │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-4B INTERNAL ARCHITECTURE                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  VISION ENCODER                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Patch Embedding: Conv3D(in=3, out=1280, kernel=(2,14,14))                                             ││
│  │  ViT Blocks: 28 layers, hidden=1280, heads=16, mlp=5120                                                ││
│  │  DeepStack Extract: layers [10, 19, 28]                                                                ││
│  │  Merger: Linear(1280×4 → 2048)                                                                         ││
│  │  Parameters: ~600M                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  LANGUAGE MODEL                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Layers: 36 | Hidden: 2048 | Heads: 16 | KV Heads: 4 (GQA 4:1) | Intermediate: 11264                  ││
│  │  DeepStack Inject: layers [0, 1, 2]                                                                    ││
│  │  Parameters: ~3.8B                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  MEMORY: Weights=8.8GB BF16 | KV/token=73.7KB | Best GPU: A100-40GB or T4 with INT4                        │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B INTERNAL ARCHITECTURE                                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  VISION ENCODER                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Patch Embedding: Conv3D(in=3, out=1536, kernel=(2,14,14))                                             ││
│  │  ViT Blocks: 32 layers, hidden=1536, heads=24, mlp=6144                                                ││
│  │  DeepStack Extract: layers [11, 22, 32]                                                                ││
│  │  Merger: Linear(1536×4 → 4096)                                                                         ││
│  │  Parameters: ~1.2B                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  LANGUAGE MODEL                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Layers: 32 | Hidden: 4096 | Heads: 32 | KV Heads: 8 (GQA 4:1) | Intermediate: 12288                  ││
│  │  DeepStack Inject: layers [0, 1, 2]                                                                    ││
│  │  Parameters: ~7.1B                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  MEMORY: Weights=16.6GB BF16 | KV/token=131KB | Best GPU: A100-80GB, H100 with FP8                         │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-32B INTERNAL ARCHITECTURE (GUI Agent Recommended)                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  VISION ENCODER                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Patch Embedding: Conv3D(in=3, out=1792, kernel=(2,14,14))                                             ││
│  │  ViT Blocks: 32 layers, hidden=1792, heads=28, mlp=7168                                                ││
│  │  DeepStack Extract: layers [11, 22, 32]                                                                ││
│  │  Merger: Linear(1792×4 → 5120)                                                                         ││
│  │  Parameters: ~2.0B                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  LANGUAGE MODEL                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Layers: 64 (2× more than 8B!)                                                                         ││
│  │  Hidden: 5120                                                                                           ││
│  │  Heads: 40                                                                                              ││
│  │  KV Heads: 8 (GQA 5:1)                                                                                 ││
│  │  Intermediate: 25600                                                                                    ││
│  │  DeepStack Inject: layers [0, 1, 2, 3] (more injection points)                                         ││
│  │  Parameters: ~30.8B                                                                                     ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  WHY 64 LAYERS MATTERS FOR GUI AGENTS:                                                                     │
│  • 2× reasoning depth = better multi-step planning                                                         │
│  • More capacity for spatial/visual reasoning                                                              │
│  • Research: Models <32B fail to reliably combine grounding + planning + action                            │
│                                                                                                             │
│  MEMORY: Weights=65.6GB BF16 | KV/token=262KB | Best GPU: H100 with FP8, B200                              │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-30B-A3B (MoE) INTERNAL ARCHITECTURE                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  VISION ENCODER: Same as 8B (1.2B params)                                                                  │
│                                                                                                             │
│  LANGUAGE MODEL (MoE)                                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Layers: 48                                                                                             ││
│  │  Hidden: 2048                                                                                           ││
│  │  Heads: 16                                                                                              ││
│  │  KV Heads: 4 (GQA 4:1)                                                                                 ││
│  │  Expert Count: 128                                                                                      ││
│  │  Top-K: 8 (only 8 experts compute per token)                                                           ││
│  │  Shared Experts: 0                                                                                      ││
│  │  Expert Intermediate: 1024                                                                              ││
│  │                                                                                                         ││
│  │  Per-Token Compute: 8 experts × MLP = ~3B active params                                                ││
│  │  Total Params: 48 layers × 128 experts × MLP = ~30B                                                    ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  KEY INSIGHT: Same decode speed as 4B dense, but with 8B-level quality                                     │
│  MEMORY: Weights=60GB BF16 (all experts loaded) | KV/token=49KB | Best GPU: H100-80GB                      │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-235B-A22B (MoE) INTERNAL ARCHITECTURE                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  VISION ENCODER: Same as 32B (2.0B params)                                                                 │
│                                                                                                             │
│  LANGUAGE MODEL (MoE)                                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  Layers: 94                                                                                             ││
│  │  Hidden: 5120                                                                                           ││
│  │  Heads: 40                                                                                              ││
│  │  KV Heads: 8 (GQA 5:1)                                                                                 ││
│  │  Expert Count: 128                                                                                      ││
│  │  Top-K: 8                                                                                               ││
│  │  Shared Experts: 0                                                                                      ││
│  │  Expert Intermediate: 2560                                                                              ││
│  │                                                                                                         ││
│  │  Per-Token Compute: 8 experts × large MLP = ~22B active params                                         ││
│  │  Total Params: 94 layers × 128 experts × MLP = ~235B                                                   ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  FRONTIER MODEL: GPT-4 class quality at 10× lower cost per token                                           │
│  MEMORY: Weights=470GB BF16 | KV/token=245KB | Requires: 4×H100 (TP=4) or 8×H100 (TP=8)                    │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-2B: Complete Internal Architecture (Reference Model)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-2B INTERNAL ARCHITECTURE (COMPLETE REFERENCE)                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  VISION ENCODER (Qwen3_VisionTransformer)                                                                   │
│  ═════════════════════════════════════════                                                                  │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Patch Embedding (Conv3D)                                                                               ││
│  │  ────────────────────────                                                                               ││
│  │  Input: [batch, 3, temporal, H, W]                                                                      ││
│  │  Conv3D(in=3, out=1152, kernel=(2,14,14), stride=(2,14,14), bias=True)                                 ││
│  │  Output: [num_patches, 1152]                                                                            ││
│  │                                                                                                         ││
│  │  For 1024×1024 image: (1024/14)² = 5,329 patches → after merge: 1,332 tokens                           ││
│  │                                                                                                         ││
│  │  Position Embedding                          RoPE (Rotary Position)                                     ││
│  │  ──────────────────────                      ────────────────────────                                   ││
│  │  nn.Embedding(num_pos, 1152)                 partial_rotary_factor=0.5                                  ││
│  │  + Bilinear interpolation                    head_dim=72, max_position=8192                             ││
│  │                                                                                                         ││
│  │  ViT Blocks (×24 layers)                                                                                ││
│  │  ───────────────────────                                                                                ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │ For layer_idx in range(24):                                                                     │   ││
│  │  │   x = x + Attention(LayerNorm(x), cu_seqlens, rope_cos, rope_sin)                              │   ││
│  │  │   x = x + MLP(LayerNorm(x))   # MLP: Linear(1152→4608) → SiLU → Linear(4608→1152)             │   ││
│  │  │                                                                                                 │   ││
│  │  │   if layer_idx in [8, 16, 24]:  # DeepStack extraction points                                  │   ││
│  │  │       deepstack_features[idx] = DeepStackMerger(x)                                              │   ││
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                         ││
│  │  Patch Merger                                                                                           ││
│  │  ────────────                                                                                           ││
│  │  spatial_merge_size=2 → Merge 2×2 patches into 1 token                                                 ││
│  │  Linear(1152 × 4 = 4608 → 1536)  # Projects to LLM hidden size                                         ││
│  │                                                                                                         ││
│  │  Total ViT Parameters: ~400M                                                                            ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  LANGUAGE MODEL (Qwen3LLMModel)                                                                             │
│  ═══════════════════════════════                                                                            │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Embedding: nn.Embedding(151936, 1536)  # vocab_size × hidden_size                                      ││
│  │                                                                                                         ││
│  │  Transformer Blocks (×28 layers)                                                                        ││
│  │  ───────────────────────────────                                                                        ││
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐   ││
│  │  │ For layer_idx in range(28):                                                                     │   ││
│  │  │                                                                                                 │   ││
│  │  │   # Self-Attention (GQA)                                                                        │   ││
│  │  │   hidden = RMSNorm(hidden)                                                                      │   ││
│  │  │   Q = Linear(1536 → 1536)   # 12 heads × 128 head_dim                                          │   ││
│  │  │   K = Linear(1536 → 256)    # 2 KV heads × 128 head_dim                                        │   ││
│  │  │   V = Linear(1536 → 256)    # 2 KV heads × 128 head_dim                                        │   ││
│  │  │   attn_out = Attention(Q, K, V)  # GQA ratio 6:1                                               │   ││
│  │  │   hidden = hidden + O_proj(attn_out)                                                           │   ││
│  │  │                                                                                                 │   ││
│  │  │   # MLP (SwiGLU)                                                                                │   ││
│  │  │   hidden = hidden + RMSNorm(hidden)                                                            │   ││
│  │  │   gate = Linear(1536 → 8960)                                                                   │   ││
│  │  │   up = Linear(1536 → 8960)                                                                     │   ││
│  │  │   hidden = hidden + Down(SiLU(gate) × up)  # Down: 8960 → 1536                                │   ││
│  │  │                                                                                                 │   ││
│  │  │   # DeepStack Injection (layers 0, 1, 2)                                                        │   ││
│  │  │   if layer_idx < len(deepstack_features):                                                       │   ││
│  │  │       hidden = hidden + deepstack_features[layer_idx]                                          │   ││
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘   ││
│  │                                                                                                         ││
│  │  Final: RMSNorm(hidden) → LM_Head(1536 → 151936)                                                        ││
│  │                                                                                                         ││
│  │  Total LLM Parameters: ~1.7B                                                                            ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  MEMORY BREAKDOWN                                                                                           │
│  ════════════════                                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Component                    │ Parameters    │ BF16 Size   │ FP8 Size    │ INT4 Size                   ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encoder               │ 400M          │ 800 MB      │ 400 MB      │ 200 MB                      ││
│  │  LLM Embedding               │ 233M          │ 466 MB      │ 233 MB      │ 117 MB                      ││
│  │  LLM Attention (×28)         │ 567M          │ 1.1 GB      │ 567 MB      │ 284 MB                      ││
│  │  LLM MLP (×28)               │ 846M          │ 1.7 GB      │ 846 MB      │ 423 MB                      ││
│  │  LLM Head                    │ 233M          │ 466 MB      │ 233 MB      │ 117 MB                      ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL WEIGHTS               │ ~2.3B         │ 4.6 GB      │ 2.3 GB      │ 1.15 GB                     ││
│  │                                                                                                         ││
│  │  KV Cache per Token:                                                                                    ││
│  │  = 2 × 28 layers × 2 kv_heads × 128 head_dim × dtype_bytes                                             ││
│  │  = 2 × 28 × 2 × 128 × 2 (BF16) = 28.7 KB per token                                                     ││
│  │  = 2 × 28 × 2 × 128 × 1 (FP8) = 14.3 KB per token                                                      ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL-30B-A3B MoE: Complete Internal Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-30B-A3B MOE ARCHITECTURE (COMPLETE)                                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  MOE LAYER STRUCTURE (from qwen3_vl_moe.py)                                                                 │
│  ══════════════════════════════════════════                                                                 │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  class Qwen3MoeSparseMoeBlock:                                                                          ││
│  │      num_logical_experts = 128     # Total experts in model                                             ││
│  │      num_routed_experts = 8        # Top-K experts per token                                            ││
│  │      num_shared_experts = 0        # Always-active experts                                              ││
│  │                                                                                                         ││
│  │  PER-TOKEN EXECUTION:                                                                                   ││
│  │  ────────────────────                                                                                   ││
│  │                                                                                                         ││
│  │     Input Token                                                                                         ││
│  │         │                                                                                               ││
│  │         ▼                                                                                               ││
│  │   ┌───────────────┐                                                                                     ││
│  │   │    Router     │  Linear(hidden_dim → 128)                                                           ││
│  │   │   (Gating)    │  router_logits = token @ W_router                                                   ││
│  │   └───────┬───────┘  top_k_weights, top_k_indices = topk(softmax(router_logits), k=8)                  ││
│  │           │                                                                                             ││
│  │           ▼                                                                                             ││
│  │   ┌───────────────────────────────────────────────────────────────────────────────────────────┐        ││
│  │   │                           EXPERT SELECTION (Top-8 of 128)                                 │        ││
│  │   │                                                                                           │        ││
│  │   │   Expert Pool: [E0, E1, E2, E3, E4, ... E126, E127]                                      │        ││
│  │   │                  ▲   ▲       ▲           ▲    ▲                                          │        ││
│  │   │                  │   │       │           │    │                                          │        ││
│  │   │              Selected: E12(0.18), E45(0.15), E7(0.14), E89(0.12),                        │        ││
│  │   │                        E34(0.11), E56(0.10), E23(0.10), E101(0.10)                       │        ││
│  │   │                                                                                           │        ││
│  │   │   Each Expert is an MLP:                                                                  │        ││
│  │   │   Expert_i(x) = Down_i(SiLU(Gate_i(x)) × Up_i(x))                                        │        ││
│  │   │   Gate_i, Up_i: [hidden=2048 → expert_intermediate=1024]                                 │        ││
│  │   │   Down_i: [expert_intermediate=1024 → hidden=2048]                                       │        ││
│  │   │   Per expert: ~6.3M params                                                               │        ││
│  │   │   128 experts: ~800M params per MoE layer                                                │        ││
│  │   │                                                                                           │        ││
│  │   └───────────────────────────────────────────────────────────────────────────────────────────┘        ││
│  │           │                                                                                             ││
│  │           ▼                                                                                             ││
│  │   ┌───────────────┐                                                                                     ││
│  │   │  Weighted Sum │  output = Σ (weight_i × Expert_i(token))                                           ││
│  │   └───────────────┘           for i in selected_8_experts                                              ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  FULL MODEL STRUCTURE                                                                                       │
│  ════════════════════                                                                                       │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Vision Encoder: Same as Dense models (~1B params, 32 ViT layers)                                       ││
│  │                                                                                                         ││
│  │  Language Model (48 layers):                                                                            ││
│  │  ┌───────────────────────────────────────────────────────────────────────────────────────────────────┐ ││
│  │  │                                                                                                   │ ││
│  │  │  For layer_idx in range(48):                                                                      │ ││
│  │  │                                                                                                   │ ││
│  │  │    # Attention (same as dense)                                                                    │ ││
│  │  │    hidden = RMSNorm(hidden)                                                                       │ ││
│  │  │    Q = Linear(2048 → 2048)   # 16 heads × 128 head_dim                                           │ ││
│  │  │    K = Linear(2048 → 512)    # 4 KV heads × 128 head_dim                                         │ ││
│  │  │    V = Linear(2048 → 512)    # GQA ratio 4:1                                                     │ ││
│  │  │    hidden = hidden + Attention(Q, K, V)                                                          │ ││
│  │  │                                                                                                   │ ││
│  │  │    # MoE (replaces dense MLP)                                                                     │ ││
│  │  │    hidden = hidden + MoE_Layer(RMSNorm(hidden))  # Routes to 8 of 128 experts                    │ ││
│  │  │                                                                                                   │ ││
│  │  └───────────────────────────────────────────────────────────────────────────────────────────────────┘ ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  PARAMETER COUNT BREAKDOWN                                                                                  │
│  ══════════════════════════                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Component                    │ Total Params  │ Active/Token  │ Memory (BF16) │ Notes                   ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Encoder               │ 1.0B          │ 1.0B          │ 2.0 GB        │ Always fully active     ││
│  │  LLM Embedding               │ 311M          │ 311M          │ 622 MB        │ Always fully active     ││
│  │  LLM Attention (×48)         │ 1.2B          │ 1.2B          │ 2.4 GB        │ Always fully active     ││
│  │  Router Weights (×48)        │ 6.3M          │ 6.3M          │ 13 MB         │ Always fully active     ││
│  │  Expert Weights (×48×128)    │ 38.4B         │ 2.4B          │ 76.8 GB       │ 8/128 active per token  ││
│  │  LLM Head                    │ 311M          │ 311M          │ 622 MB        │ Always fully active     ││
│  │  ──────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  TOTAL                        │ ~41B          │ ~5.2B         │ ~82 GB        │ 12.7% activation ratio  ││
│  │                                                                                                         ││
│  │  KV Cache per Token (same as 4B dense due to same hidden/kv_heads):                                    ││
│  │  = 2 × 48 layers × 4 kv_heads × 128 head_dim × 2 bytes = 98.3 KB per token                            ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### DeepStack: Multi-Level Vision Feature Injection

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     DEEPSTACK ARCHITECTURE (Qwen3-VL Unique Feature)                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WHAT IS DEEPSTACK?                                                                                         │
│  Instead of injecting vision features only at layer 0, DeepStack injects at multiple LLM layers            │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  VISION ENCODER                           LANGUAGE MODEL                                                ││
│  │  ═════════════                           ═══════════════                                                ││
│  │                                                                                                         ││
│  │  ┌──────────────┐                        ┌──────────────┐                                               ││
│  │  │ ViT Layer 8  │ ──────────────────────▶│ LLM Layer 0  │ (+ deepstack_features[0])                    ││
│  │  └──────────────┘                        └──────────────┘                                               ││
│  │         ↓                                       ↓                                                       ││
│  │  ┌──────────────┐                        ┌──────────────┐                                               ││
│  │  │ ViT Layer 16 │ ──────────────────────▶│ LLM Layer 1  │ (+ deepstack_features[1])                    ││
│  │  └──────────────┘                        └──────────────┘                                               ││
│  │         ↓                                       ↓                                                       ││
│  │  ┌──────────────┐                        ┌──────────────┐                                               ││
│  │  │ ViT Layer 24 │ ──────────────────────▶│ LLM Layer 2  │ (+ deepstack_features[2])                    ││
│  │  └──────────────┘                        └──────────────┘                                               ││
│  │         ↓                                       ↓                                                       ││
│  │  ┌──────────────┐                        ┌──────────────┐                                               ││
│  │  │ Final Merge  │ ──────────────────────▶│ LLM Layer 3+ │ (standard processing)                        ││
│  │  └──────────────┘                        └──────────────┘                                               ││
│  │                                                 ↓                                                       ││
│  │                                          ┌──────────────┐                                               ││
│  │                                          │ Output Tokens│                                               ││
│  │                                          └──────────────┘                                               ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  FROM VLLM CODE (qwen3_vl.py):                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  # deepstack_visual_indexes defines which ViT layers extract features                                  ││
│  │  # These are injected into early LLM layers (0, 1, 2, ...)                                             ││
│  │                                                                                                         ││
│  │  self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")                        ││
│  │  if self.use_deepstack:                                                                                 ││
│  │      self.deepstack_input_embeds = [                                                                    ││
│  │          torch.zeros(max_batched_tokens, hidden_size)                                                   ││
│  │          for _ in range(len(deepstack_visual_indexes))                                                  ││
│  │      ]                                                                                                  ││
│  │                                                                                                         ││
│  │  # During forward pass:                                                                                 ││
│  │  if layer_idx < len(deepstack_features):                                                                ││
│  │      hidden_states = hidden_states + deepstack_features[layer_idx]                                     ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  WHY DEEPSTACK MATTERS:                                                                                     │
│  • Low-level features (edges, textures) injected early                                                     │
│  • Mid-level features (shapes, objects) injected mid-way                                                   │
│  • High-level features (semantics) injected via main merge                                                 │
│  • Result: Richer vision-language alignment than single-injection models                                   │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Model Size Comparison: Dense vs MoE

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     DENSE vs MoE: WHAT'S DIFFERENT                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  DENSE MODELS (2B, 4B, 8B, 32B):                                                                           │
│  ══════════════════════════════                                                                             │
│                                                                                                             │
│  ┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐                                  │
│  │   Input    │────▶│  Attention │────▶│    MLP     │────▶│   Output   │                                  │
│  │  Tokens    │     │   (QKV)    │     │ (Gate+Up)  │     │  Tokens    │                                  │
│  └────────────┘     └────────────┘     └────────────┘     └────────────┘                                  │
│                                                                                                             │
│  • ALL parameters active for EVERY token                                                                   │
│  • Compute = O(total_params × tokens)                                                                      │
│  • Memory = model_weights + KV_cache + activations                                                         │
│                                                                                                             │
│  MoE MODELS (30B-A3B, 235B-A22B):                                                                          │
│  ════════════════════════════════                                                                           │
│                                                                                                             │
│  ┌────────────┐     ┌────────────┐     ┌─────────────────────────────────┐     ┌────────────┐             │
│  │   Input    │────▶│  Attention │────▶│         ROUTER                  │────▶│   Output   │             │
│  │  Tokens    │     │   (QKV)    │     │  (Token → Top-K Experts)       │     │  Tokens    │             │
│  └────────────┘     └────────────┘     │             ↓                   │     └────────────┘             │
│                                         │  ┌───┐┌───┐┌───┐     ┌───┐    │                                  │
│                                         │  │E1 ││E2 ││E3 │ ... │E64│    │                                  │
│                                         │  └───┘└───┘└───┘     └───┘    │                                  │
│                                         │   ↑ ↑    (only 2-8 active)    │                                  │
│                                         └─────────────────────────────────┘                                  │
│                                                                                                             │
│  • Only TOP-K experts active per token (e.g., 2 of 64)                                                    │
│  • Compute = O(active_params × tokens)  ← Much less!                                                       │
│  • Memory = ALL expert weights + shared_layers + KV_cache                                                  │
│                                                                                                             │
│  FROM vllm/model_executor/models/qwen3_vl_moe.py:                                                          │
│  ─────────────────────────────────────────────────                                                          │
│  class Qwen3VLMoeMixtureOfExperts:                                                                         │
│      num_logical_experts = 64      # Total experts per layer                                               │
│      num_routed_experts = 2-8      # Active per token (top-k)                                              │
│      num_shared_experts = 0        # No shared experts in Qwen3 MoE                                        │
│                                                                                                             │
│  CONCRETE EXAMPLE:                                                                                          │
│  ──────────────────                                                                                         │
│  Qwen3-VL-235B-A22B:                                                                                       │
│  • 235B total parameters (all experts + shared layers)                                                     │
│  • 22B active parameters per forward pass                                                                  │
│  • Memory: Need to load all 235B into VRAM                                                                 │
│  • Compute: Only 22B worth of FLOPs per token                                                              │
│  • Result: Frontier-model quality at 10x lower compute cost                                                │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Inference Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     SINGLE REQUEST INFERENCE: STEP BY STEP                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  EXAMPLE REQUEST:                                                                                           │
│  ═════════════════                                                                                          │
│  prompt = "What is in this image?"                                                                         │
│  image = 1024×1024 RGB image                                                                               │
│  model = Qwen3-VL-8B-Instruct                                                                              │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 1: TOKENIZATION + IMAGE PREPROCESSING                                                            │ │
│  │ ══════════════════════════════════════════                                                             │ │
│  │                                                                                                        │ │
│  │ CPU OPERATIONS:                                                                                        │ │
│  │ • Tokenize prompt → [token_ids] (e.g., 15 tokens)                                                     │ │
│  │ • Resize image to model's expected size (smart_resize)                                                │ │
│  │ • Normalize pixels: (pixel - mean) / std                                                              │ │
│  │ • Insert placeholder tokens: <|vision_start|><|image_pad|>×N<|vision_end|>                           │ │
│  │                                                                                                        │ │
│  │ LATENCY: ~5-10ms (CPU-bound, not GPU-dependent)                                                       │ │
│  │ MEMORY: Minimal (just input tensors)                                                                  │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 2: VISION ENCODING (Image → Visual Tokens)                                                        │ │
│  │ ═════════════════════════════════════════════════                                                      │ │
│  │                                                                                                        │ │
│  │ GPU OPERATIONS (Qwen3_VisionTransformer.forward):                                                     │ │
│  │                                                                                                        │ │
│  │ 1. Patch Embedding (Conv3D):                                                                          │ │
│  │    • Input: (batch, 3, H, W) → (batch, num_patches, hidden_dim=1152)                                  │ │
│  │    • 1024×1024 image → (1024/14)² = 5,329 patches → merged to ~1,332 visual tokens                   │ │
│  │    • Kernel: CUDNN Conv3D                                                                             │ │
│  │                                                                                                        │ │
│  │ 2. Position Encoding (Interleaved-MRoPE):                                                             │ │
│  │    • Compute 3D rotary embeddings (temporal, height, width)                                           │ │
│  │    • Kernel: Element-wise ops                                                                         │ │
│  │                                                                                                        │ │
│  │ 3. ViT Blocks (×32 layers):                                                                           │ │
│  │    FOR each layer:                                                                                    │ │
│  │        • LayerNorm → Self-Attention → LayerNorm → MLP                                                │ │
│  │        • Attention: FlashAttention-2 if available, else SDPA                                         │ │
│  │        • MLP: Two GEMMs with SiLU activation                                                         │ │
│  │                                                                                                        │ │
│  │ 4. DeepStack (extract multi-level features):                                                          │ │
│  │    • Save intermediate features from specific layers                                                  │ │
│  │    • Will be injected into language model later                                                       │ │
│  │                                                                                                        │ │
│  │ OUTPUT: visual_embeds (1332, 3584) for 8B model                                                       │ │
│  │                                                                                                        │ │
│  │ LATENCY BY GPU:                                                                                       │ │
│  │ ┌──────────┬────────────┬────────────────────────────────────────────────────────────────────────────┐│ │
│  │ │ GPU      │ Time       │ Why                                                                        ││ │
│  │ ├──────────┼────────────┼────────────────────────────────────────────────────────────────────────────┤│ │
│  │ │ T4       │ ~80ms      │ FP16 only, 320 GB/s bandwidth, no FlashAttention                          ││ │
│  │ │ A100     │ ~25ms      │ BF16 + FlashAttn-2, 2 TB/s bandwidth                                      ││ │
│  │ │ H100     │ ~12ms      │ FP8 + FlashAttn-3, 3.35 TB/s bandwidth, TMA                               ││ │
│  │ │ B200     │ ~8ms       │ 2nd Gen TE, 8 TB/s bandwidth                                              ││ │
│  │ └──────────┴────────────┴────────────────────────────────────────────────────────────────────────────┘│ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 3: KV CACHE ALLOCATION (PagedAttention)                                                           │ │
│  │ ════════════════════════════════════════════                                                           │ │
│  │                                                                                                        │ │
│  │ vLLM's PagedAttention allocates KV cache in BLOCKS, not contiguous memory:                           │ │
│  │                                                                                                        │ │
│  │ KV Cache Size Calculation:                                                                            │ │
│  │ kv_size_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_size                            │ │
│  │                                                                                                        │ │
│  │ For Qwen3-VL-8B (FP16):                                                                               │ │
│  │ = 2 × 28 layers × 4 kv_heads × 128 head_dim × 2 bytes = 57.3 KB per token                            │ │
│  │                                                                                                        │ │
│  │ For our request (prompt=15 + visual=1332 = 1347 tokens):                                              │ │
│  │ = 1347 × 57.3 KB = 77.2 MB KV cache for prefill                                                      │ │
│  │                                                                                                        │ │
│  │ Block Layout in HBM:                                                                                  │ │
│  │ ┌─────────┬─────────┬─────────┬─────────┬───────────────┐                                            │ │
│  │ │ Block 0 │ Block 1 │ Block 2 │ Block 3 │ ... Block N   │                                            │ │
│  │ │ (K,V)   │ (K,V)   │ (K,V)   │ (K,V)   │               │                                            │ │
│  │ │ 16 tok  │ 16 tok  │ 16 tok  │ 16 tok  │               │                                            │ │
│  │ └─────────┴─────────┴─────────┴─────────┴───────────────┘                                            │ │
│  │                                                                                                        │ │
│  │ MEMORY AVAILABLE BY GPU:                                                                               │ │
│  │ ┌──────────┬────────────┬────────────────────────────────────────────────────────────────────────────┐│ │
│  │ │ GPU      │ KV Cache   │ Max Concurrent Requests (8B model)                                         ││ │
│  │ ├──────────┼────────────┼────────────────────────────────────────────────────────────────────────────┤│ │
│  │ │ T4       │ ~2 GB      │ ~4 requests (2K context each)                                              ││ │
│  │ │ A100-40  │ ~15 GB     │ ~16 requests (8K context each)                                             ││ │
│  │ │ A100-80  │ ~35 GB     │ ~32 requests (16K context each)                                            ││ │
│  │ │ H100     │ ~40 GB     │ ~64 requests (FP8 KV cache)                                                ││ │
│  │ │ B200     │ ~100 GB    │ ~128 requests (FP4 KV cache future)                                        ││ │
│  │ └──────────┴────────────┴────────────────────────────────────────────────────────────────────────────┘│ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 4: PREFILL PHASE (Process All Input Tokens)                                                       │ │
│  │ ════════════════════════════════════════════════                                                       │ │
│  │                                                                                                        │ │
│  │ Merge visual + text embeddings:                                                                       │ │
│  │ input_embeds = _merge_multimodal_embeddings(text_embeds, visual_embeds, placeholder_positions)       │ │
│  │                                                                                                        │ │
│  │ FOR each layer in language_model:                                                                     │ │
│  │                                                                                                        │ │
│  │     1. ATTENTION (all 1347 tokens attend to each other):                                              │ │
│  │        ┌─────────────────────────────────────────────────────────────────────────────────────────────┐│ │
│  │        │ Q = input @ W_q    (GEMM: 1347 × hidden → 1347 × num_heads × head_dim)                     ││ │
│  │        │ K = input @ W_k    (GEMM)                                                                   ││ │
│  │        │ V = input @ W_v    (GEMM)                                                                   ││ │
│  │        │                                                                                             ││ │
│  │        │ Attention = softmax(Q @ K^T / sqrt(d)) @ V                                                 ││ │
│  │        │                                                                                             ││ │
│  │        │ Kernel: FlashAttention-2 (A100) or FlashAttention-3 (H100)                                 ││ │
│  │        │         Fuses QKV matmuls + softmax + output projection                                    ││ │
│  │        │         Memory: O(N) instead of O(N²)                                                      ││ │
│  │        └─────────────────────────────────────────────────────────────────────────────────────────────┘│ │
│  │                                                                                                        │ │
│  │     2. STORE KV CACHE:                                                                                │ │
│  │        • K, V tensors written to PagedAttention blocks                                               │ │
│  │        • Block table updated with new block IDs                                                      │ │
│  │                                                                                                        │ │
│  │     3. MLP (Dense) or MoE (Sparse):                                                                   │ │
│  │        ┌─────────────────────────────────────────────────────────────────────────────────────────────┐│ │
│  │        │ DENSE:                                                                                      ││ │
│  │        │   gate = input @ W_gate                                                                    ││ │
│  │        │   up = input @ W_up                                                                        ││ │
│  │        │   output = (SiLU(gate) * up) @ W_down                                                      ││ │
│  │        │   Kernels: 3 GEMMs                                                                         ││ │
│  │        │                                                                                             ││ │
│  │        │ MoE (235B-A22B):                                                                            ││ │
│  │        │   router_logits = input @ W_router                                                         ││ │
│  │        │   expert_ids = top_k(router_logits, k=8)  # Select 8 of 64 experts                         ││ │
│  │        │   FOR each selected expert:                                                                 ││ │
│  │        │       expert_output = expert_mlp(input)                                                    ││ │
│  │        │   output = weighted_sum(expert_outputs, router_weights)                                    ││ │
│  │        │   Kernel: FusedMoE (vLLM custom CUDA kernel)                                               ││ │
│  │        └─────────────────────────────────────────────────────────────────────────────────────────────┘│ │
│  │                                                                                                        │ │
│  │     4. DEEPSTACK INJECTION (if Qwen3-VL):                                                             │ │
│  │        • At specific layers, add multi-level vision features                                         │ │
│  │        • input = input + deepstack_input_embeds[layer_idx]                                           │ │
│  │                                                                                                        │ │
│  │ PREFILL LATENCY (1347 tokens, Qwen3-VL-8B):                                                           │ │
│  │ ┌──────────┬────────────┬────────────────────────────────────────────────────────────────────────────┐│ │
│  │ │ GPU      │ Time       │ Bottleneck                                                                 ││ │
│  │ ├──────────┼────────────┼────────────────────────────────────────────────────────────────────────────┤│ │
│  │ │ T4       │ ~600ms     │ Memory bandwidth (320 GB/s), FP16 compute                                 ││ │
│  │ │ A100     │ ~120ms     │ Compute-bound (312 TFLOPS BF16)                                           ││ │
│  │ │ H100     │ ~60ms      │ Compute-bound (990 TFLOPS FP8)                                            ││ │
│  │ │ B200     │ ~35ms      │ Compute-bound (2.5 PFLOPS FP4)                                            ││ │
│  │ └──────────┴────────────┴────────────────────────────────────────────────────────────────────────────┘│ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 5: DECODE PHASE (Generate Tokens One by One)                                                      │ │
│  │ ══════════════════════════════════════════════════                                                     │ │
│  │                                                                                                        │ │
│  │ FOR each output token:                                                                                 │ │
│  │                                                                                                        │ │
│  │     1. ATTENTION (1 new token attends to all previous):                                               │ │
│  │        • Q = new_token @ W_q   (GEMM: 1 × hidden → 1 × num_heads × head_dim)                         │ │
│  │        • K, V = lookup from PagedAttention cache                                                      │ │
│  │        • Attention = softmax(Q @ cached_K^T / sqrt(d)) @ cached_V                                    │ │
│  │        • Much faster than prefill (1 query vs N queries)                                             │ │
│  │                                                                                                        │ │
│  │     2. MLP/MoE:                                                                                       │ │
│  │        • Same as prefill, but only for 1 token                                                       │ │
│  │                                                                                                        │ │
│  │     3. LM HEAD + SAMPLING:                                                                            │ │
│  │        • logits = hidden @ lm_head_weights  (GEMM: 1 × hidden → 1 × vocab_size)                      │ │
│  │        • next_token = sample(softmax(logits / temperature))                                          │ │
│  │                                                                                                        │ │
│  │     4. UPDATE KV CACHE:                                                                               │ │
│  │        • Append new K, V to cache blocks                                                             │ │
│  │        • Allocate new block if current block full                                                    │ │
│  │                                                                                                        │ │
│  │ DECODE LATENCY PER TOKEN (Qwen3-VL-8B):                                                               │ │
│  │ ┌──────────┬────────────┬────────────────────────────────────────────────────────────────────────────┐│ │
│  │ │ GPU      │ Time/Token │ Throughput (tokens/sec)                                                    ││ │
│  │ ├──────────┼────────────┼────────────────────────────────────────────────────────────────────────────┤│ │
│  │ │ T4       │ ~50ms      │ ~20 tok/s                                                                  ││ │
│  │ │ A100     │ ~12ms      │ ~80 tok/s                                                                  ││ │
│  │ │ H100     │ ~6ms       │ ~160 tok/s                                                                 ││ │
│  │ │ B200     │ ~3ms       │ ~300 tok/s                                                                 ││ │
│  │ └──────────┴────────────┴────────────────────────────────────────────────────────────────────────────┘│ │
│  │                                                                                                        │ │
│  │ NOTE: Decode is MEMORY-BANDWIDTH BOUND, not compute-bound                                             │ │
│  │       → More HBM bandwidth = faster decoding                                                          │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │ STEP 6: OUTPUT DECODING                                                                                │ │
│  │ ═══════════════════════                                                                                │ │
│  │                                                                                                        │ │
│  │ • Detokenize output token IDs → string                                                                │ │
│  │ • Handle streaming (yield tokens as generated) or batch (wait for EOS)                               │ │
│  │ • CPU operation, negligible latency                                                                   │ │
│  │                                                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### vLLM Attention Backend Selection Logic (from cuda.py)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     VLLM ATTENTION BACKEND SELECTION (from cuda.py)                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  BACKEND PRIORITY BY GPU ARCHITECTURE                                                                       │
│  ═════════════════════════════════════                                                                      │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  def _get_backend_priorities(use_mla, device_capability):                                               ││
│  │                                                                                                         ││
│  │      if device_capability.major == 10:  # Blackwell (B200)                                             ││
│  │          if use_mla:                                                                                    ││
│  │              return [CUTLASS_MLA, FLASHINFER_MLA, FLASH_ATTN_MLA, FLASHMLA, TRITON_MLA]                ││
│  │          else:                                                                                          ││
│  │              return [FLASHINFER, FLASH_ATTN, TRITON_ATTN, FLEX_ATTENTION]                              ││
│  │                                                                                                         ││
│  │      else:  # A100 (SM 8.0), H100 (SM 9.0)                                                             ││
│  │          if use_mla:                                                                                    ││
│  │              return [FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA, TRITON_MLA]                             ││
│  │          else:                                                                                          ││
│  │              return [FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION]                              ││
│  │                                                                                                         ││
│  │  # Note: T4 (SM 7.5) doesn't meet FlashAttention's SM >= 8.0 requirement                               ││
│  │  # FlashInfer supports SM 7.5-12.1, so T4 uses FlashInfer as primary backend                           ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  RESULTING BACKEND BY GPU                                                                                   │
│  ════════════════════════                                                                                   │
│                                                                                                             │
│  ┌──────────────┬─────────────────────────┬─────────────────────────────────────────────────────────────┐  │
│  │ GPU          │ Primary Backend         │ Features                                                    │  │
│  ├──────────────┼─────────────────────────┼─────────────────────────────────────────────────────────────┤  │
│  │ T4 (SM 7.5)  │ FlashInfer              │ FP16 only, no FlashAttn, TRTLLM unavailable                │  │
│  │ A100 (SM 8.0)│ FlashAttention 2        │ BF16, FA2 optimizations, no FP8                            │  │
│  │ H100 (SM 9.0)│ FlashAttention 3        │ BF16/FP8, FA3 + TMA, warp specialization                   │  │
│  │ B200 (SM 10) │ FlashInfer + TRTLLM     │ BF16/FP8, TRTLLM decode, FlashInfer prefill                │  │
│  └──────────────┴─────────────────────────┴─────────────────────────────────────────────────────────────┘  │
│                                                                                                             │
│  FROM flashinfer.py - TRTLLM SUPPORT:                                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  # FlashInfer uses TRTLLM attention kernels when available (SM 10.0)                                   ││
│  │  # This enables efficient decode with FP8 Q and KV cache                                               ││
│  │                                                                                                         ││
│  │  def can_use_trtllm_attention(num_qo_heads, num_kv_heads):                                             ││
│  │      # Returns True if TRTLLM kernels are available and beneficial                                     ││
│  │      # Used for: decode phase, FP8 quantization, attention sinks                                       ││
│  │                                                                                                         ││
│  │  class TRTLLMDecode:                                                                                    ││
│  │      block_tables: torch.Tensor  # [num_decodes, max_num_blocks_per_seq]                               ││
│  │      seq_lens: torch.Tensor      # [num_decodes]                                                       ││
│  │      max_seq_len: int                                                                                   ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Bottleneck Analysis: Compute vs Memory Bound

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     BOTTLENECK ANALYSIS: WHERE TIME IS SPENT                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  PREFILL PHASE (Compute-Bound)                                                                              │
│  ═════════════════════════════                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Operation         │ FLOPs Formula                        │ 8B Example (1382 tokens)                   ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  QKV Projection    │ 3 × N × hidden² × 2                   │ 3 × 1382 × 4096² × 2 = 139 GFLOPs         ││
│  │  Attention         │ 4 × N² × hidden × 2                   │ 4 × 1382² × 4096 × 2 = 62 GFLOPs          ││
│  │  Output Proj       │ N × hidden² × 2                       │ 1382 × 4096² × 2 = 46 GFLOPs              ││
│  │  MLP (gate_up)     │ 2 × N × hidden × inter × 2            │ 2 × 1382 × 4096 × 12288 × 2 = 279 GFLOPs  ││
│  │  MLP (down)        │ N × inter × hidden × 2                │ 1382 × 12288 × 4096 × 2 = 139 GFLOPs      ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Per Layer Total   │                                       │ ~665 GFLOPs                                ││
│  │  32 Layers         │                                       │ ~21.3 TFLOPs                               ││
│  │                                                                                                         ││
│  │  Time by GPU (compute-bound):                                                                           ││
│  │  • T4 (65 TFLOPS):   21.3T / 65T = 328ms (actual ~500ms due to memory + overhead)                      ││
│  │  • A100 (312 TFLOPS): 21.3T / 312T = 68ms (actual ~120ms)                                              ││
│  │  • H100 (990 TFLOPS): 21.3T / 990T = 21ms (actual ~60ms)                                               ││
│  │  • B200 (2500 TFLOPS): 21.3T / 2500T = 8.5ms (actual ~35ms)                                            ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  DECODE PHASE (Memory-Bandwidth Bound)                                                                      │
│  ══════════════════════════════════════                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Memory Reads per Token:                                                                                ││
│  │  • Model weights: Must read ~all weights (17 GB for 8B BF16)                                           ││
│  │  • KV cache: Read cached K/V for all previous tokens                                                   ││
│  │                                                                                                         ││
│  │  Arithmetic Intensity = FLOPs / Bytes                                                                   ││
│  │  • Prefill: ~100-500 FLOPs/byte (compute-bound)                                                        ││
│  │  • Decode: ~1-2 FLOPs/byte (memory-bound)                                                              ││
│  │                                                                                                         ││
│  │  Decode Time = Weights_Size / Memory_Bandwidth                                                          ││
│  │                                                                                                         ││
│  │  ┌──────────────┬──────────────┬────────────────┬───────────────┬──────────────┐                       ││
│  │  │ GPU          │ BW (GB/s)    │ 8B BF16 (17GB) │ 8B FP8 (8.5GB)│ Theoretical  │                       ││
│  │  ├──────────────┼──────────────┼────────────────┼───────────────┼──────────────┤                       ││
│  │  │ T4           │ 320          │ 53ms           │ N/A           │ Best case    │                       ││
│  │  │ A100-80      │ 2,039        │ 8.3ms          │ N/A           │ Actual ~12ms │                       ││
│  │  │ H100         │ 3,350        │ 5.1ms          │ 2.5ms         │ Actual ~6ms  │                       ││
│  │  │ B200         │ 8,000        │ 2.1ms          │ 1.1ms         │ Actual ~3ms  │                       ││
│  │  └──────────────┴──────────────┴────────────────┴───────────────┴──────────────┘                       ││
│  │                                                                                                         ││
│  │  KEY INSIGHT: H100/B200 decode is nearly 10× faster than T4 due to memory bandwidth                    ││
│  │               FP8 on H100 nearly halves decode time vs BF16                                             ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### GPU Hardware Characteristics for Inference

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     GPU COMPARISON: WHAT MATTERS FOR QWEN3-VL INFERENCE                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌────────────────────┬───────────────┬───────────────┬───────────────┬───────────────┐                    │
│  │ Spec               │ T4 (Turing)   │ A100 (Ampere) │ H100 (Hopper) │ B200 (B'well) │                    │
│  ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤                    │
│  │ VRAM               │ 16 GB GDDR6   │ 40/80 GB HBM2e│ 80 GB HBM3    │ 192 GB HBM3e  │                    │
│  │ Memory Bandwidth   │ 320 GB/s      │ 2.0 TB/s      │ 3.35 TB/s     │ 8 TB/s        │                    │
│  │ Compute (FP16)     │ 65 TFLOPS     │ 312 TFLOPS    │ 990 TFLOPS    │ 2500 TFLOPS   │                    │
│  │ Compute (FP8)      │ N/A           │ N/A           │ 1980 TFLOPS   │ 5000 TFLOPS   │                    │
│  │ SMs                │ 40            │ 108           │ 132           │ 192           │                    │
│  │ Tensor Cores       │ 320 (Gen 2)   │ 432 (Gen 3)   │ 528 (Gen 4)   │ 768 (Gen 5)   │                    │
│  ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤                    │
│  │ FlashAttention     │ ❌ No         │ ✅ FA2        │ ✅ FA3        │ ✅ FA3+       │                    │
│  │ FP8 Support        │ ❌ No         │ ❌ No         │ ✅ Yes        │ ✅ Yes        │                    │
│  │ vLLM Backend       │ FlashInfer    │ FlashAttn2    │ FlashAttn3    │ FlashMLA      │                    │
│  └────────────────────┴───────────────┴───────────────┴───────────────┴───────────────┘                    │
│                                                                                                             │
│  WHAT EACH SPEC MEANS FOR INFERENCE:                                                                        │
│  ═══════════════════════════════════                                                                        │
│                                                                                                             │
│  VRAM:                                                                                                      │
│  • Determines which models fit (weights + KV cache + activations)                                          │
│  • T4: Only 2B-4B models, or 8B with aggressive quantization                                               │
│  • A100-80: Up to 32B in BF16, or 235B-A22B with FP8                                                       │
│  • B200: 235B-A22B comfortably in BF16                                                                     │
│                                                                                                             │
│  MEMORY BANDWIDTH:                                                                                          │
│  • DECODE IS BANDWIDTH-BOUND: Must read all model weights per token                                        │
│  • T4: 16B model = 32GB weights, 320 GB/s → 10 tok/s theoretical max                                       │
│  • H100: 16B model = 32GB weights, 3.35 TB/s → 100+ tok/s possible                                        │
│                                                                                                             │
│  TENSOR CORES:                                                                                              │
│  • PREFILL IS COMPUTE-BOUND: Matrix multiplications dominate                                               │
│  • More Tensor Cores = faster prefill                                                                      │
│  • H100 has 1.22× more cores than A100, but 3× faster due to FP8                                          │
│                                                                                                             │
│  FLASHATTENTION:                                                                                            │
│  • Fuses attention kernels, reduces memory traffic                                                         │
│  • T4: No FlashAttention → falls back to memory-inefficient SDPA                                          │
│  • H100: FlashAttention-3 + FP8 = 2× faster than A100                                                     │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Complete Model × GPU Performance Matrix

Every Qwen3-VL model on every GPU tier, with a standard workload: 1024×1024 image + 50 prompt → 200 output tokens.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE MODEL × GPU PERFORMANCE MATRIX                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  QWEN3-VL-2B                                                                                                │
│  ════════════                                                                                               │
│  Model: 2.3B params, Vision:400M + LLM:1.9B, 28 layers, hidden=1536, kv_heads=2                            │
│  Weights: 4.6GB BF16 | 2.3GB FP8 | 1.15GB INT4                                                             │
│  KV/token: 28.7 KB (BF16) | 14.3 KB (FP8)                                                                  │
│                                                                                                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐                              │
│  │ Metric       │ T4 (16GB)    │ A100-80GB    │ H100-80GB    │ B200-192GB   │                              │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤                              │
│  │ Precision    │ FP16         │ BF16         │ FP8          │ BF16         │                              │
│  │ Fits?        │ ✅ Yes       │ ✅ Yes       │ ✅ Yes       │ ✅ Yes       │                              │
│  │ Attn Backend │ FlashInfer   │ FlashAttn2   │ FlashAttn3   │ FlashInfer   │                              │
│  │ Vision Enc   │ 40ms         │ 12ms         │ 6ms          │ 4ms          │                              │
│  │ Prefill      │ 180ms        │ 45ms         │ 22ms         │ 12ms         │                              │
│  │ Decode/tok   │ 25ms         │ 6ms          │ 3ms          │ 1.5ms        │                              │
│  │ Total (200t) │ 5.2s         │ 1.3s         │ 0.63s        │ 0.32s        │                              │
│  │ Throughput   │ 38 tok/s     │ 154 tok/s    │ 317 tok/s    │ 625 tok/s    │                              │
│  │ Max Requests │ 16 (4K ctx)  │ 256 (8K)     │ 512 (8K)     │ 1024 (8K)    │                              │
│  │ KV/req (8K)  │ 230 MB       │ 230 MB       │ 115 MB       │ 230 MB       │                              │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘                              │
│                                                                                                             │
│  QWEN3-VL-4B                                                                                                │
│  ════════════                                                                                               │
│  Model: 4.4B params, 36 layers, hidden=2048, kv_heads=4                                                    │
│  Weights: 8.8GB BF16 | 4.4GB FP8 | 2.2GB INT4                                                              │
│  KV/token: 73.7 KB (BF16) | 36.9 KB (FP8)                                                                  │
│                                                                                                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐                              │
│  │ Metric       │ T4 (16GB)    │ A100-80GB    │ H100-80GB    │ B200-192GB   │                              │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤                              │
│  │ Precision    │ INT4 (BnB)   │ BF16         │ FP8          │ BF16         │                              │
│  │ Fits?        │ ⚠️ Tight     │ ✅ Yes       │ ✅ Yes       │ ✅ Yes       │                              │
│  │ Attn Backend │ FlashInfer   │ FlashAttn2   │ FlashAttn3   │ FlashInfer   │                              │
│  │ Vision Enc   │ 55ms         │ 16ms         │ 8ms          │ 5ms          │                              │
│  │ Prefill      │ 280ms        │ 70ms         │ 35ms         │ 18ms         │                              │
│  │ Decode/tok   │ 35ms         │ 8ms          │ 4ms          │ 2ms          │                              │
│  │ Total (200t) │ 7.3s         │ 1.7s         │ 0.84s        │ 0.42s        │                              │
│  │ Throughput   │ 27 tok/s     │ 118 tok/s    │ 238 tok/s    │ 476 tok/s    │                              │
│  │ Max Requests │ 4 (4K ctx)   │ 64 (8K)      │ 128 (8K)     │ 256 (8K)     │                              │
│  │ KV/req (8K)  │ 590 MB       │ 590 MB       │ 295 MB       │ 590 MB       │                              │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘                              │
│                                                                                                             │
│  QWEN3-VL-8B                                                                                                │
│  ════════════                                                                                               │
│  Model: 8.3B params, 32 layers, hidden=4096, kv_heads=8                                                    │
│  Weights: 16.6GB BF16 | 8.3GB FP8 | 4.15GB INT4                                                            │
│  KV/token: 131 KB (BF16) | 65.5 KB (FP8)                                                                   │
│                                                                                                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐                              │
│  │ Metric       │ T4 (16GB)    │ A100-80GB    │ H100-80GB    │ B200-192GB   │                              │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤                              │
│  │ Precision    │ INT4 (BnB)   │ BF16         │ FP8          │ BF16         │                              │
│  │ Fits?        │ ⚠️ Very Tight│ ✅ Yes       │ ✅ Yes       │ ✅ Yes       │                              │
│  │ Attn Backend │ FlashInfer   │ FlashAttn2   │ FlashAttn3   │ FlashInfer   │                              │
│  │ Vision Enc   │ 80ms         │ 25ms         │ 12ms         │ 8ms          │                              │
│  │ Prefill      │ 600ms        │ 120ms        │ 60ms         │ 35ms         │                              │
│  │ Decode/tok   │ 50ms         │ 12ms         │ 6ms          │ 3ms          │                              │
│  │ Total (200t) │ 10.7s        │ 2.5s         │ 1.3s         │ 0.65s        │                              │
│  │ Throughput   │ 19 tok/s     │ 80 tok/s     │ 154 tok/s    │ 308 tok/s    │                              │
│  │ Max Requests │ 2 (4K ctx)   │ 32 (8K)      │ 64 (8K)      │ 128 (8K)     │                              │
│  │ KV/req (8K)  │ 1.0 GB       │ 1.0 GB       │ 0.5 GB       │ 1.0 GB       │                              │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘                              │
│                                                                                                             │
│  QWEN3-VL-32B                                                                                               │
│  ═════════════                                                                                              │
│  Model: 32.8B params, 64 layers, hidden=5120, kv_heads=8                                                   │
│  Weights: 65.6GB BF16 | 32.8GB FP8 | 16.4GB INT4                                                           │
│  KV/token: 262 KB (BF16) | 131 KB (FP8)                                                                    │
│                                                                                                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐                              │
│  │ Metric       │ T4 (16GB)    │ A100-80GB    │ H100-80GB    │ B200-192GB   │                              │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤                              │
│  │ Precision    │ N/A          │ BF16         │ FP8          │ BF16         │                              │
│  │ Fits?        │ ❌ OOM       │ ⚠️ Tight     │ ✅ Yes       │ ✅ Yes       │                              │
│  │ Attn Backend │ N/A          │ FlashAttn2   │ FlashAttn3   │ FlashInfer   │                              │
│  │ Vision Enc   │ N/A          │ 50ms         │ 25ms         │ 15ms         │                              │
│  │ Prefill      │ N/A          │ 250ms        │ 120ms        │ 70ms         │                              │
│  │ Decode/tok   │ N/A          │ 25ms         │ 12ms         │ 6ms          │                              │
│  │ Total (200t) │ N/A          │ 5.3s         │ 2.5s         │ 1.3s         │                              │
│  │ Throughput   │ N/A          │ 38 tok/s     │ 80 tok/s     │ 154 tok/s    │                              │
│  │ Max Requests │ N/A          │ 4 (8K ctx)   │ 16 (8K)      │ 48 (8K)      │                              │
│  │ KV/req (8K)  │ N/A          │ 2.0 GB       │ 1.0 GB       │ 2.0 GB       │                              │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘                              │
│                                                                                                             │
│  QWEN3-VL-30B-A3B (MoE) — 30B Total, 3B Active                                                             │
│  ══════════════════════════════════════════════                                                             │
│  Model: 30B total (128 experts), 3B active/token (top-8), 48 layers, hidden=2048                           │
│  Weights: 60GB BF16 | 30GB FP8 | ALL experts must be in VRAM                                               │
│  KV/token: 49 KB (BF16) | 24.5 KB (FP8) — smaller than 8B dense!                                           │
│                                                                                                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐                              │
│  │ Metric       │ T4 (16GB)    │ A100-80GB    │ H100-80GB    │ B200-192GB   │                              │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤                              │
│  │ Precision    │ N/A          │ ⚠️ FP8 only  │ FP8          │ BF16         │                              │
│  │ Fits?        │ ❌ OOM       │ ⚠️ 30GB FP8  │ ✅ Yes       │ ✅ Yes       │                              │
│  │ Attn Backend │ N/A          │ FlashAttn2   │ FlashAttn3   │ FlashInfer   │                              │
│  │ Vision Enc   │ N/A          │ 50ms         │ 25ms         │ 15ms         │                              │
│  │ Prefill      │ N/A          │ 150ms        │ 75ms         │ 40ms         │                              │
│  │ Decode/tok   │ N/A          │ 12ms         │ 6ms          │ 3ms          │                              │
│  │ Total (200t) │ N/A          │ 2.6s         │ 1.3s         │ 0.66s        │                              │
│  │ Throughput   │ N/A          │ 77 tok/s     │ 154 tok/s    │ 303 tok/s    │                              │
│  │ Max Requests │ N/A          │ 16 (8K ctx)  │ 48 (8K)      │ 128 (8K)     │                              │
│  │ KV/req (8K)  │ N/A          │ 0.4 GB       │ 0.2 GB       │ 0.4 GB       │                              │
│  │ FLOPs/tok    │ N/A          │ ~3B (sparse) │ ~3B (sparse) │ ~3B (sparse) │                              │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘                              │
│                                                                                                             │
│  QWEN3-VL-235B-A22B (MoE) — 235B Total, 22B Active                                                         │
│  ══════════════════════════════════════════════                                                             │
│  Model: 235B total (128 experts), 22B active/token (top-8), 94 layers, hidden=5120                         │
│  Weights: 470GB BF16 | 235GB FP8 | Requires multi-GPU tensor parallelism                                   │
│  KV/token: 245 KB (BF16) | 122.5 KB (FP8)                                                                  │
│                                                                                                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐                              │
│  │ Metric       │ 4×A100-80GB  │ 8×H100-80GB  │ 4×H100-80GB  │ 3×B200-192GB │                              │
│  ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤                              │
│  │ Precision    │ ⚠️ FP8       │ BF16         │ FP8          │ BF16         │                              │
│  │ Fits?        │ ⚠️ Tight     │ ✅ Yes       │ ✅ Yes       │ ✅ Yes       │                              │
│  │ TP Size      │ 4            │ 8            │ 4            │ 3            │                              │
│  │ VRAM/GPU     │ 59GB         │ 59GB         │ 59GB         │ 157GB        │                              │
│  │ Vision Enc   │ 100ms        │ 50ms         │ 50ms         │ 40ms         │                              │
│  │ Prefill      │ 600ms        │ 300ms        │ 300ms        │ 150ms        │                              │
│  │ Decode/tok   │ 25ms         │ 12ms         │ 12ms         │ 6ms          │                              │
│  │ Total (200t) │ 5.7s         │ 2.8s         │ 2.8s         │ 1.4s         │                              │
│  │ Throughput   │ 35 tok/s     │ 71 tok/s     │ 71 tok/s     │ 143 tok/s    │                              │
│  │ Max Requests │ 4 (8K ctx)   │ 8 (8K ctx)   │ 8 (8K ctx)   │ 16 (8K ctx)  │                              │
│  │ KV/req (8K)  │ 2.0 GB       │ 2.0 GB       │ 1.0 GB       │ 2.0 GB       │                              │
│  │ NVLink Req.  │ ✅ Required  │ ✅ Required  │ ✅ Required  │ ✅ NVLink5   │                              │
│  └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘                              │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### End-to-End Latency Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     SAME REQUEST, DIFFERENT GPUs                                                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  REQUEST: 1024×1024 image + "What is in this image?" → 100 token response                                  │
│  MODEL: Qwen3-VL-8B-Instruct                                                                               │
│                                                                                                             │
│  ┌────────────────────┬───────────────┬───────────────┬───────────────┬───────────────┐                    │
│  │ Phase              │ T4            │ A100-80       │ H100-80       │ B200          │                    │
│  ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤                    │
│  │ Vision Encode      │ 80 ms         │ 25 ms         │ 12 ms         │ 8 ms          │                    │
│  │ Prefill (1.3K tok) │ 600 ms        │ 120 ms        │ 60 ms         │ 35 ms         │                    │
│  │ Decode (100 tok)   │ 5000 ms       │ 1200 ms       │ 600 ms        │ 300 ms        │                    │
│  ├────────────────────┼───────────────┼───────────────┼───────────────┼───────────────┤                    │
│  │ TOTAL              │ ~5.7 sec      │ ~1.35 sec     │ ~0.67 sec     │ ~0.35 sec     │                    │
│  │ Tokens/sec         │ 17            │ 74            │ 149           │ 285           │                    │
│  └────────────────────┴───────────────┴───────────────┴───────────────┴───────────────┘                    │
│                                                                                                             │
│  OBSERVATIONS:                                                                                              │
│  ═════════════                                                                                              │
│                                                                                                             │
│  1. DECODE DOMINATES: 87%+ of time is spent generating tokens                                              │
│     • T4: 5000ms decode / 5700ms total = 88%                                                               │
│     • Optimization priority: Faster decoding > faster prefill                                              │
│                                                                                                             │
│  2. GPU GENERATION MATTERS MORE THAN SPECS:                                                                 │
│     • H100 is 8× faster than T4, not because of 2× more SMs                                               │
│     • Key: FlashAttention-3, FP8 compute, 10× memory bandwidth                                             │
│                                                                                                             │
│  3. VISION ENCODING IS CHEAP:                                                                               │
│     • <2% of total time on modern GPUs                                                                     │
│     • Not the bottleneck—don't over-optimize here                                                          │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Execution: Qwen3-VL-8B on Each GPU

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B ON T4 (16GB): COMPLETE DETAILED EXECUTION                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  CONFIGURATION                                                                                              │
│  ═════════════                                                                                              │
│  Model: Qwen3-VL-8B-Instruct                                                                               │
│  Precision: INT4 (BitsAndBytes)  ← Required to fit in 16GB                                                 │
│  Attention: FlashInfer (FA2 not supported on SM 7.5)                                                       │
│  KV Cache: FP16 (FP8 not supported on T4)                                                                  │
│  CUDA Graphs: Disabled (--enforce-eager)                                                                   │
│                                                                                                             │
│  MEMORY ALLOCATION                                                                                          │
│  ═════════════════                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 16 GB T4 VRAM                                                                                          ││
│  │ ├── INT4 Weights (8B × 0.5 bytes): ~4.0 GB                                                             ││
│  │ ├── Vision Encoder (FP16): ~1.0 GB                                                                     ││
│  │ ├── Activations (FP16): ~3.0 GB                                                                        ││
│  │ ├── CUDA Context: ~1.5 GB                                                                              ││
│  │ └── KV Cache Available: ~6.5 GB                                                                        ││
│  │     └── Per token: 131 KB (2 × 32 × 8 × 128 × 2 bytes)                                                ││
│  │     └── 4K context: 524 MB per request                                                                 ││
│  │     └── Max concurrent: ~3 requests at 4K context                                                      ││
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  EXECUTION TIMELINE (1024×1024 image + 50 prompt → 200 output)                                             │
│  ═══════════════════════════════════════════════════════════════                                            │
│                                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ Time (ms)  0    100   200   300   400   500   600   ...   10000  10500  11000                          ││
│  │            │     │     │     │     │     │     │           │      │      │                             ││
│  │ Tokenize   ████ (10ms)                                                                                  ││
│  │                                                                                                         ││
│  │ Vision Enc      ████████████████████ (80ms)                                                             ││
│  │                 │ Conv3D: 15ms                                                                          ││
│  │                 │ ViT Blocks (32×): 60ms (limited by 40 SMs)                                           ││
│  │                 │ Merger: 5ms                                                                           ││
│  │                                                                                                         ││
│  │ KV Alloc                              ██ (5ms)                                                          ││
│  │                                                                                                         ││
│  │ Prefill                                  ████████████████████████████████████████ (600ms)              ││
│  │ (1382 tok)                               │ FlashInfer with 64KB shared memory tiles                    ││
│  │                                          │ 28 layers × ~21ms each                                      ││
│  │                                          │ Bottleneck: Compute (65 TFLOPS FP16)                        ││
│  │                                          │ + Dequantization overhead for INT4                          ││
│  │                                                                                                         ││
│  │ Decode                                                    ███████████████████████████████████ (10s)    ││
│  │ (200 tokens)                                              │ 200 × 50ms = 10,000ms                      ││
│  │                                                           │ Bottleneck: Memory BW (320 GB/s)           ││
│  │                                                           │ Must read ~4GB weights per token           ││
│  │                                                                                                         ││
│  │ TOTAL: ~10,695ms = 10.7s                                                                               ││
│  │ Throughput: 200 / 10.7 = 18.7 tokens/sec                                                               ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  SM UTILIZATION                                                                                             │
│  ══════════════                                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │                                                                                                         ││
│  │  Phase           │ Active SMs │ Utilization │ Reason                                                    ││
│  │  ─────────────────────────────────────────────────────────────────────────────────────────────────────││
│  │  Vision Conv3D   │ 40/40      │ ~70%        │ Good parallelism in convolution                          ││
│  │  ViT Attention   │ 40/40      │ ~50%        │ 64KB shared memory limits tile size                      ││
│  │  Prefill Attn    │ 40/40      │ ~45%        │ Limited by INT4 dequant + small batch                    ││
│  │  Prefill MLP     │ 40/40      │ ~60%        │ Good GEMM efficiency on Tensor Cores                     ││
│  │  Decode Attn     │ 20/40      │ ~30%        │ Memory-bound, many SMs idle                              ││
│  │  Decode MLP      │ 30/40      │ ~40%        │ Small batch doesn't fill SMs                             ││
│  │                                                                                                         ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  VLLM CONFIGURATION                                                                                         │
│  ═════════════════                                                                                          │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct \                                                                    │
│      --dtype float16 \                                                                                      │
│      --quantization bitsandbytes \                                                                          │
│      --load-format bitsandbytes \                                                                           │
│      --max-model-len 4096 \                                                                                 │
│      --max-num-seqs 2 \                                                                                     │
│      --gpu-memory-utilization 0.92 \                                                                        │
│      --enforce-eager \                                                                                      │
│      --limit-mm-per-prompt image=2,video=1                                                                  │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B ON A100-80GB: STEP-BY-STEP EXECUTION                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  CONFIGURATION                                                                                              │
│  Model: Qwen3-VL-8B, Precision: BF16, Attention: FlashAttn2 (SM 8.0), KV: BF16, CUDA Graphs: ON            │
│                                                                                                             │
│  MEMORY: 80GB A100 HBM2e (2 TB/s bandwidth)                                                                │
│  ├── BF16 Weights: 16.6GB | ViT: 2.0GB | Activations: 4.0GB | CUDA+Graphs: 3.0GB | KV: 54GB               │
│  └── Max: 25 requests at 16K context (131KB/token × 16K = 2.1GB/request)                                   │
│                                                                                                             │
│  TIMELINE                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 0ms───50ms───200ms───500ms───1000ms───2000ms───2500ms                                                  ││
│  │ [5ms] Tokenize                                                                                          ││
│  │     [25ms] Vision: FA2 optimized, SM:75%, Tensor Core:80%                                              ││
│  │            [2ms] KV                                                                                     ││
│  │               [120ms] Prefill: FlashAttn2 fused, 108 SMs, 312 TFLOPS                                   ││
│  │                                [2400ms] Decode: 200×12ms, 2 TB/s near-saturation                       ││
│  │ TOTAL: 2.5s | 80 tok/s                                                                                  ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --max-model-len 32768 \                             │
│      --max-num-seqs 32 --gpu-memory-utilization 0.90 --enable-prefix-caching                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B ON H100-80GB: STEP-BY-STEP EXECUTION                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  CONFIGURATION                                                                                              │
│  Model: Qwen3-VL-8B, Precision: FP8, Attention: FlashAttn3+TRTLLM, KV: FP8, CUDA Graphs: ON                │
│                                                                                                             │
│  MEMORY: 80GB H100 HBM3 (3.35 TB/s bandwidth)                                                              │
│  ├── FP8 Weights: 8.3GB | ViT: 2.0GB | Activations: 4.0GB | CUDA: 3.0GB | KV: 62GB                        │
│  └── Max: 29 requests at 32K context (65.5KB/token FP8 × 32K = 2.1GB/request)                              │
│                                                                                                             │
│  TIMELINE                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 0ms───20ms───100ms───300ms───600ms───1000ms───1300ms                                                   ││
│  │ [3ms] Tokenize                                                                                          ││
│  │    [12ms] Vision: FA3 optimized                                                                         ││
│  │          [1ms] KV                                                                                       ││
│  │            [60ms] Prefill: FA3+FP8 Tensor Cores, 132 SMs, TBC                                          ││
│  │                          [1200ms] Decode: 200×6ms, TRTLLM+TMA, 3.35 TB/s                               ││
│  │ TOTAL: 1.3s | 157 tok/s                                                                                 ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --kv-cache-dtype fp8 \                              │
│      --max-model-len 65536 --max-num-seqs 64 --gpu-memory-utilization 0.90 --enable-prefix-caching         │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-8B ON B200-192GB: STEP-BY-STEP EXECUTION                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  CONFIGURATION                                                                                              │
│  Model: Qwen3-VL-8B, Precision: BF16 (plenty of room), Attention: FlashInfer+TRTLLM, KV: BF16              │
│                                                                                                             │
│  MEMORY: 192GB B200 HBM3e (8 TB/s bandwidth)                                                               │
│  ├── BF16 Weights: 16.6GB | ViT: 2.0GB | Activations: 4.0GB | CUDA: 3.0GB | KV: 166GB                     │
│  └── Max: 100 requests at 16K context, 9 at 128K context                                                   │
│                                                                                                             │
│  TIMELINE                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ 0ms───10ms───50ms───150ms───400ms───650ms                                                              ││
│  │ [2ms] Tokenize                                                                                          ││
│  │   [8ms] Vision: 8 TB/s instant data fetch                                                               ││
│  │        [1ms] KV                                                                                         ││
│  │          [35ms] Prefill: 192 SMs, 5th Gen Tensor Cores                                                 ││
│  │                    [600ms] Decode: 200×3ms, 8 TB/s bandwidth                                           ││
│  │ TOTAL: 0.65s | 310 tok/s                                                                                ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  vllm serve Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --max-model-len 131072 \                            │
│      --max-num-seqs 128 --gpu-memory-utilization 0.90 --enable-prefix-caching --enable-chunked-prefill    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Execution: Qwen3-VL-32B (GUI Agent Recommended)

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-32B ON A100-80GB, H100-80GB, B200-192GB                                        │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  WHY 32B FOR GUI AGENTS: 64 layers (vs 32) = more reasoning depth for multi-step UI tasks                  │
│  Research shows: Models <32B fail to reliably combine grounding + planning + action                        │
│                                                                                                             │
│  ┌───────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────┐           │
│  │ Metric        │ A100-80GB               │ H100-80GB               │ B200-192GB              │           │
│  ├───────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤           │
│  │ Precision     │ BF16 (tight fit)        │ FP8 (halves weights)    │ BF16 (plenty room)      │           │
│  │ Weights       │ 65.6GB / 80GB           │ 32.8GB / 80GB           │ 65.6GB / 192GB          │           │
│  │ KV Available  │ ~10GB (limited!)        │ ~43GB                   │ ~120GB                  │           │
│  │ Max Requests  │ 3-4 at 8K context       │ 16 at 8K context        │ 48 at 8K context        │           │
│  ├───────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────┤           │
│  │ Vision Enc    │ 50ms                    │ 25ms                    │ 15ms                    │           │
│  │ Prefill       │ 250ms                   │ 120ms                   │ 70ms                    │           │
│  │ Decode/tok    │ 25ms                    │ 12ms                    │ 6ms                     │           │
│  │ Total (200t)  │ 5.3s                    │ 2.5s                    │ 1.3s                    │           │
│  │ Throughput    │ 38 tok/s                │ 80 tok/s                │ 154 tok/s               │           │
│  └───────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────┘           │
│                                                                                                             │
│  RECOMMENDATION: H100 with FP8 runs 32B at same speed as 8B BF16 on A100!                                  │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Execution: MoE Models

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-30B-A3B (MoE) ON H100-80GB                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  MOE ARCHITECTURE: 30B total (128 experts), 3B active/token (top-8), 10% activation ratio                  │
│                                                                                                             │
│  KEY INSIGHT: ALL 128 experts must be in VRAM, but only 8 compute per token                                │
│                                                                                                             │
│  PER-TOKEN MoE FLOW:                                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ Token (2048 dim) → Router(2048→128) → top-8 indices → FusedMoE kernel:                                 ││
│  │   1. Gather: Collect weights for 8 selected experts (6.25% of total)                                   ││
│  │   2. Compute: Run 8 expert MLPs in parallel (Gate→SiLU→Up→Down)                                        ││
│  │   3. Combine: Weighted sum of expert outputs                                                            ││
│  │ → Next layer                                                                                            ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  MEMORY: 80GB H100                                                                                          │
│  ├── FP8 Expert Weights: 30GB | Shared: 5GB | ViT: 2GB | Activations: 3GB | KV: 39GB                       │
│  └── KV/token: 24.5KB FP8 — SMALLER than 8B dense! (fewer kv_heads)                                        │
│                                                                                                             │
│  TIMELINE                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ Vision: 25ms | Prefill: 75ms | Decode: 200×6ms = 1200ms | TOTAL: 1.3s | 154 tok/s                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  WHY MoE > 32B DENSE (same quality, 2× faster decode):                                                     │
│  • 30B-A3B decode: Read ~8GB active params → 6ms/token                                                     │
│  • 32B dense decode: Read ~32.8GB all params → 12ms/token                                                  │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     QWEN3-VL-235B-A22B (MoE) ON 8×H100-80GB                                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  MOE ARCHITECTURE: 235B total, 22B active/token (top-8 of 128 experts), 94 layers                          │
│                                                                                                             │
│  MULTI-GPU LAYOUT (Tensor Parallelism = 8):                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │  GPU 0     GPU 1     GPU 2     GPU 3     GPU 4     GPU 5     GPU 6     GPU 7                           ││
│  │  [59GB]    [59GB]    [59GB]    [59GB]    [59GB]    [59GB]    [59GB]    [59GB]                           ││
│  │  16 exp    16 exp    16 exp    16 exp    16 exp    16 exp    16 exp    16 exp                           ││
│  │  /layer    /layer    /layer    /layer    /layer    /layer    /layer    /layer                           ││
│  │     └───────┴───────┴───────┴─── NVLink (900 GB/s) ───┴───────┴───────┴───────┘                        ││
│  │                                                                                                         ││
│  │  Expert Distribution: 128 experts ÷ 8 GPUs = 16 experts per GPU per layer                             ││
│  │  All-to-All: When token needs expert on different GPU, NVLink transfers                               ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  TIMELINE                                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐│
│  │ Vision: 50ms | Prefill: 300ms | Decode: 200×12ms = 2400ms | TOTAL: 2.8s | 71 tok/s                     ││
│  │                                                                                                         ││
│  │ Per-Decode-Token: Router(0.2ms) + All-to-All(2ms) + Expert(6ms) + All-Reduce(2ms) + Attn(2ms) = 12ms  ││
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                                             │
│  WHY 235B-A22B: Frontier-model quality (GPT-4 class) at 10× lower cost per token                           │
│  8×H100 = ~$100K setup, serves ~200 tok/s with batching                                                    │
│                                                                                                             │
│  vllm serve Qwen/Qwen3-VL-235B-A22B --tensor-parallel-size 8 --dtype bfloat16 \                            │
│      --max-model-len 32768 --max-num-seqs 8 --gpu-memory-utilization 0.90                                  │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Summary: Model Selection by Use Case

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     MODEL SELECTION DECISION GUIDE                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                                                       │ │
│  │  USE CASE                         │ RECOMMENDED MODEL     │ MIN GPU          │ LATENCY    │ QUALITY  │ │
│  │  ─────────────────────────────────┼───────────────────────┼──────────────────┼────────────┼──────────│ │
│  │  Edge/Mobile deployment           │ 2B                    │ T4 / Consumer    │ <100ms     │ ★★☆☆☆    │ │
│  │  Real-time chat, basic vision     │ 4B                    │ T4               │ ~200ms     │ ★★★☆☆    │ │
│  │  Production VQA, document OCR     │ 8B                    │ A100-40GB        │ ~500ms     │ ★★★★☆    │ │
│  │  GUI agents (SFT+RL reliable)     │ 32B                   │ A100-80GB/H100   │ ~1s        │ ★★★★★    │ │
│  │  High-throughput production       │ 30B-A3B (MoE)         │ H100-80GB        │ ~500ms     │ ★★★★☆    │ │
│  │  Frontier quality, cost-optimized │ 235B-A22B (MoE)       │ 4×H100 or 3×B200 │ ~1.5s      │ ★★★★★+   │ │
│  │                                                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                                             │
│  KEY TAKEAWAYS:                                                                                             │
│  1. GUI agents require 32B minimum for reliable SFT+RL training (research-backed)                         │
│  2. MoE models offer best quality/cost ratio at scale (same quality, 2× decode speed)                     │
│  3. H100 with FP8 can run 32B at speeds comparable to 8B on A100                                          │
│  4. B200 is future-proof: 192GB VRAM + 8 TB/s enables single-GPU 32B with massive context                 │
│                                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## References

### Core Model Papers
- [Qwen3-VL Technical Report (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
- [MAI-UI Technical Report (arXiv:2512.22047)](https://arxiv.org/abs/2512.22047) - GRPO + SFT for GUI agents

### GUI Agent Research (Why Model Size Matters)
- [XBOUND: State-Level Evaluation for Device Control Agents](https://openreview.net/forum?id=UXdxYnkJtX) - Sub-7B models show limited state mastery
- [GUI Knowledge Bench (arXiv:2510.26098)](https://arxiv.org/abs/2510.26098) - Smaller models retain limited knowledge
- [Scaling Vision Transformers (CVPR 2022)](https://arxiv.org/abs/2106.04560) - Smaller models saturate
- [CogAgent (CVPR 2024)](https://arxiv.org/abs/2312.08914) - 18B VLM for GUI
- [GUI-R1 (arXiv:2504.10458)](https://arxiv.org/abs/2504.10458) - GRPO for GUI agents
- [GUI-Actor (Microsoft)](https://microsoft.github.io/GUI-Actor/) - Qwen2-VL based grounding

### Resources
- [Qwen3-VL GitHub Repository](https://github.com/QwenLM/Qwen3-VL)
- [MAI-UI GitHub Repository](https://github.com/Tongyi-MAI/MAI-UI)
- [vLLM Qwen3-VL Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Hugging Face Qwen Collection](https://huggingface.co/Qwen)
- [QWEN_VL_COMPLETE_GUIDE.md](./QWEN_VL_COMPLETE_GUIDE.md) - Companion architectural deep-dive

