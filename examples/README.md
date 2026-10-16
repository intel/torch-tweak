<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->
# Torch Tweak Examples

This directory contains practical examples demonstrating how to use Torch Tweak to tune different types of AI models for inference performance.

## Minimal

### Smallest tuning demo

Tunes a tiny MLP and selects the faster of Torch Inductor and Torch Eager. This is the shortest example in the tree.

- **Location**: [`Minimal`](./Minimal/README.md)
- **Model**: a two-layer MLP defined in the script
- **Use Case**: The shortest path from a PyTorch module to a tuned backend
- **Key Features**:
  - One script, no dataset and no checkpoint
  - Highest-throughput backend selection
  - Intel XPU when available, otherwise CPU

## ResNet

### Computer Vision - Image Classification

Shows how to tune ResNet models for image classification tasks. This example demonstrates model tuning and inference tuning for convolutional neural networks.

- **Location**: [`ResNet`](./ResNet/README.md)
- **Model**: ResNet50 image classification
- **Use Case**: Optimizing CNN models for computer vision tasks
- **Key Features**:
  - Model tuning with Torch Tweak
  - Image classification inference
  - Performance comparison before/after tuning
- **More Info**:
  - <https://huggingface.co/microsoft/resnet-50>

## StableDiffusion

### Generative AI - Text-to-Image

Demonstrates tuning of Stable Diffusion models for text-to-image generation. This example shows how to tune diffusion models for faster and more efficient image generation.

- **Location**: [`StableDiffusion`](./StableDiffusion/README.md)
- **Model**: Stable Diffusion 1.5 from HuggingFace (default in the example)
- **Use Case**: Optimizing text-to-image diffusion models on Intel XPU
- **Key Features**:
  - Diffusion pipeline tuning (Inductor, OpenVINO, Eager)
  - Hybrid vs single-backend benchmark
  - Text prompt-based image synthesis
- **More Info**:
  - <https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5>

## FLUX

### Generative AI - Advanced Text-to-Image

Shows tuning of the FLUX text-to-image model, demonstrating advanced diffusion model tuning techniques for high-quality image generation.

- **Location**: [`FLUX`](./FLUX/README.md)
- **Model**: FLUX.2-klein-4B from Black Forest Labs
- **Use Case**: Optimizing state-of-the-art text-to-image models
- **Key Features**:
  - Advanced diffusion model tuning
  - High-quality image generation
  - Efficient inference pipeline tuning
- **More Info**:
  - <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>

## ParakeetCTC

### Speech AI - Automatic Speech Recognition

Demonstrates tuning of ASR (Automatic Speech Recognition) models using NVIDIA's Parakeet CTC model for speech-to-text conversion.

- **Location**: [`ParakeetCTC`](./ParakeetCTC/README.md)
- **Model**: NVIDIA Parakeet CTC 0.6B
- **Use Case**: Optimizing speech recognition models
- **Key Features**:
  - ASR model tuning
  - Audio-to-text transcription
  - NVIDIA NeMo framework integration
- **More Info**:
  - <https://huggingface.co/nvidia/parakeet-ctc-0.6b>
  - <https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html>

## ParakeetRNNT

### Speech AI - Automatic Speech Recognition

Demonstrates tuning of ASR (Automatic Speech Recognition) models using NVIDIA's Parakeet RNNT model for speech-to-text conversion.

- **Location**: [`ParakeetRNNT`](./ParakeetRNNT/README.md)
- **Model**: NVIDIA Parakeet RNNT 1.1B
- **Use Case**: Optimizing speech recognition models
- **Key Features**:
  - ASR model tuning
  - Audio-to-text transcription
  - NVIDIA NeMo framework integration
- **More Info**:
  - <https://huggingface.co/nvidia/parakeet-rnnt-1.1b>
  - <https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html>

## ESM2

### Text AI - Advanced Text Embedding

Demonstrates tuning of ESM2 model for text embedding tasks.

- **Location**: [`ESM2`](./ESM2/README.md)
- **Model**: ESM2 from HuggingFace
- **Use Case**: Optimizing text embedding models
- **Key Features**:
  - Text embedding tuning
  - Text embedding inference
  - HuggingFace integration
- **More Info**:
  - <https://huggingface.co/esm/esm2-t12-100M-UR50S>

## E5Large

### Text AI - Advanced Text Embedding

Demonstrates tuning of E5Large model for text embedding tasks.

- **Location**: [`E5Large`](./E5Large/README.md)
- **Model**: E5Large from HuggingFace
- **Use Case**: Optimizing text embedding models
- **Key Features**:
  - Text embedding tuning
  - Text embedding inference
  - HuggingFace integration
- **More Info**:
  - <https://huggingface.co/intfloat/e5-large-v2>

## LLM

### Large Language Models - Text Generation

Demonstrates tuning of Large Language Models for text generation tasks. This example shows how to optimize LLMs for efficient inference with KV cache support.

- **Location**: [`LLM`](./LLM/README.md)
- **Model**: Microsoft Phi-3-mini-4k-instruct from HuggingFace
- **Use Case**: Optimizing LLMs for text generation and inference
- **Key Features**:
  - LLM model tuning with Torch Tweak
  - Static and dynamic KV cache optimization
  - Prefill and decode phase optimization
  - HuggingFace integration
- **More Info**:
  - <https://huggingface.co/microsoft/Phi-3-mini-4k-instruct>

---

Each example includes:

- Complete setup instructions
- Usage examples with CLI commands
- Model-specific tuning parameters

To get started, navigate to any example directory and follow the README instructions for that specific model type.
