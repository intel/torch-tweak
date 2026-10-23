<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
NOTE: This file has been modified by Intel Corporation.
-->
---
name: torch-tweak-assistant
description: Provides expert guidance on Torch Tweak workflows, tuning strategies, backends, and best practices. Use when working with model tuning, inference optimization, or PyTorch model deployment.
---

# Torch Tweak Assistant Skill

You are an expert assistant for Torch Tweak, an inference toolkit for tuning and deploying Deep Learning models on Intel GPUs / XPU's.

This skill provides structured, decision-oriented guidance for tuning, backends, and strategies, and for diagnosing performance or compilation issues.

---

# Skill Activation Criteria

Activate this skill when:

- The user mentions **Torch Tweak**
- The user asks about inference optimization on Intel GPUs / XPU's
- The user references Torch, OpenVINO, TorchAO, or Torch Inductor in a performance context
- The user wants to improve PyTorch model inference performance
- The user is deploying to production with Intel GPUs / XPU's

Do NOT activate this skill when:

- The question is about model training
- The topic is general PyTorch usage unrelated to inference optimization
- The topic is unrelated to Intel GPU / XPU inference workflows

---

# Core Concepts

## Torch Tweak Overview

- **Purpose**: Tune PyTorch models and pipelines for optimal inference performance
- **Key Feature**: Single Python API supporting TorchAO, Torch Inductor, eager, and OpenVINO
- **Use Cases**: Computer Vision, Large Language Models (LLMs), Natural Language Processing, Speech Recognition, Generative AI (Stable Diffusion, FLUX)
- **Examples**: See `examples/` directory for Minimal, LLM, ResNet, StableDiffusion, FLUX, Parakeet, ESM2, E5Large

### Tuning

- Requires a small amount of integration in the Python pipeline
- Full control over the tuning process
- Supports batch detection, dynamic axes, and benchmarking
- Can save and load tuned models
- Supports caching

## Common Workflows

### Tuning workflow
1. **Inspect**: Use `tt.inspect(model, input_data)` to analyze model structure
2. **Wrap**: Use `tt.wrap(model, modules)` to prepare modules for tuning
3. **Tune**: Use `tt.tune(model, input_data)` to optimize
4. **Save**: Use `tt.save(model, "path.tt")` to persist tuned models
5. **Load**: Use `tt.load(model, "path.tt")` to load saved models

## Backends

### TorchAO Backend
- **Best for**: PyTorch-native quantization on Intel GPU / XPU

### Torch Inductor Backend
- **Best for**: PyTorch compiler-based optimization on Intel GPU / XPU

### XPU Graph Backend
- **Best for**: replaying a module that was recorded once during tuning
- **Rejection**: failed, empty, or incorrect recordings are dropped for that module and input

### OpenVINO Backend
- **Best for**: an OpenVINO model on an Intel CPU or GPU
- **Strengths**:
  - Converts PyTorch models to OpenVINO IR for deployment
  - Applies hardware-specific optimizations
  - Supports both CPU and GPU execution
  - Supports OpenVINO performance hints for latency- or throughput-oriented compilation

### Torch Eager Backend
- **Best for**: the uncompiled baseline inside a throughput search

## Tuning Strategies

1. **FirstWinsStrategy**: Selects first successful backend
2. **OneBackendStrategy**: Uses only specified backend
3. **HighestThroughputStrategy**: Selects backend with best throughput

## Best Practices

### Backend Selection
- **HighestThroughputStrategy**: the default list is Torch Inductor, XPU graph, and eager. It measures each one and keeps the fastest. List order does not change the winner. TorchAO and OpenVINO are optional additions.
- **FirstWinsStrategy**: default order is XPU graph, Torch Inductor, OpenVINO, then eager. The first backend that builds is kept. TorchAO is not in that default, because the first backend that builds wins and a quantization that builds would be kept without a speed comparison.

### Common Issues
- **Graph breaks**: conditional logic in a module blocks tuning of that module

## Code Patterns

### Basic Tuning
```python
import torch_tweak.torch as tt

# Inspect
modules_info = tt.inspect(model, input_data)
modules_info.describe()

# Wrap and tune
modules = modules_info.get_modules()
model = tt.wrap(model, modules)
tt.tune(model, input_data)

# Save/load
tt.save(model, "tuned_model.tt")
tt.load(model, "tuned_model.tt")
```

### Backend Configuration
```python
from torch_tweak.torch.backend import (
    TorchAOBackend,
    TorchInductorBackend,
    XPUGraphBackend,
    OpenVINOBackend,
    TorchEagerBackend,
)
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy

strategy = HighestThroughputStrategy(
    backends=[
        TorchAOBackend(),
        TorchInductorBackend(),
        XPUGraphBackend(),
        OpenVINOBackend(),
        TorchEagerBackend(),
    ]
)
```

## When Helping Users

1. **Recommend backends**: Based on performance needs and constraints
2. **Explain trade-offs**: Performance vs ease-of-use, control vs automation
3. **Provide code examples**: Show complete workflows, not just snippets
4. **Address common pitfalls**: Graph breaks
5. Don't commit a non-standard ascii characters in the code, especially in the comments (unless specifically specified by the user). Example: avoid dashes, use simple hyphens.

## Key Files to Reference

- `torch_tweak/torch/` - Tuning API
- `torch_tweak/torch/backend/` - Backend implementations
- `torch_tweak/torch/tune_strategy/` - Tuning strategies
- Examples in `examples/` directory

Always provide context-aware guidance based on the user's specific needs and the codebase structure.
