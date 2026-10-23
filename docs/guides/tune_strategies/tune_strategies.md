<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# Tune Strategies Guide

Tune strategies determine how Torch Tweak selects and configures backends during the tuning process. They provide flexibility in balancing performance, reliability, and tuning time.

## Overview

Torch Tweak provides three built-in strategies:

- **OneBackendStrategy**: Uses a single specified backend
- **FirstWinsStrategy**: Tries backends in order, uses the first that succeeds
- **HighestThroughputStrategy**: Profiles all backends, selects the fastest

## Why Backends Can Fail

Not every backend can successfully tune every model. Each backend relies on a different compilation or export technology, and each has its own limitations:

- **OpenVINO** requires converting the model to an OpenVINO representation. Models with unsupported operators, complex dynamic control flow, or symbolic shape constraints may fail during conversion or model compilation. Memory constraints can also prevent the model from being compiled.
- **Torch Inductor** uses `torch.compile`, which may encounter *graph breaks* on unsupported Python constructs or operations, causing partial or failed compilation.
- **XPU Graph** records the module for replay, so it requires static tensor shapes. A failed recording or a mismatch against eager output rejects the backend for that module and input.
- **TorchAO** applies quantization transformations that may not support all layer types or model architectures.
- **Torch Eager** runs the module unchanged, so it always succeeds, which makes it a useful last-resort fallback.

Because of these differences, a backend that fails on one model may succeed on another, and vice versa. This is the core motivation behind strategies like `FirstWinsStrategy`: by trying multiple backends in priority order, you get automatic fallback when your preferred backend cannot handle a particular model.

## Choosing a Strategy

Use the table below as a quick decision guide. If you already know a backend is compatible and stable in production, start with `OneBackendStrategy`. If you want a safer default with minimal tuning time, `FirstWinsStrategy` balances reliability and speed. When absolute throughput matters and you can afford longer tuning, choose `HighestThroughputStrategy`.

| Strategy                  | When to Use                     | Tuning Time | Reliability             | Performance        |
|---------------------------|---------------------------------|-------------|-------------------------|--------------------|
| OneBackendStrategy        | Known backend works, production | Fast        | High (if backend works) | Depends on backend |
| FirstWinsStrategy         | Want reliability, quick tuning  | Fast-Medium | Very High               | Good               |
| HighestThroughputStrategy | Maximum performance, have time  | Slow        | High                    | Best               |

## OneBackendStrategy

Uses exactly one backend, failing immediately with the original error if it cannot build. Use this when you have already validated that a backend works and want deterministic, reproducible behavior in production.

!!! note
    `OneBackendStrategy` may look equivalent to `FirstWinsStrategy` with a single backend, but the key difference is error handling: `OneBackendStrategy` raises the backend's original exception on failure, while `FirstWinsStrategy` catches errors and tries the next candidate.

### Usage

```python
from torch_tweak.torch.backend import TorchInductorBackend, TorchInductorBackendConfig
import torch_tweak.torch as tt

# Configure backend
config = TorchInductorBackendConfig(mode="default")
backend = TorchInductorBackend(config)

# Create strategy
strategy = tt.OneBackendStrategy(backend=backend)

# Use in tuning
model = tt.Module(model, "my-model", strategy=strategy)
tt.tune(model, input_data)
```

### When to Use

✅ **Good for**:

- Production environments with validated backends
- Reproducible results
- Fast tuning cycles
- Specific backend requirements

❌ **Not ideal for**:

- Experimentation (no fallback)
- Unknown models (might fail)
- Maximum performance discovery

## FirstWinsStrategy

Tries backends in priority order and returns the first one that successfully builds and validates. If a backend fails, the strategy moves on to the next candidate instead of aborting. If all backends fail, the error is caught and the original model is used as-is. The default order is XPU graph, Torch Inductor, OpenVINO, then eager. Put a backend earlier only when you want that one to win as soon as it builds.

### Usage

```python
from torch_tweak.torch.backend import (
    XPUGraphBackend,
    TorchAOBackend,
    TorchAOBackendConfig,
    TorchInductorBackend,
)
import torch_tweak.torch as tt

# List backends in priority order (fastest → most compatible)
backends = [
    XPUGraphBackend(),
    TorchAOBackend(config=TorchAOBackendConfig(quantization="fp8wo")),  # Quantization: fast inference
    TorchInductorBackend(),  # Good performance, broader compatibility
]

# Create strategy
strategy = tt.FirstWinsStrategy(backends=backends)

# Use in tuning
model = tt.Module(model, "my-model", strategy=strategy)
tt.tune(model, input_data)
```

### How It Works

1. Tries the first backend (for example, Torch Inductor)
2. If successful → uses it, done
3. If fails (e.g., unsupported op, conversion error, memory limit) → tries next backend
4. Repeats until a backend succeeds or all fail

### When to Use

✅ **Good for**:

- Experimentation with unknown or diverse models
- CI/CD pipelines where different models may need different backends
- Maximum reliability (if all backends fail, the original model is used as-is)
- Quick validation that *something* works before investing in backend-specific tuning

❌ **Not ideal for**:

- Maximum performance (stops at first success, not the fastest)
- When you already know which backend works (use `OneBackendStrategy` instead)
- Detailed performance comparison (use `HighestThroughputStrategy` instead)

### Best Practices

1. **Order by Performance**: Put fastest backends first
2. **Automatic Fallback**: If all backends fail, the original model is used as-is
3. **Similar Configurations**: Use compatible configs (e.g., all FP16)

## HighestThroughputStrategy

Tries all backends, profiles their performance, and selects the fastest.

### Usage

```python
from torch_tweak.torch.backend import (
    TorchInductorBackendConfig,
    TorchInductorBackend,
    TorchAOBackend,
    TorchAOBackendConfig,
    XPUGraphBackend,
)
import torch_tweak.torch as tt

# List all candidate backends
backends = [
    TorchInductorBackend(config=TorchInductorBackendConfig(mode="default")),
    XPUGraphBackend(),
    TorchAOBackend(config=TorchAOBackendConfig(quantization="fp8wo")),
]

# Create strategy
strategy = tt.HighestThroughputStrategy(
    backends=backends,
)

# Use in tuning
model = tt.Module(model, "my-model", strategy=strategy)
tt.tune(model, input_data)
```

### How It Works

1. Tries to build with each backend
2. For successful backends:
   - Runs warmup iterations
   - Measures throughput over N iterations
   - Records performance metrics
3. Selects backend with highest throughput
4. Returns best performing backend

### When to Use

✅ **Good for**:

- Production deployment planning
- Maximum performance requirements
- Comparing backend options
- When tuning time is not critical

❌ **Not ideal for**:

- Quick experiments (slow)
- Development iteration (overkill)
- Memory-constrained systems (keeps multiple builds)

## Best Practices

1. **Development**: Use `FirstWinsStrategy` with fallback
2. **Production**: Use `OneBackendStrategy` with validated backend
3. **Benchmarking**: Use `HighestThroughputStrategy` to find the best option
4. **Always Validate**: Test tuned models before deployment
5. **Cache Results**: Save tuned models to avoid re-tuning

## Next Steps

- Learn about specific backends: [TorchAO](../backends/torchao_backend.md), [Inductor](../backends/torch_inductor_backend.md), [XPU Graph](../backends/xpu_graph_backend.md)
- Explore [Deployment Guide](../deployment/deployment.md)
- Review the [Tuning Guide](../tuning.md)
