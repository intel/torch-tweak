<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# Torch Inductor Backend Guide

The Torch Inductor backend uses PyTorch's built-in compiler (`torch.compile` with `backend="inductor"`) for model tuning. It provides automatic kernel fusion and optimization without external dependencies.

## Overview

- **Pure PyTorch**: No external dependencies
- **Automatic Optimization**: Kernel fusion and code generation
- **Multiple Modes**: Default, reduce-overhead, max-autotune
- **Dynamic Shapes**: Configurable dynamic shape support
- **Cross-Platform**: Works on CPU and XPU

## Quick Start

```python
from torch_tweak.torch.backend import TorchInductorBackend, TorchInductorBackendConfig
import torch_tweak.torch as tt
import torch

# Configure backend
config = TorchInductorBackendConfig(mode="default")
backend = TorchInductorBackend(config)

# Use in tuning
from torch_tweak.torch.tune_strategy import OneBackendStrategy

strategy = tt.OneBackendStrategy(backend=backend)

model = tt.TunableModule(model, "my-model", strategy=strategy)
tt.tune(model, input_data)
```

## Configuration Options

### TorchInductorBackendConfig

```python
@dataclass
class TorchInductorBackendConfig(BackendConfig):
    fullgraph: bool = False
    dynamic: bool | None = None
    mode: str | None = None
    options: dict | None = None
    autocast_enabled: bool = False
    autocast_dtype: torch.dtype | None = None
```

### mode

Predefined optimization modes:

```python
# Default mode (balanced)
config = TorchInductorBackendConfig(mode="default")
```

**Mode Details**:

- **default**: Good balance, general purpose; works on all devices including Intel XPU

### fullgraph

Require complete graph capture:

```python
config = TorchInductorBackendConfig(
    fullgraph=True,  # Error if graph breaks occur
    mode="max-autotune",
)
```

### dynamic

Control dynamic shape behavior:

```python
# Always generate dynamic kernels
config = TorchInductorBackendConfig(dynamic=True)

# Never generate dynamic kernels (always specialize)
config = TorchInductorBackendConfig(dynamic=False)

# Auto-detect (default)
config = TorchInductorBackendConfig(dynamic=None)
```

### options

Custom inductor options:

```python
# See all options: torch._inductor.list_options()
config = TorchInductorBackendConfig(
    options={
        "triton.cudagraphs": False,
        "max_autotune": False,
        "coordinate_descent_tuning": True,
    }
)
```

**Note**: Cannot use both `mode` and `options`.

### autocast

Enable automatic mixed precision:

```python
config = TorchInductorBackendConfig(
    mode="max-autotune",
    autocast_enabled=True,
    autocast_dtype=torch.float16,
)
```

## Debugging

### Enable Logging

```python
# Set environment variables before running
import os

os.environ["TORCH_LOGS"] = "dynamic,perf_hints,graph_breaks"

# Then run tuning
tt.tune(wrapped_model, input_data)
```

### Check Optimizations

```python
# See what mode does
import torch

print(torch._inductor.list_mode_options())

# See all available options
print(torch._inductor.list_options())
```

## Best Practices

1. **Start with default**: Good baseline, works across all devices including Intel XPU
2. **Enable Autocast**: Free performance boost with FP16
3. **Set Autocast dtype**: Set the dtype to BFloat16 for better performance on XPU
4. **Dynamic Shapes**: Only when necessary (adds overhead - larger, slower kernels)
5. **Warmup**: Run a few iterations before benchmarking

## Troubleshooting

### Issue: Graph breaks

**Check where breaks occur**:

```bash
TORCH_LOGS=graph_breaks python your_script.py
```

**Solution**: Use `fullgraph=False` (default) to allow partial compilation.

### Issue: Slow compilation

**Solution**: Reduce auto-tuning:

```python
config = TorchInductorBackendConfig(mode="default")
```

### Issue: Variable shape recompilations

**Solution**: Enable dynamic shapes:

```python
config = TorchInductorBackendConfig(
    mode="default",
    dynamic=True,
)
```

## Comparison with Other Backends

| Feature          | Inductor  | OpenVINO   | TorchAO   |
|------------------|-----------|------------|-----------|
| **Dependencies** | None      | openvino   | torchao   |
| **Setup**        | Easy      | Moderate   | Easy      |
| **Quantization** | Limited   | Advanced   | Extensive |
| **Portability**  | Excellent | Intel only | Good      |

## Next Steps

- Explore [TorchAO Backend](torchao_backend.md) for quantization
- Learn about [Tune Strategies](../tune_strategies/tune_strategies.md)
- Review [Deployment Guide](../deployment/deployment.md)
