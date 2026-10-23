<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# OpenVINO Backend Guide

The OpenVINO backend converts a PyTorch module to an OpenVINO model and runs it on an Intel CPU or GPU.

## Overview

- **Conversion**: `torch.export` by default, with a legacy trace path when export cannot capture the module
- **Devices**: `cpu` compiles for the OpenVINO CPU device, `xpu` compiles for the OpenVINO GPU that is the same physical device as the PyTorch XPU device
- **Compile hint**: `LATENCY` (default), `THROUGHPUT`, or `CUMULATIVE_THROUGHPUT`

## Quick Start

```python
from torch_tweak.torch.backend import OpenVINOBackend, OpenVINOBackendConfig
import torch_tweak.torch as tt

config = OpenVINOBackendConfig(use_dynamo=False, performance_hint="THROUGHPUT")
backend = OpenVINOBackend(config)

strategy = tt.OneBackendStrategy(backend=backend)
model = tt.Module(model, "my-model", strategy=strategy)
tt.tune(model, input_data)
```

`OpenVINOBackend()` with no config uses `torch.export` and the `LATENCY` hint.

## Configuration

```python
config = OpenVINOBackendConfig(
    use_dynamo=True,              # False selects the legacy trace conversion
    performance_hint="THROUGHPUT",
)
```

Use `use_dynamo=False` when `torch.export` cannot capture the module.

## Next Steps

- Compare with the [Torch Inductor backend](torch_inductor_backend.md)
- API reference: [OpenVINO Backend](../../api/backends/openvino_backend.md)
