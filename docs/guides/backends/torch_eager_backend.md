<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Torch Eager Backend Guide

The Torch Eager backend runs the module in eager mode, with no compilation and no graph conversion.
Use it as the uncompiled baseline, for example inside `HighestThroughputStrategy`, so the profile includes the original PyTorch execution.

## Overview

- **No compile step**: the module is called directly
- **Optional autocast**: enable autocast and pick a dtype
- **Baseline**: other backends are only worth keeping when they beat this one

## Quick Start

```python
from torch_tweak.torch.backend import (
    TorchAOBackend,
    TorchInductorBackend,
    OpenVINOBackend,
    TorchEagerBackend,
)
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy

strategy = HighestThroughputStrategy(
    backends=[
        TorchAOBackend(),
        TorchInductorBackend(),
        OpenVINOBackend(),
        TorchEagerBackend(),
    ]
)
```

## Configuration

Autocast is off unless you set it:

```python
import torch
from torch_tweak.torch.backend import TorchEagerBackend, TorchEagerBackendConfig

config = TorchEagerBackendConfig(autocast_enabled=True, autocast_dtype=torch.float16)
backend = TorchEagerBackend(config)
```

## Next Steps

- Compare with the [OpenVINO backend](openvino_backend.md)
- API reference: [Torch Eager Backend](../../api/backends/torch_eager_backend.md)
