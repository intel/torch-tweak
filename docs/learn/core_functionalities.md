<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->
# Core Functionalities

## Inspect

The `inspect` function allows you to analyze PyTorch models and pipelines to understand their structure, parameters, and execution flow. It provides detailed insights into model architecture and helps identify tuning opportunities.

```python
import torch_tweak.torch as tt
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(100, 10)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()

# Inspect the model
tt.inspect(model, dataset)
```

## Tune

The `tune` function is the core functionality that automatically tunes your PyTorch models and pipelines for optimal inference performance. It supports various backends and automatically selects the best performing configuration.

```python
import torch_tweak.torch as tt
import torch

# Define your model
model = SimpleModel()

# Wrap the model
model = tt.Module(model)


# Define inference function
def inference_fn(x):
    return model(x)


# Tune the model
tt.tune(
    func=inference_fn,
    dataset=torch.randn(1, 100),
)
```

## Save

The `save` function allows you to persist tuned models for later use. It stores tuned and original module weights together in a single file with a `.tt` extension. Apart from the checkpoint file, there is also a SHA hash file.

```python
# Save the tuned model
import torch_tweak.torch as tt

tt.save(model, "tuned_model.tt")
```

Example output:

```bash
checkpoints/
├── tuned_model
├── tuned_model.tt
└── tuned_model_sha256_sums.txt
```

You can copy the checkpoint file `tuned_model.tt` and SHA sums file to a target host or folder to use it for inference.

*Note:* We recommend deploying the `*.tt` package on the same hardware used for tuning to ensure functional and performance compatibility.

## Load

The `load` function enables you to load previously tuned models from a checkpoint file.

```python
# Load the tuned model
import torch_tweak.torch as tt

tuned_model = tt.load(model, "tuned_model.tt")
```

On first load, the checkpoint file is decompressed and the tuned and original module weights are loaded. Subsequent loads will use the decompressed weights from the same folder.

# Tune Strategies

Torch Tweak provides different strategies for selecting the optimal backend configuration. The strategies align with a common interface for the tuning process.

Not every backend can tune every model - each relies on different compilation technology with its own limitations (e.g., model conversion for OpenVINO, graph breaks in Torch Inductor, unsupported layers in TorchAO). Strategies control how Torch Tweak handles this.

## FirstWinsStrategy

Tries backends in priority order and returns the first one that succeeds. If a backend fails, the strategy moves on to the next candidate instead of aborting.

```python
from torch_tweak.torch.tune_strategy import FirstWinsStrategy

strategy = FirstWinsStrategy(
    backends=[
        XPUGraphBackend(),
        TorchInductorBackend(),
        OpenVINOBackend(),
        TorchEagerBackend(),
    ]
)
```

That order is also the default when you construct `FirstWinsStrategy` with no backend list. The first backend that builds and validates is kept.

## OneBackendStrategy

Uses exactly one backend, failing immediately with the original error if it cannot build. Use this when you have already validated that a backend works and want deterministic behavior. Unlike `FirstWinsStrategy` with a single backend, `OneBackendStrategy` surfaces the original exception rather than catching it.

```python
from torch_tweak.torch.tune_strategy import OneBackendStrategy

strategy = OneBackendStrategy(backend=OpenVINOBackend())
```

## HighestThroughputStrategy

Profiles all compatible backends and selects the fastest. Use this when maximum throughput matters and you can afford longer tuning time. `HighestThroughputStrategy()` with no backend list compares Torch Inductor, XPU graph, and eager.

```python
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
