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
model = tt.TunableModule(model, "my-model", strategy=strategy)
tt.tune(model, input_data)
```

`OpenVINOBackend()` with no config uses `torch.export` and the `LATENCY` hint.

## XPU

On `xpu` the backend binds PyTorch tensors as OpenVINO GPU tensors.
Inputs and outputs stay in device memory and nothing is copied through the host.

This needs the Intel GPU driver to run OpenCL on Level Zero (LEO).

On Battlemage (Intel Arc B-series), LEO is off by default.
Start the process with `EnableLEO=1`:

```bash
EnableLEO=1 python my_pipeline.py
```

The driver reads the variable when the process first uses OpenCL, so setting it later has no effect.
Without LEO, building the backend on XPU raises `LevelZeroInteropError`.

On newer Intel GPUs (driver 26.35 or newer), the driver enables LEO by default and the variable is not needed.

## Configuration

```python
config = OpenVINOBackendConfig(
    use_dynamo=True,  # False selects the legacy trace conversion
    performance_hint="THROUGHPUT",
)
```

Use `use_dynamo=False` when `torch.export` cannot capture the module.

The backend moves the module to CPU after conversion, since the compiled model holds its own copy of the weights.
`offload_module=False` leaves the module on its device, for a framework that keeps calling it.
The weights then exist twice on the device: in the module and in the compiled model.

The flag controls only the backend build.
`TunableModule` replaces the module's forward, so after tuning it still moves the module to `config.device_after_tuning`.

`reuse_ir=True` compiles the IR already in the build directory instead of converting the module.
The IR holds the weights, so the caller keys that directory by the weights and the shapes, as `OnlineTunedModule` does.

## Next Steps

- Compare with the [Torch Inductor backend](torch_inductor_backend.md)
- API reference: [OpenVINO Backend](../../api/backends/openvino_backend.md)
