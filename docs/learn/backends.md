<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->
# Backends

Torch Tweak supports multiple tuning backends, each with different characteristics and use cases. The backends align with a common interface for the build and inference process.

## TorchAO Backend

The TorchAO backend leverages PyTorch's AO (Accelerated Optimization) framework for model tuning.

```python
from torch_tweak.torch.backend import TorchAOBackend

backend = TorchAOBackend()
```

Guide: [TorchAO Backend](../guides/backends/torchao_backend.md).

## Torch Inductor Backend

The Torch Inductor backend uses PyTorch's Inductor compiler for model tuning.

```python
from torch_tweak.torch.backend import TorchInductorBackend

backend = TorchInductorBackend()
```

Guide: [Torch Inductor Backend](../guides/backends/torch_inductor_backend.md).

## OpenVINO Backend

The OpenVINO backend converts the module to an OpenVINO model and runs it on an Intel CPU or GPU.

```python
from torch_tweak.torch.backend import OpenVINOBackend

backend = OpenVINOBackend()
```

Conversion defaults to `torch.export`. For modules that `torch.export` cannot capture, switch to the
legacy trace-based conversion and optionally change the compile hint:

```python
from torch_tweak.torch.backend import OpenVINOBackend, OpenVINOBackendConfig

backend = OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False, performance_hint="THROUGHPUT"))
```

Guide: [OpenVINO Backend](../guides/backends/openvino_backend.md).

## Torch Eager Backend

The Torch Eager backend runs the module with no compilation and no conversion. It is the baseline `HighestThroughputStrategy` compares the other backends against.

```python
from torch_tweak.torch.backend import TorchEagerBackend, TorchEagerBackendConfig
import torch

backend = TorchEagerBackend(TorchEagerBackendConfig(autocast_enabled=True, autocast_dtype=torch.float16))
```

Guide: [Torch Eager Backend](../guides/backends/torch_eager_backend.md).
