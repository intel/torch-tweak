<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# XPU Graph Backend Guide

The XPU graph backend records the module on an Intel GPU once during tuning and replays that recording on later inference calls.
Recording is not repeated for each inference call.

## Overview

- **One recording**: warmup and capture run in `build`, while the module is being tuned
- **Replay**: inference copies new tensor values into the recorded inputs and calls `replay`
- **Rejection**: a failed recording, an empty recording, or outputs that disagree with eager drop this backend for that module and input
- **Static shapes**: a batch axis or another dynamic axis rejects the backend. Replay cannot change tensor shapes
- **Fallback**: an inference call whose inputs differ from the recorded call in shape, dtype, device, or a non-tensor value runs the original module

## Quick Start

```python
from torch_tweak.torch.backend import XPUGraphBackend
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy

strategy = HighestThroughputStrategy(backends=[XPUGraphBackend()])
```

The backend is first in the default `FirstWinsStrategy` list, before Torch Inductor.
It is also in the default `HighestThroughputStrategy` list.

PyTorch records a native SYCL graph when it was built with a SYCL compiler of 2026.1 or newer, except on PVC.
An older build records the legacy SYCL command graph.
This backend uses the recording that build already performs.

## Next Steps

- API reference: [XPU Graph Backend](../../api/backends/xpu_graph_backend.md)
