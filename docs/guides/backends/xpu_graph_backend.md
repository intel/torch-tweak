<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# XPU Graph Backend Guide

The XPU graph backend records the module on an Intel GPU once during tuning and replays that recording on later inference calls.
Recording is not repeated for each inference call.

## Overview

- **Recording**: warmup and capture run in `build`, while the module is being tuned
- **Replay**: inference copies new tensor values into the recorded inputs and calls `replay`
- **Rejection**: a failed recording, an empty recording, or outputs that disagree with eager drop this backend for that module and input
- **Dynamic axes**: an axis that changes without following the batch size, such as sequence length, rejects the backend
- **Fallback**: an inference call whose inputs differ from the recordings in shape, dtype, device, or a non-tensor value runs the original module

## Batch Sizes

A graph spec with static shapes records one graph.
A graph spec with a batch axis records graphs for batch sizes 1, 2, 4, and then every multiple of 8, up to the largest batch size seen during tuning.
`max_recorded_batch_size` caps that size. The default is 256, which records 35 graphs.

A call pads its batch with zeros up to the smallest recorded size that holds it, replays that graph, and returns only the real rows.
A batch larger than the largest recorded size replays in chunks of that size.
Padding costs compute. A batch of 3 runs the graph recorded for 4, and a batch of 17 runs the graph recorded for 24.

Padding is only correct when every row of the batch is computed on its own.
Tuning checks that: it runs eager once on the largest recorded batch size.
Every graph then replays a prefix of that batch, full and partly padded with zeros and with copies of a real row, and is compared with the matching rows of eager.
The backend is rejected when:

- an output tensor has no batch axis, for example `x.sum()`
- one row of the output depends on another row, for example `x - x.mean(dim=0)`
- the module rearranges the rows of its batch, for example `torch.cat([y, y])`

All graphs of one module share one memory pool and one input buffer sized for the largest recorded graph.
Peak memory follows the largest recorded graph.
Tuning time grows with the number of graphs: each one is warmed up, recorded, and replayed by the check.
An out-of-memory error while recording a size skips that size and keeps the smaller graphs.

```python
from torch_tweak.torch.backend import XPUGraphBackend, XPUGraphBackendConfig

backend = XPUGraphBackend(XPUGraphBackendConfig(max_recorded_batch_size=64))
```

## Inductor Kernels

`compile_with_inductor=True` records the kernels Torch Inductor generates instead of the eager kernels.
Replay then runs the fused Inductor kernels without launching each one from Python.
On CUDA, `torch.compile(mode="reduce-overhead")` does the same with CUDA graphs. That mode has no effect on XPU.

Compilation runs during the warmup before each recording.
Inductor compiles about three times for all recorded sizes: once for the largest size, once with a dynamic batch axis, and once for batch size 1.
Replay is still checked against the eager module.

```python
from torch_tweak.torch.backend import XPUGraphBackend, XPUGraphBackendConfig

backend = XPUGraphBackend(XPUGraphBackendConfig(compile_with_inductor=True))
```

## Quick Start

```python
from torch_tweak.torch.backend import XPUGraphBackend
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy

strategy = HighestThroughputStrategy(backends=[XPUGraphBackend()])
```

On XPU the backend is first in `default_backends(device)`, before Torch Inductor.
`FirstWinsStrategy`, `HighestThroughputStrategy`, and `tt.default_strategy(device)` use that list when no backends are given.
On CPU the list leaves this backend out.
An XPU graph often wins at small batch sizes and loses at large ones.
`HighestThroughputStrategy` then sends small batches to it and large batches to the faster backend.

PyTorch records a native SYCL graph when it was built with a SYCL compiler of 2026.1 or newer, except on Intel Data Center GPU Max Series.
An older build records the legacy SYCL command graph.
This backend uses the recording that build already performs.

## Next Steps

- API reference: [XPU Graph Backend](../../api/backends/xpu_graph_backend.md)
