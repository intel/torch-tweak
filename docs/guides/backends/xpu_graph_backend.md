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

A recording fails when the module reads a GPU tensor from Python during capture.
`tensor.all()` in a mask check does this.
Level Zero reports error 44.

## Batch Sizes

A graph spec with static shapes records one graph.
A graph spec with a batch axis records graphs for batch sizes 1, 2, 4, every multiple of 8 below the largest batch size seen during tuning, and that size itself.
`max_recorded_batch_size` caps that size. The default is 256, which records 35 graphs.

A call pads its batch with zeros up to the smallest recorded size that holds it, replays that graph, and returns only the real rows.
A batch larger than the largest recorded size replays in chunks of that size.
Padding costs compute. A batch of 3 runs the graph recorded for 4, and a batch of 17 runs the graph recorded for 24.

Padding and chunking are only correct when every row of the batch is computed on its own.
Tuning checks that against one eager run.
Every graph replays its batch in full, a shorter prefix padded with zeros, and, when chunking is possible, a window that starts after the first row.
Chunking copies that window to the front of the graph. The rows before it are not in the replay.
The backend is rejected when:

- an output tensor has no batch axis, for example `x.sum()`
- an output hides a batched tensor inside another object, for example a cache
- one row of the output depends on another row, for example `x - x.mean(dim=0)` or `x + x[:1]`
- the module rearranges the rows of its batch, for example `torch.cat([y, y])`

All graphs of one module share one memory pool and one input buffer sized for the largest recorded graph.
Peak memory follows the largest recorded graph.
Tuning time grows with the number of graphs: each one is warmed up, recorded, and replayed by the check.
An out-of-memory error while recording a size skips that size and keeps the smaller graphs.

```python
from torch_tweak.torch.backend import XPUGraphBackend, XPUGraphBackendConfig

backend = XPUGraphBackend(XPUGraphBackendConfig(max_recorded_batch_size=64))
```

## Quick Start

```python
from torch_tweak.torch.backend import XPUGraphBackend
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy

strategy = HighestThroughputStrategy(backends=[XPUGraphBackend()])
```

The backend is first in the default `FirstWinsStrategy` list, before Torch Inductor.
It is also in the default `HighestThroughputStrategy` list.
An XPU graph often wins at small batch sizes and loses at large ones.
`HighestThroughputStrategy` then sends small batches to it and large batches to the faster backend.

PyTorch records a native SYCL graph when it was built with a SYCL compiler of 2026.1 or newer, except on Intel Data Center GPU Max Series.
An older build records the legacy SYCL command graph.
This backend uses the recording that build already performs.

## Next Steps

- API reference: [XPU Graph Backend](../../api/backends/xpu_graph_backend.md)
