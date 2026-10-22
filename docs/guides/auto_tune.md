<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Auto Tune Guide

`auto_tune` picks the depth at which a model is cut into tuned modules.
It tunes the model at every depth from `min_depth` to `max_depth` and keeps the fastest one.

A shallow cut gives a backend one large graph.
A deep cut tunes many small modules separately, and each wrapped call adds overhead.
The best depth depends on the model, the backends, and the inputs, so `auto_tune` measures it.

## Basic Usage

```python
import torch_tweak.torch as tt

result = tt.auto_tune(model, input_data, strategy=tt.HighestThroughputStrategy(), max_depth=5)
print(result)

model = result.obj
tt.save(model, "tuned_model.tt")
```

Use `result.obj` after tuning.
It is a `Module` wrapper when the whole model wins, and `model` itself otherwise.
When the whole model wins, `model` stays untuned.

`print(result)` shows the latency of every depth, the speedup over the untuned model, and the selected depth.
Every parameter after `dataset` is keyword-only.

## How It Works

1. One inspection finds the executed modules and their depth, as in [Inspect](inspect.md).
2. Every depth gives a cut. A cut equal to the previous one is skipped.
3. The untuned model is measured as the baseline.
4. For every cut, the modules are wrapped, tuned, and the whole model is measured on `dataset`.
   The wrappers are then detached, so the next depth starts from the original model.
5. The depth with the lowest sum of median latencies over all samples wins.
   A shallower depth wins when it is within `tie_tolerance` (3% by default) of the fastest one.
   When no depth is faster than the baseline, the model is left untuned and `best_depth` is None.

A module that appears in several cuts is tuned once.
For example, a leaf module such as `head` is in every cut below its own depth.

Weights stay on the device during the search.
Only the winning modules are offloaded at the end, as `tune` does.

Without `strategy`, `auto_tune` uses `FirstWinsStrategy` without the search for the maximum batch size.
That search runs for every module at every depth and probes batch sizes far above `batch_sizes`.
Pass a strategy to enable it explicitly.

## Inputs

`dataset` and `batch_sizes` are used for tuning and for measurement.
`print(result)` also shows the fastest depth for every sample.
When different inputs prefer different depths, run `auto_tune` separately for each group of inputs and save each result to its own checkpoint.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `min_depth` | 0 | Shallowest depth to try |
| `max_depth` | 5 | Deepest depth to try |
| `min_execution_percentage` | None | Modules below this fraction (0 to 1) of inspected time are not wrapped |
| `measure_iterations` | 10 | Timed runs per sample at every depth |
| `tie_tolerance` | 0.03 | Relative latency difference treated as a tie |
| `time_budget` | None | Seconds after which no new depth is started |
| `dry_run` | False | Only report the modules of each cut |

A depth that fails is reported in `result.depths` and skipped.
A module that fails to tune runs its original forward and is not installed.
A depth where every module fails to tune is skipped.

Backends such as OpenVINO move a module to CPU when they build it.
`auto_tune` moves the weights back after every depth, so deeper cuts start on the original device.

## Cost

Every depth runs a full tuning pass for its new modules.
Start with `dry_run=True` to see how many modules each depth wraps.

`time_budget` limits the search.
Depths run from the shallowest, and the depth in progress always finishes.
Depths not started are reported in `result.depths` with the error `time budget used up`.
