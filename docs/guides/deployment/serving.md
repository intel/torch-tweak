<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Serving Frameworks

`tt.ServedModule` tunes modules inside a framework that loads the model and runs the requests itself.
Torch Tweak never sees a dataset or a training loop.
It sees the calls the framework makes.

```python
import torch_tweak.torch as tt

config = tt.ServingConfig(workdir="/var/cache/my-server/torch-tweak")
tt.ServedModule(pipeline.transformer, "wan2.1-1.3b.transformer.rank0", config, device=torch.device("xpu"))
```

The module is wrapped in place.
The framework keeps calling `pipeline.transformer` as before.

## Modes

The work directory holds one directory per module name.
The default mode, `auto`, picks the step from what that directory holds.

| Directory holds | Mode | First call | Later calls |
|---|---|---|---|
| nothing | `record` | original module, inputs saved | original module, inputs of new shapes saved |
| a recording | `tune` | tunes every recorded shape, saves the choice | tuned backends |
| a tuned choice | `serve` | loads the choice and builds its backends | tuned backends |

Run the server once with the requests it will serve, so every shape is recorded.
Restart it to tune.
Every later start serves the saved choice.

Pass `mode="record"`, `"tune"`, or `"serve"` to force a step.
`tune` without a recording and `serve` without a choice raise `FileNotFoundError`.

## What is saved

Each tensor shape is its own graph.
A shape that was not recorded runs the fallback and logs one warning.
The default fallback, `inductor`, compiles the module with dynamic shapes on the first such call.
`fallback="eager"` runs the original module instead.

The tuned choice does not hold the module weights.
On `serve`, the backends use the weights the framework already loaded.

Tuning writes its log to `tune.log` next to the choice.

## Backends

`backends` lists names from `torch_tweak.torch.serving.BACKEND_FACTORIES`.
`HighestThroughputStrategy` measures each one on every recorded shape and keeps the fastest.

| Name | Backend |
|---|---|
| `xpu_graph` | XPU graph of the eager kernels |
| `xpu_graph_inductor` | XPU graph of the kernels Torch Inductor generates |
| `inductor` | Torch Inductor |
| `eager` | the original module |

The default is all four.

Tuning and loading run inside the first call.
They use the device, the streams, and the framework context of a real request.
