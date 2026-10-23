<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# Tuning Guide

You choose which modules to tune, which strategy to use, and which backends to try.

## Overview

Tuning follows a four-step workflow:

1. **Inspect**: Analyze your model or pipeline to identify tuneable modules
2. **Wrap**: Wrap selected modules for tuning
3. **Tune**: Execute the tuning process across different backends
4. **Persist**: Save and load tuned models for later deployment

Tuning provides:

- **Control**: Explicitly choose which modules to tune, pick strategies and backends, and mix different technologies
- **Performance**: Benchmark and select optimal configurations
- **Speed**: Save the tuned model to a deployable artifact to be loaded on the production environment
- **Reproducibility**: Deterministic tuning results

## Quick Start

Here's a complete example using Stable Diffusion:

```python
import torch_tweak.torch as tt
from diffusers import DiffusionPipeline

# Initialize pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("xpu")

# Prepare input data
input_data = [{"prompt": "A beautiful landscape with mountains and a lake"}]

# Step 1: Inspect pipeline to discover modules
modules_info = tt.inspect(pipe, input_data)

# Display discovered modules
modules_info.describe()

# Step 2: Wrap modules for tuning
modules = modules_info.get_modules()
pipe = tt.wrap(pipe, modules)

# Step 3: Tune the pipeline
tt.tune(pipe, input_data)

# Step 4: Save the tuned pipeline
tt.save(pipe, "tuned_pipe.tt")

# Use the tuned pipeline
images = pipe(["A beautiful landscape with mountains and a lake"])
```

## Detailed Workflow

### 1. Inspection Phase

The `inspect` function analyzes your model or pipeline to identify PyTorch modules that can be tuned. For a detailed guide on inspection, see the [Inspect Guide](inspect.md).

### 2. Wrapping Phase

Given the list of modules from the previous step, you can wrap them for tuning. Under the hood, each `torch.nn.Module` is wrapped (imagine a proxy object) with Torch Tweak `Module` which intercepts all `forward` calls to get data, tune the module and serve the tuned version.

The following line shows how to wrap modules.

```python
model = tt.wrap(model, modules)
```

You can also specify tuning strategies during wrapping:

```python
import torch_tweak.torch as tt

strategy = tt.OneBackendStrategy(backend=tt.backend.TorchInductorBackend())
model = tt.wrap(model, modules, strategy=strategy)
```

If you would like to have more control over picking the modules, you can manually wrap `torch.nn.Module`. When wrapping, you can specify a strategy for each module separately; i.e., you can combine different strategies backends into one model.

```python
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
pipe.to("xpu")

pipe.unet = tt.Module(pipe.unet, strategy=strategy_for_unet)
pipe.transformer = tt.Module(pipe.transformer, strategy=strategy_for_transformer)
```


### 3. Tuning Phase

The `tune` function executes the actual tuning:

```python
tt.tune(
    func=model,  # The wrapped callable module or pipeline to tune
    dataset=input_data,  # Dataset to use for tuning (list, Dataset, DataLoaderFactory, or Tensor)
    batch_sizes=[1, 4, 8],  # Optional: Multiple batch sizes. Defaults to [1, 2]
    max_num_batches_per_batch_size=10,  # Max batches per size. Defaults to None (all)
    device="xpu",  # Device for tuning. Defaults to "xpu:0"
    dry_run=False,  # Set True to test without tuning
    disable_external_logging=False,  # Disable third-party logs
    clear_cache=False,  # Clear Torch Tweak cache before tuning
)
```

#### Tuning Parameters

- **func**: The wrapped callable (model or pipeline)
- **dataset**: Dataset for tuning. It can be a list of samples, `torch.utils.data.Dataset`, `DataLoaderFactory`,  `Tensor` or sequence of tensors, dictionaries, strings
- **batch_sizes**: List of batch sizes to tune against. If not specified, values [1, 2] will be used
- **max_num_batches_per_batch_size**: Maximum number of batches per batch size. If None, all batches will be used
- **device**: Device to use for tuning. Defaults to "xpu:0"
- **dry_run**: If True, performs a dry run without actual tuning
- **disable_external_logging**: Disable logging from external libraries
- **clear_cache**: Clear Torch Tweak cache before tuning

Tuning time depends on the tuned modules' size, used strategy, and number of backends. Modules are tuned one by one. If a strategy has many backends to pick from, it takes the one that fulfills specific strategy criteria. Each backend is validated against returning proper numeric results (check against NANs and infinity) and output shapes.

Note: If you specify a batch size that is not a power of 2, it will be used to gather samples but the actual search for the highest throughput will round it up to the nearest power of 2.

### 4. Persistence Phase

Once tuned, you can save your model for later use. This is crucial for production deployments to avoid re-tuning every time:

```python
# Save the tuned model/pipeline
tt.save(pipe, "tuned_model.tt")
```

The tuned artifact will be saved in the `checkpoints` folder. The `save` function creates several files:

- `tuned_model.tt`: The compressed checkpoint containing tuned and original weights
- `tuned_model_sha256_sums.txt`: SHA256 hashes for verification

To do inference, you can load the tuned model/pipeline:

```python
# Note: Initializing the original object is required before loading
pipe = DiffusionPipeline.from_pretrained(...)
pipe = tt.load(pipe, "tuned_model.tt")
# pipe is ready for use
```

## Custom Inference Functions

For complex pipelines, you can provide a custom inference function:

```python
def custom_inference(prompt, num_steps=50):
    """Function forces width, height and number of steps."""
    return pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        height=1024,
        width=1024,
    )


modules_info = tt.inspect(pipe, input_data, inference_function=custom_inference)
```

## Configuration Options

Torch Tweak has configuration for the tuning process, and each backend has its configuration.

### Global Configuration

You can configure Torch Tweak globally:

```python
from torch_tweak.torch import config

# Set cache directory
config.cache_dir = "/path/to/cache"

# Set minimum samples for tuning
config.min_num_samples = 5

# Set maximum stored samples per graph
config.max_num_samples_stored = 100

# Device to move model after tuning
config.device_after_tuning = "xpu"

# Enable/disable strict mode for input validation
config.strict_mode = True

# Enable HuggingFace integrations
config.enable_hf_integrations = True
```

### Backend-Specific Configuration

Each backend has its own corresponding configuration:

```python
from torch_tweak.torch.backend import TorchInductorBackendConfig, TorchInductorBackend

config = TorchInductorBackendConfig(mode="default")
backend = TorchInductorBackend(config)
```

See backend-specific documentation:

- [Torch-Inductor Backend](backends/torch_inductor_backend.md)
- [TorchAO Backend](backends/torchao_backend.md)

## Dry Run Mode

You can run tuning in dry-run mode. It records samples of data, detects batch and dynamic axes, and detects graphs of execution but does not call the actual backend to tune. This allows debugging if everything is working as expected.

The dry-run mode can be turned on with the `dry_run` argument. This example tunes a small MLP on Intel XPU and stops before any backend is built:

```python
import torch
from torch import nn

import torch_tweak.torch as tt

model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval()
module = tt.Module(model, name="mlp")
sample = torch.randn(32)
tt.tune(module, sample, batch_sizes=[1, 4], device="xpu", dry_run=True)
```

Example output from that command, with the cache directory set to `/tmp/torch-tweak`:

```text
2026-10-01 06:19:51,358 - INFO - ════════════════════════════════════════════════════════════════
2026-10-01 06:19:51,358 - INFO - 🎯 Tuning module: `mlp` (all graphs)
2026-10-01 06:19:51,359 - INFO - ------------------------------------------------------------
2026-10-01 06:19:51,359 - INFO - 🚀 Tuning graph `0` for module `mlp` (DRY RUN):
2026-10-01 06:19:51,359 - INFO -   number of parameters: 1320
2026-10-01 06:19:51,359 - INFO -   number of layers: 3
2026-10-01 06:19:51,359 - INFO -   precisions: torch.float32
2026-10-01 06:19:51,359 - INFO -   graph_spec:
2026-10-01 06:19:51,359 - INFO -     input_spec:
 Tensors:
╒═══════════╤════════╤════════════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name   │ Shape          │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪════════╪════════════════╪═════════════╪═════════════╪═══════════════╡
│ [0]       │ args_0 │ ['batch0', 32] │ [1, 32]     │ [4, 32]     │ torch.float32 │
╘═══════════╧════════╧════════════════╧═════════════╧═════════════╧═══════════════╛

2026-10-01 06:19:51,359 - INFO -     output_spec:
 Tensors:
╒═══════════╤═════════╤═══════════════╤═════════════╤═════════════╤═══════════════╕
│ Locator   │ Name    │ Shape         │ Min Shape   │ Max Shape   │ Dtype         │
╞═══════════╪═════════╪═══════════════╪═════════════╪═════════════╪═══════════════╡
│           │ outputs │ ['batch0', 8] │ [1, 8]      │ [4, 8]      │ torch.float32 │
╘═══════════╧═════════╧═══════════════╧═════════════╧═════════════╧═══════════════╛

2026-10-01 06:19:51,359 - INFO -   num samples: 1
2026-10-01 06:19:51,359 - INFO -   device: xpu:0
2026-10-01 06:19:51,359 - INFO -   cache_dir: /tmp/torch-tweak/mlp/0
2026-10-01 06:19:51,359 - INFO -   strategy:
2026-10-01 06:19:51,359 - INFO -     name: First Wins Strategy
2026-10-01 06:19:51,359 - INFO -     description: evaluate backends in order, return first working backend
2026-10-01 06:19:51,360 - INFO -     backends:
2026-10-01 06:19:51,360 - INFO -       XPUGraphBackend()
2026-10-01 06:19:51,360 - INFO -       TorchInductorBackend()
2026-10-01 06:19:51,360 - INFO -       OpenVINOBackend()
2026-10-01 06:19:51,360 - INFO -       TorchEagerBackend()
2026-10-01 06:19:51,360 - INFO - ✅ Tuning module: `mlp` (all graphs) completed.
```

## Next Steps

- Learn about [Inspect](inspect.md) for detailed module analysis
- Explore [Backend Configuration](backends/torch_inductor_backend.md)
- Review [Tune Strategies](tune_strategies/tune_strategies.md)
- See [Deployment Guide](deployment/deployment.md)
