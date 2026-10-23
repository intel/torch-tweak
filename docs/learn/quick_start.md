<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->
# Quick Start

This quick start provides examples of tuning and deployment path available in Torch Tweak.

Torch Tweak tunes models for deployment (for example, converting them to OpenVINO) with a small amount of integration in your Python pipeline. You provide a model or a pipeline, and a dataset or dataloader. You can rely on `inspect` to detect promising modules to tune, or select them manually.

## Enabling logging

The tuning process guides the user through decisions and steps that are performed to tune every selected module.

We recommend to enable the INFO logging level for better verbosity in the quick start steps:
```python
import logging

logging.basicConfig(level=logging.INFO, force=True)
```

Learn about more options in [observability](observability.md).

## Tuning

The code below demonstrates Stable Diffusion pipeline tuning.

You can annotate `torch.nn.Module`s manually or use the `inspect` functionality to have modules picked automatically; you can then verify them and schedule them for tuning.

First, install the required third-party dependencies:

```bash
pip install transformers diffusers torch
```

Then initialize the pipeline:

```python
import torch
from diffusers import DiffusionPipeline

import torch_tweak.torch as tt

# Initialize pipeline
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("xpu")
```

Next, `inspect` the pipeline components and display the summary:

```python
# Prepare input data
input_data = [{"prompt": "A beautiful landscape with mountains and a lake"}]

# Inspect pipeline to get modules
modules_info = tt.inspect(pipe, input_data)


# Optional: inference function, if you need more control over execution
def infer(prompt):
    return pipe(prompt, width=1024, height=1024, num_inference_steps=10)


# modules_info = tt.inspect(pipe, input_data, inference_function=infer)

# Display modules info
modules_info.describe()
```

Finally, `wrap` the selected modules and `tune` within the pipeline:

```python
# Wrap modules for tuning
modules = modules_info.get_modules()
pipe = tt.wrap(pipe, modules)

# Tune pipeline
tt.tune(pipe, input_data)
```

At this point, you can use the pipeline to generate predictions with the tuned models directly in Python:

```python
# Run inference on tuned pipeline
images = pipe(["A beautiful landscape with mountains and a lake"])
image = images[0][0]

# Save image for preview
image.save("landscape.png")
```

Once the pipeline has been tuned, you can save the best-performing version of the modules for later deployment:

```python
tt.save(pipe, "tuned_pipe.tt")
```

And load the tuned pipeline directly:

```python
pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.to("xpu")
tt.load(pipe, "tuned_pipe.tt")
```

## Summary

Tuning gives you control over the process:

* it detects the batch axis and dynamic axes (axes that change shape independently of batch size, e.g., sequence length in LLMs)
* allows picking modules to tune
* you can pick a tuning strategy (e.g., best throughput) for the whole process or per-module
* you can pick tuning backends (for example, XPU graph, TorchAO, Torch Inductor, OpenVINO, and eager) which will be used by the strategy
* you can mix different backends in the same model/pipeline
* you can manually verify the tuning process (note: Torch Tweak performs basic checks for NaNs and errors)
* you can save the resulting artifact and later read it from disk
