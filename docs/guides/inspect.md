<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# Inspect Guide

The `inspect` function is a powerful tool for analyzing PyTorch models and pipelines. It helps you understand model structure, identify tuneable modules, and gather execution statistics. It can also be a first step to pick modules before tuning.

## Overview

Inspection provides:

- **Module Discovery**: Automatically finds all PyTorch modules in your model or pipeline
- **Execution Tracking**: Identifies which modules are executed during inference
- **Performance Profiling**: Measures execution time for each module
- **Input and output data types**: records model input and its results

## Basic Usage

```python
import torch_tweak.torch as tt

# Inspect a model
modules_info = tt.inspect(model, input_data)

# Display results
modules_info.describe()
```

## Inspection Parameters

### Complete Signature

```python
modules_info = tt.inspect(
    obj=model,  # Model or pipeline to inspect
    dataset=input_data,  # Representative input data
    inference_function=None,  # Optional custom inference function
    number_of_iterations=10,  # Iterations for profiling
    warmup_iterations=5,  # Warmup iterations
    depth=0,  # Depth at which the model is cut into modules
    warn_if_no_modules=True,  # Warn when the walk stops before any module is registered
)
```

### Parameter Details

#### obj (Required)

The object to inspect. Can be:

- `torch.nn.Module`: Any PyTorch module
- Callable: Any callable containing PyTorch modules (e.g., HuggingFace pipelines)

```python
# PyTorch module
model = torchvision.models.resnet50()
modules_info = tt.inspect(model, input_data)

# Diffusion pipeline
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("...")
modules_info = tt.inspect(pipe, input_data)
```

#### dataset (Required)

This is the source of data for your model. It can be:

- `torch.Tensor`
- sequence of strings, tensors, or dictionaries. The collate function is used to stack samples into batches.
- `torch.utils.data.Dataset`

```python
# Single tensor with batch dimension
input_data = torch.randn(1, 3, 224, 224)

# List of strings
input_data = ["prompt1", "prompt2"]

# List of tensors
input_data = [torch.randn(3, 224, 224) for _ in range(4)]  # 4 random images, notice there is no batch dimension

# List of dictionaries
input_data = [
    {"input_ids": torch.tensor([...]), "attention_mask": torch.tensor([...])},
]
```

For customization you can use `tt.DataLoaderFactory` class.

#### inference_function (Optional)

Custom function for running inference. Useful for complex execution logic:

```python
def custom_inference(prompt, steps=50):
    """Function forces width, height and number of steps."""
    return pipe(
        prompt=prompt,
        width=1024,
        height=1024,
        num_inference_steps=steps,
    )


modules_info = tt.inspect(pipe, input_data=[{"prompt": "test"}], inference_function=custom_inference)
```

#### number_of_iterations (Default: 10)

Number of iterations for profiling execution time:

```python
# Quick inspection
modules_info = tt.inspect(model, input_data, number_of_iterations=3)
```

#### warmup_iterations (Default: 5)

Warmup iterations before profiling to stabilize measurements:

```python
modules_info = tt.inspect(
    model,
    input_data,
    warmup_iterations=3,  # Speed up inspection by reducing warmup iterations
)
```

#### Module depth

The depth of a module is the number of executed modules above it.
Depth 0 holds the outermost executed modules.
That is the model itself, or the executed components of a pipeline object.
Containers such as `nn.ModuleList`, lists, and dicts do not add a level.
Modules whose `forward` never ran do not add a level either.

For a model whose `forward` loops over `self.layers` (an `nn.ModuleList`) and then calls `self.head`:

| Depth | Modules |
|---|---|
| 0 | `model` |
| 1 | `layers.0`, `layers.1`, `head` |
| 2 | `layers.0.mlp`, `layers.1.mlp` |

#### depth (Default: 0)

Depth at which the model is cut into modules for tuning.
Every executed module at `depth` is returned.
A shallower executed module without executed children is returned too, so no part of the model is left out.
In the example above, `depth=2` returns `layers.0.mlp`, `layers.1.mlp`, and `head`.

Increase it when the outermost modules cannot be tuned:

```python
# Cut at the outermost executed modules (default)
modules_info = tt.inspect(model, input_data, depth=0)

# Cut one level deeper
modules_info = tt.inspect(model, input_data, depth=1)
```

#### warn_if_no_modules (Default: True)

Warn when the walk stops before any module is registered.
The message says how far the walk went and to increase `depth`.
Pass `False` to silence that warning.

```python
modules_info = tt.inspect(model, input_data, warn_if_no_modules=False)
```

## Working with InspectedModulesInfo

The `inspect` function returns an `InspectedModulesInfo` object with several useful methods:

### describe()

Display comprehensive information about discovered modules:

```python
modules_info = tt.inspect(model, input_data)
modules_info.describe()
```

Output example for `stable-diffusion-3-medium-diffusers`:

```text
Module Execution Summary:
==========================================================================================================================================
    Module Name        Calls    Total Time (s)    Avg Time (s)    % of Total    # of params      # of layers           precisions
------------------------------------------------------------------------------------------------------------------------------------------
decoder                      1           0.1760           0.1760        2.80%         49545475                6              torch.float16
text_encoder                 2           0.0057           0.0029        0.09%        123650304                2              torch.float16
text_encoder_2               2           0.0124           0.0062        0.20%        694659840                2              torch.float16
text_encoder_3               2           0.0783           0.0391        1.24%       4762310656                2  torch.float16, torch.float32
transformer                 28           5.9934           0.2140       95.17%       2028328000                6              torch.float16
------------------------------------------------------------------------------------------------------------------------------------------
Total execution time: 6.297374 seconds
Number of batch iterations: 1
==========================================================================================================================================
```

### get_modules()

This function allows getting found modules. Its basic usage returns all of them:

```python
# Get all executed modules
modules = modules_info.get_modules()
```

You can place additional criteria:

- **min_execution_percentage** - minimum percentage of total execution count e.g. .9
- **limit** - maximum number of modules to return, e.g., 5

If those criteria are not sufficient, you can manually filter modules:

```python
modules_info = tt.inspect(model, input_data)

# Get only transformer blocks
transformer_modules = [m for m in modules_info.get_modules() if "transformer" in m.name.lower()]
```

## Troubleshooting

### Issue: No modules found

```python
# Solution: Change depth or check object structure
modules_info = tt.inspect(model, input_data, depth=1)
```

## Summary

Inspection can be a first step in exploring your model's structure and performance. It can also be used to select modules for tuning. For details on the tuning workflow, head to the [Tuning Guide](tuning.md).
