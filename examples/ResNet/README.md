<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# ResNet models tuning with Torch Tweak

This example demonstrates how to use Torch Tweak to tune a ResNet model.

## Environment Setup

You can use either of the following options to set up the environment:

### Option 1 - virtual environment managed by you

Activate your virtual environment and install the dependencies:

```bash
pip install --index-url https://download.pytorch.org/whl/xpu \
		--extra-index-url https://pypi.org/simple \
		.
```

### Option 2 - virtual environment managed by `uv`

Install dependencies:

```bash
uv sync
```

## Usage

To tune the ResNet model, run:

```bash
tune --model-name resnet50
```

or for uv:

```bash
uv run tune --model-name resnet50
```

After tuning, run classification with:

```bash
inference --model-name resnet50 --image-path your_image
```

or for uv:

```bash
uv run inference --model-name resnet50 --image-path your_image
```

## Model Details

* [Hugging Face Model](https://huggingface.co/timm)
* [Pytorch Documentation](https://pytorch.org/vision/stable/models.html#classification)
