<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# Nemo ASR Parakeet CTC 0.6B Pipeline Tuning with Torch Tweak

This example demonstrates how to use Torch Tweak to tune the Nemo ASR with Parakeet-CTC-0.6B model.

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

### Sample audio file

The example uses a sample audio file that is **downloaded automatically** when you run the
commands below without an explicit `--audio_path`.
You can also download it manually:

```bash
wget https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
```

### Tuning and inference the model

To tune the ASR model, run:

```bash
tune
```

or

```bash
uv run tune
```

To infer the ASR model, run:

```bash
inference
```

or

```bash
uv run inference
```

## Model Details

* [Hugging Face Model](https://huggingface.co/nvidia/parakeet-ctc-0.6b)
* [Nemo Documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/asr/models.html)
