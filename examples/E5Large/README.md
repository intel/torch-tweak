<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# E5 Large V2 Embedding

This example demonstrates how to use Torch Tweak to optimize the HuggingFace E5Large v2 embeddings.

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

### Tuning and inference the model

To optimize the embedding model, run:

```bash
tune
```

or

```bash
uv run tune
```

To infer the embedding model, run:

```bash
inference --prompt "query: What is the capital city of France?"
```

or

```bash
uv run inference --prompt "query: What is the capital city of France?"
```

### Command-Line Options

- `--model-name`: SentenceTransformer model name (default: "intfloat/e5-large-v2")
- `--tuned-model-path`: Path to save/load the tuned model (default: "e5large_tuned.pt")
- `--prompt`: Text prompt for embedding (default: "query: how much protein should a female eat")
- `--max-batch-size`: Maximum batch size (default: 4)

## Model Details

* [Hugging Face Model](https://huggingface.co/intfloat/e5-large-v2)
