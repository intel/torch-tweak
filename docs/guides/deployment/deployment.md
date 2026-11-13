<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# Deployment Guide

This guide covers deploying Torch Tweak-tuned models in production environments, from saving tuned models to loading them in production systems.

## Overview

Torch Tweak provides comprehensive tools for model deployment:

- **Save/Load**: Persist and restore tuned models
- **Storage Options**: Local and custom storage backends
- **Verification**: SHA256 checksums for integrity
- **Portability**: Deploy across different environments

## Quick Start

### Save a Tuned Model

```python
import torch_tweak.torch as tt

# After tuning
tt.save(tuned_model, "model.tt")
```

### Load a Tuned Model

```python
import torch_tweak.torch as tt

# In production
model = YourModel()
model = tt.load(model, "model.tt")
output = model(input_data)
```

## Saving Tuned Models

### Basic Save

```python
import torch_tweak.torch as tt

# Save after tuning
tt.save(model, "checkpoints/model.tt")
```

This creates:

- `checkpoints/model.tt`: Compressed checkpoint with tuned modules
- `checkpoints/model_sha256_sums.txt`: SHA256 checksums
- `checkpoints/model/`: Decompressed artifacts (after first load)

The `.tt` file starts with a format version and a record of the tuning environment:
host name, CPU, accelerators and their drivers, and the versions of Python, torch-tweak, torch, and the backend libraries.
`tt.load` raises `tt.IncompatibleCheckpointError` before it reads the tuned modules when the format version or any part of the environment differs.
Tune the model again on the target machine in that case.

### With Custom Storage

```python
from torch_tweak.torch import LocalTorchStorage

# Configure storage
storage = LocalTorchStorage(
    base_folder="production/models",
    remove_checkpoint_after_tune=False,  # Keep intermediate files
)

# Save with custom storage
tt.save(model, "model_v2.tt", storage=storage)
```

## Loading Tuned Models

### Basic Load

```python
import torch_tweak.torch as tt

# Create model instance
model = YourModel()
model.eval()
model.to("xpu")

# Load tuned version
tt.load(model, "checkpoints/model.tt")

# Ready for inference
output = model(input_data)
```

### With Custom Storage

```python
from torch_tweak.torch import LocalTorchStorage

storage = LocalTorchStorage(base_folder="production/models")
tt.load(model, "model.tt", storage=storage)
```

### Loading Process

1. First Load:

- Decompresses `.tt` file
- Extracts artifacts to `checkpoints/` directory
- Verifies checksums
- Loads backend and weights
- Slower (decompression overhead)

2. Subsequent Loads:

- Uses decompressed files from `checkpoints/`
- Skips decompression
- Faster startup

## Next Steps

- Review the [Tuning Guide](../tuning.md)
- Explore [Tune Strategies](../tune_strategies/tune_strategies.md) for optimization
- Check [Backend Guides](../backends/torch_inductor_backend.md) for backend-specific deployment notes
