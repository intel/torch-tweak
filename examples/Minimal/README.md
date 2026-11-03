<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Minimal backend selection

Tunes a small MLP on batch sizes 1 and 4 and prints each backend, fastest first.
The backends are Torch Inductor, the XPU graph, and eager.
The XPU graph is recorded when an Intel GPU is available. CPU keeps Inductor and eager.

## Run from the repository

From the repository root, after `uv sync --extra dev`:

```bash
uv run python examples/Minimal/demo.py
```

## Run this example on its own

From this directory. PyTorch comes from the `xpu` extra (Intel XPU wheels).

```bash
uv sync
uv run demo
```
