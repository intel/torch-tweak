<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Auto tune

Finds the fastest cut depth of a four-layer transformer encoder with `tweak`.
Uses an Intel XPU when one is available, and CPU otherwise.

The demo runs in three steps:

1. A dry run prints how many modules every depth wraps. This is the cost of the search.
2. `tweak` tunes every depth, measures the whole model, and prints the latency of every depth.
3. The winner is saved, loaded into a fresh model, and compared with the untuned output.

The Tweak Guide (`docs/guides/tweak.md`) explains how depths are compared.

## Run from the repository

From the repository root, after `uv sync --extra dev`:

```bash
uv run python examples/Tweak/demo.py
```

## Run this example on its own

From this directory. PyTorch comes from the `xpu` extra (Intel XPU wheels).

```bash
uv sync
uv run demo
```
