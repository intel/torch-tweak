<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
NOTE: This file has been modified by Intel Corporation.
-->
# Install

This page is the install guide. The repository [README](../../README.md) only summarizes the stable commands.

## Prerequisites

Before proceeding with the installation of Torch Tweak, ensure your system meets the following criteria:

* **Operating System**: Linux (Ubuntu 22.04+ recommended)
* **Python**: Version `3.10` or newer
* **PyTorch**: Version `2.13` or newer **with XPU support**
* **Intel GPU**: Required for XPU-accelerated tuning

## pip

PyTorch XPU wheels are not on PyPI. The `xpu` extra declares `torch`. You only have to pass the XPU index:

```bash
pip install \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple \
    -e ".[xpu,dev]"
```

Nightly stack of PyTorch **is not supported** on the pip path. Use uv for nightly wheels.

## uv (recommended)

**Stable** is the default stack; **nightly** is uv-only.
Configuration: `[dependency-groups]`, optional extra `xpu`, and `[tool.uv]` indexes are defined in `pyproject.toml`. You cannot install both PyTorch stacks in one resolved environment
(see `conflicts` in `pyproject.toml`), but you may either keep two venvs (recommended) or switch stacks in a single `.venv` with the correct `uv sync` flags each time.

**Stable (default)** uses the project environment in `.venv`:

```bash
uv sync --extra dev
uv run pytest
```

**Nightly (optional)**. The recommended option is a separate environment, so stable and nightly stay installed side by side:

```bash
UV_PROJECT_ENVIRONMENT=.venv-nightly uv sync --no-default-groups --group xpu-nightly --extra dev
UV_PROJECT_ENVIRONMENT=.venv-nightly uv run pytest
```

**Nightly in the same `.venv` as stable (optional)**. `uv sync` reconciles the environment (removes packages not in the current resolution unless you pass `--inexact`).
Switching reinstalls `torch` and related wheels (the cache speeds downloads, but it is not instant):

```bash
# nightly
uv sync --no-default-groups --group xpu-nightly --extra dev

# back to stable
uv sync --extra dev
```

Rules:

1. Never resolve with **both** `xpu-nightly` and `xpu-stable` active (same sync invocation). Two directories (`.venv` + `.venv-nightly`) are **recommended**; one `.venv` is fine if you always use the matching sync command when you switch.
2. Nightly sync must include `--no-default-groups --group xpu-nightly` so `default-groups = ["xpu-stable"]` is not applied. For a dedicated nightly tree, set `UV_PROJECT_ENVIRONMENT=.venv-nightly` (or any path) on `uv sync` / `uv run`.
3. Do not use `uv sync --only-group ...` together with `--extra`. Use `--no-default-groups --group xpu-nightly --extra dev` for nightly.
4. Do not use `uv sync --all-extras` expecting both PyTorch stacks. Do not combine the `xpu` extra with the `xpu-nightly` group in one environment (`conflicts` in `pyproject.toml`).

## Installing from Source

```bash
git clone https://github.com/intel/torch-tweak
cd torch-tweak
pip install .
```

`pip install .` / `pip install -e .` installs **torch-tweak only** (no `torch`). Use the pip or uv commands above for a full stack.
