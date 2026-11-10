<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
NOTE: This file has been modified by Intel Corporation.
-->
# Install

This guide walks you through setting up Torch Tweak together with PyTorch XPU, so you can start tuning
models on Intel GPUs.

## Prerequisites

Before proceeding with the installation of Torch Tweak, ensure your system meets the following criteria:

* **Operating System**: Linux (Ubuntu 22.04+ recommended)
* **Python**: Version `3.11` or newer
* **PyTorch**: Version `2.13` or newer. Tuning on an Intel GPU needs the **XPU build**. Tuning on CPU works with any build.
* **Intel GPU**: Required for XPU-accelerated tuning

## pip

To get **the latest release of torch-tweak**, install it from PyPI. Please note that
PyTorch XPU wheels are not published on PyPI, so the PyTorch XPU index has to be selected explicitly:

```bash
pip install \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple \
    "torch_tweak[xpu]"
```

You can also build and install local torch-tweak (from this repository), but
you still have to pass the XPU index. Please note that for development and testing
of torch-tweak we recommend uv (see uv commands below).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple \
    -e ".[xpu,dev]"
```

Nightly stack of PyTorch XPU **is not supported** on the pip path. Use uv for nightly wheels.

### CPU only

Without an Intel GPU, install the CPU build of PyTorch. It skips the GPU runtime packages:

```bash
pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    torch "torch_tweak[torchao]"
```

An existing CPU or CUDA build of PyTorch also works.
On a machine without an Intel GPU, `tune()` picks the CPU.
On a machine with an Intel GPU, `tune()` expects the XPU build. Pass `device="cpu"` to `tune()`, or set
`TORCH_TWEAK_DEVICE=cpu`, to tune on CPU anyway.

## uv (recommended for dev)

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

## Installing torch-tweak only

```bash
git clone https://github.com/intel/torch-tweak
cd torch-tweak
pip install .
```

`pip install .` or `pip install -e .` installs **torch-tweak only** (no `torch`). Use the pip or uv commands above for a full stack.

The TorchAO backend needs the `torchao` extra.
`torch_tweak[xpu]` installs it.

## Common Issues

### PyTorch is installed but XPU support is unavailable

A CPU or CUDA build of PyTorch cannot use an Intel GPU.
Check the build in the active Python environment:

```bash
python -c "import torch; print(torch.__version__, torch.version.xpu)"
```

An XPU build prints a version ending in `+xpu` and a date. Any other build prints `None`.
Reinstall torch from the XPU index:

```bash
python -m pip install --force-reinstall --index-url https://download.pytorch.org/whl/xpu torch
```

To keep the current build, tune on CPU with `device="cpu"`.

### No Intel GPU is available

Confirm that an Intel GPU is visible to the system and that its driver and runtime are functioning.
Also check that your user or container has permission to access the device. On some Linux systems, for example,
GPU access may require membership in the `render` group.

### Importing PyTorch fails

Make sure you are using the Python environment where PyTorch is installed. If importing PyTorch fails
with a missing library or another dependency error, use the original traceback to troubleshoot that
installation. Torch Tweak only reports PyTorch as missing when the `torch` module itself cannot be found.
