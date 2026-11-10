# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Torch tuning module."""

from torch_tweak.troubleshooting import COMMON_ISSUES_URL, CPU_TORCH_INSTALL, XPU_TORCH_INSTALL

try:
    import torch  # noqa: F401
except ModuleNotFoundError as err:
    if err.name != "torch":
        raise
    raise ImportError(
        "PyTorch is not installed. Torch Tweak requires PyTorch.\n"
        f"For an Intel GPU:\n    {XPU_TORCH_INSTALL}\n"
        f"For CPU only:\n    {CPU_TORCH_INSTALL}\n"
        f"See {COMMON_ISSUES_URL}"
    ) from err

# configuring some variables before importing other modules
from torch_tweak.torch.config import torch_tweak_cache_dir, config  # noqa: I001

from torch_tweak.torch.checkpoint.local_torch_storage import LocalTorchStorage
from torch_tweak.torch.dataloader import DataLoaderFactory
from torch_tweak.torch.inspecting import inspect, wrap
from torch_tweak.torch.module import TunableModule
from torch_tweak.torch.tune_strategy import (
    FirstWinsStrategy,
    HighestThroughputStrategy,
    OneBackendStrategy,
    TuneStrategy,
    default_strategy,
)
from torch_tweak.torch.tuning import load, save, tune

__all__ = [
    "torch_tweak_cache_dir",
    "config",
    "inspect",
    "wrap",
    "tune",
    "load",
    "save",
    "TunableModule",
    "TuneStrategy",
    "OneBackendStrategy",
    "FirstWinsStrategy",
    "HighestThroughputStrategy",
    "default_strategy",
    "LocalTorchStorage",
    "DataLoaderFactory",
]

if config.enable_hf_integrations:
    import torch_tweak.torch.integrations.hugging_face  # noqa: F401
