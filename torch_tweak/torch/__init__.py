# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Torch tuning module."""

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
