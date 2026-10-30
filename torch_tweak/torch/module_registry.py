# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Inplace model registry."""

import gc
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_tweak.torch.module.tunable_module import TunableModule


class ModuleRegistry:
    """Registry for inplace modules."""

    def __init__(self) -> None:
        """Initialize ModuleRegistry."""
        self._registry: OrderedDict[str, TunableModule] = OrderedDict()

    def register(self, name: str, module: "TunableModule") -> None:
        """Register a module."""
        self._registry[name] = module

    def unregister(self, name: str) -> None:
        """Unregister a module."""
        if name in self._registry:
            del self._registry[name]

    def clear(self) -> None:
        """Removes already registered modules."""
        self._registry = OrderedDict()
        gc.collect()

    @property
    def modules(self) -> dict[str, "TunableModule"]:
        """All registered modules."""
        return self._registry

    def get(self, name: str) -> "TunableModule":
        """Get a module."""
        if name not in self._registry:
            raise ValueError(f"TunableModule {name} not found in registry.")
        return self._registry[name]


MODULE_REGISTRY = ModuleRegistry()
