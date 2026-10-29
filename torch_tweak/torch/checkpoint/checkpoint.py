# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Checkpoint abstract base class."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.nn import Module

from torch_tweak.torch.module.tunable_module import TunableModule


class Checkpoint(ABC):
    """Abstract base class for checkpoint operations.

    This class defines the interface for saving and loading tuned modules.
    """

    @abstractmethod
    def save(self, module: Module | TunableModule, path: str | Path) -> None:
        """Save a module to a checkpoint.

        Args:
            module: The module to save. Can be a regular torch module or a wrapped module.
            path: The path where the module should be saved.

        Raises:
            NotImplementedError: If the concrete class does not implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        module: Module,
        path: str | Path,
        device_map: dict[str, str | torch.device] | None = None,
    ) -> Module | TunableModule:
        """Load a module from a checkpoint.

        Args:
            module: The base module to load into.
            path: The path from where the module should be loaded.
            device_map: The device map to load module to.

        Returns:
            The loaded module, which may be either the original module type or a wrapped module.

        Raises:
            NotImplementedError: If the concrete class does not implement this method.
        """
        raise NotImplementedError
