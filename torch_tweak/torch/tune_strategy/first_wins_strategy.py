# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""First Wins tune strategy."""

from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn

from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.tune_strategy.default_backends import default_backends
from torch_tweak.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from torch_tweak.utils.logging import log
from torch_tweak.utils.timer import Timer


class FirstWinsStrategy(TuneStrategyFindMaxBatchSizeExtension):
    """Runs backends in order and keeps the first one that builds and validates.

    Without `backends` it tries `default_backends(device)` for the tuning device.
    """

    def __init__(self, backends: Iterable[Backend] | None = None, **kwargs):
        """Initializes strategy."""
        super().__init__(**kwargs)
        self._backends = backends

    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ) -> Backend:
        """Tunes given torch module with provided graph_spec and data."""
        log(
            "⏳ Executing strategy `%s` on module `%s` (graph: %s)",
            self.__class__.__name__,
            name,
            graph_spec.name,
            sink=self._sink,
        )

        for backend in self._backends or default_backends(device):
            try:
                with Timer(sink=self._sink, depth=2):
                    selected = self._build_and_validate(backend, module, name, graph_spec, data, device, cache_dir)
            except Exception:
                continue
            log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
            log("✅ Selected backend: %s", selected.describe(), sink=self._sink)
            return selected

        raise RuntimeError(f"There is no valid backend for a module: {name}, graph_spec: {graph_spec}")

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        backends = (
            [backend.describe() for backend in self._backends] if self._backends else ["default_backends(device)"]
        )
        return [
            "name: First Wins Strategy",
            "description: evaluate backends in order, return first working backend",
            "backends:",
            *[f"  {backend}" for backend in backends],
        ]
