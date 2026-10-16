# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""First Wins tune strategy."""

from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from torch_tweak.torch.backend import OpenVINOBackend, TorchEagerBackend, TorchInductorBackend
from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from torch_tweak.utils.logging import control_output, log
from torch_tweak.utils.timer import Timer


class FirstWinsStrategy(TuneStrategyFindMaxBatchSizeExtension):
    """Runs backends in order and keeps the first one that builds and validates.

    The default order is Torch Inductor, OpenVINO, then eager.
    """

    def __init__(self, backends: Iterable[Backend] | None = None, **kwargs):
        """Initializes strategy."""
        super().__init__(**kwargs)
        self._backends = backends or self._default_backends()

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
        selected_backend = None
        log(
            "⏳ Executing strategy `%s` on module `%s` (graph: %s)",
            self.__class__.__name__,
            name,
            graph_spec.name,
            sink=self._sink,
        )

        # Run backend by backend until first working backend is found
        for backend in self._backends:
            backend_cache_dir = cache_dir / backend.key()
            log_file = self._log_file(backend_cache_dir, "build.log")

            with Timer(sink=self._sink, depth=2):
                try:
                    log("⚙️ backend:  %s", backend.describe(), sink=self._sink)
                    log("🔄 in progress...please wait", depth=2, sink=self._sink)
                    with control_output(log_file=log_file):
                        backend = deepcopy(backend)
                        backend = backend.build(module, graph_spec, deepcopy(data), device, backend_cache_dir)
                    log("✅ backend built", depth=2, sink=self._sink)
                    self.check_correctness(backend, name, graph_spec, data)
                    log("✅ backend validated", depth=2, sink=self._sink)
                    selected_backend = backend
                    break
                except Exception:
                    if backend.is_active:
                        backend.deactivate()
                    log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)
                    module.to(device)  # move module back to device as failed backend could move it to cpu

        if selected_backend:
            log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
            log("✅ Selected backend: %s", selected_backend.describe(), sink=self._sink)
            return selected_backend

        raise RuntimeError(f"There is no valid backend for a module: {name}, graph_spec: {graph_spec}")

    def _default_backends(self) -> list[Backend]:
        """Returns default backends."""
        return [
            TorchInductorBackend(),
            OpenVINOBackend(),
            TorchEagerBackend(),
        ]

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        return [
            "name: First Wins Strategy",
            "description: evaluate backends in order, return first working backend",
            "backends:",
            *[f"  {backend.describe()}" for backend in self._backends],
        ]
