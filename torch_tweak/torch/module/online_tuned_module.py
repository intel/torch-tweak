# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tunes every new input shape on its first call, inside the process that serves it."""

import itertools
from collections.abc import Hashable
from copy import deepcopy
from logging import getLogger
from typing import Any

import torch
import torch.nn as nn

from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.backend.openvino import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.config import torch_tweak_cache_dir
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.module.tunable_module import sanitize_model_name
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy
from torch_tweak.torch.tune_strategy.default_backends import default_backends
from torch_tweak.torch.tune_strategy.tune_strategy import TuneStrategy
from torch_tweak.torch.utils.module import get_module_device
from torch_tweak.utils.permissions import secure_mkdir

logger = getLogger(__name__)


def borrowed_module_strategy(device: torch.device) -> HighestThroughputStrategy:
    """Measures every default backend for `device` and keeps the fastest, without moving the module.

    OpenVINO keeps the module on its device after conversion, so the backends tuned for other shapes keep
    pointing at the weights.
    """
    backends = [
        OpenVINOBackend(OpenVINOBackendConfig(offload_module=False))
        if isinstance(backend, OpenVINOBackend)
        else backend
        for backend in default_backends(device)
    ]
    strategy = HighestThroughputStrategy(backends=backends)
    strategy.enable_find_max_batch_size(False)
    return strategy


class OnlineTunedModule:
    """Calls `module` through the backend tuned for the exact shapes of each call.

    The first call with new shapes runs `module`, then tunes those shapes with `strategy`.
    Later calls with the same shapes run the selected backend.
    Only the backend of the last call stays active, since an OpenVINO model holds its own copy of the weights.
    Returning to other shapes activates their backend again, which can recompile it.

    The module stays where it is and keeps its `forward`. Its owner can keep calling and patching it.
    Call `reset` after the owner moves or replaces the weights, since some backends hold their addresses.
    """

    def __init__(
        self,
        module: nn.Module,
        name: str,
        strategy: TuneStrategy | None = None,
        device: torch.device | None = None,
    ):
        """Initializes module.

        Args:
            module: Module with its weights loaded on `device`.
            name: Names the cache directory of the tuned graphs.
            strategy: Tunes each new shape. Defaults to `borrowed_module_strategy(device)`.
            device: Device to tune and run on. Defaults to the device of the module parameters.
        """
        self._module = module
        self._name = sanitize_model_name(name)
        self._device = device or get_module_device(module)
        self._strategy = strategy or borrowed_module_strategy(self._device)
        self._backends: dict[Hashable, Backend] = {}
        self._active: Backend | None = None
        self._graph_counter = itertools.count()

    @property
    def backends(self) -> dict[Hashable, Backend]:
        """The selected backend of every tuned shape."""
        return self._backends

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the backend tuned for these shapes, tuning them first if they are new."""
        metadata = SampleMetadata.from_inputs(args, kwargs, strict=True)
        key = (metadata, tuple((tuple(spec.shape), spec.dtype) for spec in metadata.tensor_specs))
        backend = self._backends.get(key)
        if backend is not None:
            if backend is not self._active:
                self._deactivate_active()
                backend.activate()
                self._active = backend
            return backend.infer(*args, **kwargs)
        sample = deepcopy((args, kwargs))
        output = self._module(*args, **kwargs)
        self._deactivate_active()
        self._active = self._backends[key] = self._tune(
            sample, metadata, SampleMetadata.from_outputs(output, strict=True)
        )
        return output

    def reset(self) -> None:
        """Drops every tuned shape. The next call with any shape tunes again."""
        self._deactivate_active()
        self._backends.clear()

    def _deactivate_active(self) -> None:
        if self._active is not None:
            self._active.deactivate()
            self._active = None

    def _tune(self, sample: tuple[tuple, dict], inputs: SampleMetadata, outputs: SampleMetadata) -> Backend:
        graph_spec = GraphSpec(name=str(next(self._graph_counter)), input_spec=inputs, output_spec=outputs)
        cache_dir = torch_tweak_cache_dir() / self._name / graph_spec.name
        secure_mkdir(cache_dir)
        return self._strategy.tune(self._module, self._name, graph_spec, [sample], self._device, cache_dir)
