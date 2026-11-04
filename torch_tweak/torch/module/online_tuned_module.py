# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tunes every new input shape on its first call, inside the process that serves it."""

import hashlib
import json
from collections.abc import Hashable
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.backend.openvino import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.config import torch_tweak_cache_dir
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.module.tunable_module import sanitize_model_name
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy, NoCorrectBackendError
from torch_tweak.torch.tune_strategy.default_backends import default_backends
from torch_tweak.torch.tune_strategy.tune_strategy import TuneStrategy
from torch_tweak.torch.utils.module import get_module_device, weights_fingerprint
from torch_tweak.utils.permissions import secure_file, secure_mkdir

logger = getLogger(__name__)

SELECTED_BACKEND_FILE = "selected_backend.txt"


def borrowed_module_strategy(device: torch.device) -> HighestThroughputStrategy:
    """Measures every default backend for `device` and keeps the fastest, without moving the module.

    OpenVINO keeps the module on its device after conversion, so the backends tuned for other shapes keep
    pointing at the weights.
    It also reuses the IR that an earlier process converted for the same weights and shapes.
    """
    backends = [
        OpenVINOBackend(OpenVINOBackendConfig(offload_module=False, reuse_ir=True))
        if isinstance(backend, OpenVINOBackend)
        else backend
        for backend in default_backends(device)
    ]
    strategy = HighestThroughputStrategy(backends=backends)
    strategy.enable_find_max_batch_size(False)
    return strategy


def _is_on(weight: torch.Tensor, device: torch.device) -> bool:
    return weight.device.type == device.type and device.index in (None, weight.device.index)


class OnlineTunedModule:
    """Calls `module` through the backend tuned for the exact shapes of each call.

    The first call with new shapes runs `module`, then tunes those shapes with `strategy`.
    Later calls with the same shapes run the selected backend.
    Only the backend of the last call stays active, since an OpenVINO model holds its own copy of the weights.
    Returning to other shapes activates their backend again, which can recompile it.

    The selection is saved under the cache directory, keyed by the weights and the shapes.
    A later process with the same weights builds the saved backend for those shapes and skips tuning.
    A selection measured while another process used the device is not saved.

    The module stays where it is and keeps its `forward`. Its owner can keep calling and patching it.
    Call `reset` after the owner changes the weights in any way: moves them, replaces them, or patches their
    values in place. A backend serves the weights it was built on. OpenVINO keeps a snapshot
    of their values, and other backends hold their addresses.
    Every parameter of `module` must already be on `device`, since the backends build on that device and leave the
    module there.

    When no backend of `strategy` runs a shape, the call still returns the module output.
    The error is logged, and later calls with that shape run `module` without a backend.
    Any other error from tuning reaches the caller, and the next call with that shape tunes again.

    Not thread safe: call it from one thread at a time.
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
            module: Module with all its parameters on `device`.
            name: Names the cache directory of the tuned graphs.
            strategy: Tunes each new shape. Defaults to `borrowed_module_strategy(device)`.
                Its backends must leave `module` and its weights on `device`.
                `OpenVINOBackendConfig(offload_module=False)` does that, and `TorchAOBackend` does not.
            device: Device to tune and run on. Defaults to the device of the module parameters.
        """
        self._module = module
        self._name = sanitize_model_name(name)
        self._device = device or get_module_device(module)
        misplaced = [name for name, weight in module.named_parameters() if not _is_on(weight, self._device)]
        if misplaced:
            raise ValueError(f"Every parameter must be on {self._device} before tuning. Not on it: {misplaced[:3]}")
        self._strategy = strategy or borrowed_module_strategy(self._device)
        self._backends: dict[Hashable, Backend] = {}
        self._active: Backend | None = None
        self._failed: set[Hashable] = set()
        self._weights_fingerprint: str | None = None

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
        if key in self._failed:
            return self._module(*args, **kwargs)
        sample = deepcopy((args, kwargs))
        output = self._module(*args, **kwargs)
        self._deactivate_active()
        # A strategy can update the shapes of the spec it gets, so the dict key must not be that object.
        graph_spec = GraphSpec(
            name=_shapes_digest(metadata, key[1]),
            input_spec=SampleMetadata.from_inputs(*sample, strict=True),
            output_spec=SampleMetadata.from_outputs(output, strict=True),
        )
        cache_dir = self._cache_dir(graph_spec)
        try:
            backend = self._restore(graph_spec, sample, cache_dir) or self._tune(graph_spec, sample, cache_dir)
        except NoCorrectBackendError:
            logger.warning("No backend runs %s for this shape. It runs without a backend.", self._name, exc_info=True)
            self._failed.add(key)
        else:
            self._active = self._backends[key] = backend
        return output

    def reset(self) -> None:
        """Drops every tuned shape. The next call with any shape tunes again."""
        self._deactivate_active()
        self._backends.clear()
        self._failed.clear()
        self._weights_fingerprint = None

    def _deactivate_active(self) -> None:
        if self._active is not None:
            self._active.deactivate()
            self._active = None

    def _cache_dir(self, graph_spec: GraphSpec) -> Path:
        if self._weights_fingerprint is None:
            self._weights_fingerprint = weights_fingerprint(self._module)
        cache_dir = torch_tweak_cache_dir() / self._name / self._weights_fingerprint / graph_spec.name
        secure_mkdir(cache_dir)
        return cache_dir

    def _restore(self, graph_spec: GraphSpec, sample: tuple[tuple, dict], cache_dir: Path) -> Backend | None:
        selected = cache_dir / SELECTED_BACKEND_FILE
        if not selected.exists():
            return None
        backend_key = selected.read_text().strip()
        for candidate in self._strategy.candidate_backends(self._device):
            if candidate.key() != backend_key:
                continue
            backend = deepcopy(candidate)
            backend_cache_dir = cache_dir / backend_key
            secure_mkdir(backend_cache_dir)
            try:
                backend.build(self._module, graph_spec, [sample], self._device, backend_cache_dir)
            except Exception:
                logger.warning("Restoring %s for %s failed, tuning again", backend.describe(), cache_dir, exc_info=True)
                return None
            logger.info("Restored %s for %s", backend.describe(), cache_dir)
            return backend
        return None

    def _tune(self, graph_spec: GraphSpec, sample: tuple[tuple, dict], cache_dir: Path) -> Backend:
        backend = self._strategy.tune(self._module, self._name, graph_spec, [sample], self._device, cache_dir)
        if not self._strategy.tuned_on_quiet_device:
            logger.warning("Another process used the device while %s was tuned. Its selection is not saved", cache_dir)
            return backend
        selected = cache_dir / SELECTED_BACKEND_FILE
        selected.write_text(backend.key())
        secure_file(selected)
        return backend


def _shapes_digest(metadata: SampleMetadata, shapes: tuple) -> str:
    described = json.dumps([metadata.to_dict(), repr(shapes)], sort_keys=True, default=str)
    return hashlib.sha256(described.encode()).hexdigest()[:16]
