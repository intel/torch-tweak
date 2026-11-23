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
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy, NoCorrectBackendError
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


def _is_on(tensor: torch.Tensor, device: torch.device) -> bool:
    return tensor.device.type == device.type and device.index in (None, tensor.device.index)


def _misplaced(module: nn.Module, device: torch.device) -> list[str]:
    tensors = itertools.chain(module.named_parameters(), module.named_buffers())
    return [name for name, tensor in tensors if not _is_on(tensor, device)]


def _shapes_key(metadata: SampleMetadata) -> Hashable:
    """Names the place, shape, and dtype of every tensor, and every non-tensor value.

    `SampleMetadata` equality compares only tensor ranks, and only the LLM phase when `cache_position`
    is passed by name, so the key does not rely on it.
    """
    tensors = tuple((locator, spec.name, tuple(spec.shape), spec.dtype) for locator, spec in metadata.tensor_data)
    return metadata.llm_phase, tensors, metadata.other_data


class OnlineTunedModule:
    """Calls `module` through the backend tuned for the exact shapes of each call.

    The first call with new shapes runs `module`, then tunes those shapes with `strategy`.
    Later calls with the same shapes run the selected backend.
    Only the backend of the last call stays active, since an OpenVINO model holds its own copy of the weights.
    Returning to other shapes activates their backend again, which can recompile it.

    The module stays where it is and keeps its `forward`. Its owner can keep calling and patching it.
    Call `reset` after the owner changes the weights in any way: moves them, replaces them, or patches their
    values in place. A backend serves the weights it was built on. OpenVINO keeps a snapshot
    of their values, and other backends hold their addresses.
    Every parameter and buffer of `module` must already be on `device`, since the backends build on that device and
    leave the module there.

    When no backend of `strategy` runs a shape, or its backend fails to activate again, the call still returns
    the module output. The error is logged, and later calls with that shape run `module` without a backend.
    So do calls whose inputs cannot be copied, and calls with a non-tensor input that cannot be hashed.
    Any other error from tuning reaches the caller, and the next call with that shape tunes again.

    Not thread safe: call it from one thread at a time.
    """

    # Shared by every instance, so two modules with the same name never write to one graph directory.
    _graph_counter = itertools.count()

    def __init__(
        self,
        module: nn.Module,
        name: str,
        strategy: TuneStrategy | None = None,
        device: torch.device | None = None,
    ):
        """Initializes module.

        Args:
            module: Module with all its parameters and buffers on `device`.
            name: Names the cache directory of the tuned graphs.
            strategy: Tunes each new shape. Defaults to `borrowed_module_strategy(device)`.
                Its backends must leave `module` and its weights on `device`.
                `OpenVINOBackendConfig(offload_module=False)` and `TorchAOBackendConfig(offload_module=False)`
                do that.
                A call that tunes and finds a parameter or buffer moved off `device` raises `ValueError`.
            device: Device to tune and run on. Defaults to the device of the module parameters.
        """
        self._module = module
        self._name = sanitize_model_name(name)
        self._device = device or get_module_device(module)
        if self._device is None:
            raise ValueError("Pass `device` for a module without parameters.")
        misplaced = _misplaced(module, self._device)
        if misplaced:
            raise ValueError(
                f"Every parameter and buffer must be on {self._device} before tuning. Not on it: {misplaced[:3]}"
            )
        self._strategy = strategy or borrowed_module_strategy(self._device)
        self._backends: dict[Hashable, Backend] = {}
        self._active: Backend | None = None
        self._failed: set[Hashable] = set()
        self._warned_unkeyed = False

    @property
    def backends(self) -> dict[Hashable, Backend]:
        """The selected backend of every tuned shape."""
        return self._backends

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Runs the backend tuned for these shapes, tuning them first if they are new."""
        try:
            inputs = SampleMetadata.from_inputs(args, kwargs, strict=True)
            key = _shapes_key(inputs)
            backend = self._backends.get(key)
        except TypeError:
            if not self._warned_unkeyed:
                logger.warning(
                    "Cannot key a call to %s by its inputs. Such calls run the module without a backend.",
                    self._name,
                    exc_info=True,
                )
                self._warned_unkeyed = True
            return self._module(*args, **kwargs)
        if backend is not None:
            if backend is not self._active:
                self._deactivate_active()
                try:
                    backend.activate()
                except Exception:
                    del self._backends[key]
                    self._mark_failed(key, "Activating the backend")
                    return self._module(*args, **kwargs)
                self._active = backend
            return backend.infer(*args, **kwargs)
        if key in self._failed:
            return self._module(*args, **kwargs)
        try:
            sample = deepcopy((args, kwargs))
        except Exception:
            self._mark_failed(key, "Copying the inputs")
            return self._module(*args, **kwargs)
        output = self._module(*args, **kwargs)
        self._deactivate_active()
        try:
            backend = self._tune(sample, inputs, SampleMetadata.from_outputs(output, strict=True))
        except NoCorrectBackendError:
            self._mark_failed(key, "Every backend")
        else:
            self._active = self._backends[key] = backend
        return output

    def reset(self) -> None:
        """Drops every tuned shape. The next call with any shape tunes again."""
        self._deactivate_active()
        self._backends.clear()
        self._failed.clear()

    def _mark_failed(self, key: Hashable, step: str) -> None:
        logger.warning(
            "%s failed for %s. This shape runs the module without a backend.", step, self._name, exc_info=True
        )
        self._failed.add(key)

    def _deactivate_active(self) -> None:
        if self._active is not None:
            self._active.deactivate()
            self._active = None

    def _tune(self, sample: tuple[tuple, dict], inputs: SampleMetadata, outputs: SampleMetadata) -> Backend:
        graph_spec = GraphSpec(name=str(next(self._graph_counter)), input_spec=inputs, output_spec=outputs)
        cache_dir = torch_tweak_cache_dir() / self._name / graph_spec.name
        secure_mkdir(cache_dir)
        backend = self._strategy.tune(self._module, self._name, graph_spec, [sample], self._device, cache_dir)
        misplaced = _misplaced(self._module, self._device)
        if misplaced:
            backend.deactivate()
            raise ValueError(
                f"Tuning moved parameters or buffers off {self._device}: {misplaced[:3]}."
                " Every backend of the strategy must leave the module on its device."
            )
        return backend
