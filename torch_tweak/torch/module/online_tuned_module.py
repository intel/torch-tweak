# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tunes every new input shape on its first call, inside the process that serves it."""

import hashlib
import itertools
import os
import shutil
import tempfile
import weakref
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
from torch_tweak.torch.module.sample_metadata import CHECKPOINTABLE_VALUE_TYPES, SampleMetadata
from torch_tweak.torch.module.tunable_module import sanitize_model_name
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy, NoCorrectBackendError
from torch_tweak.torch.tune_strategy.default_backends import default_backends
from torch_tweak.torch.tune_strategy.tune_strategy import TuneStrategy
from torch_tweak.torch.utils.module import get_module_device, weights_fingerprint
from torch_tweak.utils.permissions import secure_file, secure_mkdir

logger = getLogger(__name__)

SELECTED_BACKEND_FILE = "selected_backend.txt"
PRIVATE_DIR_PREFIX = "private-"
LINKED_FILE_MIN_BYTES = 1 << 20


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
    Each new non-tensor value tunes apart, so pass a value that changes on every call as a tensor.
    The backends run under `torch.no_grad`, so only the first call with new shapes records autograd history.

    Each instance tunes in a private directory under `<cache>/<name>`, since a backend reads its files again
    when it activates. `reset` removes that directory, and so does the end of the instance or of the process.
    The selected backend and its files are then published to `<cache>/<name>/<weights>/<shapes>`.
    A later process with the same weights builds the published backend for those shapes and skips tuning.

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
            name: Names the cache directory of the private and the published graphs.
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
        self._warned_unkeyed: set[tuple[type, str]] = set()
        self._weights_fingerprint: str | None = None
        self._graph_names = itertools.count()
        self._create_private_dir()

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
        except TypeError as error:
            error_kind = (type(error), str(error))
            if error_kind not in self._warned_unkeyed:
                logger.warning(
                    "Cannot key a call to %s by its inputs. Such calls run the module without a backend.",
                    self._name,
                    exc_info=True,
                )
                self._warned_unkeyed.add(error_kind)
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
        graph_spec = GraphSpec(
            name=str(next(self._graph_names)),
            input_spec=inputs,
            output_spec=SampleMetadata.from_outputs(output, strict=True),
        )
        try:
            backend = self._restore_or_tune(key, graph_spec, sample)
        except NoCorrectBackendError:
            self._mark_failed(key, "Every backend")
        else:
            self._active = self._backends[key] = backend
        return output

    def reset(self) -> None:
        """Drops every tuned shape and removes the private graph directory.

        The next call with any shape restores a published selection for the current weights or tunes again.
        """
        self._deactivate_active()
        self._backends.clear()
        self._failed.clear()
        self._weights_fingerprint = None
        self._remove_private_dir()
        self._create_private_dir()

    def _mark_failed(self, key: Hashable, step: str) -> None:
        logger.warning(
            "%s failed for %s. This shape runs the module without a backend.", step, self._name, exc_info=True
        )
        self._failed.add(key)

    def _deactivate_active(self) -> None:
        if self._active is not None:
            self._active.deactivate()
            self._active = None

    def _create_private_dir(self) -> None:
        parent = torch_tweak_cache_dir() / self._name
        secure_mkdir(parent)
        self._private_dir = Path(tempfile.mkdtemp(prefix=PRIVATE_DIR_PREFIX, dir=parent))
        self._remove_private_dir = weakref.finalize(self, shutil.rmtree, self._private_dir, ignore_errors=True)

    def _restore_or_tune(self, key: Hashable, graph_spec: GraphSpec, sample: tuple[tuple, dict]) -> Backend:
        graph_dir = self._private_dir / graph_spec.name
        secure_mkdir(graph_dir)
        published = self._published_dir(key)
        if published is not None:
            backend = self._restore(published, graph_spec, sample, graph_dir)
            if backend is not None:
                return backend
        backend = self._strategy.tune(self._module, self._name, graph_spec, [sample], self._device, graph_dir)
        self._raise_if_moved(backend)
        if published is not None:
            try:
                _publish(graph_dir / backend.key(), backend.key(), published)
            except OSError:
                logger.warning("Publishing %s to %s failed", backend.describe(), published, exc_info=True)
        return backend

    def _published_dir(self, key: Hashable) -> Path | None:
        """The directory shared by every process for these weights, shapes, device and strategy.

        None when `key` has no stable name: a non-tensor value other than a primitive has no `repr` that names it
        across processes.
        """
        _, _, other_data = key
        if any(type(value) not in CHECKPOINTABLE_VALUE_TYPES for _, _, value in other_data):
            return None
        if self._weights_fingerprint is None:
            self._weights_fingerprint = weights_fingerprint(self._module)
        name = repr((self._device.type, self._strategy.describe(), key))
        digest = hashlib.sha256(name.encode()).hexdigest()[:16]
        return torch_tweak_cache_dir() / self._name / self._weights_fingerprint / digest

    def _restore(
        self, published: Path, graph_spec: GraphSpec, sample: tuple[tuple, dict], graph_dir: Path
    ) -> Backend | None:
        selected = published / SELECTED_BACKEND_FILE
        if not selected.exists():
            return None
        backend_key = selected.read_text(encoding="utf-8").strip()
        for candidate in self._strategy.candidate_backends(self._device):
            if candidate.key() == backend_key:
                break
        else:
            return None
        backend = deepcopy(candidate)
        backend_dir = graph_dir / backend_key
        try:
            shutil.copytree(published / backend_key, backend_dir, copy_function=_link_or_copy)
            backend.build(self._module, graph_spec, [sample], self._device, backend_dir)
        except Exception:
            logger.warning("Restoring %s from %s failed, tuning again", backend.describe(), published, exc_info=True)
            shutil.rmtree(backend_dir, ignore_errors=True)
            shutil.rmtree(published, ignore_errors=True)
            return None
        self._raise_if_moved(backend)
        logger.info("Restored %s from %s", backend.describe(), published)
        return backend

    def _raise_if_moved(self, backend: Backend) -> None:
        misplaced = _misplaced(self._module, self._device)
        if misplaced:
            backend.deactivate()
            raise ValueError(
                f"Tuning moved parameters or buffers off {self._device}: {misplaced[:3]}."
                " Every backend of the strategy must leave the module on its device."
            )


def _link_or_copy(source: str, target: str) -> None:
    """Links a large file and copies a small one.

    Backends rewrite small files such as `config.json` in place on every build, and a link would carry that
    write into the published directory.
    """
    if os.path.getsize(source) >= LINKED_FILE_MIN_BYTES:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def _publish(backend_dir: Path, backend_key: str, published: Path) -> None:
    """Publishes the build directory of the selected backend once, with a single rename.

    A published directory never changes, since other processes read the IR in it when they activate a backend.
    The first process to publish wins, and later ones drop their copy.
    """
    secure_mkdir(published.parent)
    staging = Path(tempfile.mkdtemp(dir=published.parent))
    try:
        target = staging / backend_key
        if backend_dir.is_dir():
            shutil.copytree(backend_dir, target, copy_function=_link_or_copy)
        else:
            secure_mkdir(target)
        selected = staging / SELECTED_BACKEND_FILE
        selected.write_text(backend_key, encoding="utf-8")
        secure_file(selected)
        try:
            staging.rename(published)
        except OSError:
            if not published.is_dir():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
