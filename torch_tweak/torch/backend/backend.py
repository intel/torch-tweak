# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Backend interface."""

import gc
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn

from torch_tweak.torch.checkpoint.storage_tasks import is_weights_only_safe
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.utils.hashing import hash_string
from torch_tweak.utils.permissions import secure_file
from torch_tweak.utils.trace import annotate

logger = logging.getLogger(__name__)


def samples_for_checkpoint(data: list[Sample] | None) -> list[Sample] | None:
    """The samples of `data` that a checkpoint can hold.

    For backends that replay samples only to compile ahead of the first call. A sample holding an
    object such as a transformers `DynamicCache` is left out, and the first call compiles instead.
    """
    if data is None:
        return None
    safe = [sample for sample in data if is_weights_only_safe(sample)]
    if len(safe) < len(data):
        logger.info(
            "%d of %d samples cannot be saved and are left out of the checkpoint.", len(data) - len(safe), len(data)
        )
    return safe


class BackendState(Enum):
    """Enum representing the state of a backend.

    Attributes:
        INIT: The backend is initialized, there was no tuning done yet.
        INACTIVE: The model was tuned but the backend is deactivated, you can't do inference
        ACTIVE: The model was tuned and the backend is activated, you can do inference
        CHECKPOINT_LOADED: The backend was loaded from a checkpoint, it is ready to be deployed
        DEPLOYED: The model was tuned and is ready to use, you can do inference.

    State transitions:
    ```mermaid
    stateDiagram
        [*] --> init
        init --> active: When backend is built successfully
        active --> inactive: When backend is deactivated
        inactive --> active: When backend is activated again
        checkpoint_loaded --> deployed: When backend is deployed from checkpoint
        checkpoint_loaded --> active: When backend is activated from checkpoint
        active --> deployed: When backend is deployed after being active
        deployed --> [*]
    ```

    Note:
    - Changing state to deployed should be done in `ModuleWrapper` as it does a proxy of `forward` method and reverses
    that in order to deploy backends that compile during activation.
    - After loading a checkpoint you can still activate the backend while debugging, but some backends
    compile their implementation again on activation.

    This abstract class has public interface methods which manage state of the backend:
        build(): Builds backend, changes state to ACTIVE
        activate(): Activates backend, changes state to ACTIVE
        deactivate(): Deactivates backend, changes state to INACTIVE
        deploy(): Deploys backend, changes state to DEPLOYED

    Each method calls corresponding private method which should not change state differently - with one exception:
      _build should set state to ACTIVE after building the backend. This is to ensure that rest of the system
            can use the backend in ACTIVE state and prevents unnecessary backend compilation.

    Each backend implementation should implement public methods:
        to_dict(): Returns the state_dict of the backend
        from_dict(): Creates a backend from a state_dict

    and private ones:
        _build(): Builds backend, changes state to ACTIVE
        _activate(): Activates backend, changes state to ACTIVE
        _deactivate(): Deactivates backend, changes state to INACTIVE
        _deploy(): Deploys backend, changes state to DEPLOYED
        _infer(): Runs inference with the given arguments
    """

    INIT = "init"
    INACTIVE = "inactive"
    ACTIVE = "active"
    CHECKPOINT_LOADED = "checkpoint_loaded"
    DEPLOYED = "deployed"


@dataclass
class BackendConfig:
    """Configuration for a backend."""

    def key(self) -> str:
        """Returns the keys of the backend configuration."""
        config_dict = self.to_dict()
        config_dict = self._to_json(config_dict)
        config_dict_str = json.dumps(config_dict)
        key = hash_string(config_dict_str)
        return key

    def describe(self) -> str:
        """Describe the backend configuration. Display only changed fields."""
        default = self.__class__()
        changed_fields = self._get_changed_fields(self, default)
        return ",".join(changed_fields)

    def to_dict(self):
        """Returns the state_dict of the backend."""
        return asdict(self)

    def to_json(self, path: Path):
        """Saves the backend configuration to a file."""
        config_dict = self.to_dict()
        config_dict = self._to_json(config_dict)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
        secure_file(path)

    def _to_json(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format.

        Handles the following types:
        - JSON primitives: int, float, str, bool, None
        - Collections: dict, list, tuple, set
        - Dataclasses: converted to dict
        - Enums: converted to their values
        - Other objects: converted to string representation
        """
        # Handle None explicitly
        if obj is None:
            return None

        # Handle dataclasses
        if is_dataclass(obj):
            obj = asdict(obj)

        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: self._to_json(v) for k, v in obj.items()}

        # Handle sequences (list, tuple, set)
        if isinstance(obj, list | tuple | set):
            return [self._to_json(item) for item in obj]

        # Handle Enums
        if isinstance(obj, Enum):
            return obj.value

        # Handle JSON primitives (these are already JSON-serializable)
        if isinstance(obj, int | float | str | bool):
            return obj

        # Handle other types by converting to string
        return str(obj)

    def _default_describe_fields(self) -> list[str]:
        """Returns the default fields to describe."""
        return []

    def _get_changed_fields(
        self,
        current,
        other,
        exclude: list[str] | None = None,
        include: list[str] | None = None,
    ) -> list[str]:
        """Returns the changed fields of the backend configuration."""
        if exclude is None:
            exclude = []
        if include is None:
            include = self._default_describe_fields()

        changed_fields = []
        for f in fields(current):
            if (getattr(current, f.name) != getattr(other, f.name) and f.name not in exclude) or f.name in include:
                changed_fields.append(f"{f.name}={getattr(current, f.name)}")
        return changed_fields


class Backend(ABC):
    """Backend interface for tuning a module."""

    # Supported devices types
    _devices: ClassVar[list[str]] = ["cpu", "xpu"]
    # Keep the original module after tuning. False offloads it to the meta device.
    requires_original_module: ClassVar[bool] = False

    def __init__(self):
        """Initialize the backend."""
        self.state = BackendState.INIT
        self._device = None

    def build(
        self,
        module: nn.Module,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ) -> "Backend":
        """Build the model with the given arguments.

        Building a backend should be idempotent i.e. do not cause side effects. A model is not necessarily pure
        functional and can have an internal state (like kv cache for LLMs). That is why build can call a sample of
        inputs at most once so that subsequent calls have exact same state as the first call for the given sample.

        After building, the backend should be activated.
        """
        if self.state == BackendState.INIT:
            try:
                self._assert_device(device)
                self._set_device(device)
                ready_backend = self._build(module, graph_spec, data, cache_dir)
                self.state = BackendState.ACTIVE
                return ready_backend
            except Exception as e:
                self._logger.error("Failed to build backend(%s): %s", self.__class__.__name__, e, exc_info=True)
                raise e
        else:
            raise RuntimeError(f"Backend {self.name} build should be called only once")

    @annotate(domain="Torch Tweak")
    def activate(self):
        """Activates backend.

        After activating, the backend should be ready to do inference.
        """
        if self.state == BackendState.INIT:
            raise RuntimeError(f"Cannot activate backend {self.name}, backend should be built first")
        if self.state == BackendState.DEPLOYED:
            raise RuntimeError(f"Cannot activate backend {self.name}, backend is already deployed")

        if self.state == BackendState.INACTIVE or self.state == BackendState.CHECKPOINT_LOADED:
            self._activate()
            self.state = BackendState.ACTIVE

    def deactivate(self):
        """Deactivates backend.

        After deactivating, the backend cannot be used to do inference.
        """
        if self.state == BackendState.INIT:
            raise RuntimeError(f"Cannot deactivate backend {self.name}, backend should be built first")
        if self.state == BackendState.DEPLOYED:
            raise RuntimeError(f"Cannot deactivate backend {self.name}, backend is already deployed")
        if self.state == BackendState.CHECKPOINT_LOADED:
            raise RuntimeError(f"Cannot deactivate backend {self.name}, backend has already been deployed")

        if self.state == BackendState.ACTIVE:
            self._deactivate()
            self._clean_memory()
            self.state = BackendState.INACTIVE

    def deploy(self, device: torch.device | None):
        """Deploys the backend.

        After deploying, the backend is ready to do inference. Backend cannot be deactivated anymore.

        Args:
            device: The device to deploy the backend on.
        """
        if self.state != BackendState.CHECKPOINT_LOADED:
            raise RuntimeError(f"Cannot deploy backend {self.name}, backend should be loaded from a checkpoint")

        self._set_device(device)
        self._deploy()
        self.state = BackendState.DEPLOYED

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference with the given arguments.

        Args:
            args: Variable length argument list.
            kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the inference.
        """
        if self.state != BackendState.ACTIVE and self.state != BackendState.DEPLOYED:
            raise RuntimeError(f"Cannot run inference, backend {self.name} should be activated first")

        return self._infer(*args, **kwargs)

    @abstractmethod
    def key(self) -> str:
        """Returns the key of the backend."""
        pass

    @abstractmethod
    def describe(self) -> str:
        """Returns the description of the backend."""
        pass

    @abstractmethod
    def to_dict(self):
        """Returns the state_dict of the backend.

        Note: if there any binary artifacts (files) which should be stored by a backend,
        they must be passed as Python Path object. Such objects will be bundled with a checkpoint.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, state_dict: dict):
        """Creates a backend from a state_dict."""
        pass

    @property
    def name(self) -> str:
        """Name of a backend."""
        return self.__class__.__name__

    @property
    def device(self) -> torch.device:
        """The device of the backend.

        Returns:
            The device the module is using.
        """
        return self._device

    @property
    def is_active(self) -> bool:
        """True if the backend is active."""
        return self.state == BackendState.ACTIVE

    @abstractmethod
    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> "Backend":
        """Build the model with the given arguments.

        After building, the backend should be activated.

        Args:
            module: The module to build the backend on.
            name: The name of the backend.
            graph_spec: The graph specification of the backend.
            data: The data to build the backend on.
            cache_dir: The cache directory to store the backend artifacts.
        """
        pass

    @abstractmethod
    def _activate(self):
        """Activates backend.

        After activating, the backend should be ready to do inference.
        """
        pass

    @abstractmethod
    def _deactivate(self):
        """Deactivates backend.

        After deactivating, the backend cannot be used to do inference.
        """
        pass

    @abstractmethod
    def _deploy(self):
        """Deploys the backend.

        After deploying, the backend is ready to do inference. Backend cannot be deactivated anymore.
        """
        pass

    @abstractmethod
    def _infer(self, *args, **kwargs):
        """Run inference with the given arguments.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the inference.
        """
        pass

    def _assert_device(self, device: torch.device):
        """Assert the device of the backend.

        Args:
            device: The device to assert.
        """
        if device.type not in self._devices:
            raise ValueError(f"Device {device.type} is not supported by the backend {self.name}")

    def _set_device(self, device: torch.device):
        """Set the device of the backend.

        Args:
            device: The device to set the backend on.
        """
        if device is not None:
            self._device = device

        if self._device is None:
            raise ValueError("Device is not set. Please set the device before deploying.")

    @property
    def _logger(self) -> logging.Logger:
        """A logger specific to this backend implementation."""
        return logging.getLogger(f"{self.__class__.__module__}")

    def _clean_memory(self):
        """Clean up memory."""
        torch._dynamo.reset()
        torch.compiler.reset()
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and hasattr(xpu, "empty_cache") and xpu.is_available():
            xpu.empty_cache()
        gc.collect()


class DummyBackend(Backend):
    """Dummy backend for testing purposes."""

    _devices = ["cpu", "xpu"]

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return "Dummy backend"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Build the model with the given arguments."""
        return self

    def _activate(self):
        """Activate the backend."""
        print("Activating backend")  # noqa: T201

    def infer(self, *args, **kwargs):
        """Run inference with the given arguments."""
        print("Running inference")  # noqa: T201

    def _deactivate(self):
        """Deactivate the backend."""
        print("Deactivating backend")  # noqa: T201

    def _deploy(self):
        """Deploy the backend."""
        print("Deploying backend")  # noqa: T201

    def _infer(self, *args, **kwargs):
        """Run inference with the given arguments."""
        print("Running inference with args: ", args, "and kwargs: ", kwargs)  # noqa: T201

    def to_dict(self):
        """Returns the state_dict of the backend."""
        return {}

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Creates a backend from a state_dict."""
        return cls()
