# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Torchao backend."""

import copy
import gc
import importlib
import warnings
from dataclasses import MISSING, dataclass, fields
from logging import getLogger
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torchao.core.config import ALLOWED_AO_MODULES, AOBaseConfig, config_from_dict, config_to_dict
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Float8WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    PerTensor,
    quantize_,
)

from torch_tweak.torch.backend.backend import Backend, BackendConfig, BackendState, samples_for_checkpoint
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.utils.trace import annotate

logger = getLogger(__name__)

QuantizationType = Literal["int8wo", "int8dq", "fp8wo", "fp8dq"]
DEFAULT_QUANTIZATION = "fp8wo"


# config_from_dict special cases this name and resolves it as getattr(torch, _data).
_TORCHAO_DTYPE_SENTINEL = "torch.dtype"


def _resolve_ao_classes(type_name: object, path: str) -> list[type]:
    """Resolve *type_name* against the allowed torchao modules, rejecting non-classes.

    ALLOWED_AO_MODULES whitelists entire module namespaces, which export 18 plain
    functions (quantize_, quantize_affine, int_scaled_matmul, ...) next to the config
    classes. Without this check a crafted payload can make config_from_dict call any of
    them with attacker-chosen arguments.

    It is also a set, so config_from_dict resolves a name in an order that need not match
    ours. Checking every candidate rather than the first match makes the result
    independent of iteration order: whichever object config_from_dict picks is one of
    those validated here.

    Returns:
        Every class the name resolves to across the allowed modules.

    Raises:
        ValueError: If *type_name* is not a string, resolves to nothing, or names
            something that is not a class.
    """
    if type(type_name) is not str:
        raise ValueError(f"{path}: _type must be a string, got '{type(type_name).__qualname__}'")
    resolved = []
    for module_path in ALLOWED_AO_MODULES:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        candidate = getattr(module, type_name, None)
        if candidate is None:
            continue
        if not isinstance(candidate, type):
            raise ValueError(f"{path}: type '{type_name}' names a {type(candidate).__qualname__}, not a class")
        resolved.append(candidate)
    if not resolved:
        raise ValueError(f"{path}: type '{type_name}' is not exported by any allowed torchao module")
    return resolved


def _assert_nested_ao_types(value: object, path: str) -> None:
    """Check every _type reachable from *value* without instantiating anything.

    Nested types are a legitimate mix of granularities, layouts and enums rather than
    AOBaseConfig subclasses, so they only have to be classes.

    Raises:
        ValueError: If any nested _type does not name an acceptable class, or a
            "torch.dtype" payload does not name a dtype.
    """
    if isinstance(value, dict):
        type_name = value.get("_type")
        if type_name == _TORCHAO_DTYPE_SENTINEL:
            # config_from_dict resolves this payload as getattr(torch, _data).
            dtype_name = value.get("_data")
            if type(dtype_name) is not str or not isinstance(getattr(torch, dtype_name, None), torch.dtype):
                raise ValueError(f"{path}: torch.dtype payload {dtype_name!r} does not name a torch dtype")
        elif type_name is not None:
            _resolve_ao_classes(type_name, path)
        for key, item in value.items():
            _assert_nested_ao_types(item, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_nested_ao_types(item, f"{path}[{index}]")


def _assert_ao_config_payload(data: object, path: str = "quantization_config") -> None:
    """Validate a serialized torchao config before config_from_dict instantiates it.

    config_from_dict resolves and instantiates a _type before any isinstance check can
    run on the result, so the whole payload has to be validated up front. The top-level
    _type must name an AOBaseConfig subclass; nested ones only have to name classes.

    Args:
        data: Serialized config payload, as produced by config_to_dict.
        path: Human-readable path to *data*, used in error messages.

    Raises:
        ValueError: If the payload is not a serialized config, or holds a _type that
            does not name an acceptable class.
    """
    if not isinstance(data, dict) or "_type" not in data:
        raise ValueError(
            f"{path}: expected a serialized torchao config dict with a '_type' key, got '{type(data).__qualname__}'"
        )
    type_name = data["_type"]
    not_a_config = f"{path}: type '{type_name}' is not a supported AOBaseConfig subclass"
    if type_name == _TORCHAO_DTYPE_SENTINEL:
        raise ValueError(not_a_config)
    for candidate in _resolve_ao_classes(type_name, path):
        if not issubclass(candidate, AOBaseConfig):
            raise ValueError(not_a_config)
    for key, value in data.items():
        _assert_nested_ao_types(value, f"{path}[{key!r}]")


# torchao 0.17.x still materializes deprecated PlainLayout in int8 config __post_init__.
# TODO: Remove this filter once torchao is updated to >= 0.18.
warnings.filterwarnings(
    "ignore",
    message=r"Deprecation: PlainLayout is deprecated.*",
    category=UserWarning,
)


@dataclass
class TorchAOBackendConfig(BackendConfig):
    """Configuration for TorchAOBackend."""

    quantization: QuantizationType | None = None
    quantization_config: AOBaseConfig | None = None

    _QUANTIZATION_CONFIGS = {
        "int8wo": Int8WeightOnlyConfig(),
        "int8dq": Int8DynamicActivationInt8WeightConfig(),
        "fp8wo": Float8WeightOnlyConfig(),
        "fp8dq": Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
    }

    def __post_init__(self):
        """Post init for TorchAOBackendConfig."""
        if self.quantization is not None and self.quantization_config is not None:
            raise ValueError("Only one of quantization or quantization_config should be provided.")
        if self.quantization is None and self.quantization_config is None:
            raise ValueError("Either quantization or quantization_config should be provided.")

        if not self.quantization_config:
            self.quantization_config = self._get_quantization_config(self.quantization)

    def describe(self) -> str:
        """Returns the description of the backend."""
        kwargs = {}
        for f in fields(self.quantization_config.__class__):
            if f.default is MISSING and f.default_factory is MISSING:
                kwargs[f.name] = getattr(self.quantization_config, f.name)

        changed_fields = self._get_changed_fields(
            self.quantization_config,
            self.quantization_config.__class__(*kwargs),
            include=list(kwargs.keys()),
        )
        return f"quantization_config={self.quantization_config.__class__.__name__}({','.join(changed_fields)})"

    def to_dict(self):
        """Convert TorchAOBackendConfig to dict built only from primitive values."""
        return {
            "quantization_config": config_to_dict(self.quantization_config),
        }

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Convert dict to TorchAOBackendConfig.

        The payload is validated before config_from_dict runs, because that resolver
        instantiates a type as soon as it resolves it. See _assert_ao_config_payload
        for the rules that are enforced.

        Raises:
            ValueError: If the payload is not a serialized torchao config, or holds a
                _type that does not name an acceptable class.
        """
        raw = state_dict["quantization_config"]
        _assert_ao_config_payload(raw)
        quantization_config = config_from_dict(raw)
        return cls(quantization_config=quantization_config)

    def _get_quantization_config(self, quantization):
        if quantization not in self._QUANTIZATION_CONFIGS:
            valid_quantizations = list(self._QUANTIZATION_CONFIGS.keys())
            raise ValueError(f"Invalid quantization: {quantization}. Must be one of: {valid_quantizations}")
        return self._QUANTIZATION_CONFIGS[quantization]


class TorchAOBackend(Backend):
    """Backend that does torch quantization.

    Supported quantizations:
        - int8wo
        - int8dq
        - fp8wo
        - fp8dq

    If you would like to use customize quantization, you can pass in a quantization config.
    """

    requires_original_module = True

    # State dictionary keys
    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_ORIG_MODULE = "orig_module"
    STATE_DATA = "data"
    STATE_DEVICE = "device"

    def __init__(
        self,
        config: TorchAOBackendConfig | None = None,
    ):
        """Initializes backend.

        Args:
            config: The configuration to use.
        """
        super().__init__()
        # initialize variables
        self._config = config or TorchAOBackendConfig(quantization=DEFAULT_QUANTIZATION)

        # build variables
        self._quant_module = None
        self._orig_module = None
        self._data = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        """Builds the model with torchao quantization and torch.compile."""
        self._save_config(cache_dir)

        self._orig_module = module
        self._data = data
        self._do_torchao_quantization()
        return self

    def _activate(self):
        """Activates backend."""
        self._do_torchao_quantization()

    @annotate(message="TorchAOBackend.infer", domain="Torch Tweak")
    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        """Runs inference with the given arguments.

        Args:
            *args: inference arguments
            **kwargs: inference keyword arguments

        Returns:
            Any: The result of the inference.
        """
        with torch.no_grad():
            return self._quant_module(*args, **kwargs)

    def _deactivate(self):
        """Deactivates backend."""
        self._quant_module = None

    def _deploy(self):
        """Deploys the backend."""
        self._activate()
        self._data = None
        gc.collect()

    def _save_config(self, cache_dir: Path):
        """Store the backend configuration to a file."""
        self._config.to_json(cache_dir / "config.json")

    def to_dict(self):
        """Returns the state_dict of the backend."""
        if self._orig_module is None:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_ORIG_MODULE: self._orig_module.state_dict(),
            self.STATE_DATA: samples_for_checkpoint(self._data),
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict):
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        if module is None:
            raise ValueError("Module is required to create a backend from a state_dict.")

        config = TorchAOBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])

        backend = cls(config=config)
        backend._data = state_dict[cls.STATE_DATA]
        backend._device = state_dict[cls.STATE_DEVICE]
        backend._orig_module = module
        module.load_state_dict(state_dict[cls.STATE_ORIG_MODULE], strict=False)
        backend.state = BackendState.CHECKPOINT_LOADED

        return backend

    def _do_torchao_quantization(self):
        model = copy.deepcopy(self._orig_module)
        self._orig_module.to("cpu")

        quantize_(model, config=self._config.quantization_config, device=self._device)
        self._quant_module = torch.compile(model=model, mode="default")
        with torch.no_grad():
            for args, kwargs in self._data:
                self._quant_module(*args, **kwargs)
