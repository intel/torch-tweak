# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OpenVINO backend for Intel GPU/CPU inference."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import openvino as ov
import torch
import torch.nn as nn
import torch.utils._pytree as pytree

from torch_tweak.torch.backend.backend import Backend, BackendConfig, BackendState
from torch_tweak.torch.backend.openvino.ov_model_converter import (
    OV_MODEL_XML_NAME,
    OpenVINOModelConverter,
    staticize_extra_openvino_inputs,
)
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.utils.module import offload
from torch_tweak.utils.permissions import secure_mkdir

logger = logging.getLogger(__name__)


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.detach().cpu().numpy()


def _ov_tensor_to_torch(ov_value: Any, device: torch.device, dtype: torch.dtype | None) -> torch.Tensor:
    if hasattr(ov_value, "data"):
        arr = np.array(ov_value.data)
    else:
        arr = np.asarray(ov_value)
    tensor = torch.from_numpy(arr.copy())
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device.type == "xpu" and hasattr(torch, "xpu"):
        tensor = tensor.to(device)
    elif device.type == "cpu":
        tensor = tensor.to(device)
    return tensor


def openvino_device_for_xpu(core: ov.Core, device: torch.device) -> str:
    """Returns the OpenVINO GPU that is the same physical device as the PyTorch XPU ``device``.

    OpenVINO numbers GPUs independently of PyTorch, so ``GPU`` can be an integrated GPU while PyTorch
    runs on a discrete one.
    """
    uuid = str(torch.xpu.get_device_properties(device).uuid).replace("-", "")
    gpus = [name for name in core.available_devices if name.startswith("GPU")]
    for name in gpus:
        if str(core.get_property(name, "DEVICE_UUID")) == uuid:
            return name
    raise RuntimeError(f"No OpenVINO GPU matches {device} (UUID {uuid}). OpenVINO GPUs: {gpus}.")


@dataclass
class OpenVINOBackendConfig(BackendConfig):
    """Configuration for OpenVINO backend.

    Attributes:
        use_dynamo: Use ``torch.export`` (True) or legacy trace conversion (False) in OVC.
        performance_hint: OpenVINO compile hint (LATENCY, THROUGHPUT, CUMULATIVE_THROUGHPUT).
    """

    use_dynamo: bool = True
    performance_hint: str = "LATENCY"

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Convert dict to OpenVINOBackendConfig."""
        return cls(**state_dict)


class OpenVINOBackend(Backend):
    """Runs inference via OpenVINO compiled models (Intel CPU/GPU)."""

    _devices: ClassVar[list[str]] = ["cpu", "xpu"]

    STATE_TYPE = "type"
    STATE_CONFIG = "config"
    STATE_OV_MODEL_DIR = "ov_model_dir"
    STATE_OUTPUT_SPEC = "output_spec"
    STATE_OUTPUT_LEAVES = "output_leaves"
    STATE_GRAPH_SPEC = "graph_spec"
    STATE_DEVICE = "device"

    def __init__(self, config: OpenVINOBackendConfig | None = None):
        """Initializes backend."""
        super().__init__()
        self._config = config or OpenVINOBackendConfig()
        self._ov_model_dir: Path | None = None
        self._ov_model_xml: Path | None = None
        self._graph_spec: GraphSpec | None = None
        self._output_object: Any = None
        self._compiled_model: ov.CompiledModel | None = None

    def key(self) -> str:
        """Returns the key of the backend."""
        return f"{self.__class__.__name__}_{self._config.key()}"

    def describe(self) -> str:
        """Returns the description of the backend."""
        return f"{self.__class__.__name__}({self._config.describe()})"

    def _build(self, module: nn.Module, graph_spec: GraphSpec, data: list[Sample], cache_dir: Path) -> Backend:
        logger.info("Starting OpenVINO backend build")
        self._graph_spec = graph_spec
        self._save_config(cache_dir)

        module.to(self._device)
        self._output_object = self._get_output_object(module, data[0])

        model_dir = cache_dir / "openvino"
        secure_mkdir(model_dir)
        self._ov_model_dir = model_dir
        converter = OpenVINOModelConverter(output_dir=model_dir, use_dynamo=self._config.use_dynamo)
        self._ov_model_xml = converter.convert(module=module, sample=data[0], graph_spec=graph_spec)
        offload(module, device="cpu")

        self._activate()
        logger.info("OpenVINO backend build finished, model: %s", self._ov_model_xml)
        return self

    def _compile_model(self):
        if self._ov_model_xml is None:
            raise RuntimeError("OpenVINO model path is not set")
        core = ov.Core()
        ov_model = core.read_model(self._ov_model_xml)
        if self._graph_spec is not None:
            # GraphSpec shapes are already baked into the IR; only the extra-input workaround has to be
            # re-applied here, because the IR may come from a checkpoint written without it.
            tensor_input_names = {ts.name for ts in self._graph_spec.input_spec.tensor_specs}
            staticize_extra_openvino_inputs(ov_model.inputs, tensor_input_names)
            ov_model.validate_nodes_and_infer_types()
        compile_config = {}
        if self._config.performance_hint:
            compile_config["PERFORMANCE_HINT"] = self._config.performance_hint
        ov_device = "CPU"
        if self._device.type == "xpu":
            ov_device = openvino_device_for_xpu(core, self._device)
        elif self._device.type != "cpu":
            raise ValueError(f"Device {self._device.type} is not supported by OpenVINO backend")
        self._compiled_model = core.compile_model(ov_model, ov_device, config=compile_config or None)

    def _activate(self):
        if self._compiled_model is None:
            self._compile_model()

    def _deactivate(self):
        self._compiled_model = None

    def _deploy(self):
        self._activate()

    def _infer(self, *args: Any, **kwargs: Any) -> Any:
        if self._compiled_model is None:
            raise RuntimeError("OpenVINO model is not compiled. Call build() or activate() first.")

        with torch.no_grad():
            inputs = self._prepare_inputs(args, kwargs)
            if not inputs:
                raise ValueError("No input tensors provided for OpenVINO inference")

            ov_inputs = {name: _tensor_to_numpy(tensor) for name, tensor in inputs.items()}
            ov_outputs = self._compiled_model(ov_inputs)
            if hasattr(ov_outputs, "to_dict"):
                ov_outputs = {(getattr(k, "any_name", None) or str(k)): v for k, v in ov_outputs.to_dict().items()}
            elif not isinstance(ov_outputs, dict):
                out_tensor = self._compiled_model.output(0)
                ov_outputs = {(getattr(out_tensor, "any_name", None) or str(out_tensor)): ov_outputs}

            torch_outputs: dict[str, torch.Tensor] = {}
            for _locator, tensor_spec in self._graph_spec.output_spec.tensor_data:
                if tensor_spec.name not in ov_outputs:
                    continue
                ov_out = ov_outputs[tensor_spec.name]
                torch_outputs[tensor_spec.name] = _ov_tensor_to_torch(ov_out, self._device, tensor_spec.dtype)

            return self._prepare_outputs_for_return(torch_outputs)

    @staticmethod
    def _openvino_input_name(port: Any) -> str:
        name = getattr(port, "any_name", None)
        if name:
            return name
        get_any_name = getattr(port, "get_any_name", None)
        if callable(get_any_name):
            return get_any_name()
        return str(port)

    def _prepare_inputs(self, args: tuple, kwargs: dict) -> dict[str, torch.Tensor]:
        model_input_names = {self._openvino_input_name(inp) for inp in self._compiled_model.inputs}
        inputs: dict[str, torch.Tensor] = {}
        for locator, tensor_spec in self._graph_spec.input_spec.tensor_data:
            if tensor_spec.name not in model_input_names:
                continue
            if tensor_spec.name.startswith("args"):
                tensor = locator.get_value(args)
            else:
                tensor = locator.get_value(kwargs)
            inputs[tensor_spec.name] = tensor
        return inputs

    def _prepare_outputs_for_return(self, outputs: dict[str, torch.Tensor]) -> Any:
        result = copy.deepcopy(self._output_object)
        for locator, tensor_spec in self._graph_spec.output_spec.tensor_data:
            if tensor_spec.name in outputs:
                result = locator.set_value(result, outputs[tensor_spec.name])
        return result

    def _get_output_object(self, module: nn.Module, sample: Sample) -> Any:
        module.to(self._device)
        args, kwargs = sample
        with torch.no_grad():
            output_object = module(*args, **kwargs)
        # `_prepare_outputs_for_return` replaces every tensor with an OpenVINO output.
        return copy.deepcopy(pytree.tree_map_only(torch.Tensor, lambda _: torch.empty(0), output_object))

    def _save_config(self, cache_dir: Path):
        config_path = cache_dir / "config.json"
        self._config.to_json(config_path)

    def to_dict(self):
        """Returns the state_dict of the backend."""
        if self._ov_model_dir is None or self._graph_spec is None:
            raise RuntimeError("Backend has not been properly initialized. Please call build() first.")
        # A checkpoint cannot hold output classes such as transformers `ModelOutput`. Their `TreeSpec`
        # names the class instead, and loading rebuilds it from the pytree registry.
        output_leaves, output_spec = pytree.tree_flatten(self._output_object)
        try:
            output_spec_str = pytree.treespec_dumps(output_spec)
        except NotImplementedError as e:
            raise ValueError(f"{self.STATE_OUTPUT_SPEC}: forward() output cannot be saved in a checkpoint") from e
        return {
            self.STATE_TYPE: self.__class__.__name__,
            self.STATE_CONFIG: self._config.to_dict(),
            self.STATE_OV_MODEL_DIR: self._ov_model_dir,
            self.STATE_OUTPUT_SPEC: output_spec_str,
            self.STATE_OUTPUT_LEAVES: output_leaves,
            self.STATE_GRAPH_SPEC: self._graph_spec.to_dict(),
            self.STATE_DEVICE: self._device,
        }

    @classmethod
    def from_dict(cls, module: torch.nn.Module | None, state_dict: dict):
        """Creates a backend from a state_dict."""
        if state_dict.get(cls.STATE_TYPE) != cls.__name__:
            raise ValueError(f"Invalid state_dict type: {state_dict.get(cls.STATE_TYPE)}")

        config = OpenVINOBackendConfig.from_dict(state_dict[cls.STATE_CONFIG])
        backend = cls(config=config)
        if cls.STATE_OV_MODEL_DIR not in state_dict:
            raise ValueError("Checkpoint is missing OpenVINO model path (ov_model_dir).")
        backend._ov_model_dir = Path(state_dict[cls.STATE_OV_MODEL_DIR])
        backend._ov_model_xml = backend._ov_model_dir / OV_MODEL_XML_NAME
        output_spec = pytree.treespec_loads(state_dict[cls.STATE_OUTPUT_SPEC])
        backend._output_object = pytree.tree_unflatten(state_dict[cls.STATE_OUTPUT_LEAVES], output_spec)
        backend._graph_spec = GraphSpec.from_dict(state_dict[cls.STATE_GRAPH_SPEC])
        backend._set_device(state_dict[cls.STATE_DEVICE])
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend
