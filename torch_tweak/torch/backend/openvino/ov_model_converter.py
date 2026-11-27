# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Convert PyTorch modules to OpenVINO IR via Intel OVC."""

from __future__ import annotations

import contextlib
import inspect
import logging
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import openvino as ov
import torch
import torch.nn as nn
from openvino import PartialShape, serialize
from openvino.frontend.pytorch import fx_decoder
from openvino.frontend.pytorch import utils as pytorch_frontend_utils
from openvino.tools.ovc import convert_model

from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.locator import Locator, ObjectType
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.module.tensor_spec import TensorSpec, logical_batch_bounds
from torch_tweak.utils.permissions import secure_file, secure_mkdir

OV_MODEL_XML_NAME = "model.xml"
OV_MODEL_BIN_NAME = "model.bin"

logger = logging.getLogger(__name__)


class OpenVINOModelConverter:
    """Convert a tuned PyTorch submodule to serialized OpenVINO IR (.xml/.bin)."""

    def __init__(self, output_dir: Path, use_dynamo: bool = True):
        """Configure IR output directory and conversion mode (``torch.export`` vs trace)."""
        self.output_dir = output_dir
        self.use_dynamo = use_dynamo

    def convert(self, module: nn.Module, sample: Sample, graph_spec: GraphSpec) -> Path:
        """Convert module to OpenVINO IR under ``output_dir`` and return the ``.xml`` path."""
        secure_mkdir(self.output_dir)
        xml_path = self.output_dir / OV_MODEL_XML_NAME
        bin_path = self.output_dir / OV_MODEL_BIN_NAME

        logger.info("Converting PyTorch module to OpenVINO IR: %s", xml_path)
        try:
            args, kwargs = _prepare_sample(sample, graph_spec.input_spec)
            flat_module = _FlatTensorInputs(module, args, kwargs, graph_spec.input_spec)
            with _constants_do_not_share_temporary_copies(), _root_logger_kept():
                if self.use_dynamo:
                    ov_model = _convert_dynamo(flat_module)
                else:
                    ov_model = _convert_trace(flat_module)

            _apply_graph_spec_io(ov_model, graph_spec)
            # An old IR file can be a hardlink that other processes read, and serialize writes in place.
            for path in (xml_path, bin_path):
                path.unlink(missing_ok=True)
            serialize(ov_model, xml_path.as_posix(), bin_path.as_posix())
            secure_file(xml_path)
            secure_file(bin_path)
            logger.info("OpenVINO IR saved: %s", xml_path)
            return xml_path
        except Exception as e:
            logger.error("Failed to convert model to OpenVINO IR: %s", e)
            for path in (xml_path, bin_path):
                if path.exists():
                    try:
                        path.unlink()
                    except OSError as delete_error:
                        logger.info("Failed to remove incomplete IR file %s: %s", path, delete_error)
            raise


def _convert_dynamo(flat_module: _FlatTensorInputs) -> ov.Model:
    axis_maps = _create_dynamic_shapes_from_tensor_spec(flat_module.tensor_specs)
    exported: torch.export.ExportedProgram = torch.export.export(
        flat_module,
        args=flat_module.example_inputs,
        dynamic_shapes={"flat_inputs": tuple(axis_maps[name] for name in flat_module.input_names)},
    )
    ov_model = convert_model(exported)
    # OVC leaves out an unused input and names every other port after its placeholder.
    input_name_by_placeholder = dict(zip(exported.graph_signature.user_inputs, flat_module.input_names, strict=True))
    for port in ov_model.inputs:
        placeholders = port.get_names() & input_name_by_placeholder.keys()
        if len(placeholders) != 1:
            raise RuntimeError(f"OpenVINO input {port.get_names()} matches no single torch.export input")
        port.get_tensor().set_names({input_name_by_placeholder[placeholders.pop()]})
    return ov_model


def _convert_trace(flat_module: _FlatTensorInputs) -> ov.Model:
    ov_model = convert_model(flat_module, example_input=flat_module.example_inputs, dynamo=False)
    input_names = flat_module.input_names
    if len(ov_model.inputs) != len(input_names):
        raise RuntimeError(f"OpenVINO model has {len(ov_model.inputs)} inputs, GraphSpec has {len(input_names)}")
    for port, name in zip(ov_model.inputs, input_names, strict=True):
        port.get_tensor().set_names({name})
    return ov_model


class _FlatTensorInputs(nn.Module):
    """Call *module* with the sample *args* and *kwargs*, taking their recorded tensors as inputs.

    ``forward`` takes the positional tensors first, then the keyword tensors in the parameter order of
    ``module.forward``, then the remaining ``**kwargs`` tensors in recorded order. Every non-tensor argument
    keeps its sample value, which ``torch.export`` would specialize to anyway. Each call copies only the
    containers that hold a recorded tensor.
    """

    def __init__(self, module: nn.Module, args: tuple, kwargs: dict, input_spec: SampleMetadata):
        super().__init__()
        self.module = module
        self.training = module.training
        self._sample = (args, kwargs)
        tensor_data = _tensor_data_in_forward_order(module, input_spec)
        # Each locator starts at the (args, kwargs) pair.
        self._locators = [
            Locator(((0 if in_args else 1, ObjectType.SEQUENCE), *locator.path_iter()))
            for locator, in_args, _ in tensor_data
        ]
        self.tensor_specs = tuple(tensor_spec for _, _, tensor_spec in tensor_data)
        self.input_names = tuple(tensor_spec.name for tensor_spec in self.tensor_specs)
        example_inputs: list[torch.Tensor] = []
        seen: set[int] = set()
        for locator in self._locators:
            tensor = locator.get_value(self._sample)
            # A tensor at two locators would trace as one input, so the second locator gets a copy.
            example_inputs.append(tensor.clone() if id(tensor) in seen else tensor)
            seen.add(id(tensor))
        self.example_inputs = tuple(example_inputs)

    def forward(self, *flat_inputs: torch.Tensor) -> Any:
        sample = self._sample
        for locator, tensor in zip(self._locators, flat_inputs, strict=True):
            sample = locator.with_value(sample, tensor)
        args, kwargs = sample
        return self.module(*args, **kwargs)


def _tensor_data_in_forward_order(
    module: nn.Module, input_spec: SampleMetadata
) -> list[tuple[Locator, bool, TensorSpec]]:
    ordered: list[tuple[Locator, bool, TensorSpec]] = []
    keyword_groups: dict[int | str, list[tuple[Locator, bool, TensorSpec]]] = {}
    for locator, in_args, tensor_spec in input_spec.tensor_data_by_root():
        if in_args:
            ordered.append((locator, in_args, tensor_spec))
        else:
            keyword_groups.setdefault(locator.root_name, []).append((locator, in_args, tensor_spec))

    for parameter in inspect.signature(module.forward).parameters.values():
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            ordered.extend(keyword_groups.pop(parameter.name, ()))
    for group in keyword_groups.values():
        ordered.extend(group)
    return ordered


@contextlib.contextmanager
def _root_logger_kept() -> Generator[None]:
    """Restores the level, filters, and handlers of the root logger.

    ``convert_model`` sets the root logger to ``ERROR`` and restores it only when the conversion succeeds.

    Remove this function once the minimum ``openvino`` version includes
    https://github.com/openvinotoolkit/openvino/pull/40263.
    """
    root = logging.getLogger()
    level, filters, handlers = root.level, list(root.filters), list(root.handlers)
    try:
        yield
    finally:
        root.setLevel(level)
        root.filters = filters
        root.handlers = handlers


@contextlib.contextmanager
def _constants_do_not_share_temporary_copies() -> Generator[None]:
    """Makes the PyTorch frontend copy the data of a constant unless its source is a contiguous CPU tensor.

    For bf16 and f8 the frontend shares the memory of a contiguous CPU copy of the tensor and does not
    keep that copy alive. The copy is a new tensor when the source is on another device or is not contiguous.

    Remove this function and the ``fx_decoder`` and ``pytorch_frontend_utils`` imports once the minimum
    ``openvino`` version includes https://github.com/openvinotoolkit/openvino/pull/39510.
    """
    original = pytorch_frontend_utils.torch_tensor_to_ov_const

    def to_constant(tensor: torch.Tensor, shared_memory: bool = True) -> Any:
        shares_source = tensor.device.type == "cpu" and tensor.is_contiguous()
        return original(tensor, shared_memory=shared_memory and shares_source)

    fx_decoder.torch_tensor_to_ov_const = pytorch_frontend_utils.torch_tensor_to_ov_const = to_constant
    try:
        yield
    finally:
        fx_decoder.torch_tensor_to_ov_const = pytorch_frontend_utils.torch_tensor_to_ov_const = original


def _prepare_sample(sample: Sample, input_spec: SampleMetadata) -> tuple[tuple, dict]:
    args, kwargs = sample
    if not input_spec.has_batch_axis():
        return args, kwargs
    return input_spec.make_batch(
        args,
        kwargs,
        batch_size=max(logical_batch_bounds(input_spec.tensor_specs)[0], 2),
    )


def _partial_shape_from_tensor_spec(tensor_spec: TensorSpec) -> PartialShape:
    dims: list[Any] = []
    for d_min, d_max in zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False):
        if d_min != d_max:
            dims.append(ov.Dimension(d_min, d_max))
        else:
            dims.append(d_min)
    return PartialShape(dims)


def _apply_graph_spec_io(ov_model: ov.Model, graph_spec: GraphSpec) -> None:
    output_names = graph_spec.output_spec.get_names()

    name_to_tensor_spec = {ts.name: ts for ts in graph_spec.input_spec.tensor_specs}
    for port in ov_model.inputs:
        tensor_spec = name_to_tensor_spec[port.get_any_name()]
        port.get_node().set_partial_shape(_partial_shape_from_tensor_spec(tensor_spec))

    if len(output_names) != len(ov_model.outputs):
        logger.warning(
            "GraphSpec declares %d output(s) but the OpenVINO model has %d; extra names will be ignored.",
            len(output_names),
            len(ov_model.outputs),
        )
    for idx, name in enumerate(output_names):
        if idx >= len(ov_model.outputs):
            break
        ov_model.output(idx).get_tensor().set_names({name})

    ov_model.validate_nodes_and_infer_types()


def _create_dynamic_shapes_from_tensor_spec(
    tensor_specs: Sequence[TensorSpec],
) -> dict[str, dict[int, torch.export.Dim]]:
    dynamic_shapes: dict[str, dict[int, torch.export.Dim]] = {}  # Maps each tensor name to its dynamic dimensions.
    batch_dim: torch.export.Dim | None = None  # Shared symbolic dimension for batch-related axes.
    free_dims: dict[tuple[str, int], torch.export.Dim] = {}  # Dimensions not relatable to other tensors' dimensions.

    for tensor_spec in tensor_specs:
        dynamic_shapes[tensor_spec.name] = {}
        batch_multipliers: dict[int, int] = tensor_spec.get_batch_axis_multipliers()
        for idx, (d1, d2) in enumerate(zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=True)):
            if d1 == d2:
                continue
            if idx in batch_multipliers:
                if batch_dim is None:
                    batch_min, batch_max = logical_batch_bounds(tensor_specs)
                    batch_dim = torch.export.Dim("batch", min=batch_min, max=batch_max)
                multiplier: int = batch_multipliers[idx]
                # multiplying Dim object creates a new _DerivedDim object
                dynamic_shapes[tensor_spec.name][idx] = batch_dim if multiplier == 1 else multiplier * batch_dim
            else:
                key = (tensor_spec.name, idx)
                if key not in free_dims:
                    free_dims[key] = torch.export.Dim(f"{tensor_spec.name}_dim_{idx}", min=d1, max=d2)
                dynamic_shapes[tensor_spec.name][idx] = free_dims[key]
    return dynamic_shapes
