# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Convert PyTorch modules to OpenVINO IR via Intel OVC."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import openvino as ov
import torch
import torch.nn as nn
from openvino import PartialShape, serialize
from openvino.tools.ovc import convert_model

from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.locator import Locator, ObjectType
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.module.tensor_spec import TensorSpec, logical_batch_bounds
from torch_tweak.torch.utils.module import get_arguments_names
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
            if self.use_dynamo:
                ov_model, ordered_kwarg_names = self._convert_dynamo(module, args, kwargs, graph_spec)
                ordered_input_names = _ordered_input_names_for_dynamo(graph_spec.input_spec, ordered_kwarg_names)
            else:
                ov_model = self._convert_trace(module, args, kwargs, graph_spec)
                ordered_input_names = None

            _apply_graph_spec_io(ov_model, graph_spec, ordered_input_names)
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

    def _convert_dynamo(
        self, module: nn.Module, args: tuple, kwargs: dict, graph_spec: GraphSpec
    ) -> tuple[ov.Model, list[str]]:
        posonly_arg_names, named_kwarg_names = get_arguments_names(module.forward)
        var_keyword = _var_keyword_name(module.forward)
        extra_kwarg_names = [name for name in kwargs if name not in named_kwarg_names] if var_keyword else []
        non_posonly_arg_names = named_kwarg_names + extra_kwarg_names
        dynamic_shapes: dict[str, Any] = _create_dynamic_shapes(
            graph_spec.input_spec, (posonly_arg_names, non_posonly_arg_names)
        )
        ordered_kwargs: dict[str, Any] = _ordered_kwargs_for_export(non_posonly_arg_names, kwargs)
        _fill_dynamic_shapes_for_non_tensors(non_posonly_arg_names, kwargs, dynamic_shapes)
        if dynamic_shapes and var_keyword:
            # torch.export keys the shapes of **kwargs entries under the name of that parameter.
            dynamic_shapes[var_keyword] = {
                name: dynamic_shapes.pop(name, None) for name in extra_kwarg_names if name in ordered_kwargs
            }

        exported: torch.export.ExportedProgram = torch.export.export(
            module,
            args=args,
            kwargs=ordered_kwargs,
            dynamic_shapes=dynamic_shapes,
        )
        return convert_model(exported), list(ordered_kwargs)

    def _convert_trace(self, module: nn.Module, args: tuple, kwargs: dict, graph_spec: GraphSpec) -> ov.Model:
        example_input = _example_input_for_trace(args, kwargs, graph_spec)
        return convert_model(
            module,
            example_input=example_input,
            dynamo=False,
        )


def _var_keyword_name(func: Any) -> str | None:
    for parameter in inspect.signature(func).parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return parameter.name
    return None


def _prepare_sample(sample: Sample, input_spec: SampleMetadata) -> tuple[tuple, dict]:
    args, kwargs = sample
    if not input_spec.has_batch_axis():
        return args, kwargs
    return input_spec.make_batch(
        args,
        kwargs,
        batch_size=max(logical_batch_bounds(input_spec.tensor_specs)[0], 2),
    )


def _ordered_kwargs_for_export(forward_kwargs: list[str], kwargs: dict) -> dict[str, Any]:
    """Reorder sample kwargs to match the forward signature for ``torch.export``.

    Keys are taken from *forward_kwargs* in order. Names missing from *kwargs* or
    mapped to ``None`` are skipped; extra keys in *kwargs* are ignored. *forward_kwargs* must
    exclude positional-only parameters: ``torch.export`` binds those from ``args`` and rejects
    them in ``kwargs``.

    >>> _ordered_kwargs_for_export(["alpha", "beta"], {"beta": 2, "alpha": 1})
    {'alpha': 1, 'beta': 2}
    >>> _ordered_kwargs_for_export(["alpha", "beta", "gamma"], {"alpha": 1})
    {'alpha': 1}
    >>> _ordered_kwargs_for_export(["alpha", "beta"], {"alpha": 1, "beta": None})
    {'alpha': 1}
    >>> _ordered_kwargs_for_export(["alpha"], {"alpha": 1, "extra": 99})
    {'alpha': 1}
    >>> _ordered_kwargs_for_export([], {"alpha": 1})
    {}
    >>> _ordered_kwargs_for_export(["alpha"], {})
    {}
    """
    return {k: v for k in forward_kwargs if (v := kwargs.get(k)) is not None}


def _fill_dynamic_shapes_for_non_tensors(
    forward_kwargs: list[str], kwargs: dict, dynamic_shapes: dict[str, Any]
) -> None:
    """Add placeholder entries to *dynamic_shapes* for non-tensor kwargs that ``torch.export`` requires."""
    for kwarg in forward_kwargs:
        if kwarg not in kwargs or kwargs[kwarg] is None:
            continue
        if kwarg not in dynamic_shapes and not isinstance(kwargs[kwarg], torch.Tensor):
            dynamic_shapes[kwarg] = {} if isinstance(kwargs[kwarg], (dict, list)) else None


def _example_input_for_trace(args: tuple, kwargs: dict, graph_spec: GraphSpec) -> Any:
    tensors: list[torch.Tensor] = []
    for locator, _tensor_spec in graph_spec.input_spec.tensor_data:
        if _tensor_spec.name.startswith("args"):
            tensors.append(locator.get_value(args))
        else:
            tensors.append(locator.get_value(kwargs))
    if len(tensors) == 1:
        return tensors[0]
    return tuple(tensors)


def _partial_shape_from_tensor_spec(tensor_spec: TensorSpec) -> PartialShape:
    dims: list[Any] = []
    for d_min, d_max in zip(tensor_spec.min_shape, tensor_spec.max_shape, strict=False):
        if d_min != d_max:
            dims.append(ov.Dimension(d_min, d_max))
        else:
            dims.append(d_min)
    return PartialShape(dims)


def _ordered_input_names_for_dynamo(input_spec: SampleMetadata, ordered_kwarg_names: Sequence[str]) -> list[str]:
    """Return tensor-spec names in the order torch.export assigns OV input ports.

    ``torch.export`` places positional-arg inputs first, then keyword-argument inputs in
    forward-signature order (the key order of ``_ordered_kwargs_for_export``).

    >>> # Metadata may list kwargs in arbitrary insertion order.
    >>> input_spec = SampleMetadata.from_inputs((), {"beta": torch.ones(2), "alpha": torch.zeros(2)})
    >>> # Signature order (alpha before beta), as from _ordered_kwargs_for_export.
    >>> _ordered_input_names_for_dynamo(input_spec, ["alpha", "beta"])
    ['kwargs_alpha', 'kwargs_beta']
    >>> # Names omitted from ordered_kwarg_names are skipped (optional kwargs skipped at export).
    >>> _ordered_input_names_for_dynamo(input_spec, ["alpha"])
    ['kwargs_alpha']
    >>> input_spec = SampleMetadata.from_inputs((torch.randn(2), torch.randn(3)), {"t": torch.randn(4)})
    >>> _ordered_input_names_for_dynamo(input_spec, ["t"])
    ['args_0', 'args_1', 'kwargs_t']
    """
    args_names, kwargs_names = input_spec.get_names_mapping()
    ordered: list[str] = list(args_names)
    for kwarg_name in ordered_kwarg_names:
        ordered.extend(kwargs_names.get(kwarg_name, []))
    return ordered


def _apply_graph_spec_io(
    ov_model: ov.Model, graph_spec: GraphSpec, ordered_input_names: list[str] | None = None
) -> None:
    input_names = ordered_input_names if ordered_input_names is not None else graph_spec.input_spec.get_names()
    output_names = graph_spec.output_spec.get_names()

    # Keyed lookup so partial shapes are assigned by name, not by coincidental position.
    name_to_tensor_spec = {ts.name: ts for ts in graph_spec.input_spec.tensor_specs}

    if len(input_names) != len(ov_model.inputs):
        logger.warning(
            "GraphSpec declares %d input(s) but the OpenVINO model has %d; extra names will be ignored.",
            len(input_names),
            len(ov_model.inputs),
        )
    for idx, name in enumerate(input_names):
        if idx >= len(ov_model.inputs):
            break
        port = ov_model.input(idx)
        port.get_tensor().set_names({name})
        if name in name_to_tensor_spec:
            port.get_node().set_partial_shape(_partial_shape_from_tensor_spec(name_to_tensor_spec[name]))

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

    staticize_extra_openvino_inputs(ov_model.inputs, set(name_to_tensor_spec))

    ov_model.validate_nodes_and_infer_types()


def staticize_extra_openvino_inputs(inputs: Sequence[ov.Output], tensor_input_names: set[str]) -> None:
    """Give non-tensor trace/export inputs (e.g. ``return_dict=False``) static scalar shapes.

    ``torch.export`` + OVC can expose bool/scalar kwargs as extra parameters with fully dynamic
    rank (``[...]``). Intel GPU compile rejects those; GraphSpec only tracks tensor inputs.
    """
    port: ov.Output
    for port in inputs:
        name: str = port.get_any_name()
        partial: PartialShape = port.get_partial_shape()
        if name not in tensor_input_names and partial.is_dynamic:
            logger.info(
                "Setting static scalar shape on extra OpenVINO input %r (not in GraphSpec tensors)",
                name,
            )
            node: ov.Node = port.get_node()
            node.set_partial_shape(PartialShape([]))


def _create_dynamic_shapes(
    input_spec: SampleMetadata, forward_arguments: tuple[list[str], list[str]]
) -> dict[str, Any]:
    """Build ``torch.export`` ``dynamic_shapes`` keyed by ``forward`` parameter names.

    Recorded tensor metadata uses internal names (``kwargs_alpha``, ...); this helper
    maps them onto the positional/kwarg names from ``get_arguments_names`` so
    ``torch.export.export`` receives the structure it expects.

    >>> import torch
    >>> from torch_tweak.torch.module.sample_metadata import SampleMetadata
    >>> kwargs_a = {"alpha": torch.randn(2, 4), "beta": torch.randn(2, 8)}
    >>> kwargs_b = {"alpha": torch.randn(6, 4), "beta": torch.randn(6, 8)}
    >>> input_spec = SampleMetadata.from_inputs((), kwargs_a, batch_size=2)
    >>> input_spec.update_shapes_seen(SampleMetadata.from_inputs((), kwargs_b, batch_size=6))
    >>> forward_arguments = ([], ["self", "alpha", "beta"])
    >>> shapes = _create_dynamic_shapes(input_spec, forward_arguments)
    >>> sorted(shapes)
    ['alpha', 'beta']
    >>> shapes['alpha'][0] is shapes['beta'][0]
    True
    >>> # Tensors recorded positionally are keyed by signature name, positional-only ones included.
    >>> input_spec = SampleMetadata.from_inputs((torch.randn(2, 4),), {}, batch_size=2)
    >>> input_spec.update_shapes_seen(SampleMetadata.from_inputs((torch.randn(6, 4),), {}, batch_size=6))
    >>> sorted(_create_dynamic_shapes(input_spec, (["x"], ["y"])))
    ['x']
    """
    forward_args: list[str]
    forward_kwargs: list[str]
    forward_args, forward_kwargs = forward_arguments
    dynamic_shapes: dict[str, dict[int, torch.export.Dim]] = _create_dynamic_shapes_from_tensor_spec(
        input_spec.tensor_specs
    )
    input_args: dict[Any, str]
    input_kwargs: dict[tuple[Any, int], tuple[Locator, str]]
    input_args, input_kwargs = _create_inputs_mapping(input_spec)
    _rekey_positional_arguments(input_args, forward_args, forward_kwargs)
    return _create_ordered_dynamic_shapes(forward_args, forward_kwargs, input_args, input_kwargs, dynamic_shapes)


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


def _create_inputs_mapping(
    input_spec: SampleMetadata,
) -> tuple[dict[Any, str], dict[tuple[Any, int], tuple[Locator, str]]]:
    input_args: dict[Any, str] = {}
    input_kwargs: dict[tuple[Any, int], tuple[Locator, str]] = {}
    for locator, tensor_spec in input_spec.tensor_data:
        if tensor_spec.name.startswith("args"):
            input_args[locator.leaf_name] = tensor_spec.name
        else:
            input_kwargs[(locator.leaf_name, locator.depth)] = (locator, tensor_spec.name)
    return input_args, input_kwargs


def _rekey_positional_arguments(input_args: dict[Any, str], forward_args: list[str], forward_kwargs: list[str]) -> None:
    """Rekey positionally recorded tensors from their sample index to the ``forward`` parameter name.

    ``_create_inputs_mapping`` keys positional tensors by index, while
    ``_create_ordered_dynamic_shapes`` looks them up by parameter name. Positional arguments bind
    to ``POSITIONAL_ONLY`` parameters first and then to ``POSITIONAL_OR_KEYWORD`` ones, so the i-th
    tensor takes the i-th name of ``forward_args + forward_kwargs``. Names taken from
    ``forward_kwargs`` are appended to *forward_args* to keep the export order. Tensors beyond the
    last named parameter (``*args`` forwards) cannot be named and get no dynamic shape.

    >>> input_args, forward_args = {0: "args_0", 1: "args_1"}, []
    >>> _rekey_positional_arguments(input_args, forward_args, ["x", "y"])
    >>> input_args, forward_args
    ({'x': 'args_0', 'y': 'args_1'}, ['x', 'y'])
    >>> # A positional-only parameter is already in forward_args and must not be duplicated.
    >>> input_args, forward_args = {0: "args_0", 1: "args_1"}, ["x"]
    >>> _rekey_positional_arguments(input_args, forward_args, ["y"])
    >>> input_args, forward_args
    ({'x': 'args_0', 'y': 'args_1'}, ['x', 'y'])
    >>> # Fewer positional tensors than parameters: the surplus names stay unbound.
    >>> input_args, forward_args = {0: "args_0"}, ["x"]
    >>> _rekey_positional_arguments(input_args, forward_args, ["y"])
    >>> input_args, forward_args
    ({'x': 'args_0'}, ['x'])
    >>> # *args forward exposes no parameter names, so nothing can be keyed.
    >>> input_args, forward_args = {0: "args_0"}, []
    >>> _rekey_positional_arguments(input_args, forward_args, [])
    >>> input_args, forward_args
    ({}, [])
    """
    positional_names = forward_args + forward_kwargs
    recorded = list(input_args.values())
    if len(recorded) > len(positional_names):
        logger.warning(
            "forward() exposes %d nameable positional parameter(s) but %d tensor(s) were recorded "
            "positionally; the surplus tensors are exported with static shapes.",
            len(positional_names),
            len(recorded),
        )
    input_args.clear()
    for name, tensor_spec_name in zip(positional_names, recorded, strict=False):
        input_args[name] = tensor_spec_name
        if name not in forward_args:
            forward_args.append(name)


def _create_ordered_dynamic_shapes(
    forward_args: list[str],
    forward_kwargs: list[str],
    input_args: dict[str, str],
    input_kwargs: dict[tuple[int | str, int], tuple[Locator, str]],
    dynamic_shapes: dict[str, dict[int, torch.export.Dim]],
) -> dict[str, Any]:
    ordered_dynamic_shapes: dict[str, Any] = {}
    for fwd_arg in forward_args:
        if tensor_spec_name := input_args.get(fwd_arg):
            ordered_dynamic_shapes[fwd_arg] = dynamic_shapes[tensor_spec_name]

    for kwarg in forward_kwargs:
        if loc_and_name := input_kwargs.get((kwarg, 1)):
            _locator, tensor_spec_name = loc_and_name
            ordered_dynamic_shapes[kwarg] = dynamic_shapes[tensor_spec_name]

    for _, (locator, tensor_spec_name) in input_kwargs.items():
        if locator.depth > 1:
            _raise_on_locator_user_type(locator)
            ordered_dynamic_shapes = _create_nested_structure(locator, root=ordered_dynamic_shapes)
            locator.set_value(ordered_dynamic_shapes, dynamic_shapes[tensor_spec_name])

    return ordered_dynamic_shapes


def _raise_on_locator_user_type(locator: Locator) -> None:
    if any(t in [ObjectType.USER_TYPE, ObjectType.DATACLASS] for _, t in locator.path_iter()):
        raise ValueError("Dynamic shapes does not support user types or dataclasses.")


def _create_nested_structure(locator: Locator, root: Any = None, last: Any = None) -> Any:
    parents = [[] if t == ObjectType.SEQUENCE else {} for _, t in locator.path_iter()]
    accessors = [a for a, _ in locator.path_iter()]

    if root is not None:
        parents[0] = root

    parent = parents[0]
    for accessor, current in zip(accessors, parents[1:] + [last], strict=True):
        if isinstance(parent, list) and len(parent) <= accessor:
            parent.extend([None] * (accessor - len(parent) + 1))

        if accessor in parent:
            current = parent[accessor]

        parent[accessor] = current
        parent = current

    return parents[0]
