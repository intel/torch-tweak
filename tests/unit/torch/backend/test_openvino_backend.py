# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Unit tests for OpenVINOBackend."""

import collections
import dataclasses
import gc
import logging
import shutil
import uuid
from types import SimpleNamespace

import openvino as ov
import pytest
import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from openvino.frontend import OpConversionFailure
from openvino.tools.ovc import convert_model
from transformers.modeling_outputs import BaseModelOutputWithPooling

from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import LEO_MISSING, has_leo, requires_leo, requires_xpu
from torch_tweak.torch.backend.openvino import (
    OpenVINOBackend,
    OpenVINOBackendConfig,
    ov_model_converter,
    xpu_remote,
)
from torch_tweak.torch.backend.openvino.openvino_backend import openvino_device_for_xpu
from torch_tweak.torch.backend.openvino.ov_model_converter import (
    OV_MODEL_BIN_NAME,
    OV_MODEL_XML_NAME,
    OpenVINOModelConverter,
    _create_dynamic_shapes_from_tensor_spec,
)
from torch_tweak.torch.backend.openvino.xpu_remote import (
    LEVEL_ZERO_INTEROP_MISSING,
    LevelZeroInteropError,
    ov_element_type,
)
from torch_tweak.torch.checkpoint.storage_tasks import (
    STATE_DICT_FILE,
    CopyBackendArtifactsTask,
    TorchLoadTask,
    TorchSaveTask,
    UnzipLoadTask,
    ZipSaveTask,
)
from torch_tweak.torch.config import config as global_config
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.module.tunable_module import TunableModule
from torch_tweak.torch.tune_strategy import NoCorrectBackendError, OneBackendStrategy

# ---------------------------------------------------------------------------
# Pytree node without serialized_type_name - used to test the rejection path
# in OpenVINOBackend.to_dict(). treespec_dumps raises NotImplementedError for
# any node registered without serialized_type_name.
# ---------------------------------------------------------------------------


class _PytreeNodeWithoutSerializedName:
    """Pytree node intentionally registered without serialized_type_name."""

    def __init__(self, value):
        self.value = value


pytree.register_pytree_node(
    _PytreeNodeWithoutSerializedName,
    lambda x: ([x.value], None),
    lambda values, _: _PytreeNodeWithoutSerializedName(values[0]),
    # serialized_type_name intentionally omitted
)

# ---------------------------------------------------------------------------
# Helpers for the kwargs-ordering regression tests
# ---------------------------------------------------------------------------

_KWARG_DIM = 8
_KWARG_OUT = 4


class _TwoKwargModel(nn.Module):
    """Two independent tensor kwargs (alpha, beta) with distinct fixed weights.

    proj_alpha fills outputs with +1 per input unit; proj_beta fills with -1.
    If alpha and beta are swapped during inference the sign of the output flips,
    making the regression immediately detectable.
    """

    def __init__(self):
        super().__init__()
        self.proj_alpha = nn.Linear(_KWARG_DIM, _KWARG_OUT, bias=False)
        self.proj_beta = nn.Linear(_KWARG_DIM, _KWARG_OUT, bias=False)
        nn.init.constant_(self.proj_alpha.weight, 1.0)
        nn.init.constant_(self.proj_beta.weight, -1.0)

    def forward(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return self.proj_alpha(alpha) + self.proj_beta(beta)


@pytest.fixture
def torch_device(torch_device):
    if torch_device.type == "xpu" and not has_leo:
        pytest.skip(LEO_MISSING)
    return torch_device


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(model, torch_device) -> list[Sample]:
    sample = model.sample()
    args = (sample.to(torch_device).unsqueeze(0),)
    kwargs = {}
    return [(args, kwargs)]


def _graph_spec(model: nn.Module, sample: Sample) -> GraphSpec:
    args, kwargs = sample
    with torch.no_grad():
        output = model(*args, **kwargs)
    input_spec = SampleMetadata.from_inputs(args, kwargs)
    output_spec = SampleMetadata.from_outputs(output)
    return GraphSpec(name="toy_model", input_spec=input_spec, output_spec=output_spec)


def backend_build(backend, model, sample_data, tmp_path, torch_device):
    model = model.to(torch_device, dtype=torch.float32)
    graph_spec = _graph_spec(model, sample_data[0])
    backend = backend.build(model, graph_spec=graph_spec, data=sample_data, device=torch_device, cache_dir=tmp_path)
    return backend


def _save_and_load(model, state_dict, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    TorchSaveTask().save(ckpt_dir, state_dict)
    loaded = OpenVINOBackend.from_dict(model, TorchLoadTask().load(ckpt_dir))
    loaded.activate()
    return loaded


def test_openvino_backend_build_and_infer(model, sample_data, tmp_path, torch_device):
    backend = OpenVINOBackend(config=OpenVINOBackendConfig(use_dynamo=False))
    backend = backend_build(backend, model, sample_data, tmp_path, torch_device)

    args, kwargs = sample_data[0]
    output = backend.infer(*args, **kwargs)
    assert output.shape[0] == 1
    assert output.dtype == torch.float32

    backend.deactivate()


class _Bfloat16Doubling(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1024, 1024, bias=False, dtype=torch.bfloat16, device="xpu")
        with torch.no_grad():
            self.layer.weight.copy_(2 * torch.eye(1024))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.to(torch.bfloat16)).float()


@requires_xpu
def test_openvino_backend_keeps_the_bf16_weights_of_an_xpu_module(tmp_path):
    module = _Bfloat16Doubling()
    sample = ((torch.ones(1, 1024, device="xpu"),), {})
    graph_spec = _graph_spec(module, sample)
    backend = OpenVINOBackend().build(
        module, graph_spec=graph_spec, data=[sample], device=torch.device("xpu"), cache_dir=tmp_path
    )

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.full((1, 1024), 2.0))


@requires_xpu
def test_openvino_backend_without_dynamo_keeps_the_bf16_weights_of_an_xpu_module(tmp_path):
    module = _Bfloat16Doubling()
    sample = ((torch.ones(1, 1024, device="xpu"),), {})
    graph_spec = _graph_spec(module, sample)
    backend = OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False)).build(
        module, graph_spec=graph_spec, data=[sample], device=torch.device("xpu"), cache_dir=tmp_path
    )

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.full((1, 1024), 2.0))


class _TransposedBfloat16Doubling(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter((2 * torch.eye(1024, dtype=torch.bfloat16)).t())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.float()


def test_openvino_backend_keeps_the_bf16_weights_of_a_non_contiguous_cpu_parameter(tmp_path):
    module = _TransposedBfloat16Doubling()
    sample = ((torch.ones(1, 1024),), {})
    graph_spec = _graph_spec(module, sample)
    backend = OpenVINOBackend().build(
        module, graph_spec=graph_spec, data=[sample], device=torch.device("cpu"), cache_dir=tmp_path
    )

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual, torch.full((1, 1024), 2.0))


class _Double(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


def test_openvino_backend_on_cpu_keeps_the_values_of_bf16_inputs_and_outputs(tmp_path):
    model = _Double().eval()
    sample = ((torch.tensor([[1.5, -3.0, 100.0]], dtype=torch.bfloat16),), {})
    backend = OpenVINOBackend().build(
        model, graph_spec=_graph_spec(model, sample), data=[sample], device=torch.device("cpu"), cache_dir=tmp_path
    )

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual, torch.tensor([[3.0, -6.0, 200.0]], dtype=torch.bfloat16), rtol=0, atol=0)


def test_openvino_backend_on_cpu_converts_a_float16_input_of_a_bf16_graph(tmp_path):
    model = _Double().eval()
    sample = ((torch.tensor([[1.5, -3.0, 100.0]], dtype=torch.bfloat16),), {})
    backend = OpenVINOBackend().build(
        model, graph_spec=_graph_spec(model, sample), data=[sample], device=torch.device("cpu"), cache_dir=tmp_path
    )

    actual = backend.infer(torch.tensor([[1.5, -3.0, 100.0]], dtype=torch.float16))

    torch.testing.assert_close(actual, torch.tensor([[3.0, -6.0, 200.0]], dtype=torch.bfloat16), rtol=0, atol=0)


class _VarKeywordModel(nn.Module):
    def forward(self, x: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return x - kwargs["y"] + 3 * kwargs["z"]


class _OptionalVarKeywordModel(nn.Module):
    def forward(self, x: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return x - kwargs.get("y", 1.0)


class _KeywordOnlyAndVarKeywordModel(nn.Module):
    def forward(self, x: torch.Tensor, *, scale: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        return (x - kwargs["y"]) * scale


class _OnlyVarKeywordModel(nn.Module):
    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return kwargs["x"] - kwargs["y"]


def _tensor(values, device) -> torch.Tensor:
    return torch.tensor(values, device=device)


def _ones_samples_over_batch_sizes(kwarg_names, batch_sizes, device) -> list[Sample]:
    """Samples with all-ones inputs; the values do not matter, only that the batch size varies."""
    return [
        ((torch.ones(bs, 2, device=device),), {name: torch.ones(bs, 2, device=device) for name in kwarg_names})
        for bs in batch_sizes
    ]


def _build_dynamic_backend(model, samples, tmp_path, torch_device):
    model = model.to(torch_device).eval()
    graph_spec: GraphSpec | None = None
    for args, kwargs in samples:
        batch_size = pytree.tree_leaves((args, kwargs))[0].shape[0]
        in_meta = SampleMetadata.from_inputs(args, kwargs, batch_size=batch_size)
        out_meta = SampleMetadata.from_outputs(model(*args, **kwargs), batch_size=batch_size)
        if graph_spec is None:
            graph_spec = GraphSpec(name="var_keyword", input_spec=in_meta, output_spec=out_meta)
        else:
            graph_spec.update_shapes_seen(in_meta, out_meta)
    return OpenVINOBackend().build(model, graph_spec=graph_spec, data=samples, device=torch_device, cache_dir=tmp_path)


def test_openvino_backend_maps_each_var_keyword_input_by_name(tmp_path, torch_device):
    # Keys are given in non-alphabetical order and the model is not symmetric in y and z.
    sample = (
        (_tensor([[10.0, 20.0]], torch_device),),
        {"z": _tensor([[100.0, 200.0]], torch_device), "y": _tensor([[1.0, 2.0]], torch_device)},
    )
    backend = backend_build(OpenVINOBackend(), _VarKeywordModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0], **sample[1])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[309.0, 618.0]]))


def test_openvino_backend_builds_var_keyword_forward_called_without_keywords(tmp_path, torch_device):
    sample = ((_tensor([[1.0, 2.0]], torch_device),), {})
    backend = backend_build(OpenVINOBackend(), _OptionalVarKeywordModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[0.0, 1.0]]))


def test_openvino_backend_ignores_none_valued_var_keyword_input(tmp_path, torch_device):
    sample = (
        (_tensor([[5.0, 6.0]], torch_device),),
        {"y": _tensor([[1.0, 2.0]], torch_device), "flag": None},
    )
    backend = backend_build(OpenVINOBackend(), _OptionalVarKeywordModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(sample[0][0], y=sample[1]["y"])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


def test_openvino_backend_keeps_batch_dynamic_for_var_keyword_inputs(tmp_path, torch_device):
    samples = _ones_samples_over_batch_sizes(["z", "y"], [1, 4], torch_device)
    backend = _build_dynamic_backend(_VarKeywordModel(), samples, tmp_path, torch_device)
    x = _tensor([[10.0, 20.0], [30.0, 40.0]], torch_device)
    y = _tensor([[1.0, 2.0], [3.0, 4.0]], torch_device)
    z = _tensor([[100.0, 200.0], [300.0, 400.0]], torch_device)

    actual = backend.infer(x, z=z, y=y)

    torch.testing.assert_close(actual.cpu(), torch.tensor([[309.0, 618.0], [927.0, 1236.0]]))


def test_openvino_backend_keeps_batch_dynamic_for_keyword_only_mixed_with_var_keyword(tmp_path, torch_device):
    samples = _ones_samples_over_batch_sizes(["scale", "y"], [1, 4], torch_device)
    backend = _build_dynamic_backend(_KeywordOnlyAndVarKeywordModel(), samples, tmp_path, torch_device)
    x = _tensor([[3.0, 4.0], [5.0, 6.0]], torch_device)
    y = _tensor([[1.0, 1.0], [2.0, 2.0]], torch_device)
    scale = _tensor([[10.0, 10.0], [100.0, 100.0]], torch_device)

    actual = backend.infer(x, scale=scale, y=y)

    torch.testing.assert_close(actual.cpu(), torch.tensor([[20.0, 30.0], [300.0, 400.0]]))


def test_openvino_backend_builds_forward_with_only_var_keyword(tmp_path, torch_device):
    sample = ((), {"x": _tensor([[5.0, 6.0]], torch_device), "y": _tensor([[1.0, 2.0]], torch_device)})
    backend = backend_build(OpenVINOBackend(), _OnlyVarKeywordModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(**sample[1])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


def test_openvino_backend_keeps_batch_dynamic_for_forward_with_only_var_keyword(tmp_path, torch_device):
    samples = [
        ((), {"x": torch.ones(1, 2, device=torch_device), "y": torch.ones(1, 2, device=torch_device)}),
        ((), {"x": torch.ones(4, 2, device=torch_device), "y": torch.ones(4, 2, device=torch_device)}),
    ]
    backend = _build_dynamic_backend(_OnlyVarKeywordModel(), samples, tmp_path, torch_device)
    x = _tensor([[5.0, 6.0], [7.0, 8.0]], torch_device)
    y = _tensor([[1.0, 2.0], [3.0, 4.0]], torch_device)

    actual = backend.infer(x=x, y=y)

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0], [4.0, 4.0]]))


def _scaling(factor: float) -> nn.Linear:
    layer = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(factor * torch.eye(2))
    return layer


def test_openvino_backend_with_reuse_ir_compiles_the_ir_already_in_its_directory(tmp_path, torch_device):
    sample = ((torch.tensor([[1.0, 2.0]], device=torch_device),), {})
    config = OpenVINOBackendConfig(reuse_ir=True)
    backend_build(OpenVINOBackend(config), _scaling(2.0), [sample], tmp_path, torch_device)
    backend = backend_build(OpenVINOBackend(config), _scaling(3.0), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[2.0, 4.0]]))


def test_openvino_backend_without_reuse_ir_converts_the_module_again(tmp_path, torch_device):
    sample = ((torch.tensor([[1.0, 2.0]], device=torch_device),), {})
    backend_build(OpenVINOBackend(), _scaling(2.0), [sample], tmp_path, torch_device)
    backend = backend_build(OpenVINOBackend(), _scaling(3.0), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[3.0, 6.0]]))


def test_openvino_backend_with_reuse_ir_converts_when_the_ir_misses_its_weights(tmp_path, torch_device):
    sample = ((torch.tensor([[1.0, 2.0]], device=torch_device),), {})
    config = OpenVINOBackendConfig(reuse_ir=True)
    backend_build(OpenVINOBackend(config), _scaling(2.0), [sample], tmp_path, torch_device)
    (tmp_path / "openvino" / OV_MODEL_BIN_NAME).unlink()
    backend = backend_build(OpenVINOBackend(config), _scaling(3.0), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[3.0, 6.0]]))


@torch.library.custom_op("torch_tweak_test::opaque_double", mutates_args=())
def _opaque_double(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@_opaque_double.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class _Opaque(nn.Module):
    def forward(self, x):
        return _opaque_double(x)


def test_openvino_backend_keeps_the_root_logger_level_after_a_failed_conversion(caplog, tmp_path, torch_device):
    caplog.set_level(logging.INFO)
    sample = ((torch.tensor([[1.0, 2.0]], device=torch_device),), {})

    with pytest.raises(OpConversionFailure, match="opaque_double"):
        backend_build(OpenVINOBackend(), _Opaque(), [sample], tmp_path, torch_device)

    assert logging.getLogger().level == logging.INFO


class _VarPositionalAndKeywordOnlyModel(nn.Module):
    def forward(self, x: torch.Tensor, *args: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x - args[0]


def test_openvino_backend_does_not_bind_var_positional_tensor_to_keyword_only_parameter(tmp_path, torch_device):
    sample = (
        (torch.tensor([[5.0, 6.0]], device=torch_device), torch.tensor([[1.0, 2.0]], device=torch_device)),
        {},
    )
    backend = backend_build(OpenVINOBackend(), _VarPositionalAndKeywordOnlyModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


class _UnusedVarPositionalModel(nn.Module):
    def forward(self, x: torch.Tensor, *args: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x + 1


def test_openvino_backend_exports_when_var_positional_is_empty(tmp_path, torch_device):
    sample = ((torch.tensor([[5.0, 6.0]], device=torch_device),), {})
    backend = backend_build(OpenVINOBackend(), _UnusedVarPositionalModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[6.0, 7.0]]))


def test_openvino_backend_keeps_the_batch_axis_of_a_var_positional_tensor(tmp_path, torch_device):
    samples = [
        ((torch.ones(1, 2, device=torch_device), torch.ones(1, 2, device=torch_device)), {}),
        ((torch.ones(8, 2, device=torch_device), torch.ones(8, 2, device=torch_device)), {}),
    ]
    backend = _build_dynamic_backend(_VarPositionalAndKeywordOnlyModel(), samples, tmp_path, torch_device)

    actual = backend.infer(torch.full((4, 2), 5.0, device=torch_device), torch.full((4, 2), 1.0, device=torch_device))

    torch.testing.assert_close(actual.cpu(), torch.full((4, 2), 4.0))


class _OnlyVarPositionalModel(nn.Module):
    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return args[0] - args[1]


def test_openvino_backend_builds_forward_with_only_var_positional(tmp_path, torch_device):
    sample = ((_tensor([[5.0, 6.0]], torch_device), _tensor([[1.0, 2.0]], torch_device)), {})
    backend = backend_build(OpenVINOBackend(), _OnlyVarPositionalModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


class _FlagThenVarPositionalModel(nn.Module):
    def forward(self, x: torch.Tensor, subtract: bool, *args: torch.Tensor) -> torch.Tensor:
        return x - args[0] if subtract else x


def test_openvino_backend_builds_forward_with_non_tensor_before_var_positional(tmp_path, torch_device):
    sample = ((_tensor([[5.0, 6.0]], torch_device), True, _tensor([[1.0, 2.0]], torch_device)), {})
    backend = backend_build(OpenVINOBackend(), _FlagThenVarPositionalModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


class _PairModel(nn.Module):
    def forward(self, pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return pair[0] - pair[1]


def test_openvino_backend_builds_forward_with_positional_tuple_of_tensors(tmp_path, torch_device):
    sample = (((_tensor([[5.0, 6.0]], torch_device), _tensor([[1.0, 2.0]], torch_device)),), {})
    backend = backend_build(OpenVINOBackend(), _PairModel(), [sample], tmp_path, torch_device)

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


class _ScaledDifferenceModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 2 * x - y


@pytest.mark.parametrize("use_dynamo", [True, False])
def test_openvino_backend_keeps_apart_inputs_recorded_as_the_same_tensor(tmp_path, torch_device, use_dynamo):
    x = _tensor([[1.0, 2.0]], torch_device)
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=use_dynamo)),
        _ScaledDifferenceModel(),
        [((x, x), {})],
        tmp_path,
        torch_device,
    )

    actual = backend.infer(_tensor([[5.0, 6.0]], torch_device), _tensor([[6.0, 8.0]], torch_device))

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


class _UnusedFirstInputModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y + 1


@pytest.mark.parametrize("use_dynamo", [True, False])
def test_openvino_backend_feeds_the_inputs_after_an_unused_one(tmp_path, torch_device, use_dynamo):
    sample = ((_tensor([[1.0, 1.0]], torch_device), _tensor([[1.0, 1.0]], torch_device)), {})
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=use_dynamo)),
        _UnusedFirstInputModel(),
        [sample],
        tmp_path,
        torch_device,
    )

    actual = backend.infer(_tensor([[100.0, 100.0]], torch_device), _tensor([[3.0, 3.0]], torch_device))

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


@dataclasses.dataclass
class _TensorPair:
    a: torch.Tensor
    b: torch.Tensor


class _DataclassPairModel(nn.Module):
    def forward(self, pair: _TensorPair) -> torch.Tensor:
        return pair.a - pair.b


@pytest.mark.parametrize("use_dynamo", [True, False])
def test_openvino_backend_builds_forward_with_dataclass_of_tensors(tmp_path, torch_device, use_dynamo):
    sample = ((_TensorPair(_tensor([[5.0, 6.0]], torch_device), _tensor([[1.0, 2.0]], torch_device)),), {})
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=use_dynamo)),
        _DataclassPairModel(),
        [sample],
        tmp_path,
        torch_device,
    )

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


@dataclasses.dataclass(frozen=True)
class _FrozenTensorPair:
    a: torch.Tensor
    b: torch.Tensor


class _FrozenPairModel(nn.Module):
    def forward(self, pair: _FrozenTensorPair) -> torch.Tensor:
        return pair.a - pair.b


@pytest.mark.parametrize("use_dynamo", [True, False])
def test_openvino_backend_builds_forward_with_frozen_dataclass_of_tensors(tmp_path, torch_device, use_dynamo):
    sample = ((_FrozenTensorPair(_tensor([[5.0, 6.0]], torch_device), _tensor([[1.0, 2.0]], torch_device)),), {})
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=use_dynamo)),
        _FrozenPairModel(),
        [sample],
        tmp_path,
        torch_device,
    )

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


_NamedTensorPair = collections.namedtuple("_NamedTensorPair", ["a", "b"])


class _NamedPairModel(nn.Module):
    def forward(self, pair: _NamedTensorPair) -> torch.Tensor:
        return pair.a - pair.b


@pytest.mark.parametrize("use_dynamo", [True, False])
def test_openvino_backend_builds_forward_with_namedtuple_of_tensors(tmp_path, torch_device, use_dynamo):
    sample = ((_NamedTensorPair(_tensor([[5.0, 6.0]], torch_device), _tensor([[1.0, 2.0]], torch_device)),), {})
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=use_dynamo)),
        _NamedPairModel(),
        [sample],
        tmp_path,
        torch_device,
    )

    actual = backend.infer(*sample[0])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


class _PairWithTotal(tuple):
    def __new__(cls, left, right):
        pair = super().__new__(cls, (left, right))
        pair.total = left + right
        return pair


class _TotalModel(nn.Module):
    def forward(self, pair: _PairWithTotal) -> torch.Tensor:
        return pair.total


def test_openvino_backend_rejects_a_tuple_subclass_that_derives_state_from_its_items(tmp_path, torch_device):
    sample = ((_PairWithTotal(_tensor([[1.0]], torch_device), _tensor([[2.0]], torch_device)),), {})

    with pytest.raises(TypeError, match="_PairWithTotal"):
        backend_build(OpenVINOBackend(), _TotalModel(), [sample], tmp_path, torch_device)


def test_openvino_backend_keeps_the_batch_axis_of_a_namedtuple_of_tensors(tmp_path, torch_device):
    samples = [
        ((_NamedTensorPair(torch.ones(1, 2, device=torch_device), torch.ones(1, 2, device=torch_device)),), {}),
        ((_NamedTensorPair(torch.ones(3, 2, device=torch_device), torch.ones(3, 2, device=torch_device)),), {}),
    ]
    backend = _build_dynamic_backend(_NamedPairModel(), samples, tmp_path, torch_device)

    actual = backend.infer(
        _NamedTensorPair(torch.full((2, 2), 5.0, device=torch_device), torch.ones(2, 2, device=torch_device))
    )

    torch.testing.assert_close(actual.cpu(), torch.full((2, 2), 4.0))


def test_openvino_dynamo_conversion_rejects_a_port_without_its_placeholder_name(tmp_path, monkeypatch, torch_device):
    def convert_and_rename_the_input(*args, **kwargs):
        ov_model = convert_model(*args, **kwargs)
        ov_model.input(0).get_tensor().set_names({"renamed"})
        return ov_model

    monkeypatch.setattr(ov_model_converter, "convert_model", convert_and_rename_the_input)
    sample = ((_tensor([[1.0, 2.0]], torch_device),), {})
    converter = OpenVINOModelConverter(output_dir=tmp_path, use_dynamo=True)

    with pytest.raises(RuntimeError, match="matches no single torch.export input"):
        converter.convert(_DoubleModel().to(torch_device), sample, _graph_spec(_DoubleModel(), sample))


class _NotCopyable:
    def __deepcopy__(self, memo):
        raise TypeError("_NotCopyable cannot be copied")


class _IgnoresContextModel(nn.Module):
    def forward(self, pair: _TensorPair, context: _NotCopyable) -> torch.Tensor:
        return pair.a - pair.b


@pytest.mark.parametrize("use_dynamo", [True, False])
def test_openvino_backend_does_not_copy_non_tensor_arguments(tmp_path, torch_device, use_dynamo):
    pair = _TensorPair(_tensor([[5.0, 6.0]], torch_device), _tensor([[1.0, 2.0]], torch_device))
    sample = ((pair,), {"context": _NotCopyable()})
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=use_dynamo)),
        _IgnoresContextModel(),
        [sample],
        tmp_path,
        torch_device,
    )

    actual = backend.infer(*sample[0], **sample[1])

    torch.testing.assert_close(actual.cpu(), torch.tensor([[4.0, 4.0]]))


def test_openvino_backend_serialization(model, sample_data, tmp_path, torch_device):
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False)), model, sample_data, tmp_path, torch_device
    )
    loaded = _save_and_load(model, backend.to_dict(), tmp_path)

    args, kwargs = sample_data[0]
    output = loaded.infer(*args, **kwargs)
    assert output.shape[0] == 1


class _UnregisteredDictOutput(dict):
    """A dict subclass unknown to the pytree registry.

    The explicit __init__ is required: torch.jit.trace, which the OpenVINO converter
    runs, refuses a dict subclass that only inherits dict.__init__, so without it the
    backend cannot be built and the test never reaches the save step.
    """

    def __init__(self, out):
        super().__init__(out=out)


class _DictSubclassOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(_KWARG_DIM, _KWARG_OUT)

    def forward(self, x: torch.Tensor) -> dict:
        return _UnregisteredDictOutput(self.fc(x))


def test_openvino_backend_rejects_opaque_output_object_at_save_time(tmp_path, torch_device):
    """A model whose forward() returns an unregistered dict subclass cannot be checkpointed.

    OpenVINOBackend keeps the forward() result as a template to rebuild the output
    structure on infer(). pytree keeps an unregistered class as a leaf, and loading
    that leaf with weights_only=True needs the class itself. The save has to fail with
    an actionable error instead of writing a checkpoint that only fails when loaded.
    """
    model = _DictSubclassOutputModel().to(torch_device).eval()
    sample = ((torch.randn(1, _KWARG_DIM).to(torch_device),), {})
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False)), model, [sample], tmp_path, torch_device
    )

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    with pytest.raises(ValueError, match=r"output_leaves.*weights_only=True"):
        TorchSaveTask().save(ckpt_dir, backend.to_dict())

    assert not (ckpt_dir / STATE_DICT_FILE).exists()


class _HFOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(_KWARG_DIM, _KWARG_OUT)

    def forward(self, x: torch.Tensor) -> BaseModelOutputWithPooling:
        hidden = self.fc(x)
        return BaseModelOutputWithPooling(last_hidden_state=hidden, pooler_output=hidden.sum(dim=-1))


def test_openvino_backend_serializes_transformers_model_output(tmp_path, torch_device):
    model = _HFOutputModel().to(torch_device).eval()
    x = torch.randn(1, _KWARG_DIM).to(torch_device)
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False)), model, [((x,), {})], tmp_path, torch_device
    )
    state_dict = backend.to_dict()

    output = _save_and_load(model, state_dict, tmp_path).infer(x)

    assert all(leaf.numel() == 0 for leaf in state_dict[OpenVINOBackend.STATE_OUTPUT_LEAVES])
    assert isinstance(output, BaseModelOutputWithPooling)
    assert list(output.keys()) == ["last_hidden_state", "pooler_output"]
    expected = backend.infer(x)
    torch.testing.assert_close(output.last_hidden_state, expected.last_hidden_state)
    torch.testing.assert_close(output.pooler_output, expected.pooler_output)


def test_openvino_backend_rejects_output_with_unserializable_spec(model, sample_data, tmp_path, torch_device):
    """Output whose pytree node has no serialized_type_name cannot be saved in a checkpoint.

    treespec_dumps raises NotImplementedError for such a node, and to_dict reports it as ValueError.
    """
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False)), model, sample_data, tmp_path, torch_device
    )
    backend._output_object = _PytreeNodeWithoutSerializedName(torch.empty(0))

    with pytest.raises(ValueError, match=OpenVINOBackend.STATE_OUTPUT_SPEC):
        backend.to_dict()


def test_openvino_backend_ir_survives_checkpoint_copy_and_zip(model, sample_data, tmp_path, torch_device):
    """OpenVINO IR is model.xml + model.bin; both must survive CopyBackendArtifactsTask and .tt zip."""
    tune_cache = tmp_path / "tune_cache"
    tune_cache.mkdir()
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False)),
        model,
        sample_data,
        tune_cache,
        torch_device,
    )
    state_dict = backend.to_dict()

    ckpt_dir = tmp_path / "checkpoint"
    ckpt_dir.mkdir()
    CopyBackendArtifactsTask().save(ckpt_dir, state_dict)
    TorchSaveTask().save(ckpt_dir, state_dict)

    shutil.rmtree(tune_cache)

    state_dict = TorchLoadTask().load(ckpt_dir)
    loaded = OpenVINOBackend.from_dict(model, state_dict)
    loaded.activate()
    args, kwargs = sample_data[0]
    output = loaded.infer(*args, **kwargs)
    assert output.shape[0] == 1

    ov_dir = loaded._ov_model_dir
    assert (ov_dir / OV_MODEL_XML_NAME).is_file()
    assert (ov_dir / OV_MODEL_BIN_NAME).is_file()

    ZipSaveTask().save(ckpt_dir, state_dict)
    shutil.rmtree(ckpt_dir)
    UnzipLoadTask().load(ckpt_dir)

    state_dict = TorchLoadTask().load(ckpt_dir)
    loaded_zip = OpenVINOBackend.from_dict(model, state_dict)
    loaded_zip.activate()
    output_zip = loaded_zip.infer(*args, **kwargs)
    assert output_zip.shape[0] == 1


# ---------------------------------------------------------------------------
# Regression tests: kwargs recorded out of forward-signature order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_dynamo", [True, False])
def test_kwargs_order_does_not_swap_inputs(tmp_path, torch_device, use_dynamo):
    """Conversion must not swap input tensor names when kwargs are recorded out of order.

    The model projects alpha with weight +1 and beta with weight -1.  With alpha=ones,
    beta=zeros the expected output is positive.  If conversion transposes the names the
    backend feeds ones into the -1 projection → negative output → assertion fails.
    """
    model = _TwoKwargModel().to(torch_device).eval()

    # Record sample with kwargs in reversed order (beta inserted before alpha).
    kwargs_reversed: dict = {
        "beta": torch.zeros(2, _KWARG_DIM, device=torch_device),
        "alpha": torch.ones(2, _KWARG_DIM, device=torch_device),
    }
    sample: Sample = ((), kwargs_reversed)

    with torch.no_grad():
        output_ref = model(**kwargs_reversed)

    input_spec = SampleMetadata.from_inputs((), kwargs_reversed)
    output_spec = SampleMetadata.from_outputs(output_ref)
    graph_spec = GraphSpec(name="two_kwarg", input_spec=input_spec, output_spec=output_spec)

    backend = OpenVINOBackend(config=OpenVINOBackendConfig(use_dynamo=use_dynamo))
    backend = backend.build(model, graph_spec=graph_spec, data=[sample], device=torch_device, cache_dir=tmp_path)
    ov_output = backend.infer(**kwargs_reversed)
    assert (ov_output > 0).all(), (
        "Output is not positive; alpha and beta inputs were likely swapped during OV inference."
    )


class _ExpandedStepModel(nn.Module):
    """Expands `step` to the batch of `x` the way diffusers UNets expand their timestep."""

    def forward(self, x: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        return x * step.expand(x.shape[0])[:, None].to(x.dtype)


def test_input_renamed_by_compilation_is_still_fed(tmp_path, torch_device):
    model = _ExpandedStepModel()
    args = (torch.ones(2, 3, device=torch_device), torch.tensor([5, 5], device=torch_device))
    with torch.no_grad():
        output_ref = model(*args)
    graph_spec = GraphSpec(
        name="expanded_step",
        input_spec=SampleMetadata.from_inputs(args, {}),
        output_spec=SampleMetadata.from_outputs(output_ref),
    )

    backend = OpenVINOBackend().build(model, graph_spec, [(args, {})], torch_device, tmp_path)

    torch.testing.assert_close(backend.infer(*args), output_ref, rtol=0, atol=0)


class _ReshapedDoubleModel(nn.Module):
    def forward(self, x):
        return (x * 2).reshape(2, 3)


def test_output_renamed_by_compilation_is_still_returned(tmp_path, torch_device):
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=torch_device)
    backend = backend_build(OpenVINOBackend(), _ReshapedDoubleModel(), [((x,), {})], tmp_path, torch_device)

    actual = backend.infer(x)

    torch.testing.assert_close(actual, torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], device=torch_device))


# ---------------------------------------------------------------------------
# Correlated batch-dimension tests
# ---------------------------------------------------------------------------

DIM = 8
DIM_B = DIM + 7


class _SharedBatchModel(nn.Module):
    """Two inputs, ``a`` and ``b``, that must share the same dynamic batch axis.

    ``a`` and ``b`` may differ in feature dim (DIM vs DIM_B), but ``proj_a(a) + proj_b(b)`` requires
    their batch axis to be equal. If dynamic_shapes assigns that axis two independent Dim objects
    instead of one shared Dim, torch.export detects the mismatch during tracing and raises
    ConstraintViolationError.
    """

    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(DIM, DIM)
        self.proj_b = nn.Linear(DIM_B, DIM)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.proj_a(a) + self.proj_b(b)


@pytest.fixture
def _shared_batch_artifacts(torch_device) -> tuple[nn.Module, GraphSpec, list]:
    """Return (model, graph_spec, samples) for _SharedBatchModel across a range of batch sizes."""
    model = _SharedBatchModel().to(torch_device).eval()
    graph_spec: GraphSpec | None = None
    batch_sizes = [1, 8]
    samples = []
    for bs in batch_sizes:
        args = (torch.rand(bs, DIM, device=torch_device), torch.rand(bs, DIM_B, device=torch_device))
        kwargs: dict = {}
        out = model(*args)
        in_meta = SampleMetadata.from_inputs(args, kwargs, batch_size=bs)
        out_meta = SampleMetadata.from_outputs(out, batch_size=bs)
        if graph_spec is None:
            graph_spec = GraphSpec(name="shared_batch", input_spec=in_meta, output_spec=out_meta)
        else:
            graph_spec.update_shapes_seen(in_meta, out_meta)
        samples.append((args, kwargs))
    assert graph_spec is not None
    return model, graph_spec, samples


# --- Unit tests: _create_dynamic_shapes_from_tensor_spec ---


def test_dynamic_shapes_shares_dim_object_for_correlated_batch_axes(_shared_batch_artifacts):
    """Every axis detected as *the* batch axis must map to the same Dim object."""
    _, graph_spec, _ = _shared_batch_artifacts
    shapes = _create_dynamic_shapes_from_tensor_spec(graph_spec.input_spec.tensor_specs)

    batch_dims = [spec_shapes[0] for spec_shapes in shapes.values() if 0 in spec_shapes]
    assert len(batch_dims) == 2
    unique_ids = {id(d) for d in batch_dims}
    assert len(unique_ids) == 1, (
        "All axes detected as the batch axis must share one Dim object "
        "so that torch.export can enforce the batch-equality constraint."
    )


# --- Integration tests: OpenVINOModelConverter with use_dynamo=True ---


def test_dynamo_converter_succeeds_for_shared_batch_model(tmp_path, torch_device, _shared_batch_artifacts):
    """Dynamo conversion must complete without error and produce valid IR.
    The converter should export the model to OpenVINO IR with a wide dynamic range
    and inference results must match PyTorch eager.
    Binary-exact weights and inputs keep the comparison stable with BF16 inference.
    """
    model, graph_spec, samples = _shared_batch_artifacts
    nn.init.constant_(model.proj_a.weight, 0.125)
    nn.init.constant_(model.proj_a.bias, 0.25)
    nn.init.constant_(model.proj_b.weight, -0.0625)
    nn.init.constant_(model.proj_b.bias, -0.125)
    bs = 2
    a = torch.arange(bs * DIM, device=torch_device, dtype=torch.float32).reshape(bs, DIM) / 16
    b = torch.arange(bs * DIM_B, device=torch_device, dtype=torch.float32).reshape(bs, DIM_B) / 32
    ref = model(a, b)

    converter = OpenVINOModelConverter(output_dir=tmp_path / "ov", use_dynamo=True)
    xml_path = converter.convert(module=model, sample=samples[-1], graph_spec=graph_spec)
    assert xml_path.exists(), "Converter did not produce an IR file"

    be_cache = tmp_path / "be"
    be_cache.mkdir()
    backend = OpenVINOBackend(config=OpenVINOBackendConfig(use_dynamo=True))
    backend = backend.build(model, graph_spec=graph_spec, data=samples, device=torch_device, cache_dir=be_cache)

    ov_out = backend.infer(a, b)

    torch.testing.assert_close(ov_out, ref, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------------------
# Positional-only forward parameters (before ``/``)
# ---------------------------------------------------------------------------


class _PositionalOnlyModel(nn.Module):
    """``_SharedBatchModel`` whose first input is declared positional-only."""

    def __init__(self):
        super().__init__()
        self.proj_a = nn.Linear(DIM, DIM)
        self.proj_b = nn.Linear(DIM_B, DIM)

    def forward(self, a: torch.Tensor, /, b: torch.Tensor) -> torch.Tensor:
        return self.proj_a(a) + self.proj_b(b)


def _graph_spec_over_batch_sizes(
    model: nn.Module, batch_sizes, device, b_as_kwarg: bool = False
) -> tuple[GraphSpec, list[Sample]]:
    """Record *model* at several batch sizes so the batch axis is detected as dynamic.

    With *b_as_kwarg* the second tensor is recorded as a keyword argument, mixing a
    positional-only and a keyword input in one sample.
    """
    graph_spec: GraphSpec | None = None
    samples: list[Sample] = []
    for bs in batch_sizes:
        a = torch.rand(bs, DIM, device=device)
        b = torch.rand(bs, DIM_B, device=device)
        args, kwargs = ((a,), {"b": b}) if b_as_kwarg else ((a, b), {})
        with torch.no_grad():
            out = model(*args, **kwargs)
        in_meta = SampleMetadata.from_inputs(args, kwargs, batch_size=bs)
        out_meta = SampleMetadata.from_outputs(out, batch_size=bs)
        if graph_spec is None:
            graph_spec = GraphSpec(name="positional_only", input_spec=in_meta, output_spec=out_meta)
        else:
            graph_spec.update_shapes_seen(in_meta, out_meta)
        samples.append((args, kwargs))
    assert graph_spec is not None
    return graph_spec, samples


def test_dynamo_converter_handles_positional_only_mixed_with_kwarg(tmp_path, torch_device):
    """End-to-end conversion of a forward mixing a positional-only arg and a keyword arg.

    Inference runs at batch size 2, which was never recorded, so the batch axis must have stayed dynamic.
    """
    torch.manual_seed(0)
    model = _PositionalOnlyModel().to(torch_device).eval()
    graph_spec, samples = _graph_spec_over_batch_sizes(model, [1, 8], device=torch_device, b_as_kwarg=True)

    # Compute the reference before build(), which offloads the module.
    bs = 2
    a = torch.rand(bs, DIM, device=torch_device)
    b = torch.rand(bs, DIM_B, device=torch_device)
    with torch.no_grad():
        ref = model(a, b=b)

    converter = OpenVINOModelConverter(output_dir=tmp_path / "ov", use_dynamo=True)
    assert converter.convert(module=model, sample=samples[-1], graph_spec=graph_spec).exists()

    be_cache = tmp_path / "be"
    be_cache.mkdir()
    backend = OpenVINOBackend(config=OpenVINOBackendConfig(use_dynamo=True))
    backend = backend.build(model, graph_spec=graph_spec, data=samples, device=torch_device, cache_dir=be_cache)

    torch.testing.assert_close(backend.infer(a, b=b), ref, rtol=2e-3, atol=2e-3)


class _FakeCore:
    def __init__(self, uuids: dict[str, str]):
        self._uuids = uuids
        self.available_devices = ["CPU", *uuids]

    def get_property(self, device: str, name: str) -> str:
        assert name == "DEVICE_UUID"
        return self._uuids[device]


_XPU_UUID = uuid.UUID("868023e2-0000-0000-0300-000000000000")


def test_openvino_device_for_xpu_picks_the_gpu_with_the_same_uuid(monkeypatch):
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device: SimpleNamespace(uuid=_XPU_UUID))
    core = _FakeCore({"GPU.0": "868080a7040000000002000000000000", "GPU.1": _XPU_UUID.hex})

    assert openvino_device_for_xpu(core, torch.device("xpu")) == "GPU.1"


def test_openvino_device_for_xpu_raises_when_no_gpu_matches(monkeypatch):
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device: SimpleNamespace(uuid=_XPU_UUID))
    core = _FakeCore({"GPU.0": "868080a7040000000002000000000000"})

    with pytest.raises(RuntimeError, match="No OpenVINO GPU matches"):
        openvino_device_for_xpu(core, torch.device("xpu"))


@requires_xpu
def test_openvino_device_for_xpu_is_the_device_pytorch_uses():
    core = ov.Core()
    device = torch.device("xpu", torch.xpu.current_device())

    ov_device = openvino_device_for_xpu(core, device)

    assert core.get_property(ov_device, "FULL_DEVICE_NAME").startswith(torch.xpu.get_device_name(device))


def _host_copy(*args, **kwargs):
    raise AssertionError("OpenVINO inference on XPU went through host memory")


@requires_leo
def test_openvino_xpu_infer_does_not_copy_through_the_host(tmp_path, monkeypatch):
    x = torch.ones(2, _KWARG_DIM, device="xpu")
    model = nn.Linear(_KWARG_DIM, _KWARG_OUT).to("xpu").eval()
    backend = backend_build(OpenVINOBackend(), model, [((x,), {})], tmp_path, torch.device("xpu"))
    monkeypatch.setattr(torch.Tensor, "cpu", _host_copy)
    monkeypatch.setattr(torch.Tensor, "numpy", _host_copy)
    monkeypatch.setattr(ov.CompiledModel, "__call__", _host_copy)

    output = backend.infer(x)

    assert output.device == x.device


@requires_leo
def test_openvino_xpu_infer_matches_eager(tmp_path):
    torch.manual_seed(0)
    x = torch.arange(2 * _KWARG_DIM, dtype=torch.float32, device="xpu").reshape(2, _KWARG_DIM)
    model = nn.Linear(_KWARG_DIM, _KWARG_OUT).to("xpu").eval()
    with torch.no_grad():
        expected = model(x)
    backend = backend_build(OpenVINOBackend(), model, [((x,), {})], tmp_path, torch.device("xpu"))

    actual = backend.infer(x)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


class _ArgmaxModel(nn.Module):
    def forward(self, x):
        return x.argmax(dim=-1)


class _GreaterThanTwoModel(nn.Module):
    def forward(self, x):
        return x > 2


class _DoubleModel(nn.Module):
    def forward(self, x):
        return x * 2


class _ScaleModel(nn.Module):
    def forward(self, x, scale: float = 1.0):
        return x * scale


@requires_leo
def test_openvino_xpu_writes_an_int64_output(tmp_path):
    x = torch.tensor([[1.0, 5.0, 2.0], [7.0, 0.0, 3.0]], device="xpu")
    backend = backend_build(OpenVINOBackend(), _ArgmaxModel(), [((x,), {})], tmp_path, torch.device("xpu"))

    actual = backend.infer(x)

    torch.testing.assert_close(actual, torch.tensor([1, 0], device="xpu"))


@requires_leo
def test_openvino_xpu_writes_a_bool_output(tmp_path):
    x = torch.tensor([[1.0, 5.0, 2.0], [7.0, 0.0, 3.0]], device="xpu")
    backend = backend_build(OpenVINOBackend(), _GreaterThanTwoModel(), [((x,), {})], tmp_path, torch.device("xpu"))

    actual = backend.infer(x)

    torch.testing.assert_close(actual, torch.tensor([[False, True, False], [True, False, True]], device="xpu"))


@requires_leo
def test_openvino_xpu_reads_a_transposed_input(tmp_path):
    sample = torch.zeros(2, 3, device="xpu")
    backend = backend_build(OpenVINOBackend(), _DoubleModel(), [((sample,), {})], tmp_path, torch.device("xpu"))

    actual = backend.infer(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="xpu").t())

    torch.testing.assert_close(actual, torch.tensor([[2.0, 6.0, 10.0], [4.0, 8.0, 12.0]], device="xpu"))


@requires_leo
def test_openvino_xpu_second_infer_with_the_same_shape_returns_its_own_result(tmp_path):
    sample = torch.zeros(2, 3, device="xpu")
    backend = backend_build(OpenVINOBackend(), _DoubleModel(), [((sample,), {})], tmp_path, torch.device("xpu"))
    backend.infer(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="xpu"))

    actual = backend.infer(torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device="xpu"))

    torch.testing.assert_close(actual, torch.tensor([[14.0, 16.0, 18.0], [20.0, 22.0, 24.0]], device="xpu"))


@requires_leo
def test_openvino_xpu_second_infer_with_the_same_shape_keeps_the_first_result(tmp_path):
    sample = torch.zeros(2, 3, device="xpu")
    backend = backend_build(OpenVINOBackend(), _DoubleModel(), [((sample,), {})], tmp_path, torch.device("xpu"))
    first = backend.infer(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device="xpu"))

    backend.infer(torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], device="xpu"))

    torch.testing.assert_close(first, torch.tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], device="xpu"))


@requires_leo
@pytest.mark.parametrize("use_dynamo", [True, False])
def test_openvino_xpu_compiles_a_forward_with_a_scalar_keyword_argument(tmp_path, use_dynamo):
    x = torch.tensor([[1.0, 2.0, 3.0]], device="xpu")
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=use_dynamo)),
        _ScaleModel(),
        [((x,), {"scale": 3.0})],
        tmp_path,
        torch.device("xpu"),
    )

    actual = backend.infer(x, scale=3.0)

    torch.testing.assert_close(actual, torch.tensor([[3.0, 6.0, 9.0]], device="xpu"))


@requires_leo
def test_openvino_xpu_rejects_an_input_on_the_cpu(tmp_path):
    sample = torch.zeros(2, 3, device="xpu")
    backend = backend_build(OpenVINOBackend(), _DoubleModel(), [((sample,), {})], tmp_path, torch.device("xpu"))

    with pytest.raises(ValueError, match="OpenVINO tensor args_0 is on cpu, but the model runs on xpu:0"):
        backend.infer(torch.zeros(2, 3))


@requires_leo
def test_openvino_xpu_releases_the_opencl_context_when_the_backend_is_collected(tmp_path, monkeypatch):
    # Release backends that earlier tests left for the collector.
    gc.collect()
    release_context = xpu_remote._opencl().clReleaseContext
    released = []
    monkeypatch.setattr(
        xpu_remote._opencl(), "clReleaseContext", lambda context: released.append(release_context(context))
    )
    sample = torch.zeros(2, 3, device="xpu")
    backend = backend_build(OpenVINOBackend(), _DoubleModel(), [((sample,), {})], tmp_path, torch.device("xpu"))

    del backend
    gc.collect()

    assert released == [0]


@requires_xpu
def test_openvino_xpu_backend_without_leo_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(xpu_remote, "_opencl_gpus", lambda: [])
    x = torch.tensor([[1.0, 2.0, 3.0]], device="xpu")

    with pytest.raises(LevelZeroInteropError) as error:
        backend_build(OpenVINOBackend(), nn.ReLU(), [((x,), {})], tmp_path, torch.device("xpu"))

    assert str(error.value) == LEVEL_ZERO_INTEROP_MISSING


def test_ov_element_type_rejects_a_dtype_openvino_cannot_bind():
    with pytest.raises(ValueError, match="complex64"):
        ov_element_type(torch.complex64)


@requires_leo
def test_openvino_xpu_build_moves_the_module_to_the_cpu(tmp_path):
    model = nn.Linear(_KWARG_DIM, _KWARG_OUT).to("xpu").eval()
    x = torch.ones(2, _KWARG_DIM, device="xpu")

    backend_build(OpenVINOBackend(), model, [((x,), {})], tmp_path, torch.device("xpu"))

    assert model.weight.device.type == "cpu"


@requires_leo
def test_openvino_xpu_build_with_offload_module_false_keeps_the_weights_in_place(tmp_path):
    model = nn.Linear(_KWARG_DIM, _KWARG_OUT).to("xpu").eval()
    weight_address = model.weight.data_ptr()
    x = torch.ones(2, _KWARG_DIM, device="xpu")
    backend = OpenVINOBackend(OpenVINOBackendConfig(offload_module=False))

    backend_build(backend, model, [((x,), {})], tmp_path, torch.device("xpu"))

    assert model.weight.data_ptr() == weight_address


def test_offload_module_survives_a_checkpoint(model, sample_data, tmp_path, torch_device):
    config = OpenVINOBackendConfig(use_dynamo=False, offload_module=False)
    backend = backend_build(OpenVINOBackend(config), model, sample_data, tmp_path, torch_device)

    loaded = _save_and_load(model, backend.to_dict(), tmp_path)

    assert loaded.describe() == "OpenVINOBackend(use_dynamo=False,offload_module=False)"


def test_reuse_ir_survives_a_checkpoint(model, sample_data, tmp_path, torch_device):
    config = OpenVINOBackendConfig(use_dynamo=False, reuse_ir=True)
    backend = backend_build(OpenVINOBackend(config), model, sample_data, tmp_path, torch_device)

    loaded = _save_and_load(model, backend.to_dict(), tmp_path)

    assert loaded.describe() == "OpenVINOBackend(use_dynamo=False,reuse_ir=True)"


class _RowCountModel(nn.Module):
    def forward(self, x):
        return {"y": x * 2, "rows": x.shape[0]}


def test_traced_model_that_returns_the_row_count_of_its_build_sample_is_rejected(torch_device, monkeypatch):
    monkeypatch.setattr(global_config, "max_num_samples_stored", 2)
    module = TunableModule(_RowCountModel(), "row-count")
    module(torch.ones(1, 3, device=torch_device))
    module(torch.ones(2, 3, device=torch_device))
    strategy = OneBackendStrategy(OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False)))

    with pytest.raises(NoCorrectBackendError) as raised:
        module.tune(device=torch_device, strategy=strategy.enable_find_max_batch_size(False))

    assert "value 1 is 1, expected 2" in str(raised.value.__cause__)


@requires_leo
def test_float32_model_on_xpu_reports_the_float16_tolerance_it_computes_in(tmp_path):
    x = torch.ones(1, 2, device="xpu")

    backend = backend_build(OpenVINOBackend(), nn.Linear(2, 2), [((x,), {})], tmp_path, torch.device("xpu"))

    assert backend.output_tolerance == pytest.approx(0.03125)


@pytest.mark.skipif(
    ov.Core().get_property("CPU", "INFERENCE_PRECISION_HINT") != ov.Type.f32,
    reason="OpenVINO computes float32 models in a lower precision on this CPU, such as bfloat16 with AMX",
)
def test_float32_model_on_cpu_reports_the_float32_tolerance(tmp_path):
    x = torch.ones(1, 2)

    backend = backend_build(OpenVINOBackend(), nn.Linear(2, 2), [((x,), {})], tmp_path, torch.device("cpu"))

    assert backend.output_tolerance == pytest.approx(3.45e-4, rel=1e-2)


def test_output_tolerance_of_a_backend_that_is_not_built_is_an_error():
    with pytest.raises(RuntimeError, match="OpenVINO model is not compiled"):
        OpenVINOBackend().output_tolerance  # noqa: B018
