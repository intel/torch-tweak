# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Unit tests for OpenVINOBackend."""

import shutil
import uuid
from types import SimpleNamespace

import openvino as ov
import pytest
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.backend.openvino import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.backend.openvino.openvino_backend import openvino_device_for_xpu
from torch_tweak.torch.backend.openvino.ov_model_converter import (
    OV_MODEL_BIN_NAME,
    OV_MODEL_XML_NAME,
    OpenVINOModelConverter,
    _create_dynamic_shapes,
    _create_dynamic_shapes_from_tensor_spec,
    _ordered_input_names_for_dynamo,
)
from torch_tweak.torch.checkpoint.storage_tasks import (
    STATE_DICT_FILE,
    CopyBackendArtifactsTask,
    TorchLoadTask,
    TorchSaveTask,
    UnzipLoadTask,
    ZipSaveTask,
)
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.utils.module import get_arguments_names

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
    """`torch.return_types.max` is a pytree node without a `serialized_type_name`.

    `pytree.treespec_dumps` raises NotImplementedError for such a node, and `to_dict` reports it as ValueError.
    """
    backend = backend_build(
        OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=False)), model, sample_data, tmp_path, torch_device
    )
    output_without_serialized_type_name = torch.ones(2, 2).max(dim=0)
    backend._output_object = output_without_serialized_type_name

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
# Regression tests: kwargs recorded out of forward-signature order (dynamo path)
# ---------------------------------------------------------------------------


def test_ordered_input_names_for_dynamo_reorders_by_signature():
    """_ordered_input_names_for_dynamo must return names in forward-signature order.

    Recording inserts kwargs in "beta, alpha" order; the forward signature is "alpha, beta".
    The helper must return ["kwargs_alpha", "kwargs_beta"] so that OV port 0 gets the
    name belonging to alpha, not beta.
    """
    # Intentionally recorded in reverse order.
    kwargs_reversed = {
        "beta": torch.ones(2, _KWARG_DIM),
        "alpha": torch.zeros(2, _KWARG_DIM),
    }
    input_spec = SampleMetadata.from_inputs((), kwargs_reversed)
    # Key order mirrors _ordered_kwargs_for_export: signature order (alpha first).
    names = _ordered_input_names_for_dynamo(input_spec, ["alpha", "beta"])

    assert names == ["kwargs_alpha", "kwargs_beta"], (
        f"Expected ['kwargs_alpha', 'kwargs_beta'], got {names}. "
        "This indicates the helper is not reordering by signature."
    )


def test_dynamo_kwargs_order_does_not_swap_inputs(tmp_path, torch_device):
    """Dynamo conversion must not swap input tensor names when kwargs are recorded out of order.

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

    xml_path = OpenVINOModelConverter(output_dir=tmp_path / "ov", use_dynamo=True).convert(
        module=model, sample=sample, graph_spec=graph_spec
    )

    # Verify OV port names are in forward-signature order (alpha=0, beta=1).
    ov_model = ov.Core().read_model(xml_path)
    assert len(ov_model.inputs) == 2
    assert ov_model.input(0).any_name == "kwargs_alpha"
    assert ov_model.input(1).any_name == "kwargs_beta"

    # Verify end-to-end inference produces the correct (positive) output.
    be_cache = tmp_path / "be"
    be_cache.mkdir()
    backend = OpenVINOBackend(config=OpenVINOBackendConfig(use_dynamo=True))
    backend = backend.build(model, graph_spec=graph_spec, data=[sample], device=torch_device, cache_dir=be_cache)
    ov_output = backend.infer(**kwargs_reversed)
    assert (ov_output > 0).all(), (
        "Output is not positive; alpha and beta inputs were likely swapped during OV inference."
    )


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


@pytest.fixture(scope="module")
def _shared_batch_artifacts() -> tuple[nn.Module, GraphSpec, list]:
    """Return (model, graph_spec, samples) for _SharedBatchModel across a range of batch sizes."""
    device = "xpu"
    model = _SharedBatchModel().to(device).eval()
    graph_spec: GraphSpec | None = None
    batch_sizes = [1, 8]
    samples = []
    for bs in batch_sizes:
        args = (torch.rand(bs, DIM, device=device), torch.rand(bs, DIM_B, device=device))
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


# --- Integration tests: OpenVINOModelConverter._convert_dynamo ---


def test_dynamo_converter_succeeds_for_shared_batch_model(tmp_path, torch_device, _shared_batch_artifacts):
    """Dynamo conversion must complete without error and produce valid IR.
    The converter should export the model to OpenVINO IR with a wide dynamic range
    and inference results must match PyTorch eager.
    """
    model, graph_spec, samples = _shared_batch_artifacts
    converter = OpenVINOModelConverter(output_dir=tmp_path / "ov", use_dynamo=True)
    xml_path = converter.convert(module=model, sample=samples[-1], graph_spec=graph_spec)
    assert xml_path.exists(), "Converter did not produce an IR file"

    be_cache = tmp_path / "be"
    be_cache.mkdir()
    backend = OpenVINOBackend(config=OpenVINOBackendConfig(use_dynamo=True))
    backend = backend.build(model, graph_spec=graph_spec, data=samples, device=torch_device, cache_dir=be_cache)

    bs = 2
    a = torch.rand(bs, DIM, device=torch_device)
    b = torch.rand(bs, DIM_B, device=torch_device)
    model = model.to(torch_device)
    ref = model(a, b)
    ov_out = backend.infer(a, b)

    torch.testing.assert_close(ov_out, ref, rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------------------
# Positional-only forward parameters (before ``/``)
# ---------------------------------------------------------------------------


class _PositionalOnlyModel(nn.Module):
    """``_SharedBatchModel`` whose first input is declared positional-only.

    ``a`` can only be passed positionally, so ``_create_dynamic_shapes`` has to derive its
    dynamic-shape key from the signature rather than from the recorded kwargs.
    """

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


def test_dynamic_shapes_keys_positional_only_parameters_by_name():
    """A tensor bound to a positional-only parameter must still get its dynamic batch axis.

    ``_create_inputs_mapping`` keys positionally recorded tensors by index, so they must be
    rekeyed to signature names. Skipping positional-only parameters silently drops their entry:
    torch.export then either bakes in a static shape or rejects the incomplete dict.
    """
    model = _PositionalOnlyModel().eval()
    graph_spec, _ = _graph_spec_over_batch_sizes(model, [1, 8], device="cpu")

    shapes = _create_dynamic_shapes(graph_spec.input_spec, get_arguments_names(model.forward))

    assert sorted(shapes) == ["a", "b"], (
        f"Expected dynamic shapes for both forward parameters, got {sorted(shapes)}. "
        "The positional-only parameter was likely dropped."
    )
    assert shapes["a"][0] is shapes["b"][0], "Both inputs must share one batch Dim object."


def test_var_positional_forward_is_rejected():
    """A ``*args`` forward exposes no parameter names, so conversion must fail loudly.

    Returning empty name lists would leave ``dynamic_shapes`` empty and drop every kwarg,
    making ``torch.export`` bake in static shapes instead of reporting the problem. The
    tuning strategies catch build failures, so raising only skips this backend.
    """

    class _VarArgsModel(nn.Module):
        def forward(self, *args: torch.Tensor) -> torch.Tensor:
            return args[0] * 2

    model = _VarArgsModel().eval()

    with pytest.raises(ValueError, match="has no named parameters"):
        get_arguments_names(model.forward)


def test_dynamo_converter_handles_positional_only_mixed_with_kwarg(tmp_path, torch_device):
    """End-to-end conversion of a forward mixing a positional-only arg and a keyword arg.

    ``torch.export`` requires a dict ``dynamic_shapes`` to cover *every* argument name. Dropping
    the positional-only entry leaves ``{"b": ...}`` and export rejects it outright. Inference then
    runs at batch size 2, which was never recorded, so the batch axis must have stayed dynamic.
    """
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
