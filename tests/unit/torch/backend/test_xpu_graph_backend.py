# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for XPUGraphBackend."""

import copy
import inspect
import math
from collections.abc import Iterator

import pytest
import torch
import torch.nn as nn

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.backend.xpu_graph_backend import XPUGraphBackend, XPUGraphBackendConfig
from torch_tweak.torch.checkpoint.storage_tasks import TorchLoadTask, TorchSaveTask
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.module.tunable_module import TunableModule
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy, OneBackendStrategy
from torch_tweak.torch.tuning import tune


class LinearRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return torch.relu(self.linear(x))


class Scaled(nn.Module):
    def forward(self, x, scale=1.0):
        y = x * scale
        return y, y.sum()


class NanOutput(nn.Module):
    def forward(self, x):
        return torch.sqrt(-x.abs() - 1)


class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(autouse=True)
def eager_calls(request, monkeypatch) -> Iterator[list[tuple]]:
    """Records the inputs XPUGraphBackend runs on the original module.

    The fallback output equals eager, so a test that compares to eager cannot see it.
    A test that does not take eager_calls as an argument fails when the fallback runs.
    """
    calls: list[tuple] = []
    infer_eager = XPUGraphBackend._infer_eager

    def recording(self, args, kwargs):
        calls.append(args)
        return infer_eager(self, args, kwargs)

    monkeypatch.setattr(XPUGraphBackend, "_infer_eager", recording)
    yield calls
    if "eager_calls" not in inspect.signature(request.function).parameters:
        assert not calls, "XPU graph backend ran the original module instead of replaying"


@pytest.fixture
def model(torch_device):
    return LinearRelu().to(torch_device).eval()


@pytest.fixture
def sample_data(torch_device) -> list[Sample]:
    args = (torch.randn(2, 16, device=torch_device),)
    return [(args, {})]


def move_to_dtype(sample_data, dtype):
    args, kwargs = sample_data[0]
    return [((args[0].to(dtype),), kwargs)]


def static_graph_spec(module, data) -> GraphSpec:
    args, kwargs = data[0]
    with torch.no_grad():
        outputs = module(*args, **kwargs)
    return GraphSpec(
        name="sample",
        input_spec=SampleMetadata.from_inputs(args, kwargs),
        output_spec=SampleMetadata.from_outputs(outputs),
    )


def backend_build(backend, dtype, model, sample_data, tmp_path, device="xpu"):
    device = torch.device(device)
    model = model.to(device, dtype=dtype).eval()
    data = move_to_dtype(sample_data, dtype)
    return backend.build(model, graph_spec=static_graph_spec(model, data), data=data, device=device, cache_dir=tmp_path)


@requires_xpu
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["float16", "bfloat16", "float32"],
)
def test_replay_matches_eager_on_new_inputs(dtype, model, sample_data, tmp_path):
    backend = backend_build(XPUGraphBackend(), dtype, model, sample_data, tmp_path)
    sample_data = move_to_dtype(sample_data, dtype)
    args, kwargs = sample_data[0]
    new_args = (torch.randn_like(args[0]),)

    with torch.no_grad():
        torch.testing.assert_close(backend.infer(*args, **kwargs), model(*args, **kwargs))
        torch.testing.assert_close(backend.infer(*new_args, **kwargs), model(*new_args, **kwargs))
        torch.testing.assert_close(backend.infer(*args, **kwargs), model(*args, **kwargs))

    backend.deactivate()


@requires_xpu
def test_deactivate_then_activate_records_again(model, sample_data, tmp_path):
    backend = backend_build(XPUGraphBackend(), torch.float32, model, sample_data, tmp_path)
    args, kwargs = sample_data[0]
    backend.deactivate()
    backend.activate()
    new_args = (torch.randn_like(args[0]),)
    with torch.no_grad():
        torch.testing.assert_close(backend.infer(*new_args, **kwargs), model(*new_args, **kwargs))


@requires_xpu
def test_deploy_after_load_records_and_drops_samples(model, sample_data, tmp_path, torch_device):
    backend = backend_build(XPUGraphBackend(), torch.float32, model, sample_data, tmp_path)
    backend.deactivate()
    backend.activate()
    TorchSaveTask().save(tmp_path, backend.to_dict())
    loaded = XPUGraphBackend.from_dict(model, TorchLoadTask().load(tmp_path))
    loaded.deploy(device=torch_device)
    assert loaded._data is None

    args, kwargs = sample_data[0]
    new_args = (torch.randn_like(args[0]),)
    with torch.no_grad():
        torch.testing.assert_close(loaded.infer(*new_args, **kwargs), model(*new_args, **kwargs))


@requires_xpu
def test_kwargs_and_tuple_output(torch_device, tmp_path, eager_calls):
    module = Scaled().to(torch_device).eval()
    args = (torch.randn(3, 5, device=torch_device),)
    kwargs = {"scale": 2.0}
    backend = XPUGraphBackend()
    data = [(args, kwargs)]
    backend = backend.build(module, static_graph_spec(module, data), data, torch_device, tmp_path)

    new_args = (torch.randn_like(args[0]),)
    with torch.no_grad():
        got = backend.infer(*new_args, **kwargs)
        expected = module(*new_args, **kwargs)
    torch.testing.assert_close(got[0], expected[0])
    torch.testing.assert_close(got[1], expected[1])

    with torch.no_grad():
        got = backend.infer(*new_args, scale=3.0)
        expected = module(*new_args, scale=3.0)
    torch.testing.assert_close(got[0], expected[0])
    torch.testing.assert_close(got[1], expected[1])
    assert len(eager_calls) == 1


@requires_xpu
def test_conv_replay(torch_device, tmp_path):
    module = TinyConv().to(torch_device).eval()
    args = (torch.randn(2, 1, 8, 8, device=torch_device),)
    backend = XPUGraphBackend()
    data = [(args, {})]
    backend = backend.build(module, static_graph_spec(module, data), data, torch_device, tmp_path)
    new_args = (torch.randn_like(args[0]),)
    with torch.no_grad():
        torch.testing.assert_close(backend.infer(*new_args), module(*new_args))


@requires_xpu
def test_autocast_keeps_recorded_dtype(model, sample_data, tmp_path):
    config = XPUGraphBackendConfig(autocast_enabled=True, autocast_dtype=torch.float16)
    backend = backend_build(XPUGraphBackend(config), torch.float32, model, sample_data, tmp_path)
    args, kwargs = sample_data[0]
    output = backend.infer(*args, **kwargs)
    assert output.dtype == torch.float16
    with torch.autocast(device_type="xpu", dtype=torch.float16, cache_enabled=False), torch.no_grad():
        expected = model(*args, **kwargs)
    torch.testing.assert_close(output, expected)


@requires_xpu
def test_infer_does_not_record_again(model, sample_data, tmp_path, monkeypatch):
    backend = backend_build(XPUGraphBackend(), torch.float32, model, sample_data, tmp_path)
    calls = {"n": 0}
    real_graph = torch.xpu.graph

    class CountingGraph:
        def __init__(self, *args, **kwargs):
            calls["n"] += 1
            self._graph = real_graph(*args, **kwargs)

        def __enter__(self):
            return self._graph.__enter__()

        def __exit__(self, *exc):
            return self._graph.__exit__(*exc)

    monkeypatch.setattr(torch.xpu, "graph", CountingGraph)
    args, kwargs = sample_data[0]
    backend.infer(*args, **kwargs)
    backend.infer(*(torch.randn_like(args[0]),), **kwargs)
    assert calls["n"] == 0


@requires_xpu
def test_matching_nans_are_accepted(torch_device, tmp_path):
    module = NanOutput().to(torch_device).eval()
    args = (torch.randn(4, device=torch_device),)
    backend = XPUGraphBackend()
    data = [(args, {})]
    backend = backend.build(module, static_graph_spec(module, data), data, torch_device, tmp_path)
    output = backend.infer(*args)
    assert torch.isnan(output).all()


@requires_xpu
def test_empty_recording_is_rejected(torch_device, tmp_path):
    module = nn.Identity().to(torch_device).eval()
    args = (torch.randn(4, device=torch_device),)
    backend = XPUGraphBackend()
    with pytest.raises(RuntimeError, match="no device work"):
        backend.build(module, static_graph_spec(module, [(args, {})]), [(args, {})], torch_device, tmp_path)


def seen_graph_spec(module, small: Sample, large: Sample, large_batch_size: int) -> GraphSpec:
    """Builds the graph spec tuning records after it sees small at batch size 1 and large at large_batch_size."""
    small_args, small_kwargs = small
    large_args, large_kwargs = large
    with torch.no_grad():
        small_outputs = module(*small_args, **small_kwargs)
        large_outputs = module(*large_args, **large_kwargs)
    spec = GraphSpec(
        name="seen",
        input_spec=SampleMetadata.from_inputs(small_args, small_kwargs, batch_size=1),
        output_spec=SampleMetadata.from_outputs(small_outputs, batch_size=1),
    )
    spec.update_shapes_seen(
        SampleMetadata.from_inputs(large_args, large_kwargs, batch_size=large_batch_size),
        SampleMetadata.from_outputs(large_outputs, batch_size=large_batch_size),
    )
    return spec


def rows(*shape: int, device: torch.device) -> torch.Tensor:
    """A tensor whose rows all differ: values spread evenly from -1 to 1."""
    return torch.linspace(-1, 1, math.prod(shape), device=device).reshape(shape)


def batched_build(module, torch_device, tmp_path, largest=64, config=None, features=16):
    """Builds a backend on a graph spec that saw batch sizes 1 and largest."""
    small = ((rows(1, features, device=torch_device),), {})
    large = ((rows(largest, features, device=torch_device),), {})
    spec = seen_graph_spec(module, small, large, largest)
    return XPUGraphBackend(config).build(module, spec, [small, large], torch_device, tmp_path)


def eager(module, x):
    with torch.no_grad():
        return module(x)


class RowMixing(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=0, keepdim=True)


class DependsOnFirstRow(nn.Module):
    def forward(self, x):
        return x + x[:1]


class TensorHolder:
    def __init__(self, value):
        self.value = value


class HoldsOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return TensorHolder(self.linear(x))


class Stacked(nn.Module):
    """Takes a batch stacked twice along axis 0, like classifier-free guidance."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        y = self.linear(x)
        return torch.cat([y, y], dim=0)


class BatchLast(nn.Module):
    """Takes features first and the batch on axis 1."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x.T)


class WideToOne(nn.Module):
    """A large input row and a one-value output row, so input buffers dominate the recorded memory."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4096, 1)

    def forward(self, x):
        return self.linear(x)


@requires_xpu
def test_batch_axis_records_the_size_grid(model, torch_device, tmp_path):
    backend = batched_build(model, torch_device, tmp_path, largest=64)

    assert backend.recorded_batch_sizes == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]


@requires_xpu
def test_grid_ends_at_the_largest_batch_seen(model, torch_device, tmp_path):
    backend = batched_build(model, torch_device, tmp_path, largest=20)

    assert backend.recorded_batch_sizes == [1, 2, 4, 8, 16, 20]


@requires_xpu
def test_max_recorded_batch_size_ends_the_grid(model, torch_device, tmp_path):
    config = XPUGraphBackendConfig(max_recorded_batch_size=12)

    backend = batched_build(model, torch_device, tmp_path, largest=64, config=config)

    assert backend.recorded_batch_sizes == [1, 2, 4, 8, 12]


def test_max_recorded_batch_size_below_1_is_rejected():
    with pytest.raises(ValueError, match="at least 1, got 0"):
        XPUGraphBackendConfig(max_recorded_batch_size=0)


@requires_xpu
@pytest.mark.parametrize("batch_size", [1, 3, 17, 64])
def test_padded_batch_matches_eager(model, torch_device, tmp_path, batch_size):
    x = rows(batch_size, 16, device=torch_device)
    expected = eager(model, x)
    backend = batched_build(model, torch_device, tmp_path, largest=64)

    actual = backend.infer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_batch_above_the_largest_graph_replays_in_chunks(model, torch_device, tmp_path):
    x = rows(11, 16, device=torch_device)
    expected = eager(model, x)
    backend = batched_build(model, torch_device, tmp_path, config=XPUGraphBackendConfig(max_recorded_batch_size=4))

    actual = backend.infer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_padding_does_not_leak_rows_of_a_larger_graph(model, torch_device, tmp_path):
    x = rows(5, 16, device=torch_device)
    expected = eager(model, x)
    backend = batched_build(model, torch_device, tmp_path, largest=64)
    backend.infer(torch.full((64, 16), 1e4, device=torch_device))

    actual = backend.infer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_graphs_share_the_input_memory_of_the_largest_batch(torch_device, tmp_path):
    # A row of 4096 float32 values takes 16 KiB. The tuning samples (65 rows) and one shared input
    # buffer (64 rows) take 2 MiB. One input tensor per graph would take 295 rows instead of 64.
    module = WideToOne().to(torch_device).eval()
    before = torch.xpu.memory_allocated()

    # _backend keeps the graphs and their buffers allocated while memory is read.
    _backend = batched_build(module, torch_device, tmp_path, largest=64, features=4096)

    assert torch.xpu.memory_allocated() - before < 3 * 2**20


@requires_xpu
def test_out_of_memory_keeps_smaller_graphs(model, torch_device, tmp_path, monkeypatch):
    record_size = XPUGraphBackend._record_size

    def out_of_memory_above_8(self, size, *rest):
        if size > 8:
            raise torch.OutOfMemoryError("fake")
        return record_size(self, size, *rest)

    monkeypatch.setattr(XPUGraphBackend, "_record_size", out_of_memory_above_8)

    backend = batched_build(model, torch_device, tmp_path, largest=64)

    assert backend.recorded_batch_sizes == [1, 2, 4, 8]


@requires_xpu
def test_output_without_batch_axis_is_rejected(torch_device, tmp_path):
    with pytest.raises(RuntimeError, match="no batch axis"):
        batched_build(Scaled().to(torch_device), torch_device, tmp_path, largest=4, features=5)


@requires_xpu
def test_rows_that_depend_on_other_rows_are_rejected(torch_device, tmp_path):
    with pytest.raises(RuntimeError, match="differs from eager"):
        batched_build(RowMixing().to(torch_device), torch_device, tmp_path, largest=4)


@requires_xpu
def test_rows_that_depend_on_the_first_row_are_rejected(torch_device, tmp_path):
    with pytest.raises(RuntimeError, match="differs from eager"):
        batched_build(DependsOnFirstRow().to(torch_device), torch_device, tmp_path, largest=4)


@requires_xpu
def test_first_row_dependence_is_rejected_when_one_row_is_recorded(torch_device, tmp_path):
    config = XPUGraphBackendConfig(max_recorded_batch_size=1)

    with pytest.raises(RuntimeError, match="differs from eager"):
        batched_build(DependsOnFirstRow().to(torch_device), torch_device, tmp_path, largest=4, config=config)


@requires_xpu
def test_batched_tensor_inside_an_object_is_rejected(torch_device, tmp_path):
    with pytest.raises(RuntimeError, match="not returned as tensors"):
        batched_build(HoldsOutput().to(torch_device).eval(), torch_device, tmp_path, largest=4)


@requires_xpu
def test_object_holding_a_tensor_is_copied(torch_device, tmp_path):
    module = HoldsOutput().to(torch_device).eval()
    first = rows(2, 16, device=torch_device)
    second = first.roll(1, 0)
    expected = eager(module, first).value
    sample = ((first,), {})
    backend = XPUGraphBackend().build(module, static_graph_spec(module, [sample]), [sample], torch_device, tmp_path)

    actual = backend.infer(first)
    again = backend.infer(second)

    torch.testing.assert_close(actual.value, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(again.value, eager(module, second).value, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_one_row_graph_replays_a_larger_batch_in_chunks(model, torch_device, tmp_path):
    x = rows(5, 16, device=torch_device)
    expected = eager(model, x)
    config = XPUGraphBackendConfig(max_recorded_batch_size=1)
    backend = batched_build(model, torch_device, tmp_path, largest=8, config=config)

    actual = backend.infer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_module_that_stacks_its_batch_is_rejected(torch_device, tmp_path):
    with pytest.raises(RuntimeError, match="differs from eager"):
        batched_build(Stacked().to(torch_device).eval(), torch_device, tmp_path, largest=8)


@requires_xpu
def test_input_with_two_rows_per_sample_matches_eager(model, torch_device, tmp_path):
    x = rows(10, 16, device=torch_device)
    expected = eager(model, x)
    small = ((rows(2, 16, device=torch_device),), {})
    large = ((rows(16, 16, device=torch_device),), {})
    spec = seen_graph_spec(model, small, large, 8)
    backend = XPUGraphBackend().build(model, spec, [small, large], torch_device, tmp_path)

    actual = backend.infer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_input_batched_on_axis_1_matches_eager(torch_device, tmp_path):
    module = BatchLast().to(torch_device).eval()
    x = rows(16, 5, device=torch_device)
    expected = eager(module, x)
    small = ((rows(16, 1, device=torch_device),), {})
    large = ((rows(16, 8, device=torch_device),), {})
    spec = seen_graph_spec(module, small, large, 8)
    backend = XPUGraphBackend().build(module, spec, [small, large], torch_device, tmp_path)

    actual = backend.infer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_dynamic_axis_is_rejected(model, torch_device, tmp_path):
    short = ((torch.zeros(1, 3, 16, device=torch_device),), {})
    long = ((torch.zeros(2, 7, 16, device=torch_device),), {})
    spec = seen_graph_spec(model, short, long, 2)

    with pytest.raises(RuntimeError, match="dynamic axis"):
        XPUGraphBackend().build(model, spec, [short, long], torch_device, tmp_path)


@requires_xpu
def test_batch_axis_checkpoint_round_trip(model, torch_device, tmp_path):
    x = rows(6, 16, device=torch_device)
    expected = eager(model, x)
    backend = batched_build(model, torch_device, tmp_path, largest=8)
    TorchSaveTask().save(tmp_path, backend.to_dict())

    loaded = XPUGraphBackend.from_dict(model, TorchLoadTask().load(tmp_path))
    loaded.deploy(device=torch_device)
    actual = loaded.infer(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_tune_with_two_batch_sizes_replays_a_third(model, torch_device, tmp_path):
    x = rows(5, 16, device=torch_device)
    expected = eager(model, x)
    strategy = HighestThroughputStrategy(backends=[XPUGraphBackend()]).enable_find_max_batch_size(False)
    wrapped = TunableModule(model, "xpu-graph-batched", strategy=strategy)
    tune(wrapped, rows(16, device=torch_device), batch_sizes=[1, 8], device=torch_device, ignore_failing_modules=False)

    with torch.no_grad():
        actual = wrapped(x)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@requires_xpu
def test_wrong_replay_is_rejected(torch_device, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.xpu.XPUGraph, "replay", lambda graph: None)
    module = LinearRelu().to(torch_device).eval()
    args = (torch.randn(2, 16, device=torch_device),)
    backend = XPUGraphBackend()
    with pytest.raises(RuntimeError, match="differs from eager"):
        backend.build(module, static_graph_spec(module, [(args, {})]), [(args, {})], torch_device, tmp_path)


def test_cpu_is_rejected(tmp_path):
    module = LinearRelu()
    args = (torch.randn(2, 16),)
    backend = XPUGraphBackend()
    with pytest.raises(ValueError, match="not supported"):
        backend.build(
            module,
            GraphSpec(
                name="cpu",
                input_spec=SampleMetadata.from_inputs(args, {}),
                output_spec=SampleMetadata.from_outputs(args[0]),
            ),
            [(args, {})],
            torch.device("cpu"),
            tmp_path,
        )


def test_default_backend_has_no_recording_switch():
    assert XPUGraphBackend().describe() == "XPUGraphBackend()"


@requires_xpu
def test_serialization(model, sample_data, tmp_path):
    backend = backend_build(XPUGraphBackend(), torch.float16, model, sample_data, tmp_path)
    TorchSaveTask().save(tmp_path, backend.to_dict())
    loaded = XPUGraphBackend.from_dict(model, TorchLoadTask().load(tmp_path))
    loaded.activate()

    sample_data = move_to_dtype(sample_data, torch.float16)
    args, kwargs = sample_data[0]
    new_args = (torch.randn_like(args[0]),)
    torch.testing.assert_close(backend.infer(*new_args, **kwargs), loaded.infer(*new_args, **kwargs))


@requires_xpu
def test_tune_module(torch_device, tmp_path, eager_calls):
    torch.manual_seed(0)
    module = LinearRelu().to(torch_device).eval()
    # tune routes later calls of module through the selected backend.
    reference = copy.deepcopy(module)
    sample = torch.randn(16, device=torch_device)
    wrapped = TunableModule(
        module,
        "xpu-graph-mlp",
        strategy=OneBackendStrategy(XPUGraphBackend()).enable_find_max_batch_size(False),
    )
    try:
        tune(wrapped, sample, batch_sizes=[2], device=torch_device, ignore_failing_modules=False)
        larger = torch.randn(4, 16, device=torch_device)
        batch = torch.randn(2, 16, device=torch_device)
        with torch.no_grad():
            torch.testing.assert_close(wrapped(larger), reference(larger))
            torch.testing.assert_close(wrapped(batch), reference(batch))
        assert [call[0].shape for call in eager_calls] == [larger.shape]
    finally:
        MODULE_REGISTRY.clear()
