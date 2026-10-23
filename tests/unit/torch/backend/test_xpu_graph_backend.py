# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for XPUGraphBackend."""

import copy
import inspect
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
from torch_tweak.torch.module.wrapper_module import Module
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy import FirstWinsStrategy, HighestThroughputStrategy, OneBackendStrategy
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


def seen_graph_spec(module, samples: list[tuple[int, Sample]]) -> GraphSpec:
    """Builds the graph spec tuning records after it sees every (batch size, sample) pair."""
    spec = None
    for batch_size, (args, kwargs) in samples:
        with torch.no_grad():
            outputs = module(*args, **kwargs)
        inputs = SampleMetadata.from_inputs(args, kwargs, batch_size=batch_size)
        outputs = SampleMetadata.from_outputs(outputs, batch_size=batch_size)
        if spec is None:
            spec = GraphSpec(name="seen", input_spec=inputs, output_spec=outputs)
        else:
            spec.update_shapes_seen(inputs, outputs)
    return spec


def batched_build(module, torch_device, tmp_path, largest=64, config=None, features=16):
    """Builds a backend on a graph spec that saw batch sizes 1 and largest."""
    module = module.to(torch_device).eval()
    small = ((torch.randn(1, features, device=torch_device),), {})
    large = ((torch.randn(largest, features, device=torch_device),), {})
    spec = seen_graph_spec(module, [(1, small), (largest, large)])
    return XPUGraphBackend(config).build(module, spec, [small, large], torch_device, tmp_path)


class RowMixing(nn.Module):
    def forward(self, x):
        return x - x.mean(dim=0, keepdim=True)


class Stacked(nn.Module):
    """Takes a batch stacked twice along axis 0, like classifier-free guidance."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        y = self.linear(x)
        return torch.cat([y, y], dim=0)


@requires_xpu
def test_batch_axis_records_the_size_grid(torch_device, tmp_path):
    module = LinearRelu()
    backend = batched_build(module, torch_device, tmp_path)
    assert backend.recorded_batch_sizes == [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    with torch.no_grad():
        for batch_size in (1, 3, 5, 8, 17, 64, 100):
            x = torch.randn(batch_size, 16, device=torch_device)
            torch.testing.assert_close(backend.infer(x), module(x))


@requires_xpu
def test_grid_stops_at_the_first_size_that_holds_the_largest_batch(torch_device, tmp_path):
    assert batched_build(LinearRelu(), torch_device, tmp_path, largest=20).recorded_batch_sizes == [1, 2, 4, 8, 16, 24]
    config = XPUGraphBackendConfig(max_recorded_batch_size=12)
    backend = batched_build(LinearRelu(), torch_device, tmp_path, config=config)
    assert backend.recorded_batch_sizes == [1, 2, 4, 8, 12]


@requires_xpu
def test_graphs_share_one_input_buffer(torch_device, tmp_path):
    backend = batched_build(LinearRelu(), torch_device, tmp_path)
    pointers = {recording.static_leaves[0].data_ptr() for recording in backend._recordings.values()}
    assert len(pointers) == 1


@requires_xpu
def test_padding_does_not_leak_rows_of_a_larger_graph(torch_device, tmp_path):
    module = LinearRelu()
    backend = batched_build(module, torch_device, tmp_path)
    with torch.no_grad():
        backend.infer(torch.full((64, 16), 1e4, device=torch_device))
        x = torch.randn(5, 16, device=torch_device)
        torch.testing.assert_close(backend.infer(x), module(x))


@requires_xpu
def test_max_recorded_batch_size_caps_recording_and_chunks(torch_device, tmp_path):
    module = LinearRelu()
    config = XPUGraphBackendConfig(max_recorded_batch_size=4)
    backend = batched_build(module, torch_device, tmp_path, config=config)
    assert backend.recorded_batch_sizes == [1, 2, 4]
    assert backend.describe() == "XPUGraphBackend(max_recorded_batch_size=4)"
    with torch.no_grad():
        x = torch.randn(11, 16, device=torch_device)
        torch.testing.assert_close(backend.infer(x), module(x))


@requires_xpu
def test_out_of_memory_keeps_smaller_graphs(torch_device, tmp_path, monkeypatch):
    record_size = XPUGraphBackend._record_size

    def failing(self, size, *rest):
        if size > 8:
            raise torch.OutOfMemoryError("fake")
        return record_size(self, size, *rest)

    monkeypatch.setattr(XPUGraphBackend, "_record_size", failing)
    module = LinearRelu()
    backend = batched_build(module, torch_device, tmp_path)
    assert backend.recorded_batch_sizes == [1, 2, 4, 8]
    with torch.no_grad():
        x = torch.randn(30, 16, device=torch_device)
        torch.testing.assert_close(backend.infer(x), module(x))


@requires_xpu
def test_output_without_batch_axis_is_rejected(torch_device, tmp_path):
    with pytest.raises(RuntimeError, match="no batch axis"):
        batched_build(Scaled(), torch_device, tmp_path, largest=4, features=5)


@requires_xpu
def test_rows_that_depend_on_other_rows_are_rejected(torch_device, tmp_path):
    with pytest.raises(RuntimeError, match="differs from eager"):
        batched_build(RowMixing(), torch_device, tmp_path, largest=4)


@requires_xpu
def test_module_that_stacks_its_batch_is_rejected(torch_device, tmp_path):
    with pytest.raises(RuntimeError, match="differs from eager"):
        batched_build(Stacked(), torch_device, tmp_path, largest=8)


@requires_xpu
def test_input_with_two_rows_per_sample(torch_device, tmp_path):
    module = LinearRelu().to(torch_device).eval()
    small = ((torch.randn(2, 16, device=torch_device),), {})
    large = ((torch.randn(16, 16, device=torch_device),), {})
    spec = seen_graph_spec(module, [(1, small), (8, large)])
    backend = XPUGraphBackend().build(module, spec, [small, large], torch_device, tmp_path)
    assert backend.recorded_batch_sizes == [1, 2, 4, 8]
    with torch.no_grad():
        x = torch.randn(10, 16, device=torch_device)
        torch.testing.assert_close(backend.infer(x), module(x))


@requires_xpu
def test_dynamic_axis_is_rejected(torch_device, tmp_path):
    module = LinearRelu().to(torch_device).eval()
    short = ((torch.randn(1, 3, 16, device=torch_device),), {})
    long = ((torch.randn(2, 7, 16, device=torch_device),), {})
    spec = seen_graph_spec(module, [(1, short), (2, long)])
    with pytest.raises(RuntimeError, match="dynamic axis"):
        XPUGraphBackend().build(module, spec, [short, long], torch_device, tmp_path)


@requires_xpu
def test_compile_with_inductor_records_the_compiled_kernels(torch_device, tmp_path):
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    module = LinearRelu()
    config = XPUGraphBackendConfig(compile_with_inductor=True)
    backend = batched_build(module, torch_device, tmp_path, largest=16, config=config)

    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0
    assert backend.recorded_batch_sizes == [1, 2, 4, 8, 16]
    assert backend.describe() == "XPUGraphBackend(compile_with_inductor=True)"
    # Replay runs the recorded kernels without calling the compiled module.
    backend._recorded_module = None
    with torch.no_grad():
        for batch_size in (1, 3, 16, 20):
            x = torch.randn(batch_size, 16, device=torch_device)
            torch.testing.assert_close(backend.infer(x), module(x), rtol=1e-4, atol=1e-4)


@requires_xpu
def test_batch_axis_serialization(torch_device, tmp_path):
    module = LinearRelu()
    backend = batched_build(module, torch_device, tmp_path, largest=8)
    TorchSaveTask().save(tmp_path, backend.to_dict())
    loaded = XPUGraphBackend.from_dict(module, TorchLoadTask().load(tmp_path))
    loaded.deploy(device=torch_device)
    assert loaded.recorded_batch_sizes == [1, 2, 4, 8]
    with torch.no_grad():
        x = torch.randn(6, 16, device=torch_device)
        torch.testing.assert_close(loaded.infer(x), module(x))


@requires_xpu
def test_tune_with_two_batch_sizes_records_the_batch_axis(torch_device, tmp_path):
    module = LinearRelu().to(torch_device).eval()
    reference = copy.deepcopy(module)
    wrapped = Module(
        module,
        "xpu-graph-batched",
        strategy=HighestThroughputStrategy(backends=[XPUGraphBackend()]).enable_find_max_batch_size(False),
    )
    try:
        tune(
            wrapped,
            torch.randn(16, device=torch_device),
            batch_sizes=[1, 8],
            device=torch_device,
            ignore_failing_modules=False,
        )
        (backend,) = wrapped._self_wrapper.backends.values()
        assert isinstance(backend, XPUGraphBackend)
        assert backend.recorded_batch_sizes == [1, 2, 4, 8]
        x = torch.randn(5, 16, device=torch_device)
        with torch.no_grad():
            torch.testing.assert_close(wrapped(x), reference(x))
    finally:
        MODULE_REGISTRY.clear()


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


def test_first_wins_tries_the_graph_backend_before_inductor():
    text = FirstWinsStrategy().describe()
    assert text.index("XPUGraphBackend()") < text.index("TorchInductorBackend()")


def test_highest_throughput_lists_the_graph_backend():
    assert "XPUGraphBackend()" in HighestThroughputStrategy().describe()


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
    wrapped = Module(
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
