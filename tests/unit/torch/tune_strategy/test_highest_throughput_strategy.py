# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

import time
from typing import cast
from unittest.mock import MagicMock

import pytest
import torch

from tests.toy_backends import AllocateThenFailBackend, BuildFailsBackend, MemoryProbeBackend, SleepBackend
from tests.toy_models.torch_models import INPUT_SIZE, ToyTorchModel, toy_on_xpu
from tests.utilities.device_probes import ScriptedProbe
from tests.utilities.helpers import requires_xpu
from torch_tweak.torch import TunableModule
from torch_tweak.torch.backend import Backend
from torch_tweak.torch.backend.torch_eager import TorchEagerBackend
from torch_tweak.torch.backend.torch_inductor_backend import TorchInductorBackend
from torch_tweak.torch.module.tunable_module import ModuleState
from torch_tweak.torch.task.correctness import CorrectnessValueError
from torch_tweak.torch.task.profiling import NumStepsMeasuringStopStrategy, StableWindowMeasuringStopStrategy
from torch_tweak.torch.task.profiling.profiling_stop_strategy import (
    AllSamplesProfilingStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from torch_tweak.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from torch_tweak.torch.tune_strategy.highest_throughput_strategy import HighestThroughputStrategy
from torch_tweak.torch.tuning import tune


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock(spec=Backend)
    backend.name = "mock_backend"
    backend.describe.return_value = "mock_backend"
    return backend


def test_describe(mock_backend):
    """Test describe method."""
    backends = [mock_backend, mock_backend]
    strategy = HighestThroughputStrategy(backends)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    assert (
        strategy.describe()
        == "name: Highest Throughput Strategy\ndescription: evaluate all backends, return backend with highest throughput\nbackends:\n  mock_backend\n  mock_backend"
    )


def test_sanity_batch_sizes_generator():
    max_batch_size = 2**20
    batch_sizes = [2**n for n in range(max_batch_size.bit_length() + 1)]
    assert 2**20 in batch_sizes


def test_highest_throughput_strategy_tune_highest_throughput_backend(torch_device, tmp_path):
    profiling_config = TuneStrategyFindMaxBatchSizeExtension.default_profiling_config(max_batch_size=8)
    slower = SleepBackend(sleep_time=1e-2)
    faster = SleepBackend(sleep_time=1e-5)
    strategy = HighestThroughputStrategy(backends=[slower, faster])
    strategy.set_find_max_batch_size_profiling_config(profiling_config)
    strategy.enable_correctness_check(False)

    model = ToyTorchModel()
    sample = model.sample().unsqueeze(0)  # as dataloader make batches, we need to unsqueeze the sample

    def forward_slow(x):
        time.sleep(1e-3)
        return x

    model.forward = forward_slow

    highest_throughput_backend = strategy.tune(
        model, "test", model.graph_spec(batch_sizes=[1, 2]), [((sample,), {})], torch_device, tmp_path
    )
    highest_throughput_backend = cast(SleepBackend, highest_throughput_backend)

    assert highest_throughput_backend.sleep_time == faster.sleep_time


def test_without_backends_tunes_with_the_default_backends_for_the_device(torch_device, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "torch_tweak.torch.tune_strategy.highest_throughput_strategy.default_backends",
        lambda device: {torch_device: [SleepBackend(sleep_time=2e-5)]}[device],
    )
    strategy = HighestThroughputStrategy()
    strategy.set_find_max_batch_size_profiling_config(
        TuneStrategyFindMaxBatchSizeExtension.default_profiling_config(max_batch_size=8)
    )
    strategy.enable_correctness_check(False)
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().unsqueeze(0).to(torch_device)
    graph_spec = model.graph_spec(batch_sizes=[1, 2], device=torch_device)

    backend = strategy.tune(model, "test", graph_spec, [((sample,), {})], torch_device, tmp_path)

    assert cast(SleepBackend, backend).sleep_time == 2e-5


def test_highest_throughput_strategy_max_batch_size_in_graph_spec(torch_device, tmp_path):
    profiling_config = TuneStrategyFindMaxBatchSizeExtension.default_profiling_config(max_batch_size=8)
    strategy = HighestThroughputStrategy(backends=[SleepBackend(sleep_time=1e-5)])
    strategy.set_find_max_batch_size_profiling_config(profiling_config)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().unsqueeze(0).to(torch_device)  # as dataloader make batches, we need to unsqueeze the sample
    graph_spec = model.graph_spec(batch_sizes=[1, 2, 4, 8], device=torch_device)

    # sanity heck that graph spec input_spec was updated with max batch size
    max_batch_sample = graph_spec.input_spec.make_batch(args=(sample,), kwargs={}, batch_size=8)
    assert max_batch_sample[0][0].shape[0] == 8

    # Run tuning to update graph spec with max batch size
    strategy.tune(model, "test", graph_spec, [((sample,), {})], torch_device, cache_dir=tmp_path)

    # Verify tensor specs were updated with max batch size info
    for tensor_spec in graph_spec.input_spec.tensor_specs:
        assert tensor_spec.max_shape[0] == 8


@requires_xpu
def test_highest_throughput_strategy_num_steps_all_samples(torch_device):
    find_profiling_config = TuneStrategyFindMaxBatchSizeExtension.default_profiling_config(max_batch_size=16)
    find_profiling_config.measurement_stop_strategy = NumStepsMeasuringStopStrategy(num_steps=10)

    strategy = HighestThroughputStrategy(
        backends=[
            TorchInductorBackend(),
            TorchEagerBackend(),
        ],
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=10),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    )
    strategy.set_find_max_batch_size_profiling_config(find_profiling_config)
    strategy.enable_correctness_check(True)

    model = TunableModule(ToyTorchModel().eval().to(torch_device), strategy=strategy)
    sample = model.sample().to(torch_device)
    batch_sizes = [1, 2, 4, 8, 16]
    n_backends = len(strategy._backends)

    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert len(strategy.results) == 1  # for 1 graph spec

    assert strategy.results[0].graph_spec_name == "0"

    backend_results = strategy.results[0].highest_throughput_results
    assert len(backend_results) == n_backends
    assert set(backend_results[0].throughput_per_batch_size) == set(batch_sizes)
    assert set(backend_results[1].throughput_per_batch_size) == set(batch_sizes)

    # check graph spec
    graph_specs = list(model._self_wrapper._backends.keys())
    assert len(graph_specs) == 1
    assert graph_specs[0].tensor_specs[0].max_shape[0] == 16


@requires_xpu
def test_highest_throughput_strategy_stable_window(torch_device):
    strategy = HighestThroughputStrategy(
        backends=[
            TorchInductorBackend(),
        ],
        measurement_stop_strategy=StableWindowMeasuringStopStrategy(window_size=10, stability_percentage=90),
        profiling_stop_strategy=ThroughputSaturatedProfilingStopStrategy(throughput_cutoff_threshold=0.99),
    ).enable_find_max_batch_size(False)
    model = TunableModule(ToyTorchModel().eval().to(torch_device), strategy=strategy)
    sample = model.sample().to(torch_device)

    tune(model, sample, batch_sizes=list(range(1, 17)), device=torch_device, disable_external_logging=False)

    assert len(strategy.results) == 1  # for 1 graph spec

    assert strategy.results[0].graph_spec_name == "0"

    assert len(strategy.results[0].highest_throughput_results) == 1
    assert len(strategy.results[0].highest_throughput_results[0].throughput_per_batch_size) >= 1


def _tune_with_probe(backends, probe, torch_device, tmp_path):
    strategy = HighestThroughputStrategy(backends=backends, device_probe=probe)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel()
    sample = torch.zeros(1, INPUT_SIZE)
    strategy.tune(model, "test", model.graph_spec(batch_sizes=[1]), [((sample,), {})], torch_device, tmp_path)
    return strategy


def test_a_measurement_on_a_quiet_device_runs_once(torch_device, tmp_path):
    probe = ScriptedProbe([1.0, 1.0])

    _tune_with_probe([SleepBackend()], probe, torch_device, tmp_path)

    assert probe.calls == 2


def test_a_measurement_on_a_shared_device_runs_again(torch_device, tmp_path):
    probe = ScriptedProbe([1.0, 20.0, 1.0, 1.0])

    strategy = _tune_with_probe([SleepBackend()], probe, torch_device, tmp_path)

    assert (probe.calls, strategy.tuned_on_quiet_device) == (4, True)


def test_a_device_shared_in_every_attempt_is_reported(torch_device, tmp_path):
    probe = ScriptedProbe([1.0, 20.0, 1.0, 20.0, 1.0, 20.0])

    strategy = _tune_with_probe([SleepBackend()], probe, torch_device, tmp_path)

    assert strategy.tuned_on_quiet_device is False


def test_a_device_that_gets_quieter_after_a_measurement_is_reported(torch_device, tmp_path):
    probe = ScriptedProbe([20.0, 20.0, 1.0, 1.0])

    strategy = _tune_with_probe([SleepBackend(), SleepBackend()], probe, torch_device, tmp_path)

    assert strategy.tuned_on_quiet_device is False


class ActivateFailsBackend(SleepBackend):
    def __init__(self):
        super().__init__()

    def _activate(self):
        raise RuntimeError("Activate failed")


class ReactivateFailsBackend(SleepBackend):
    """Builds without activating and fails every activate after the build."""

    def _build(self, module, graph_spec, data, cache_dir):
        self._module = module
        return self

    def _activate(self):
        raise RuntimeError("Reactivate failed")


def test_next_fastest_backend_is_selected_when_the_fastest_fails_to_reactivate(torch_device, tmp_path):
    strategy = HighestThroughputStrategy(
        backends=[
            ReactivateFailsBackend(sleep_time=1e-5),
            SleepBackend(sleep_time=1e-2),
            SleepBackend(sleep_time=1e-3),
        ]
    )
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel()
    sample = torch.zeros(1, INPUT_SIZE)

    selected = strategy.tune(
        model, "test", model.graph_spec(batch_sizes=[1]), [((sample,), {})], torch_device, tmp_path
    )

    assert selected.name == "SleepBackend with sleep time 0.001"


def test_tune_fails_with_the_last_error_when_no_backend_reactivates(torch_device, tmp_path):
    strategy = HighestThroughputStrategy(backends=[ReactivateFailsBackend(sleep_time=1e-5)])
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel()
    sample = torch.zeros(1, INPUT_SIZE)

    with pytest.raises(RuntimeError, match="No profiled backend could be activated") as raised:
        strategy.tune(model, "test", model.graph_spec(batch_sizes=[1]), [((sample,), {})], torch_device, tmp_path)

    assert str(raised.value.__cause__) == "Reactivate failed"


class InferFailsBackend(SleepBackend):
    def _infer(self, *args, **kwargs):
        raise RuntimeError("Infer failed")


def test_failure_after_the_build_is_written_to_the_backend_log(torch_device, tmp_path):
    backend = InferFailsBackend()
    strategy = HighestThroughputStrategy(backends=[backend, SleepBackend(sleep_time=1e-2)])
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel().to(torch_device)
    sample = model.sample().unsqueeze(0).to(torch_device)
    graph_spec = model.graph_spec(batch_sizes=[1], device=torch_device)

    strategy.tune(model, "test", graph_spec, [((sample,), {})], torch_device, tmp_path)

    assert "RuntimeError: Infer failed" in (tmp_path / backend.key() / "build.log").read_text()


@requires_xpu
def test_next_backend_builds_without_the_memory_a_failed_backend_held(tmp_path):
    model, data, graph_spec = toy_on_xpu()
    failed_bytes = 2**30
    reserved_at_build = []
    strategy = HighestThroughputStrategy(
        backends=[AllocateThenFailBackend(failed_bytes), MemoryProbeBackend(on_build=reserved_at_build.append)]
    )
    strategy.enable_find_max_batch_size(False)
    torch.xpu.empty_cache()
    reserved_before = torch.xpu.memory_reserved()

    strategy.tune(model, "test", graph_spec, data, torch.device("xpu"), tmp_path)

    assert reserved_at_build[0] < reserved_before + failed_bytes


def test_highest_throughput_strategy_fails_backend_if_all_of_backends_fails(torch_device):
    """If all backends fails it should raise an error."""
    strategy = HighestThroughputStrategy(
        backends=[
            BuildFailsBackend(RuntimeError),
            BuildFailsBackend(MemoryError),
            BuildFailsBackend(CorrectnessValueError),
            ActivateFailsBackend(),
        ],
    )
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    batch_sizes = list(range(1, 17))

    model = TunableModule(model, strategy=strategy)
    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert model.state == ModuleState.PASSTHROUGH


def test_highest_throughput_strategy_select_backend_if_one_of_backends_succeeds(torch_device):
    """If backend fails it should be skipped and TorchEagerBackend should be used as a fallback."""
    strategy = HighestThroughputStrategy(
        backends=[
            BuildFailsBackend(RuntimeError),
            BuildFailsBackend(MemoryError),
            BuildFailsBackend(CorrectnessValueError),
            ActivateFailsBackend(),
            TorchEagerBackend(),
        ],
    )
    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    batch_sizes = list(range(1, 17))

    model = TunableModule(model, strategy=strategy)
    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)

    assert model(sample.unsqueeze(0)) is not None  # Note: does not work without unsqueeze


def test_highest_throughput_strategy_find_max_batch_size_fails(torch_device):
    """If find max batch size fails, there is no recovery."""
    strategy = HighestThroughputStrategy(
        backends=[
            SleepBackend(),
        ],
    )

    # failing backend
    strategy.find_config.default_backend_class = lambda: BuildFailsBackend(RuntimeError)
    strategy.enable_find_max_batch_size(True)

    model = ToyTorchModel().eval().to(torch_device)
    sample = model.sample().to(torch_device)
    assert model(sample) is not None

    batch_sizes = list(range(1, 17))

    model = TunableModule(model, strategy=strategy)

    tune(model, sample, batch_sizes=batch_sizes, device=torch_device, disable_external_logging=False)
    assert model.state == ModuleState.PASSTHROUGH
