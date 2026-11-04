# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file has been modified by Intel Corporation.
import time

import pytest

from tests.toy_models.torch_models import ToyTorchModel
from torch_tweak.torch import TunableModule
from torch_tweak.torch.backend import TorchEagerBackend
from torch_tweak.torch.task.profiling.config import ProfilingConfig
from torch_tweak.torch.task.profiling.events import ProfilingResultEvent
from torch_tweak.torch.task.profiling.measuring_stop_strategy import (
    NumStepsMeasuringStopStrategy,
    StableWindowMeasuringStopStrategy,
)
from torch_tweak.torch.task.profiling.measuring_strategy import (
    MeasuringStrategy,
    ModelExecutionTimeMeasuringStrategy,
)
from torch_tweak.torch.task.profiling.metrics import get_throughput, is_throughput_saturated
from torch_tweak.torch.task.profiling.profiling import ProfilingResult, profile, profile_backend
from torch_tweak.torch.task.profiling.profiling_stop_strategy import ThroughputSaturatedProfilingStopStrategy


def new_event(
    batch_size: int,
    execution_time: float,
    phase: str = "inference",
    backend_details: str = "torch",
    measurement_id: int = 0,
) -> ProfilingResultEvent:
    return ProfilingResultEvent(
        measurement_id=measurement_id,
        timestamp=time.monotonic_ns(),
        batch_size=batch_size,
        execution_time=execution_time,
        phase=phase,
        backend_details=backend_details,
        model_name="test_model:module_name",
    )


def test_get_throughput_batch_size_1():
    profiling_result = [
        new_event(1, 100e6),
        new_event(1, 100e6),
    ]
    assert int(get_throughput(profiling_result)) == 10


def test_get_throughput_batch_size_20():
    profiling_result = [
        new_event(20, 201e6),
        new_event(20, 201e6),
    ]
    assert int(get_throughput(profiling_result)) == 99


def test_first_throughput_is_not_saturated():
    assert is_throughput_saturated(10.0, 0.01, None) is False


def test_smaller_throughput_is_saturated():
    assert is_throughput_saturated(9.95, 0.01, 10.0) is True


def test_throughput_saturated_threshold_strategy():
    strategy = ThroughputSaturatedProfilingStopStrategy(throughput_cutoff_threshold=0.01)

    assert strategy.update_and_should_stop(10.0) is False
    assert strategy.update_and_should_stop(9.95) is True


def test_backoff_stops_on_the_third_saturated_batch():
    strategy = ThroughputSaturatedProfilingStopStrategy(throughput_cutoff_threshold=0.05, throughput_backoff_limit=2)

    assert strategy.update_and_should_stop(100) is False  # no best yet, 100 is stored
    assert strategy.update_and_should_stop(200) is False  # 200 is a new best
    assert strategy.update_and_should_stop(200) is False  # saturated, backoff count becomes 1
    assert strategy.update_and_should_stop(200) is False  # saturated, backoff count becomes 2
    assert strategy.update_and_should_stop(200) is True  # count is already 2, so profiling stops


def test_a_better_batch_resets_the_backoff():
    strategy = ThroughputSaturatedProfilingStopStrategy(throughput_cutoff_threshold=0.05, throughput_backoff_limit=1)

    assert strategy.update_and_should_stop(100) is False  # no best yet, 100 is stored
    assert strategy.update_and_should_stop(200) is False  # 200 is a new best
    assert strategy.update_and_should_stop(200) is False  # saturated, backoff count becomes 1
    assert strategy.update_and_should_stop(400) is False  # 400 is a new best, count returns to 0
    assert strategy.update_and_should_stop(400) is False  # saturated, backoff count becomes 1
    assert strategy.update_and_should_stop(400) is True  # count is already 1, so profiling stops


def test_measurement_stop_strategy_num_steps():
    strategy = NumStepsMeasuringStopStrategy(num_steps=10)
    assert not strategy.update_and_should_stop([new_event(1, 100e6)])
    for i in range(8):
        assert not strategy.update_and_should_stop([new_event(1, 100e6, measurement_id=i)])
    assert strategy._steps_counter == 9
    assert strategy.update_and_should_stop([new_event(1, 100e6)])
    assert strategy.update_and_should_stop([new_event(1, 100e6)])


def test_measurement_stop_strategy_stable_window():
    strategy = StableWindowMeasuringStopStrategy(window_size=10, stability_percentage=90)
    assert not strategy.update_and_should_stop([new_event(1, 100e6)])
    for i in range(8):
        assert not strategy.update_and_should_stop([new_event(1, 100e6, measurement_id=i)])
    assert len(strategy._window) == 9
    assert strategy.update_and_should_stop([new_event(1, 100e6)])
    assert strategy.update_and_should_stop([new_event(1, 100e6)])


def test_measurement_stop_strategy_stable_window_unstable():
    strategy = StableWindowMeasuringStopStrategy(window_size=10, stability_percentage=90)
    assert not strategy.update_and_should_stop([new_event(1, 0)])

    # unstable measurement
    for i in range(8):
        assert not strategy.update_and_should_stop([new_event(1, i, measurement_id=i)])

    # stable measurement
    for i in range(8, 17):
        assert not strategy.update_and_should_stop([new_event(1, 1, measurement_id=i)])

    assert strategy.update_and_should_stop([new_event(1, 1, measurement_id=17)])


@pytest.mark.parametrize(
    "window_size,stability_percentage",
    [
        (10, 90),
        (100, 99),
        (1000, 99.9),
    ],
)
def test_measurement_stop_strategy_stable_window_stability(window_size: int, stability_percentage: float):
    strategy = StableWindowMeasuringStopStrategy(window_size=window_size, stability_percentage=stability_percentage)

    assert not strategy.update_and_should_stop([new_event(1, 300e6, measurement_id=0)])

    for i in range(1, window_size):
        assert not strategy.update_and_should_stop([new_event(1, 100e6, measurement_id=i)])

    assert strategy.update_and_should_stop([new_event(1, 100e6, measurement_id=window_size)])


def test_profile_toy_model():
    profile_config = ProfilingConfig(batch_sizes=[2**n for n in range(1, 5)])

    model = TunableModule(ToyTorchModel())
    dataset = model.samples()

    result = profile(model, dataset, profile_config)
    assert result.status == ProfilingResult.Status.SUCCESS
    assert 2 in result.throughput_per_batch_size


def test_profile_toy_model_no_batching():
    profile_config = ProfilingConfig(batch_sizes=[2**n for n in range(1, 5)], batching=False)

    model = TunableModule(ToyTorchModel())
    dataset = model.samples()

    result = profile(model, dataset, profile_config)
    assert result.status == ProfilingResult.Status.SUCCESS
    assert 2 in result.throughput_per_batch_size


@pytest.mark.parametrize(("batching", "batch_sizes"), [(True, {1, 2}), (False, {1})])
def test_profile_backend(batching: bool, batch_sizes: set[int]):
    model = ToyTorchModel()
    graph_spec = model.graph_spec(batch_sizes=[1])

    class MockBackend(TorchEagerBackend):
        def __init__(self, model):
            self.model = model

        def infer(self, *args, **kwargs):
            """The following function imitates adding something to cache each time it is called.

            If the profile is idempotent, the cache argument should always be empty i.e. any changes should be discarded.
            """
            assert len(kwargs["cache"]) == 0, "Cache should be empty"
            kwargs["cache"].append("not important, should be discarded")
            return self.model(*args)

        def describe(self):
            return "mock_backend"

    backend = MockBackend(model)
    profile_config = ProfilingConfig(batch_sizes=[1, 2], batching=batching)

    samples = [(model.inputs(batch_sizes=[1]), {"cache": []})]
    result = profile_backend(backend, "test_model", graph_spec, samples, profile_config)  # type: ignore

    assert result.status == ProfilingResult.Status.SUCCESS
    assert set(result.throughput_per_batch_size) == batch_sizes


def test_measurement_waits_for_the_device(mocker):
    order = []
    mocker.patch(
        "torch_tweak.torch.task.profiling.measuring_strategy.synchronize",
        side_effect=lambda: order.append("sync"),
    )

    def model(*_args, **_kwargs):
        order.append("model")

    events = ModelExecutionTimeMeasuringStrategy().do_measurement(1, model, ([], {}))

    assert order == ["sync", "model", "sync"]
    assert events[0].execution_time >= 0


def test_compilation_of_a_new_batch_size_does_not_stop_profiling():
    steady_ns = 5_000_000
    compilation_ns = 100_000_000
    times_ns = {
        1: [steady_ns, steady_ns, steady_ns],
        2: [compilation_ns, steady_ns, steady_ns, steady_ns],
        4: [compilation_ns, steady_ns, steady_ns, steady_ns],
        8: [compilation_ns, steady_ns, steady_ns, steady_ns],
    }

    class ScriptedExecutionTime(MeasuringStrategy):
        def do_measurement(self, batch_size, _model, _sample, **_kwargs):
            return [new_event(batch_size, times_ns[batch_size].pop(0))]

    model = ToyTorchModel()
    profile_config = ProfilingConfig(
        batch_sizes=[1, 2, 4, 8],
        measuring_strategy=ScriptedExecutionTime(),
        measurement_stop_strategy=StableWindowMeasuringStopStrategy(window_size=3, stability_percentage=80),
        profiling_stop_strategy=ThroughputSaturatedProfilingStopStrategy(
            throughput_cutoff_threshold=0.05,
            throughput_backoff_limit=0,
        ),
    )

    result = profile(model, model.samples(), profile_config)

    assert result.throughput_per_batch_size == {1: 200.0, 2: 400.0, 4: 800.0, 8: 1600.0}
