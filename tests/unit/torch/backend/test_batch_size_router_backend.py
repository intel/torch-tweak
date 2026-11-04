# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.toy_models.torch_models import ToyTorchModel
from torch_tweak.torch.backend import Backend, BatchSizeRouterBackend
from torch_tweak.torch.backend.backend import BackendState
from torch_tweak.torch.backend.batch_size_router_backend import routes_by_throughput
from torch_tweak.torch.checkpoint.storage_tasks import TorchLoadTask, TorchSaveTask
from torch_tweak.torch.task.profiling import NumStepsMeasuringStopStrategy
from torch_tweak.torch.task.profiling.profiling_stop_strategy import AllSamplesProfilingStopStrategy
from torch_tweak.torch.tune_strategy.highest_throughput_strategy import HighestThroughputStrategy


class ConstantBackend(Backend):
    """Stub. Returns one row of value per input row, so the output names the backend that ran.

    It sleeps per_call plus per_row times the batch size, so profiling sees a cost model.
    """

    def __init__(self, value: float = 0.0, per_call: float = 0.0, per_row: float = 0.0):
        super().__init__()
        self.value, self.per_call, self.per_row = value, per_call, per_row

    def key(self) -> str:
        return f"ConstantBackend_{self.value}"

    def describe(self) -> str:
        return f"ConstantBackend({self.value})"

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        return self

    def _activate(self):
        pass

    def _deactivate(self):
        pass

    def _deploy(self):
        pass

    def _infer(self, x):
        time.sleep(self.per_call + self.per_row * x.shape[0])
        return torch.full((x.shape[0], 1), self.value, device=x.device)

    def to_dict(self):
        return {"type": "ConstantBackend", "value": self.value}

    @classmethod
    def from_dict(cls, module, state_dict: dict):
        backend = cls(state_dict["value"])
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend


def built(backend: ConstantBackend) -> ConstantBackend:
    backend.build(None, None, [((torch.zeros(1, 4),), {})], torch.device("cpu"), Path())
    return backend


@pytest.fixture
def input_spec():
    return ToyTorchModel().graph_spec(batch_sizes=[1, 16]).input_spec


def router(input_spec) -> BatchSizeRouterBackend:
    """Batch sizes 1 to 4 return 1.0. Larger batch sizes return 2.0."""
    return BatchSizeRouterBackend([(4, built(ConstantBackend(1.0))), (None, built(ConstantBackend(2.0)))], input_spec)


@pytest.mark.parametrize(("batch_size", "value"), [(1, 1.0), (4, 1.0), (5, 2.0), (64, 2.0)])
def test_each_batch_size_runs_on_its_route(input_spec, batch_size, value):
    backend = router(input_spec)

    actual = backend.infer(torch.zeros(batch_size, 4))

    torch.testing.assert_close(actual, torch.full((batch_size, 1), value))


def test_describe_lists_the_batch_sizes_of_every_route(input_spec):
    assert router(input_spec).describe() == (
        "BatchSizeRouterBackend(1-4: ConstantBackend(1.0), 5+: ConstantBackend(2.0))"
    )


def test_last_route_must_take_every_larger_batch_size(input_spec):
    with pytest.raises(ValueError, match="every larger batch size"):
        BatchSizeRouterBackend([(4, built(ConstantBackend()))], input_spec)


def test_checkpoint_round_trip_keeps_the_routes(input_spec, tmp_path):
    TorchSaveTask().save(tmp_path, router(input_spec).to_dict())

    loaded = BatchSizeRouterBackend.from_dict(None, TorchLoadTask().load(tmp_path))
    loaded.deploy(torch.device("cpu"))
    actual = loaded.infer(torch.zeros(5, 4))

    torch.testing.assert_close(actual, torch.full((5, 1), 2.0))


def test_router_needs_the_original_module_when_a_route_needs_it(input_spec):
    class NeedsModule(ConstantBackend):
        requires_original_module = True

    backend = BatchSizeRouterBackend([(4, built(ConstantBackend())), (None, built(NeedsModule()))], input_spec)

    assert backend.requires_original_module is True


def test_one_backend_faster_at_every_batch_size_is_one_route():
    fast, slow = ConstantBackend(1.0), ConstantBackend(2.0)

    routes = routes_by_throughput([(fast, {1: 200.0, 8: 1600.0}), (slow, {1: 100.0, 8: 800.0})], fast)

    assert routes == [(None, fast)]


def test_batch_sizes_where_another_backend_is_faster_get_their_own_route():
    best, small = ConstantBackend(1.0), ConstantBackend(2.0)
    throughputs = [
        (best, {1: 100.0, 2: 200.0, 4: 400.0, 8: 800.0}),
        (small, {1: 500.0, 2: 600.0, 4: 420.0, 8: 100.0}),
    ]

    routes = routes_by_throughput(throughputs, best)

    assert routes == [(2, small), (None, best)]


def test_a_backend_within_the_noise_margin_does_not_get_a_route():
    best, other = ConstantBackend(1.0), ConstantBackend(2.0)

    routes = routes_by_throughput([(best, {1: 100.0}), (other, {1: 109.0})], best)

    assert routes == [(None, best)]


def test_strategy_routes_small_batches_to_the_backend_faster_there(torch_device, tmp_path):
    # 1 ms per row against a fixed 6 ms per call: per row is faster up to batch size 4.
    per_row = ConstantBackend(1.0, per_row=1e-3)
    per_call = ConstantBackend(2.0, per_call=6e-3)
    model = ToyTorchModel().eval().to(torch_device)
    strategy = HighestThroughputStrategy(
        backends=[per_row, per_call],
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=3),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    ).enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    sample = ((model.sample(device=torch_device).unsqueeze(0),), {})
    graph_spec = model.graph_spec(batch_sizes=[1, 16], device=torch_device)

    tuned = strategy.tune(model, "test", graph_spec, [sample], torch_device, tmp_path)

    assert tuned.describe() == "BatchSizeRouterBackend(1-4: ConstantBackend(1.0), 5+: ConstantBackend(2.0))"
