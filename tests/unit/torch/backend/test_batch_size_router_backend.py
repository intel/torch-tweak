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
from torch_tweak.torch.task.profiling import NumStepsMeasuringStopStrategy
from torch_tweak.torch.task.profiling.profiling_stop_strategy import AllSamplesProfilingStopStrategy
from torch_tweak.torch.tune_strategy.highest_throughput_strategy import HighestThroughputStrategy


class TimedBackend(Backend):
    """Sleeps per_call plus per_row times the batch size, then runs the module."""

    requires_original_module = True

    def __init__(self, label: str = "", per_call: float = 0.0, per_row: float = 0.0):
        super().__init__()
        self.label, self.per_call, self.per_row = label, per_call, per_row
        self.calls: list[int] = []
        self._module: Any = None

    def key(self) -> str:
        return f"TimedBackend_{self.label}"

    def describe(self) -> str:
        return f"TimedBackend({self.label})"

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        self._module = module
        return self

    def _activate(self):
        pass

    def _deactivate(self):
        pass

    def _deploy(self):
        pass

    def _infer(self, *args, **kwargs):
        batch_size = args[0].shape[0]
        self.calls.append(batch_size)
        time.sleep(self.per_call + self.per_row * batch_size)
        return self._module(*args, **kwargs)

    def to_dict(self):
        return {"type": "TimedBackend", "label": self.label}

    @classmethod
    def from_dict(cls, module, state_dict: dict):
        backend = cls(label=state_dict["label"])
        backend._module = module
        backend.state = BackendState.CHECKPOINT_LOADED
        return backend


def built(backend: TimedBackend, module, torch_device, tmp_path) -> TimedBackend:
    sample = ((torch.rand(1, 4, device=torch_device),), {})
    backend.build(module, None, [sample], torch_device, tmp_path)
    return backend


@pytest.fixture
def model(torch_device):
    return ToyTorchModel().eval().to(torch_device)


@pytest.fixture
def graph_spec(model, torch_device):
    return model.graph_spec(batch_sizes=[1, 16], device=torch_device)


def test_routes_send_each_batch_size_to_its_backend(model, graph_spec, torch_device, tmp_path):
    small = built(TimedBackend("small"), model, torch_device, tmp_path)
    large = built(TimedBackend("large"), model, torch_device, tmp_path)
    router = BatchSizeRouterBackend([(4, small), (None, large)], graph_spec.input_spec)

    assert router.is_active
    for batch_size in (1, 4, 5, 64):
        router.infer(model.samples([batch_size], device=torch_device)[0][0][0])
    assert small.calls == [1, 4]
    assert large.calls == [5, 64]
    assert router.describe() == "BatchSizeRouterBackend(1-4: TimedBackend(small), 5+: TimedBackend(large))"
    assert router.requires_original_module


def test_last_route_must_take_every_larger_batch_size(model, graph_spec, torch_device, tmp_path):
    backend = built(TimedBackend(), model, torch_device, tmp_path)
    with pytest.raises(ValueError, match="every larger batch size"):
        BatchSizeRouterBackend([(4, backend)], graph_spec.input_spec)


def test_serialization(model, graph_spec, torch_device, tmp_path):
    small = built(TimedBackend("small"), model, torch_device, tmp_path)
    large = built(TimedBackend("large"), model, torch_device, tmp_path)
    router = BatchSizeRouterBackend([(4, small), (None, large)], graph_spec.input_spec)

    loaded = BatchSizeRouterBackend.from_dict(model, router.to_dict())

    assert loaded.state == BackendState.CHECKPOINT_LOADED
    assert loaded.describe() == router.describe()
    loaded.deploy(torch_device)
    sample = model.samples([2], device=torch_device)[0][0][0]
    torch.testing.assert_close(loaded.infer(sample), model(sample))
    assert loaded.backend_for(2).calls == [2]


def tune(backends, model, graph_spec, torch_device, tmp_path) -> Backend:
    strategy = HighestThroughputStrategy(
        backends=backends,
        measurement_stop_strategy=NumStepsMeasuringStopStrategy(num_steps=3),
        profiling_stop_strategy=AllSamplesProfilingStopStrategy(),
    ).enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    sample = ((model.sample(device=torch_device).unsqueeze(0),), {})
    return strategy.tune(model, "test", graph_spec, [sample], torch_device, tmp_path)


def test_strategy_routes_small_batches_to_the_backend_faster_there(model, graph_spec, torch_device, tmp_path):
    # Per call: per_row 1 ms per row against a fixed 6 ms. Rows are faster up to batch size 4.
    rows = TimedBackend("rows", per_row=1e-3)
    fixed = TimedBackend("fixed", per_call=6e-3)

    tuned = tune([rows, fixed], model, graph_spec, torch_device, tmp_path)

    assert isinstance(tuned, BatchSizeRouterBackend)
    assert [(largest, backend.describe()) for largest, backend in tuned.routes] == [
        (4, "TimedBackend(rows)"),
        (None, "TimedBackend(fixed)"),
    ]


def test_strategy_keeps_one_backend_when_it_wins_every_batch_size(model, graph_spec, torch_device, tmp_path):
    tuned = tune(
        [TimedBackend("slow", per_call=5e-3), TimedBackend("fast", per_call=1e-3)],
        model,
        graph_spec,
        torch_device,
        tmp_path,
    )

    assert tuned.describe() == "TimedBackend(fast)"
