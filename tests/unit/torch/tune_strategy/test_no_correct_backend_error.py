# SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Contract of `NoCorrectBackendError`, the same for every strategy."""

import gc
import weakref
from collections.abc import Callable

import pytest
import torch

from tests.toy_backends import BuildFailsBackend, SleepBackend
from tests.toy_models.torch_models import INPUT_SIZE, ToyTorchModel
from torch_tweak.torch.tune_strategy import (
    FirstWinsStrategy,
    HighestThroughputStrategy,
    NoCorrectBackendError,
    OneBackendStrategy,
    TuneStrategy,
)
from torch_tweak.torch.utils import memory


class _FailedState:
    """Stands for the device memory a failed backend holds."""


class _KeepsStateAndFails(SleepBackend):
    """Keeps a `_FailedState` on itself and passes a weak reference to it to `on_state`.

    `copy.deepcopy` keeps a bound `list.append` as the same object, so the copy a strategy builds appends to the
    caller's list.
    """

    def __init__(self, on_state: Callable[[weakref.ref], None]):
        super().__init__()
        self.on_state = on_state

    def _keep_state(self):
        state = _FailedState()
        self.on_state(weakref.ref(state))
        self._state = state


class _KeepsStateAndFailsBuild(_KeepsStateAndFails):
    def _build(self, module, graph_spec, data, cache_dir):
        self._keep_state()
        raise ValueError("Export failed")


class _KeepsStateAndFailsReactivate(_KeepsStateAndFails):
    """Builds without activating and fails every activate after the build."""

    def _build(self, module, graph_spec, data, cache_dir):
        self._module = module
        return self

    def _activate(self):
        self._keep_state()
        raise RuntimeError("Reactivate failed")


def _first_wins():
    return FirstWinsStrategy(backends=[BuildFailsBackend(RuntimeError), BuildFailsBackend(MemoryError)])


def _highest_throughput():
    return HighestThroughputStrategy(backends=[BuildFailsBackend(RuntimeError), BuildFailsBackend(MemoryError)])


def _one_backend():
    return OneBackendStrategy(BuildFailsBackend(MemoryError))


STRATEGIES = [_first_wins, _highest_throughput, _one_backend]

STRATEGIES_WITH_FAILED_STATE = [
    pytest.param(lambda keep: FirstWinsStrategy(backends=[_KeepsStateAndFailsBuild(keep)]), id="first_wins"),
    pytest.param(
        lambda keep: HighestThroughputStrategy(backends=[_KeepsStateAndFailsBuild(keep)]), id="highest_throughput"
    ),
    pytest.param(
        lambda keep: HighestThroughputStrategy(backends=[_KeepsStateAndFailsReactivate(keep)]),
        id="highest_throughput_reactivate",
    ),
    pytest.param(lambda keep: OneBackendStrategy(_KeepsStateAndFailsBuild(keep)), id="one_backend"),
]


@pytest.fixture
def no_garbage_collection(monkeypatch):
    """Leaves only reference counting to free objects, so a reference cycle stays alive."""
    monkeypatch.setattr(memory, "free_device_memory", lambda device: None)
    gc.disable()
    yield
    gc.enable()


def _tune(strategy, device, cache_dir):
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    model = ToyTorchModel()
    sample = torch.zeros(1, INPUT_SIZE)
    return strategy.tune(model, "test", model.graph_spec(batch_sizes=[1]), [((sample,), {})], device, cache_dir)


@pytest.mark.parametrize("make_strategy", STRATEGIES)
def test_raises_no_correct_backend_error_with_the_backend_error_as_cause(make_strategy, torch_device, tmp_path):
    with pytest.raises(NoCorrectBackendError) as raised:
        _tune(make_strategy(), torch_device, tmp_path)

    assert "Build failed" in str(raised.value.__cause__)


@pytest.mark.parametrize("make_strategy", STRATEGIES)
def test_error_outside_backends_is_not_wrapped(make_strategy, torch_device, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(TuneStrategy, "_log_file", fail)

    with pytest.raises(OSError, match="disk full"):
        _tune(make_strategy(), torch_device, tmp_path)


@pytest.mark.parametrize("make_strategy", STRATEGIES_WITH_FAILED_STATE)
def test_caught_error_does_not_keep_the_failed_backend_alive(
    make_strategy, torch_device, tmp_path, no_garbage_collection
):
    states = []

    # `raised` keeps the caught error alive during the assert.
    with pytest.raises(NoCorrectBackendError) as raised:  # noqa: F841
        _tune(make_strategy(states.append), torch_device, tmp_path)

    assert [state() for state in states] == [None]
