# SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Contract of `NoCorrectBackendError`, the same for every strategy."""

import gc
import weakref

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

FAILED_STATE: list[weakref.ref] = []


class _FailedState:
    """Stands for the device memory a failed backend holds."""


class _KeepsStateAndFailsBuild(SleepBackend):
    def _build(self, module, graph_spec, data, cache_dir):
        state = _FailedState()
        FAILED_STATE.append(weakref.ref(state))
        self._state = state
        raise ValueError("Export failed")


class _KeepsStateAndFailsReactivate(SleepBackend):
    """Builds without activating and fails every activate after the build."""

    def _build(self, module, graph_spec, data, cache_dir):
        self._module = module
        return self

    def _activate(self):
        state = _FailedState()
        FAILED_STATE.append(weakref.ref(state))
        self._state = state
        raise RuntimeError("Reactivate failed")


def _first_wins():
    return FirstWinsStrategy(backends=[BuildFailsBackend(RuntimeError), BuildFailsBackend(MemoryError)])


def _highest_throughput():
    return HighestThroughputStrategy(backends=[BuildFailsBackend(RuntimeError), BuildFailsBackend(MemoryError)])


def _one_backend():
    return OneBackendStrategy(BuildFailsBackend(MemoryError))


STRATEGIES = [_first_wins, _highest_throughput, _one_backend]

STRATEGIES_WITH_FAILED_STATE = [
    pytest.param(lambda: FirstWinsStrategy(backends=[_KeepsStateAndFailsBuild()]), id="first_wins"),
    pytest.param(lambda: HighestThroughputStrategy(backends=[_KeepsStateAndFailsBuild()]), id="highest_throughput"),
    pytest.param(
        lambda: HighestThroughputStrategy(backends=[_KeepsStateAndFailsReactivate()]),
        id="highest_throughput_reactivate",
    ),
    pytest.param(lambda: OneBackendStrategy(_KeepsStateAndFailsBuild()), id="one_backend"),
]


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
def test_caught_error_does_not_keep_the_failed_backend_alive(make_strategy, torch_device, tmp_path, monkeypatch):
    # `release_device_memory` collects garbage, which would free a reference cycle that holds the failed state.
    monkeypatch.setattr(memory, "free_device_memory", lambda device: None)
    FAILED_STATE.clear()
    gc.disable()
    try:
        with pytest.raises(NoCorrectBackendError) as raised:
            _tune(make_strategy(), torch_device, tmp_path)

        assert raised.value.__cause__ is not None
        assert FAILED_STATE
        assert all(state() is None for state in FAILED_STATE)
    finally:
        gc.enable()
