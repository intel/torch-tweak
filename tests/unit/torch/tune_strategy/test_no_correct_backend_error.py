# SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Contract of `NoCorrectBackendError`, the same for every strategy."""

import pytest
import torch

from tests.toy_backends import BuildFailsBackend
from tests.toy_models.torch_models import INPUT_SIZE, ToyTorchModel
from torch_tweak.torch.tune_strategy import (
    FirstWinsStrategy,
    HighestThroughputStrategy,
    NoCorrectBackendError,
    OneBackendStrategy,
)
from torch_tweak.torch.tune_strategy import tune_strategy as tune_strategy_module


def _first_wins():
    return FirstWinsStrategy(backends=[BuildFailsBackend(RuntimeError), BuildFailsBackend(MemoryError)])


def _highest_throughput():
    return HighestThroughputStrategy(backends=[BuildFailsBackend(RuntimeError), BuildFailsBackend(MemoryError)])


def _one_backend():
    return OneBackendStrategy(BuildFailsBackend(MemoryError))


STRATEGIES = [_first_wins, _highest_throughput, _one_backend]


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

    monkeypatch.setattr(tune_strategy_module, "release_device_memory", fail)

    with pytest.raises(OSError, match="disk full"):
        _tune(make_strategy(), torch_device, tmp_path)
