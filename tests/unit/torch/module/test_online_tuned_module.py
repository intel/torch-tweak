# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for OnlineTunedModule."""

import pytest
import torch

from tests.toy_backends import BuildFailsBackend, SleepBackend
from tests.utilities.helpers import requires_leo
from torch_tweak.torch import HighestThroughputStrategy, OnlineTunedModule


class CountingStrategy(HighestThroughputStrategy):
    """Counts how many times it tunes."""

    def __init__(self, backends):
        super().__init__(backends=backends)
        self.tune_calls = 0

    def tune(self, *args, **kwargs):
        self.tune_calls += 1
        return super().tune(*args, **kwargs)


def sleep_strategy(backends=None) -> CountingStrategy:
    strategy = CountingStrategy(backends=backends or [SleepBackend()])
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    return strategy


def doubling() -> torch.nn.Linear:
    layer = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    return layer


def test_first_call_returns_the_module_output():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())

    actual = online(torch.tensor([[1.0, 3.0]]))

    torch.testing.assert_close(actual, torch.tensor([[2.0, 6.0]]))


def test_first_call_tunes_its_shapes():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())

    online(torch.tensor([[1.0, 3.0]]))

    assert [type(backend) for backend in online.backends.values()] == [SleepBackend]


def test_shapes_of_the_same_rank_are_tuned_apart():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())

    online(torch.tensor([[1.0, 3.0]]))
    online(torch.tensor([[1.0, 3.0], [5.0, 7.0]]))

    assert len(online.backends) == 2


def test_tuned_backend_returns_the_module_output():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())
    online(torch.tensor([[1.0, 3.0]]))

    actual = online(torch.tensor([[4.0, 5.0]]))

    torch.testing.assert_close(actual, torch.tensor([[8.0, 10.0]]))


def test_tuning_new_shapes_deactivates_the_backend_of_the_last_call():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())
    online(torch.tensor([[1.0, 3.0]]))

    online(torch.tensor([[1.0, 3.0], [5.0, 7.0]]))

    assert [backend.is_active for backend in online.backends.values()] == [False, True]


def test_calling_tuned_shapes_again_activates_their_backend():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())
    online(torch.tensor([[1.0, 3.0]]))
    online(torch.tensor([[1.0, 3.0], [5.0, 7.0]]))

    online(torch.tensor([[4.0, 5.0]]))

    assert [backend.is_active for backend in online.backends.values()] == [True, False]


def test_reactivated_backend_returns_the_module_output():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())
    online(torch.tensor([[1.0, 3.0]]))
    online(torch.tensor([[1.0, 3.0], [5.0, 7.0]]))

    actual = online(torch.tensor([[4.0, 5.0]]))

    torch.testing.assert_close(actual, torch.tensor([[8.0, 10.0]]))


def test_reset_drops_the_tuned_shapes():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())
    online(torch.tensor([[1.0, 3.0]]))

    online.reset()

    assert online.backends == {}


def test_tuned_shapes_are_not_tuned_again():
    strategy = sleep_strategy()
    online = OnlineTunedModule(doubling(), "doubling", strategy=strategy)

    online(torch.tensor([[1.0, 3.0]]))
    online(torch.tensor([[4.0, 5.0]]))
    online(torch.tensor([[6.0, 7.0]]))

    assert strategy.tune_calls == 1


def test_call_after_reset_tunes_again():
    strategy = sleep_strategy()
    online = OnlineTunedModule(doubling(), "doubling", strategy=strategy)
    online(torch.tensor([[1.0, 3.0]]))
    online.reset()

    online(torch.tensor([[1.0, 3.0]]))

    assert strategy.tune_calls == 2


def test_same_shape_with_another_dtype_is_tuned_apart():
    online = OnlineTunedModule(torch.nn.ReLU(), "relu", strategy=sleep_strategy(), device=torch.device("cpu"))

    online(torch.tensor([[1.0, -3.0]]))
    online(torch.tensor([[1.0, -3.0]], dtype=torch.bfloat16))

    assert len(online.backends) == 2


def test_same_shape_in_a_keyword_is_tuned_apart_from_a_positional_one():
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())

    online(torch.tensor([[1.0, 3.0]]))
    online(input=torch.tensor([[1.0, 3.0]]))

    assert len(online.backends) == 2


def test_failed_tuning_still_returns_the_module_output():
    strategy = sleep_strategy(backends=[BuildFailsBackend(RuntimeError)])
    online = OnlineTunedModule(doubling(), "doubling", strategy=strategy)

    actual = online(torch.tensor([[1.0, 3.0]]))

    torch.testing.assert_close(actual, torch.tensor([[2.0, 6.0]]))


def test_shape_that_failed_tuning_is_not_tuned_again():
    strategy = sleep_strategy(backends=[BuildFailsBackend(RuntimeError)])
    online = OnlineTunedModule(doubling(), "doubling", strategy=strategy)
    online(torch.tensor([[1.0, 3.0]]))

    actual = online(torch.tensor([[4.0, 5.0]]))

    torch.testing.assert_close(actual, torch.tensor([[8.0, 10.0]]))
    assert (strategy.tune_calls, online.backends) == (1, {})


@requires_leo
def test_default_strategy_leaves_the_weights_where_they_are():
    module = doubling().to("xpu")
    weight = module.weight.data_ptr()
    online = OnlineTunedModule(module, "doubling")

    online(torch.tensor([[1.0, 3.0]], device="xpu"))

    assert (module.weight.device.type, module.weight.data_ptr()) == ("xpu", weight)


def test_module_with_a_weight_on_another_device_is_rejected():
    module = torch.nn.Linear(2, 2, device="meta")

    with pytest.raises(ValueError, match="weight"):
        OnlineTunedModule(module, "linear", strategy=sleep_strategy(), device=torch.device("cpu"))


def test_instances_with_one_name_use_separate_graph_directories(torch_tweak_cache_dir):
    first = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())
    second = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())

    first(torch.tensor([[1.0, 3.0]]))
    second(torch.tensor([[1.0, 3.0]]))

    assert len(list((torch_tweak_cache_dir / "doubling").iterdir())) == 2
