# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for OnlineTunedModule."""

import pytest
import torch

from tests.toy_backends import BuildFailsBackend, SleepBackend
from tests.utilities.device_probes import ScriptedProbe
from tests.utilities.helpers import requires_leo
from torch_tweak.torch import HighestThroughputStrategy, OneBackendStrategy, OnlineTunedModule
from torch_tweak.torch.module import online_tuned_module
from torch_tweak.torch.module.online_tuned_module import SELECTED_BACKEND_FILE
from torch_tweak.torch.utils.device_load import DeviceLoadProbe


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_TWEAK_CACHE_DIR", str(tmp_path))


class CountingStrategy(HighestThroughputStrategy):
    """Counts how many times it tunes."""

    def __init__(self, backends, device_probe=None):
        super().__init__(backends=backends, device_probe=device_probe)
        self.tune_calls = 0

    def tune(self, *args, **kwargs):
        self.tune_calls += 1
        return super().tune(*args, **kwargs)


def sleep_strategy(backends=None, device_probe: DeviceLoadProbe | None = None) -> CountingStrategy:
    strategy = CountingStrategy(backends=backends or [SleepBackend()], device_probe=device_probe)
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    return strategy


def doubling() -> torch.nn.Linear:
    layer = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0]]))
    return layer


def tripling() -> torch.nn.Linear:
    layer = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[3.0, 0.0], [0.0, 3.0]]))
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


def test_a_later_module_with_the_same_weights_restores_the_selection_without_tuning():
    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())(torch.tensor([[1.0, 3.0]]))
    strategy = sleep_strategy()
    later = OnlineTunedModule(doubling(), "doubling", strategy=strategy)

    later(torch.tensor([[1.0, 3.0]]))

    assert (strategy.results, [type(backend) for backend in later.backends.values()]) == ([], [SleepBackend])


def test_restored_backend_returns_the_module_output():
    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())(torch.tensor([[1.0, 3.0]]))
    later = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())
    later(torch.tensor([[1.0, 3.0]]))

    actual = later(torch.tensor([[4.0, 5.0]]))

    torch.testing.assert_close(actual, torch.tensor([[8.0, 10.0]]))


def test_a_later_module_with_other_weights_tunes_again():
    OnlineTunedModule(doubling(), "linear", strategy=sleep_strategy())(torch.tensor([[1.0, 3.0]]))
    strategy = sleep_strategy()

    OnlineTunedModule(tripling(), "linear", strategy=strategy)(torch.tensor([[1.0, 3.0]]))

    assert len(strategy.results) == 1


def test_a_selection_tuned_on_a_shared_device_is_tuned_again_later():
    shared = sleep_strategy(device_probe=ScriptedProbe([1.0, 20.0, 1.0, 20.0, 1.0, 20.0]))
    OnlineTunedModule(doubling(), "doubling", strategy=shared)(torch.tensor([[1.0, 3.0]]))
    strategy = sleep_strategy()

    OnlineTunedModule(doubling(), "doubling", strategy=strategy)(torch.tensor([[1.0, 3.0]]))

    assert len(strategy.results) == 1


def test_a_later_module_in_the_same_environment_restores_the_selection_without_tuning():
    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy(), environment={"torch": "2.15.0"})(
        torch.tensor([[1.0, 3.0]])
    )
    strategy = sleep_strategy()

    OnlineTunedModule(doubling(), "doubling", strategy=strategy, environment={"torch": "2.15.0"})(
        torch.tensor([[1.0, 3.0]])
    )

    assert strategy.results == []


def test_a_later_module_in_another_environment_tunes_again():
    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy(), environment={"torch": "2.15.0"})(
        torch.tensor([[1.0, 3.0]])
    )
    strategy = sleep_strategy()

    OnlineTunedModule(doubling(), "doubling", strategy=strategy, environment={"torch": "2.16.0"})(
        torch.tensor([[1.0, 3.0]])
    )

    assert len(strategy.results) == 1


def test_tuning_again_in_another_environment_logs_the_entries_that_differ(caplog):
    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy(), environment={"torch": "2.15.0"})(
        torch.tensor([[1.0, 3.0]])
    )

    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy(), environment={"torch": "2.16.0"})(
        torch.tensor([[1.0, 3.0]])
    )

    assert "torch: tuned with '2.15.0', running '2.16.0'" in caplog.text


def test_tuning_again_in_another_environment_removes_the_artifacts_of_the_earlier_one(tmp_path):
    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy(), environment={"torch": "2.15.0"})(
        torch.tensor([[1.0, 3.0]])
    )
    earlier_ir = next(tmp_path.rglob(SELECTED_BACKEND_FILE)).parent / "model.xml"
    earlier_ir.write_text("converted by the earlier environment")

    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy(), environment={"torch": "2.16.0"})(
        torch.tensor([[1.0, 3.0]])
    )

    assert not earlier_ir.exists()


def test_a_selection_saved_without_a_format_version_tunes_again(tmp_path):
    OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())(torch.tensor([[1.0, 3.0]]))
    next(tmp_path.rglob(SELECTED_BACKEND_FILE)).write_text("SleepBackend")
    strategy = sleep_strategy()

    OnlineTunedModule(doubling(), "doubling", strategy=strategy)(torch.tensor([[1.0, 3.0]]))

    assert len(strategy.results) == 1


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


def test_call_after_reset_with_the_same_weights_restores_without_tuning():
    strategy = sleep_strategy()
    online = OnlineTunedModule(doubling(), "doubling", strategy=strategy)
    online(torch.tensor([[1.0, 3.0]]))
    online.reset()

    online(torch.tensor([[1.0, 3.0]]))

    assert strategy.tune_calls == 1


def test_call_after_reset_with_patched_weights_tunes_again():
    module = doubling()
    strategy = sleep_strategy()
    online = OnlineTunedModule(module, "doubling", strategy=strategy)
    online(torch.tensor([[1.0, 3.0]]))
    with torch.no_grad():
        module.weight.mul_(3.0)
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


def test_failed_export_of_a_one_backend_strategy_still_returns_the_module_output():
    strategy = OneBackendStrategy(BuildFailsBackend(ValueError))
    strategy.enable_find_max_batch_size(False)
    online = OnlineTunedModule(doubling(), "doubling", strategy=strategy)

    actual = online(torch.tensor([[1.0, 3.0]]))

    torch.testing.assert_close(actual, torch.tensor([[2.0, 6.0]]))


def test_tuning_error_outside_the_backends_reaches_the_caller(monkeypatch):
    def full_disk(path):
        raise OSError("No space left on device")

    monkeypatch.setattr(online_tuned_module, "secure_mkdir", full_disk)
    online = OnlineTunedModule(doubling(), "doubling", strategy=sleep_strategy())

    with pytest.raises(OSError, match="No space left on device"):
        online(torch.tensor([[1.0, 3.0]]))


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
