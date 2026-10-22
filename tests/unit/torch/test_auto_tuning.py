# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import Counter

import pytest
import torch
import torch.nn as nn

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch import auto_tuning
from torch_tweak.torch.auto_tuning import auto_tune
from torch_tweak.torch.backend import TorchEagerBackend
from torch_tweak.torch.module.wrapper_module import Module
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy import FirstWinsStrategy, HighestThroughputStrategy, default_backends


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(8, 8)

    def forward(self, x):
        return self.mlp(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


class _EagerWithoutOriginalModule(TorchEagerBackend):
    requires_original_module = False


@pytest.fixture(autouse=True)
def _cache_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_TWEAK_CACHE_DIR", str(tmp_path))


@pytest.fixture
def model():
    return _Model().eval()


@pytest.fixture
def sample():
    return torch.randn(8)


def _auto_tune(model, sample, **kwargs):
    kwargs.setdefault("strategy", FirstWinsStrategy(backends=[TorchEagerBackend()]))
    return auto_tune(
        model,
        sample,
        batch_sizes=[1, 2],
        device="cpu",
        inspect_iterations=1,
        measure_iterations=1,
        warmup_iterations=1,
        **kwargs,
    )


def _fake_measure(mocker, totals):
    """Make the baseline and the depths measure `totals`, in that order, split over two samples."""
    latencies = [{(1, 0): total / 2, (2, 0): total / 2} for total in totals]
    return mocker.patch("torch_tweak.torch.auto_tuning._measure", side_effect=latencies)


def test_dry_run_reports_distinct_cuts_without_wrapping(model, sample):
    result = _auto_tune(model, sample, dry_run=True)

    assert [(r.depth, sorted(r.module_paths)) for r in result.depths] == [
        (0, [""]),
        (1, [".head", ".layers.0", ".layers.1"]),
        (2, [".head", ".layers.0.mlp", ".layers.1.mlp"]),
    ]
    assert result.obj is model
    assert not isinstance(model.layers[0], Module)


def test_installs_the_fastest_depth(mocker, model, sample):
    expected = model(sample.unsqueeze(0))
    _fake_measure(mocker, [10.0, 5.0, 1.0, 3.0])

    result = _auto_tune(model, sample)

    assert result.best_depth == 1
    assert [type(m) for m in (model.layers[0], model.layers[1], model.head)] == [Module] * 3
    assert not isinstance(model.layers[0].mlp, Module)
    assert sorted(MODULE_REGISTRY.modules) == ["head", "layers.0", "layers.1"]
    torch.testing.assert_close(result.obj(sample.unsqueeze(0)), expected)


def test_root_module_wins(mocker, model, sample):
    _fake_measure(mocker, [10.0, 1.0, 5.0, 5.0])

    result = _auto_tune(model, sample)

    assert result.best_depth == 0
    assert isinstance(result.obj, Module)
    assert not isinstance(model.head, Module)


def test_tie_prefers_the_shallower_depth(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 1.02, 1.0])

    assert _auto_tune(model, sample).best_depth == 1


def test_module_shared_by_several_cuts_is_tuned_once(mocker, model, sample):
    tune = mocker.spy(Module, "tune")

    _auto_tune(model, sample)

    tuned_names = Counter(call.args[0].name for call in tune.call_args_list)
    assert tuned_names == Counter(["_Model", "layers.0", "layers.1", "head", "layers.0.mlp", "layers.1.mlp"])


def test_failed_depth_is_skipped_and_undone(mocker, model, sample):
    mocker.patch(
        "torch_tweak.torch.auto_tuning._measure",
        side_effect=[
            {(1, 0): 5.0, (2, 0): 5.0},
            {(1, 0): 2.5, (2, 0): 2.5},
            RuntimeError("measurement failed"),
            {(1, 0): 1.5, (2, 0): 1.5},
        ],
    )

    result = _auto_tune(model, sample)

    assert "measurement failed" in result.depths[1].error
    assert result.best_depth == 2
    assert not isinstance(model.layers[0], Module)
    assert isinstance(model.layers[0].mlp, Module)


def test_weights_are_offloaded_only_after_the_search(mocker, model, sample):
    """Offloading the root at depth 0 would break every deeper depth."""
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    strategy = FirstWinsStrategy(backends=[_EagerWithoutOriginalModule()])

    result = _auto_tune(model, sample, strategy=strategy)

    assert all(r.error is None for r in result.depths)
    assert result.best_depth == 2
    assert model.layers[0].mlp.weight.device.type == "meta"


class _EagerMovingModuleToCPU(_EagerWithoutOriginalModule):
    def _build(self, module, graph_spec, data, cache_dir):
        super()._build(copy.deepcopy(module), graph_spec, data, cache_dir)
        module.to("cpu")
        return self


@requires_xpu
def test_deeper_depths_get_weights_back_after_a_backend_moved_them(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    strategy = FirstWinsStrategy(backends=[_EagerMovingModuleToCPU()])

    result = auto_tune(
        model.to("xpu"),
        sample.to("xpu"),
        strategy=strategy,
        device="xpu",
        inspect_iterations=1,
        measure_iterations=1,
        warmup_iterations=1,
    )

    assert all(r.error is None for r in result.depths)
    assert result.best_depth == 2


def test_model_is_left_untuned_when_no_depth_beats_the_baseline(mocker, model, sample):
    _fake_measure(mocker, [1.0, 5.0, 3.0, 1.0])

    result = _auto_tune(model, sample)

    assert result.best_depth is None
    assert result.obj is model
    assert not isinstance(model.layers[0].mlp, Module)
    assert MODULE_REGISTRY.modules == {}


def test_default_strategy_measures_every_backend_for_the_device(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    wrap = mocker.spy(auto_tuning, "wrap")
    mocker.patch.object(HighestThroughputStrategy, "tune", side_effect=RuntimeError("not needed"))

    auto_tune(model, sample, device="cpu", inspect_iterations=1, measure_iterations=1, warmup_iterations=1)

    strategy = wrap.call_args.kwargs["strategy"]
    assert isinstance(strategy, HighestThroughputStrategy)
    assert [type(b) for b in strategy._backends] == [type(b) for b in default_backends("cpu")]
    assert not strategy.find_config.enable_find_max_batch_size


def test_min_execution_percentage_drops_small_modules(model, sample):
    result = _auto_tune(model, sample, dry_run=True, min_execution_percentage=1.1)

    assert result.depths == []


def test_min_execution_percentage_dropping_every_module_leaves_the_model_untuned(mocker, model, sample):
    measure = _fake_measure(mocker, [])

    result = _auto_tune(model, sample, min_execution_percentage=1.1)

    assert result.obj is model
    assert result.best_depth is None
    measure.assert_not_called()


class _FailingBackend(TorchEagerBackend):
    def _build(self, module, graph_spec, data, cache_dir):
        raise RuntimeError("build failed")


def test_depth_where_every_module_failed_to_tune_is_not_a_candidate(mocker, model, sample):
    measure = _fake_measure(mocker, [10.0])
    strategy = FirstWinsStrategy(backends=[_FailingBackend()])

    result = _auto_tune(model, sample, strategy=strategy)

    assert [r.error for r in result.depths] == ["every module failed to tune"] * 3
    assert result.best_depth is None
    assert result.obj is model
    assert measure.call_count == 1
    assert MODULE_REGISTRY.modules == {}


class _FailingOnHeadBackend(TorchEagerBackend):
    def _build(self, module, graph_spec, data, cache_dir):
        if isinstance(module, nn.Linear) and module.out_features == 4:
            raise RuntimeError("build failed")
        return super()._build(module, graph_spec, data, cache_dir)


def test_modules_that_failed_to_tune_are_not_installed(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 1.0, 3.0])
    strategy = FirstWinsStrategy(backends=[_FailingOnHeadBackend()])

    result = _auto_tune(model, sample, strategy=strategy)

    assert result.best_depth == 1
    assert isinstance(model.layers[0], Module)
    assert not isinstance(model.head, Module)
    assert sorted(MODULE_REGISTRY.modules) == ["layers.0", "layers.1"]


def test_parameters_after_dataset_are_keyword_only(model, sample):
    with pytest.raises(TypeError):
        auto_tune(model, sample, None)


def test_result_prints_every_depth_and_the_selected_one(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])

    result = _auto_tune(model, sample)

    text = str(result)
    assert "baseline" in text
    assert [line.split()[0] for line in text.splitlines() if line.strip().startswith(("0 ", "1 ", "2 "))] == [
        "0",
        "1",
        "2",
    ]
    assert "selected" in next(line for line in text.splitlines() if line.strip().startswith("2 "))


def test_time_budget_stops_starting_new_depths(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    clock = iter([0.0, 1.0, 100.0, 200.0])
    mocker.patch.object(auto_tuning.time, "monotonic", side_effect=lambda: next(clock))

    result = _auto_tune(model, sample, time_budget=50.0)

    assert [r.error for r in result.depths] == [None, auto_tuning.TIME_BUDGET_ERROR, auto_tuning.TIME_BUDGET_ERROR]
    assert result.best_depth == 0
