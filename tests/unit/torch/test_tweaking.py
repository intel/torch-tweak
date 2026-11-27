# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import Counter

import pytest
import torch
import torch.nn as nn

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch import tweaking
from torch_tweak.torch.backend import TorchEagerBackend
from torch_tweak.torch.config import torch_tweak_cache_dir
from torch_tweak.torch.dataloader import DataLoaderFactory
from torch_tweak.torch.module.tunable_module import TunableModule
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy import FirstWinsStrategy, HighestThroughputStrategy
from torch_tweak.torch.tweaking import tweak


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


@pytest.fixture
def model():
    return _Model().eval()


@pytest.fixture
def sample():
    return torch.arange(8, dtype=torch.float32)


def _tweak(model, sample, **kwargs):
    kwargs.setdefault("strategy", FirstWinsStrategy(backends=[TorchEagerBackend()]))
    kwargs.setdefault("batch_sizes", [1, 2])
    return tweak(
        model,
        sample,
        device="cpu",
        inspect_iterations=1,
        measure_iterations=1,
        warmup_iterations=1,
        **kwargs,
    )


def _fake_measure(mocker, totals):
    """Make the baseline and the depths measure `totals`, in that order, split over two samples.

    The second baseline measurement after the depths repeats the first one. A test where no depth is
    measured does not use it.
    """
    latencies = [{(1, 0): total / 2, (2, 0): total / 2} for total in [*totals, *totals[:1]]]
    return mocker.patch("torch_tweak.torch.tweaking._measure", side_effect=latencies)


def test_dry_run_reports_distinct_cuts_without_wrapping(model, sample):
    result = _tweak(model, sample, dry_run=True)

    assert [(r.depth, sorted(r.module_paths)) for r in result.depths] == [
        (0, [""]),
        (1, [".head", ".layers.0", ".layers.1"]),
        (2, [".head", ".layers.0.mlp", ".layers.1.mlp"]),
    ]
    assert result.obj is model
    assert not isinstance(model.layers[0], TunableModule)


def test_installs_the_fastest_depth(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 1.0, 3.0])

    result = _tweak(model, sample)

    assert result.best_depth == 1
    assert [type(m) for m in (model.layers[0], model.layers[1], model.head)] == [TunableModule] * 3
    assert not isinstance(model.layers[0].mlp, TunableModule)
    assert sorted(MODULE_REGISTRY.modules) == ["head", "layers.0", "layers.1"]


def test_tweaked_model_matches_eager(mocker, model, sample):
    eager = model(sample.unsqueeze(0))
    _fake_measure(mocker, [10.0, 5.0, 1.0, 3.0])

    result = _tweak(model, sample)

    torch.testing.assert_close(result.obj(sample.unsqueeze(0)), eager)


def test_root_module_wins(mocker, model, sample):
    _fake_measure(mocker, [10.0, 1.0, 5.0, 5.0])

    result = _tweak(model, sample)

    assert result.best_depth == 0
    assert isinstance(result.obj, TunableModule)
    assert not isinstance(model.head, TunableModule)


def test_tie_prefers_the_shallower_depth(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 1.02, 1.0])

    assert _tweak(model, sample).best_depth == 1


def test_tie_ignores_a_shallower_depth_that_is_not_a_candidate(mocker, model, sample):
    _fake_measure(mocker, [100.0, 150.0, 97.5, 95.0])

    assert _tweak(model, sample).best_depth == 2


def test_depth_faster_than_the_baseline_within_the_tie_tolerance_is_not_installed(mocker, model, sample):
    _fake_measure(mocker, [100.0, 150.0, 150.0, 99.0])

    assert _tweak(model, sample).best_depth is None


def test_module_shared_by_several_cuts_is_tuned_once(mocker, model, sample):
    tune = mocker.spy(TunableModule, "tune")

    _tweak(model, sample)

    tuned_names = Counter(call.args[0].name for call in tune.call_args_list)
    assert tuned_names == Counter(["_Model", "layers.0", "layers.1", "head", "layers.0.mlp", "layers.1.mlp"])


def test_failed_depth_is_skipped_and_undone(mocker, model, sample):
    mocker.patch(
        "torch_tweak.torch.tweaking._measure",
        side_effect=[
            {(1, 0): 5.0, (2, 0): 5.0},
            {(1, 0): 2.5, (2, 0): 2.5},
            RuntimeError("measurement failed"),
            {(1, 0): 1.5, (2, 0): 1.5},
            {(1, 0): 5.0, (2, 0): 5.0},
        ],
    )

    result = _tweak(model, sample)

    assert result.depths[1].error == "RuntimeError: measurement failed"
    assert result.best_depth == 2
    assert not isinstance(model.layers[0], TunableModule)
    assert isinstance(model.layers[0].mlp, TunableModule)


def test_weights_are_offloaded_only_after_the_search(mocker, model, sample):
    """Offloading the root at depth 0 would break every deeper depth."""
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    strategy = FirstWinsStrategy(backends=[_EagerWithoutOriginalModule()])

    result = _tweak(model, sample, strategy=strategy)

    assert all(r.error is None for r in result.depths)
    assert result.best_depth == 2
    assert model.layers[0].mlp.weight.device.type == "meta"


class _EagerMovingModuleToCPUOnActivation(_EagerWithoutOriginalModule):
    """Moves the original module to CPU on every activation, as `TorchAOBackend` does."""

    def _build(self, module, graph_spec, data, cache_dir):
        self._original = module
        return super()._build(copy.deepcopy(module), graph_spec, data, cache_dir)

    def _activate(self):
        self._original.to("cpu")


@requires_xpu
def test_detached_modules_get_weights_back_after_a_backend_moved_them_on_activation(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    strategy = FirstWinsStrategy(backends=[_EagerMovingModuleToCPUOnActivation()])

    result = tweak(
        model.to("xpu"),
        sample.to("xpu"),
        strategy=strategy,
        device="xpu",
        inspect_iterations=1,
        measure_iterations=1,
        warmup_iterations=1,
    )

    assert [r.error for r in result.depths] == [None, None, None]


class _Offset(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.ones(8))

    def forward(self, x):
        return x + self.offset


@requires_xpu
def test_untuned_model_runs_after_a_backend_moved_buffers_of_a_module_without_parameters(mocker, sample):
    model = nn.Sequential(_Offset(), _Block()).eval().to("xpu")
    eager = model(sample.to("xpu").unsqueeze(0))
    _fake_measure(mocker, [1.0, 5.0, 3.0, 2.0])
    strategy = FirstWinsStrategy(backends=[_EagerMovingModuleToCPUOnActivation()])

    result = tweak(
        model,
        sample.to("xpu"),
        strategy=strategy,
        device="xpu",
        max_depth=2,
        inspect_iterations=1,
        measure_iterations=1,
        warmup_iterations=1,
    )

    assert result.best_depth is None
    torch.testing.assert_close(result.obj(sample.to("xpu").unsqueeze(0)), eager)


class _EagerMovingModuleToCPU(_EagerWithoutOriginalModule):
    def _build(self, module, graph_spec, data, cache_dir):
        super()._build(copy.deepcopy(module), graph_spec, data, cache_dir)
        module.to("cpu")
        return self


@requires_xpu
def test_deeper_depths_get_weights_back_after_a_backend_moved_them(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    strategy = FirstWinsStrategy(backends=[_EagerMovingModuleToCPU()])

    result = tweak(
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

    result = _tweak(model, sample)

    assert result.best_depth is None
    assert result.obj is model
    assert not isinstance(model.layers[0].mlp, TunableModule)
    assert MODULE_REGISTRY.modules == {}


def test_default_strategy_measures_every_backend_for_the_device(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    wrap = mocker.spy(tweaking, "wrap")
    mocker.patch.object(HighestThroughputStrategy, "tune", side_effect=RuntimeError("not needed"))

    tweak(model, sample, device="cpu", inspect_iterations=1, measure_iterations=1, warmup_iterations=1)

    strategy = wrap.call_args.kwargs["strategy"]
    assert strategy.describe().splitlines()[0] == "name: Highest Throughput Strategy"
    assert strategy.describe().splitlines()[-4:] == [
        "backends:",
        "  TorchInductorBackend()",
        "  OpenVINOBackend()",
        "  TorchEagerBackend()",
    ]
    assert not strategy.find_config.enable_find_max_batch_size


def test_min_execution_percentage_drops_small_modules(model, sample):
    result = _tweak(model, sample, dry_run=True, min_execution_percentage=1.1)

    assert result.depths == []


def test_min_execution_percentage_dropping_every_module_leaves_the_model_untuned(mocker, model, sample):
    measure = _fake_measure(mocker, [])

    result = _tweak(model, sample, min_execution_percentage=1.1)

    assert result.obj is model
    assert result.best_depth is None
    measure.assert_not_called()


class _FailingBackend(TorchEagerBackend):
    def _build(self, module, graph_spec, data, cache_dir):
        raise RuntimeError("build failed")


def test_depth_where_every_module_failed_to_tune_is_not_a_candidate(mocker, model, sample):
    measure = _fake_measure(mocker, [10.0])
    strategy = FirstWinsStrategy(backends=[_FailingBackend()])

    result = _tweak(model, sample, strategy=strategy)

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

    result = _tweak(model, sample, strategy=strategy)

    assert result.best_depth == 1
    assert isinstance(model.layers[0], TunableModule)
    assert not isinstance(model.head, TunableModule)
    assert sorted(MODULE_REGISTRY.modules) == ["layers.0", "layers.1"]


def test_parameters_after_dataset_are_keyword_only(model, sample):
    with pytest.raises(TypeError, match="positional"):
        tweak(model, sample, None)


def test_result_prints_every_depth_and_the_selected_one(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])

    result = _tweak(model, sample)

    assert str(result).splitlines() == [
        "Tweak Summary:",
        "=" * 80,
        " Depth   # modules    Total latency (ms)   Speedup  Status",
        "-" * 80,
        " eager           -             10000.000      1.00  baseline",
        "     0           1              5000.000      2.00",
        "     1           3              3000.000      3.33",
        "     2           3              1000.000     10.00  selected",
        "-" * 80,
        "Fastest for batch size 1, batch 0: depth 2",
        "Fastest for batch size 2, batch 0: depth 2",
        "=" * 80,
    ]


def test_dry_run_result_prints_every_depth_as_not_measured(model, sample):
    result = _tweak(model, sample, dry_run=True)

    assert str(result).splitlines() == [
        "Tweak Summary:",
        "=" * 80,
        " Depth   # modules    Total latency (ms)   Speedup  Status",
        "-" * 80,
        "     0           1                     -         -  not measured",
        "     1           3                     -         -  not measured",
        "     2           3                     -         -  not measured",
        "-" * 80,
        "=" * 80,
    ]


def test_result_prints_failed_depths(mocker, model, sample):
    _fake_measure(mocker, [10.0])
    strategy = FirstWinsStrategy(backends=[_FailingBackend()])

    result = _tweak(model, sample, strategy=strategy)

    assert str(result).splitlines() == [
        "Tweak Summary:",
        "=" * 80,
        " Depth   # modules    Total latency (ms)   Speedup  Status",
        "-" * 80,
        " eager           -             10000.000      1.00  baseline",
        "     0           1                     -         -  failed: every module failed to tune",
        "     1           3                     -         -  failed: every module failed to tune",
        "     2           3                     -         -  failed: every module failed to tune",
        "-" * 80,
        "=" * 80,
    ]


def test_clear_cache_removes_the_cache_directory(model, sample):
    stale = torch_tweak_cache_dir() / "stale"
    stale.mkdir(parents=True)

    _tweak(model, sample, dry_run=True, clear_cache=True)

    assert not stale.exists()


def test_time_budget_stops_starting_new_depths(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0])
    clock = mocker.patch("torch_tweak.torch.tweaking.time")
    clock.monotonic.side_effect = [0.0, 1.0, 100.0, 200.0]

    result = _tweak(model, sample, time_budget=50.0)

    assert [r.error for r in result.depths] == [None, tweaking.TIME_BUDGET_ERROR, tweaking.TIME_BUDGET_ERROR]
    assert result.best_depth == 0


def test_baseline_is_the_faster_of_the_measurements_before_and_after_the_search(mocker, model, sample):
    latencies = [{(1, 0): total / 2, (2, 0): total / 2} for total in [10.0, 5.0, 3.0, 4.0, 2.0]]
    mocker.patch("torch_tweak.torch.tweaking._measure", side_effect=latencies)

    result = _tweak(model, sample)

    assert sum(result.baseline.values()) == 2.0
    assert result.best_depth is None


def test_fastest_per_sample_is_untuned_when_the_untuned_model_is_faster(mocker, model, sample):
    mocker.patch(
        "torch_tweak.torch.tweaking._measure",
        side_effect=[
            {(1, 0): 1.0, (2, 0): 9.0},
            {(1, 0): 2.0, (2, 0): 3.0},
            {(1, 0): 2.0, (2, 0): 4.0},
            {(1, 0): 2.0, (2, 0): 5.0},
            {(1, 0): 1.0, (2, 0): 9.0},
        ],
    )

    result = _tweak(model, sample)

    assert result.best_depth_per_sample() == {(1, 0): None, (2, 0): 0}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_depth": -1}, "min_depth=-1, max_depth=5"),
        ({"min_depth": 3, "max_depth": 2}, "min_depth=3, max_depth=2"),
        ({"measure_iterations": 0}, "measure_iterations must be at least 1, got 0"),
        ({"tie_tolerance": -0.01}, "tie_tolerance must be non-negative, got -0.01"),
        ({"tie_tolerance": float("nan")}, "tie_tolerance must be non-negative, got nan"),
    ],
)
def test_invalid_arguments_are_rejected_before_tuning(model, sample, kwargs, message):
    with pytest.raises(ValueError, match=message):
        tweak(model, sample, **kwargs)


def test_every_batch_of_a_dataloader_factory_is_measured(model, sample):
    dataset = DataLoaderFactory([sample, sample + 1, sample + 2])

    result = _tweak(model, dataset, batch_sizes=[1])

    assert sorted(result.baseline) == [(1, 0), (1, 1), (1, 2)]


class _MlpFailingAfterSearchBackend(TorchEagerBackend):
    """Fails to activate a `.mlp` Linear once `search_done` is set."""

    search_done = False

    def _activate(self):
        module = self._orig_module
        if self.search_done and isinstance(module, nn.Linear) and module.out_features == 8:
            raise RuntimeError("activation failed")


def _measure_then_end_search(mocker, totals):
    """Measure `totals` like `_fake_measure`, and set `search_done` before the second baseline measurement."""

    def latencies():
        yield from ({(1, 0): total / 2, (2, 0): total / 2} for total in totals)
        mocker.patch.object(_MlpFailingAfterSearchBackend, "search_done", True)
        yield {(1, 0): totals[0] / 2, (2, 0): totals[0] / 2}

    mocker.patch("torch_tweak.torch.tweaking._measure", side_effect=latencies())


def test_next_depth_is_installed_when_the_winner_fails_to_install(mocker, model, sample):
    _measure_then_end_search(mocker, [10.0, 5.0, 3.0, 1.0])
    strategy = FirstWinsStrategy(backends=[_MlpFailingAfterSearchBackend()])

    result = _tweak(model, sample, strategy=strategy)

    assert result.best_depth == 1
    assert [type(m) for m in (model.layers[0], model.layers[1], model.head)] == [TunableModule] * 3
    assert not isinstance(model.layers[0].mlp, TunableModule)


def test_depth_that_failed_to_install_reports_the_error(mocker, model, sample):
    _measure_then_end_search(mocker, [10.0, 5.0, 3.0, 1.0])
    strategy = FirstWinsStrategy(backends=[_MlpFailingAfterSearchBackend()])

    result = _tweak(model, sample, strategy=strategy)

    assert [r.error for r in result.depths] == [None, None, "RuntimeError: activation failed"]


class _FailingToDeactivateBackend(TorchEagerBackend):
    def _deactivate(self):
        raise RuntimeError("deactivation failed")


def test_model_is_restored_when_deactivation_fails(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0])
    strategy = FirstWinsStrategy(backends=[_FailingToDeactivateBackend()])

    with pytest.raises(RuntimeError, match="deactivation failed"):
        _tweak(model, sample, strategy=strategy, min_depth=1, max_depth=1)

    assert [type(m) for m in (model.layers[0], model.layers[1], model.head)] == [_Block, _Block, nn.Linear]


class _MovingModuleToCPUAndFailingToDeactivateAgain(_EagerMovingModuleToCPUOnActivation):
    """Deactivates once. `_activate` moves the module only on a reactivation, so the second deactivation fails."""

    def _deactivate(self):
        if getattr(self, "_deactivated", False):
            raise RuntimeError("deactivation failed")
        self._deactivated = True


@requires_xpu
def test_model_keeps_its_weights_on_the_device_when_deactivation_fails(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0])
    strategy = FirstWinsStrategy(backends=[_MovingModuleToCPUAndFailingToDeactivateAgain()])

    with pytest.raises(RuntimeError, match="deactivation failed"):
        tweak(
            model.to("xpu"),
            sample.to("xpu"),
            strategy=strategy,
            device="xpu",
            min_depth=1,
            max_depth=2,
            inspect_iterations=1,
            measure_iterations=1,
            warmup_iterations=1,
        )

    assert {p.device.type for p in model.parameters()} == {"xpu"}


def test_registry_is_empty_when_the_search_fails(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0])
    strategy = FirstWinsStrategy(backends=[_FailingToDeactivateBackend()])

    with pytest.raises(RuntimeError, match="deactivation failed"):
        _tweak(model, sample, strategy=strategy, min_depth=1, max_depth=1)

    assert MODULE_REGISTRY.modules == {}


def test_winner_stays_installed_when_offloading_fails(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0, 3.0, 1.0])
    mocker.patch.object(TunableModule, "offload_after_tuning", side_effect=RuntimeError("offload failed"))

    _tweak(model, sample)

    assert [type(m) for m in (model.layers[0], model.layers[0].mlp, model.head)] == [
        _Block,
        TunableModule,
        TunableModule,
    ]


def test_model_is_detached_when_moving_it_back_to_its_device_fails(mocker, model, sample):
    _fake_measure(mocker, [10.0, 5.0])
    mocker.patch.object(TunableModule, "restore_device", side_effect=RuntimeError("out of memory"))

    with pytest.raises(RuntimeError, match="out of memory"):
        _tweak(model, sample, min_depth=1, max_depth=1)

    assert [type(m) for m in (model.layers[0], model.layers[1], model.head)] == [_Block, _Block, nn.Linear]


class _ResizingInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        y = self.linear(x)
        x.resize_(x.shape[0], 8)
        return y


def test_a_call_that_changes_its_input_in_place_does_not_change_the_next_call():
    result = _tweak(_ResizingInput(), torch.zeros(4), max_depth=0)

    assert [r.error for r in result.depths] == [None]


def test_sample_that_is_part_of_an_autograd_graph_is_rejected_with_a_tip(model):
    sample = nn.Linear(8, 8)(torch.ones(8))

    with pytest.raises(RuntimeError, match=r"x\.detach\(\)"):
        _tweak(model, sample)


def test_out_of_memory_while_copying_a_sample_is_not_reported_as_an_autograd_problem(mocker, model, sample):
    copy_module = mocker.patch("torch_tweak.torch.tweaking.copy")
    copy_module.deepcopy.side_effect = torch.OutOfMemoryError("out of memory")

    with pytest.raises(torch.OutOfMemoryError):
        _tweak(model, sample)
