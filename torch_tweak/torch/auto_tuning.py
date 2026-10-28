# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tune a model at several cut depths and keep the fastest one."""

import math
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch

from torch_tweak.torch.config import DEFAULT_DEVICE
from torch_tweak.torch.dataloader import DataLoaderFactory, DatasetLike, samples_generator
from torch_tweak.torch.inspecting.inspecting import (
    DEFAULT_INSPECT_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS,
    modules_info_for,
    run_inspection,
)
from torch_tweak.torch.inspecting.module_info import ModuleInfo
from torch_tweak.torch.inspecting.module_inspector import ModuleInspector, cut_at
from torch_tweak.torch.inspecting.wrapping import wrap
from torch_tweak.torch.module.wrapper_module import Module, ModuleState
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy import default_strategy
from torch_tweak.torch.tune_strategy.tune_strategy import TuneStrategy
from torch_tweak.torch.tuning import (
    _clear_cache,
    _record_and_tune_registered_modules,
    _validate_and_normalize_batch_sizes,
)
from torch_tweak.torch.utils.xpu import synchronize
from torch_tweak.utils.logging import libraries_logging, setup_logging
from torch_tweak.utils.trace import annotate

logger = getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

DEFAULT_MAX_DEPTH = 5
DEFAULT_MEASURE_ITERATIONS = 10
DEFAULT_TIE_TOLERANCE = 0.03
TIME_BUDGET_ERROR = "time budget used up"

SampleKey = tuple[int, int]
"""Batch size and the index of the batch among batches of that size."""


@dataclass
class DepthResult:
    """Outcome of tuning the model cut at one depth."""

    depth: int
    module_paths: list[str]
    latencies: dict[SampleKey, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def total_latency(self) -> float:
        """Sum of median latencies over all samples in seconds, infinite when the depth failed."""
        if self.error is not None or not self.latencies:
            return math.inf
        return sum(self.latencies.values())


@dataclass
class AutoTuneResult:
    """Outcome of `auto_tune`."""

    obj: Any
    """The object to call for inference and to pass to `save`. It is a `Module` when the root module won."""
    best_depth: int | None
    baseline: dict[SampleKey, float]
    """Median latency per sample in seconds before tuning."""
    depths: list[DepthResult]

    def best_depth_per_sample(self) -> dict[SampleKey, int]:
        """The fastest depth for each sample, ignoring the tie tolerance."""
        measured = [result for result in self.depths if result.error is None and result.latencies]
        if not measured:
            return {}
        return {key: min(measured, key=lambda result: result.latencies[key]).depth for key in measured[0].latencies}

    def __str__(self) -> str:
        """The latency of every depth and the winner, as a table."""
        lines = ["Auto Tune Summary:", "=" * 80]
        lines.append(f"{'Depth':>6}  {'# modules':>10}  {'Total latency (ms)':>20}  {'Speedup':>8}  {'Status':<20}")
        lines.append("-" * 80)
        baseline_total = sum(self.baseline.values())
        if self.baseline:
            lines.append(f"{'eager':>6}  {'-':>10}  {baseline_total * 1e3:>20.3f}  {1.0:>8.2f}  {'baseline':<20}")
        for result in self.depths:
            if result.error is not None:
                status, total, speedup = f"failed: {result.error}"[:40], "-", "-"
            elif not result.latencies:
                status, total, speedup = "not measured", "-", "-"
            else:
                status = "selected" if result.depth == self.best_depth else ""
                total = f"{result.total_latency * 1e3:.3f}"
                speedup = f"{baseline_total / result.total_latency:.2f}" if baseline_total else "-"
            lines.append(f"{result.depth:>6}  {len(result.module_paths):>10}  {total:>20}  {speedup:>8}  {status:<20}")
        lines.append("-" * 80)
        for (batch_size, index), depth in self.best_depth_per_sample().items():
            lines.append(f"Fastest depth for batch size {batch_size}, batch {index}: {depth}")
        lines.append("=" * 80)
        return "\n".join(lines)

    def describe(self) -> None:
        """Print the latency of every depth and the winner."""
        print(self)  # noqa: T201


@annotate(domain="Torch Tweak")
def auto_tune(
    obj: Callable | torch.nn.Module,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    *,
    inference_function: Callable | None = None,
    strategy: TuneStrategy | None = None,
    min_depth: int = 0,
    max_depth: int = DEFAULT_MAX_DEPTH,
    batch_sizes: list[int] | None = None,
    max_num_batches_per_batch_size: int | None = None,
    device: str | torch.device | None = DEFAULT_DEVICE,
    min_execution_percentage: float | None = None,
    inspect_iterations: int = DEFAULT_INSPECT_ITERATIONS,
    measure_iterations: int = DEFAULT_MEASURE_ITERATIONS,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    time_budget: float | None = None,
    dry_run: bool = False,
    disable_external_logging: bool = False,
    clear_cache: bool = False,
) -> AutoTuneResult:
    """Tune `obj` cut at every depth from `min_depth` to `max_depth` and keep the fastest cut.

    Depth is defined as in `inspect`. Each distinct cut is wrapped, tuned, and measured end to end
    on `dataset`. A module tuned at one depth is reused at deeper depths where it appears again.
    The untuned model is measured before and after the search, and the lower latency is the baseline.
    The second measurement is skipped when no depth was measured.
    The winner has the lowest sum of median latencies over all samples. A shallower depth wins
    when it is within `tie_tolerance` of the fastest one. The model is left untuned when no depth
    is faster than the untuned model.

    Args:
        obj: Model or pipeline to tune.
        dataset: Samples for inspection, tuning, and measurement.
        inference_function: Function that runs inference on `obj`, `obj` is called by default.
        strategy: Strategy for every wrapped module. When None, `default_strategy(device)` without the search
            for the maximum batch size is used: every backend for the device is measured and the fastest wins.
        min_depth: Shallowest depth to try.
        max_depth: Deepest depth to try.
        batch_sizes: Batch sizes for tuning and measurement.
        max_num_batches_per_batch_size: Maximum number of batches per batch size.
        device: Device to tune on.
        min_execution_percentage: Modules below this fraction, from 0 to 1, of inspected execution time are
            not wrapped.
        inspect_iterations: Number of inference runs during inspection.
        measure_iterations: Number of timed runs per sample at every depth.
        warmup_iterations: Number of untimed runs before inspection and before every measurement.
        tie_tolerance: Relative latency difference treated as a tie.
        time_budget: Seconds after which no new depth is started. The depth in progress finishes.
        dry_run: If True, only report the modules of each cut.
        disable_external_logging: If True, libraries logging will be suppressed.
        clear_cache: If True, the cache will be cleared before tuning.

    Returns:
        The result. Use `result.obj` for inference and for `save`: when the whole model wins, `result.obj` is a
        new `Module` and `obj` stays untuned. `print(result)` shows the latency of every depth.
    """
    start = time.monotonic()
    setup_logging(format_string=LOG_FORMAT)
    if clear_cache:
        _clear_cache()

    func = obj if inference_function is None else inference_function
    batch_sizes = _validate_and_normalize_batch_sizes(batch_sizes)
    if strategy is None:
        # Searching the maximum batch size of every small module at every depth costs more than the tuning
        # itself, and batch sizes far above `batch_sizes` can hang the device.
        strategy = default_strategy(device or DEFAULT_DEVICE).enable_find_max_batch_size(False)

    with libraries_logging(disable_external_logging):
        inspector, total_execution_time = run_inspection(
            obj, dataset, inference_function, inspect_iterations, warmup_iterations, max_depth
        )
        cuts = _distinct_cuts(
            inspector, min_depth, max_depth, total_execution_time, inspect_iterations, min_execution_percentage
        )
        for depth, modules in cuts:
            logger.info("Depth %d cuts the model into %d modules: %s", depth, len(modules), _paths(modules))
        if not cuts:
            logger.warning(
                "No executed module passes min_execution_percentage=%s. The model is left untuned.",
                min_execution_percentage,
            )
            return AutoTuneResult(obj, None, {}, [])
        if dry_run:
            return AutoTuneResult(obj, None, {}, [DepthResult(depth, _paths(modules)) for depth, modules in cuts])

        def measure() -> dict[SampleKey, float]:
            return _measure(
                func, dataset, batch_sizes, max_num_batches_per_batch_size, warmup_iterations, measure_iterations
            )

        logger.info("Measuring the model before tuning.")
        baseline = measure()

        tuned: dict[str, Module] = {}

        def tune_depth(depth: int, modules: list[ModuleInfo]) -> DepthResult:
            return _tune_and_measure_depth(
                obj,
                func,
                depth,
                modules,
                tuned,
                strategy,
                measure,
                dataset,
                batch_sizes,
                max_num_batches_per_batch_size,
                device,
            )

        results = _tune_every_depth(cuts, tune_depth, start, time_budget)
        if any(result.latencies for result in results):
            # The first measurement can run before the device reaches its full clock, which favors every depth.
            logger.info("Measuring the model before tuning again.")
            baseline = _faster(baseline, measure())
        return _install_best(obj, cuts, tuned, baseline, results, tie_tolerance)


def _install_best(
    obj: Any,
    cuts: list[tuple[int, list[ModuleInfo]]],
    tuned: dict[str, Module],
    baseline: dict[SampleKey, float],
    results: list[DepthResult],
    tie_tolerance: float,
) -> AutoTuneResult:
    """Install the best depth when it is faster than `baseline`, otherwise leave `obj` untuned."""
    best = _pick_best(results, tie_tolerance)
    MODULE_REGISTRY.clear()
    if best is None:
        logger.warning("No depth was tuned successfully. The model is left untuned.")
        return AutoTuneResult(obj, None, baseline, results)
    if best.total_latency >= sum(baseline.values()):
        logger.warning("No depth is faster than the untuned model. The model is left untuned.")
        return AutoTuneResult(obj, None, baseline, results)
    logger.info("Selected depth %d.", best.depth)
    best_modules = next(modules for depth, modules in cuts if depth == best.depth)
    return AutoTuneResult(_install(obj, best_modules, tuned), best.depth, baseline, results)


def _distinct_cuts(
    inspector: ModuleInspector,
    min_depth: int,
    max_depth: int,
    total_execution_time: float,
    inspect_iterations: int,
    min_execution_percentage: float | None,
) -> list[tuple[int, list[ModuleInfo]]]:
    """Cuts from `min_depth` to `max_depth`. A cut equal to the previous one is skipped."""
    cuts: list[tuple[int, list[ModuleInfo]]] = []
    previous_cut: list[ModuleInfo] | None = None
    for depth in range(min_depth, max_depth + 1):
        cut = cut_at(inspector.modules, depth)
        if previous_cut is not None and _paths(cut) == _paths(previous_cut):
            # Every module of `cut` is a leaf, so deeper cuts are equal too.
            break
        previous_cut = cut
        selected = modules_info_for(cut, total_execution_time, inspect_iterations).get_modules(
            min_execution_percentage=min_execution_percentage
        )
        if selected and (not cuts or set(_paths(selected)) != set(_paths(cuts[-1][1]))):
            cuts.append((depth, selected))
    return cuts


def _tune_every_depth(
    cuts: list[tuple[int, list[ModuleInfo]]],
    tune_depth: Callable[[int, list[ModuleInfo]], DepthResult],
    start: float,
    time_budget: float | None,
) -> list[DepthResult]:
    """Tune every cut from the shallowest. No cut is started after `time_budget` seconds from `start`."""
    results = []
    for depth, modules in cuts:
        if time_budget is not None and time.monotonic() - start > time_budget:
            logger.warning("Time budget of %ss is used up. Depth %d is not tuned.", time_budget, depth)
            results.append(DepthResult(depth, _paths(modules), error=TIME_BUDGET_ERROR))
            continue
        logger.info("Tuning the model cut at depth %d.", depth)
        results.append(tune_depth(depth, modules))
    return results


def _tune_and_measure_depth(
    obj: Any,
    func: Callable,
    depth: int,
    modules: list[ModuleInfo],
    tuned: dict[str, Module],
    strategy: TuneStrategy | None,
    measure: Callable[[], dict[SampleKey, float]],
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    batch_sizes: list[int],
    max_num_batches_per_batch_size: int | None,
    device: str | torch.device | None,
) -> DepthResult:
    """Tune the modules of one cut that are not in `tuned`, measure the model, and detach every wrapper.

    The model is left as it was before the call. Newly tuned wrappers are added to `tuned`.
    """
    result = DepthResult(depth, _paths(modules))
    new_modules = [m for m in modules if _path(m) not in tuned]
    reused_modules = [m for m in modules if _path(m) in tuned]
    installed: dict[str, Module] = {}
    # Backends such as OpenVINO and TorchAO move the module to CPU when they build. Deeper cuts tune
    # the same tensors again, so they must be back on their device.
    devices = _tensor_devices(modules)
    try:
        if new_modules:
            wrap(obj, new_modules, strategy=strategy)
            installed.update(zip(_paths(new_modules), MODULE_REGISTRY.modules.values(), strict=True))
            _record_and_tune_registered_modules(
                func,
                dataset,
                batch_sizes,
                max_num_batches_per_batch_size,
                device,
                dry_run=False,
                ignore_failing_modules=True,
                offload=False,
            )
            tuned.update(installed)
        for module_info in reused_modules:
            wrapper = tuned[_path(module_info)]
            _attach(module_info, wrapper)
            installed[_path(module_info)] = wrapper
            wrapper.activate()
        failed = [path for path, wrapper in installed.items() if wrapper.state != ModuleState.TUNED]
        if len(failed) == len(modules):
            result.error = "every module failed to tune"
            logger.warning("Depth %d: every module failed to tune.", depth)
            return result
        if failed:
            logger.warning("Depth %d: %d modules failed to tune and run eagerly: %s", depth, len(failed), failed)
        result.latencies = measure()
    except Exception as e:
        logger.exception("Depth %d failed.", depth)
        result.error = f"{type(e).__name__}: {e}"
    finally:
        for module_info in modules:
            wrapper = installed.get(_path(module_info))
            if wrapper is not None:
                wrapper.deactivate()
                _detach(module_info, wrapper)
        _restore_devices(devices)
    return result


TensorDevices = list[tuple[torch.nn.Module, str, torch.device]]


def _tensor_devices(modules: list[ModuleInfo]) -> TensorDevices:
    """The device of every parameter and buffer in the subtrees of `modules`."""
    return [
        (submodule, name, tensor.device)
        for module_info in modules
        for submodule in module_info.module.modules()
        for name, tensor in [*submodule.named_parameters(recurse=False), *submodule.named_buffers(recurse=False)]
    ]


def _restore_devices(devices: TensorDevices) -> None:
    for module, name, device in devices:
        tensor = getattr(module, name)
        if tensor.device == device:
            continue
        if isinstance(tensor, torch.nn.Parameter):
            tensor.data = tensor.data.to(device)
        else:
            setattr(module, name, tensor.to(device))


def _measure(
    func: Callable,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    batch_sizes: list[int],
    max_num_batches_per_batch_size: int | None,
    warmup_iterations: int,
    iterations: int,
) -> dict[SampleKey, float]:
    """Median latency of `func` in seconds for every batch produced from `dataset`."""
    latencies: dict[SampleKey, float] = {}
    batch_index: defaultdict[int, int] = defaultdict(int)
    for batch_size, args, kwargs in samples_generator(dataset, batch_sizes, max_num_batches_per_batch_size):
        key = (batch_size, batch_index[batch_size])
        batch_index[batch_size] += 1
        times = []
        with torch.no_grad():
            for _ in range(warmup_iterations):
                func(*args, **kwargs)
            for _ in range(iterations):
                synchronize()
                start = time.perf_counter()
                func(*args, **kwargs)
                synchronize()
                times.append(time.perf_counter() - start)
        latencies[key] = statistics.median(times)
    return latencies


def _faster(first: dict[SampleKey, float], second: dict[SampleKey, float]) -> dict[SampleKey, float]:
    """The lower latency of every sample."""
    return {key: min(first[key], second[key]) for key in first}


def _pick_best(results: list[DepthResult], tie_tolerance: float) -> DepthResult | None:
    """The shallowest result within `tie_tolerance` of the fastest one."""
    fastest = min((result.total_latency for result in results), default=math.inf)
    if math.isinf(fastest):
        return None
    return next(result for result in results if result.total_latency <= fastest * (1 + tie_tolerance))


def _install(obj: Any, modules: list[ModuleInfo], tuned: dict[str, Module]) -> Any:
    """Attach and activate the tuned wrappers of `modules`, then offload what they no longer need.

    Modules that failed to tune keep their original forward.
    """
    result = obj
    for module_info in modules:
        wrapper = tuned[_path(module_info)]
        if wrapper.state != ModuleState.TUNED:
            continue
        _attach(module_info, wrapper)
        MODULE_REGISTRY.register(wrapper.name, wrapper)
        wrapper.activate()
        wrapper.offload()
        if module_info.parent is None:
            result = wrapper
    return result


def _attach(module_info: ModuleInfo, wrapper: Module) -> None:
    wrapper.attach()
    if module_info.parent is not None:
        module_info.parent.set_wrapped(module_info.name, wrapper)


def _detach(module_info: ModuleInfo, wrapper: Module) -> None:
    wrapper.detach()
    if module_info.parent is not None:
        module_info.parent.set_wrapped(module_info.name, module_info.module)


def _path(module_info: ModuleInfo) -> str:
    return module_info.object_path or ""


def _paths(modules: list[ModuleInfo]) -> list[str]:
    return [_path(m) for m in modules]
