# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Highest throughput tune strategy.

1. Finds max batch size for Torch Eager as a baseline.
2. Runs all backends with given max batch size.
3. Finds the highest throughput for each backend.
4. Returns the backend with the highest throughput that activates again.

A probe before and after each measurement shows whether another process used the device.
A measurement on a shared device runs again, up to `QUIET_DEVICE_ATTEMPTS` times.
"""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from operator import attrgetter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torch_tweak.torch.backend import Backend
from torch_tweak.torch.config import (
    DEFAULT_STABILITY_PERCENTAGE,
    DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
)
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.task.find_max_batch_size import calculate_highest_throughput_for_backend
from torch_tweak.torch.task.profiling import (
    MeasuringStopStrategy,
    ModelExecutionTimeMeasuringStrategy,
    ProfilingConfig,
    ProfilingStopStrategy,
    StableWindowMeasuringStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from torch_tweak.torch.tune_strategy.default_backends import default_backends
from torch_tweak.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from torch_tweak.torch.utils.device_load import DeviceLoadProbe, is_quiet
from torch_tweak.utils.logging import control_output, log
from torch_tweak.utils.timer import Timer

QUIET_DEVICE_ATTEMPTS = 3


@dataclass
class HighestThroughputResult:
    """Result of one backend in the highest throughput strategy."""

    max_batch_size: int
    throughput: float
    backend_details: str
    throughput_per_batch_size: dict[int, float]


@dataclass
class HighestThroughputStrategyResult:
    """Result of highest throughput strategy."""

    graph_spec_name: str
    highest_throughput_results: list[HighestThroughputResult] = field(default_factory=list)


@dataclass
class _ProfiledBackend:
    """A backend that passed profiling, with the batch size of its highest throughput."""

    backend: Backend
    throughput: float
    batch_size: int


class HighestThroughputStrategy(TuneStrategyFindMaxBatchSizeExtension):
    """Searches and selects the backend with the highest throughput."""

    def __init__(
        self,
        backends: Iterable[Backend] | None = None,
        measurement_stop_strategy: MeasuringStopStrategy | None = None,
        profiling_stop_strategy: ProfilingStopStrategy | None = None,
        device_probe: DeviceLoadProbe | None = None,
        **kwargs: Any,
    ):
        """Initializes strategy.

        Args:
            backends: List of backends to tune. Without it, `default_backends(device)` for the tuning device.
            measurement_stop_strategy: Measurement stop strategy.
            profiling_stop_strategy: Profiling stop strategy.
            device_probe: Times a fixed workload around each measurement. Defaults to `DeviceLoadProbe()`.
            kwargs: Additional arguments for the parent class
        """
        super().__init__(**kwargs)
        self._backends = backends
        self._measurement_stop_strategy = measurement_stop_strategy or StableWindowMeasuringStopStrategy(
            window_size=DEFAULT_WINDOW_SIZE,
            stability_percentage=DEFAULT_STABILITY_PERCENTAGE,
        )
        self._profiling_stop_strategy = profiling_stop_strategy or ThroughputSaturatedProfilingStopStrategy(
            throughput_cutoff_threshold=DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
            throughput_backoff_limit=DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
        )
        self._device_probe = device_probe or DeviceLoadProbe()
        self.results: list[HighestThroughputStrategyResult] = []

    def _tune(
        self,
        module: nn.Module,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        device: torch.device,
        cache_dir: Path,
    ) -> Backend:
        """Tunes given torch module with provided graph_spec and data."""
        log(
            "⏳ Executing strategy `%s` on module `%s` (graph: %s)",
            self.__class__.__name__,
            name,
            graph_spec.name,
            sink=self._sink,
        )
        highest_throughput_results = []

        # Verify if model has batching
        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1

        # Note: As this strategy is extended with FindMaxBatchSizeTuneStrategyExtension,
        # we can assume that max batch size is included in the graph spec.
        max_batch_size = graph_spec.get_max_batch_size()

        profiled: list[_ProfiledBackend] = []
        probes: list[float] = []
        measured_windows: list[tuple[float, float]] = []

        backends = self.candidate_backends(device)

        # Run all backends with given max batch size.
        for backend in backends:
            backend_cache_dir = cache_dir / backend.key()
            log_file = self._log_file(backend_cache_dir, "build.log")

            with Timer(sink=self._sink, depth=2):
                try:
                    log("🤖 backend: %s", backend.describe(), sink=self._sink)
                    log("🔄 in progress...please wait", depth=2, sink=self._sink)
                    with control_output(log_file=log_file):
                        backend = deepcopy(backend)
                        backend.build(module, graph_spec, deepcopy(data), device, backend_cache_dir)
                    log("✅ backend built", depth=2, sink=self._sink)
                    self.check_correctness(backend, name, graph_spec, data)
                    log("✅ backend validated", depth=2, sink=self._sink)
                    batch_size, throughput, throughput_per_batch_size = self._measure_on_quiet_device(
                        backend,
                        name,
                        graph_spec,
                        data,
                        self._get_profiling_config(batching, max_batch_size),
                        device,
                        probes,
                        measured_windows,
                    )
                    highest_throughput_results.append(
                        HighestThroughputResult(
                            max_batch_size=batch_size,
                            throughput=throughput,
                            backend_details=backend.describe(),
                            throughput_per_batch_size=throughput_per_batch_size,
                        )
                    )
                    log(
                        "✅ backend profiled - throughput: %.2f samples/s, batch size: %s",
                        throughput,
                        batch_size,
                        depth=2,
                        sink=self._sink,
                    )

                    if not profiled or throughput > max(entry.throughput for entry in profiled):
                        log(
                            "🎯 new best throughput for %s is %.2f samples/s, batch size: %s",
                            backend.describe(),
                            throughput,
                            batch_size,
                            depth=2,
                            sink=self._sink,
                        )
                    profiled.append(_ProfiledBackend(backend, throughput, batch_size))
                    backend.deactivate()

                except Exception:
                    if backend.is_active:
                        backend.deactivate()
                    log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)
                    module.to(device)  # move module back to device as failed backend could move it to cpu
        self._device_probe.release()
        self.tuned_on_quiet_device = all(is_quiet(window, min(probes)) for window in measured_windows)
        if not self.tuned_on_quiet_device:
            log("⚠️ another process used the device while backends were measured", sink=self._sink)

        if not profiled:
            log(
                "ℹ️ No correct backend found. Backends considered: %s",
                ", ".join([backend.describe() for backend in backends]),
                sink=self._sink,
            )
            raise RuntimeError("No correct backend found")

        self.results.append(
            HighestThroughputStrategyResult(
                graph_spec_name=graph_spec.name,
                highest_throughput_results=highest_throughput_results,
            )
        )
        best = self._activate_fastest(module, device, profiled)
        log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
        log(
            "✅ Selected %s for module %s and graph spec %s.",
            best.backend.describe(),
            name,
            graph_spec,
            sink=self._sink,
        )
        log("   Batch size: %s, throughput: %.2f samples/s", best.batch_size, best.throughput, sink=self._sink)

        return best.backend

    def _measure_on_quiet_device(
        self,
        backend: Backend,
        name: str,
        graph_spec: GraphSpec,
        data: list[Sample],
        profiling_config: ProfilingConfig,
        device: torch.device,
        probes: list[float],
        measured_windows: list[tuple[float, float]],
    ) -> tuple[int, float, dict[int, float]]:
        """Measures `backend` until the probes around the measurement show a quiet device.

        Appends every probe to `probes` and the probes around the kept measurement to `measured_windows`.
        """
        for attempt in range(1, QUIET_DEVICE_ATTEMPTS + 1):
            before = self._device_probe.seconds(device)
            measured = calculate_highest_throughput_for_backend(backend, name, graph_spec, data, profiling_config)
            window = (before, self._device_probe.seconds(device))
            probes.extend(window)
            if is_quiet(window, min(probes)):
                break
            log(
                "⚠️ another process used the device during measurement %d of %d: probes took %.1f and %.1f ms, "
                "the fastest %.1f ms",
                attempt,
                QUIET_DEVICE_ATTEMPTS,
                window[0] * 1e3,
                window[1] * 1e3,
                min(probes) * 1e3,
                depth=2,
                sink=self._sink,
            )
        measured_windows.append(window)
        return measured

    def _activate_fastest(
        self, module: nn.Module, device: torch.device, profiled: list[_ProfiledBackend]
    ) -> _ProfiledBackend:
        """Activates the fastest profiled backend. A rebuild can fail where the profiled build passed."""
        error: Exception | None = None
        for candidate in sorted(profiled, key=attrgetter("throughput"), reverse=True):
            # A later build can move `module`.
            # `OpenVINOBackend._build` offloads it to CPU.
            # `activate` rebuilds `candidate.backend` on the weights `module` holds now.
            module.to(device)
            try:
                candidate.backend.activate()
            except Exception as exc:
                log("❌ %s failed to activate: %s", candidate.backend.describe(), exc, sink=self._sink)
                error = exc
            else:
                return candidate
        raise RuntimeError("No profiled backend could be activated") from error

    def _get_profiling_config(self, batching: bool, max_batch_size: int) -> ProfilingConfig:
        """Gets profiling configuration."""
        return ProfilingConfig(
            batching=batching,
            batch_sizes=[2**n for n in range(max_batch_size.bit_length())],
            measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
            measurement_stop_strategy=self._measurement_stop_strategy,
            profiling_stop_strategy=self._profiling_stop_strategy,
        )

    def candidate_backends(self, device: torch.device) -> list[Backend]:
        """The backends this strategy measures on `device`."""
        return list(self._backends or default_backends(device))

    def _describe_parts(self) -> list[str]:
        """Returns the parts of the description."""
        backends = (
            [backend.describe() for backend in self._backends] if self._backends else ["default_backends(device)"]
        )
        return [
            "name: Highest Throughput Strategy",
            "description: evaluate all backends, return backend with highest throughput",
            "backends:",
            *[f"  {backend}" for backend in backends],
        ]
