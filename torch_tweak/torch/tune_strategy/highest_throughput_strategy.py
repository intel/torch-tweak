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
   When another backend is faster for some batch sizes, returns a BatchSizeRouterBackend
   that sends those batch sizes to it.
"""

import gc
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torch_tweak.torch.backend import Backend, BatchSizeRouterBackend
from torch_tweak.torch.backend.batch_size_router_backend import routes_by_throughput
from torch_tweak.torch.config import (
    DEFAULT_STABILITY_PERCENTAGE,
    DEFAULT_THROUGHPUT_BACKOFF_LIMIT,
    DEFAULT_THROUGHPUT_CUTOFF_THRESHOLD,
    DEFAULT_WINDOW_SIZE,
)
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.task.find_max_batch_size import (
    calculate_highest_throughput_for_backend,
    get_throughput_per_batch_size,
)
from torch_tweak.torch.task.profiling import (
    MeasuringStopStrategy,
    ModelExecutionTimeMeasuringStrategy,
    ProfilingConfig,
    ProfilingResultEvent,
    ProfilingStopStrategy,
    StableWindowMeasuringStopStrategy,
    ThroughputSaturatedProfilingStopStrategy,
)
from torch_tweak.torch.task.profiling.events import get_inference_events
from torch_tweak.torch.tune_strategy.default_backends import default_backends
from torch_tweak.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from torch_tweak.utils.logging import control_output, log
from torch_tweak.utils.timer import Timer


@dataclass
class HighestThroughputResult:
    """Result of highest throughput strategy."""

    max_batch_size: int
    throughput: float
    backend_details: str


@dataclass
class HighestThroughputStrategyResult:
    """Result of highest throughput strategy."""

    graph_spec_name: str
    highest_throughput_results: list[HighestThroughputResult] = field(default_factory=list)
    measurements: list[ProfilingResultEvent] = field(default_factory=list)


class HighestThroughputStrategy(TuneStrategyFindMaxBatchSizeExtension):
    """Searches and selects the backend with the highest throughput."""

    def __init__(
        self,
        backends: Iterable[Backend] | None = None,
        measurement_stop_strategy: MeasuringStopStrategy | None = None,
        profiling_stop_strategy: ProfilingStopStrategy | None = None,
        **kwargs: Any,
    ):
        """Initializes strategy.

        Args:
            backends: List of backends to tune. Without it, `default_backends(device)` for the tuning device.
            measurement_stop_strategy: Measurement stop strategy.
            profiling_stop_strategy: Profiling stop strategy.
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
        measurements = []
        highest_throughput_results = []

        # Verify if model has batching
        batching = graph_spec.input_spec.has_batch_axis() and graph_spec.get_max_batch_size() > 1

        # Note: As this strategy is extended with FindMaxBatchSizeTuneStrategyExtension,
        # we can assume that max batch size is included in the graph spec.
        max_batch_size = graph_spec.get_max_batch_size()

        profiled: list[tuple[float, int, Backend]] = []
        throughputs: list[tuple[Backend, dict[int, float]]] = []

        backends = list(self._backends or default_backends(device))

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
                    batch_size, throughput, results = calculate_highest_throughput_for_backend(
                        backend, name, graph_spec, data, self._get_profiling_config(batching, max_batch_size)
                    )
                    highest_throughput_results.append(
                        HighestThroughputResult(
                            max_batch_size=batch_size, throughput=throughput, backend_details=backend.describe()
                        )
                    )
                    measurements += results.entries
                    per_batch_size = get_throughput_per_batch_size(
                        get_inference_events(results.entries), self._measurement_stop_strategy
                    )
                    throughputs.append((backend, dict(per_batch_size)))
                    log(
                        "✅ backend profiled - throughput: %.2f samples/s, batch size: %s",
                        throughput,
                        batch_size,
                        depth=2,
                        sink=self._sink,
                    )

                    if not profiled or throughput > max(entry[0] for entry in profiled):
                        log(
                            "🎯 new best throughput for %s is %.2f samples/s, batch size: %s",
                            backend.describe(),
                            throughput,
                            batch_size,
                            depth=2,
                            sink=self._sink,
                        )
                    profiled.append((throughput, batch_size, backend))
                    backend.deactivate()

                except Exception:
                    if backend.is_active:
                        backend.deactivate()
                    log("❌ backend failed (log file: %s)", log_file, depth=2, sink=self._sink)
                    module.to(device)  # move module back to device as failed backend could move it to cpu
            _release_device_memory(device, self._sink)

        if not profiled:
            log(
                "ℹ️ No correct backend found. Backends considered: %s",
                ", ".join([backend.describe() for backend in backends]),
                sink=self._sink,
            )
            raise RuntimeError("No correct backend found with throughput > 0")

        self.results.append(
            HighestThroughputStrategyResult(
                graph_spec_name=graph_spec.name,
                highest_throughput_results=highest_throughput_results,
                measurements=measurements,
            )
        )
        best_backend, best_throughput, best_batch_size = self._activate_fastest(
            module, graph_spec, device, profiled, throughputs
        )
        log("🎯 Strategy %s execution finished:", self.__class__.__name__, sink=self._sink)
        log(
            "✅ Selected %s for module %s and graph spec %s.",
            best_backend.describe(),
            name,
            graph_spec,
            sink=self._sink,
        )
        log("   Batch size: %s, throughput: %.2f samples/s", best_batch_size, best_throughput, sink=self._sink)

        return best_backend

    def _activate_fastest(
        self,
        module: nn.Module,
        graph_spec: GraphSpec,
        device: torch.device,
        profiled: list[tuple[float, int, Backend]],
        throughputs: list[tuple[Backend, dict[int, float]]],
    ) -> tuple[Backend, float, int]:
        """Activates the fastest profiled backend. A rebuild can fail where the profiled build passed."""
        for throughput, batch_size, candidate in sorted(profiled, key=itemgetter(0), reverse=True):
            routes = routes_by_throughput(throughputs, candidate)
            backend = candidate if len(routes) == 1 else BatchSizeRouterBackend(routes, graph_spec.input_spec)
            # A later build can move `module`.
            # `OpenVINOBackend._build` offloads it to CPU.
            # `activate` rebuilds the winner on the weights `module` holds now.
            module.to(device)
            try:
                backend.activate()
            except Exception as exc:
                log("❌ %s failed to activate: %s", backend.describe(), exc, sink=self._sink)
                if backend.is_active:
                    backend.deactivate()
            else:
                return backend, throughput, batch_size
        raise RuntimeError("No profiled backend could be activated")

    def _get_profiling_config(self, batching: bool, max_batch_size: int) -> ProfilingConfig:
        """Gets profiling configuration."""
        return ProfilingConfig(
            batching=batching,
            batch_sizes=[2**n for n in range(max_batch_size.bit_length())],
            measuring_strategy=ModelExecutionTimeMeasuringStrategy(),
            measurement_stop_strategy=self._measurement_stop_strategy,
            profiling_stop_strategy=self._profiling_stop_strategy,
        )

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


def _release_device_memory(device: torch.device, sink) -> None:
    """Frees the cached blocks a deactivated backend left, so the next backend profiles with the same free memory."""
    if torch.device(device).type != "xpu":
        return
    gc.collect()
    torch.xpu.synchronize(device)
    torch.xpu.empty_cache()
    log(
        "💾 device memory: %.2f GiB allocated, %.2f GiB reserved, %.2f GiB peak",
        torch.xpu.memory_allocated(device) / 2**30,
        torch.xpu.memory_reserved(device) / 2**30,
        torch.xpu.max_memory_reserved(device) / 2**30,
        depth=2,
        sink=sink,
    )
