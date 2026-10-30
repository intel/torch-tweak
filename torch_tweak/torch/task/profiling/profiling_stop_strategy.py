# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Profiling stop strategies for profiling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from torch_tweak.torch.task.profiling.metrics import is_throughput_saturated


class ProfilingStopStrategy(ABC):
    """Strategy when to stop profiling."""

    @abstractmethod
    def update_and_should_stop(self, throughput: float) -> bool:
        """Record this throughput and report whether profiling should stop.

        Args:
            throughput: Throughput of the batch size just measured, in samples per second.

        Returns:
            True if the profiling should be stopped, False otherwise.
        """
        pass


@dataclass
class AllSamplesProfilingStopStrategy(ProfilingStopStrategy):
    """Strategy to do not stop profiling until all samples are processed."""

    def update_and_should_stop(self, _throughput: float) -> bool:
        """Record this throughput and report whether profiling should stop."""
        return False


@dataclass
class ThroughputSaturatedProfilingStopStrategy(ProfilingStopStrategy):
    """Strategy to stop profiling when throughput is saturated.

    Additionally, backoff policy is introduced, allowing for few more measurements after saturation is detected.
    This is useful to avoid stopping profiling too early, when the model is still not fully optimized.

    Args:
        throughput_cutoff_threshold: Threshold for throughput saturation.
        throughput_backoff_limit: Number of measurements after saturation is detected to continue profiling. 0 or negative value means no backoff.
    """

    throughput_cutoff_threshold: float
    throughput_backoff_limit: int = 0

    _best_throughput: float | None = None
    _backoff_counter: int = 0

    def update_and_should_stop(self, throughput: float) -> bool:
        """Record this throughput and report whether profiling should stop."""
        is_saturated = is_throughput_saturated(throughput, self.throughput_cutoff_threshold, self._best_throughput)

        if not is_saturated:
            self._best_throughput = throughput

        if self.throughput_backoff_limit <= 0:
            return is_saturated

        if is_saturated:
            if self._backoff_counter >= self.throughput_backoff_limit:
                return True  # stop profiling
            self._backoff_counter += 1
        else:
            # resetting backoff counter, maybe better results are coming
            self._backoff_counter = 0

        return False  # continue profiling
