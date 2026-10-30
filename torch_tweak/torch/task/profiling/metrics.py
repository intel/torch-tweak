# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Metrics for profiling."""

import numpy as np

from torch_tweak.torch.task.profiling.events import ProfilingResultEvent


def is_throughput_saturated(
    throughput: float,
    throughput_cutoff_threshold: float,
    previous_throughput: float | None,
) -> bool:
    """Validate if throughput saturated between consecutive batch sizes.

    Args:
        throughput: Throughput of the batch size just measured, in samples per second.
        throughput_cutoff_threshold: Threshold for throughput saturation.
        previous_throughput: Best throughput so far, or None before the first batch size.

    Returns:
        True when throughput saturated between consecutive samples. False when not yet saturated.
    """
    if previous_throughput is None:
        return False

    return throughput * (1 - throughput_cutoff_threshold) < previous_throughput


def get_batch_size(results: list[ProfilingResultEvent]) -> int:
    """Get batch size from profiling results.

    Args:
        results: List of profiling results.

    Returns:
        Batch size.
    """
    batch_sizes = [result.batch_size for result in results]
    if len(set(batch_sizes)) != 1:
        raise RuntimeError("All batch sizes must be the same")
    return batch_sizes[0]


def get_mean_executions_per_second(results: list[ProfilingResultEvent]) -> float:
    """Get execution time per second from profiling results.

    Args:
        results: List of profiling results.

    Returns:
        Executions per second based on mean execution time.
    """
    mean_execution_time = np.mean([result.execution_time for result in results])
    return 1e9 / mean_execution_time


def get_throughput(results: list[ProfilingResultEvent], batch_size: int | None = None) -> float:
    """Get throughput from profiling results.

    Args:
        results: List of profiling results.
        batch_size: Batch size to use for throughput calculation. If not provided, it will be inferred from the results.

    Returns:
        Throughput in samples per second.
    """
    batch_size = batch_size or get_batch_size(results)
    execution_time_per_second = get_mean_executions_per_second(results)
    return batch_size * execution_time_per_second
