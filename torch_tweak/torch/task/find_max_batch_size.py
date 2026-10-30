# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Find max batch size for a model."""

import logging
from pathlib import Path

import torch
import torch.nn as nn

from torch_tweak.torch.backend import TorchEagerBackend
from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.task.profiling.config import ProfilingConfig
from torch_tweak.torch.task.profiling.profiling import ProfilingResult, profile_backend
from torch_tweak.utils.logging import control_output
from torch_tweak.utils.permissions import secure_mkdir
from torch_tweak.utils.trace import annotate

logger = logging.getLogger(__name__)


@annotate(domain="Torch Tweak")
def find_max_batch_size(
    module: nn.Module,
    name: str,
    graph_spec: GraphSpec,
    data: list[Sample],
    profiling_config: ProfilingConfig,
    device: torch.device,
    cache_dir: Path,
) -> tuple[int, float, dict[int, float]]:
    """Finds max batch size for Torch Compile as a baseline.

    Uses profiling with highest throughput strategy to find max batch size.

    Note: This function expects user to set profiling_config.max_batch_size to the highest batch size they want to profile.

    Args:
        module: Model to find max batch size for.
        name: Name of the model.
        graph_spec: Graph spec of the model.
        data: Data to profile.
        profiling_config: Profiling configuration.
        torch_backend: Backend to use for the find max batch size. If not provided, Torch Eager backend will be used.
        device: Device to use for the calculation.
        cache_dir: Cache directory to store the backend artifacts.
    """
    backend = TorchEagerBackend()
    backend_cache_dir = cache_dir / backend.key()
    log_file = _log_file(backend_cache_dir, "build.log")
    with control_output(log_file=log_file):
        backend.build(module, graph_spec, data, device, backend_cache_dir)
    return calculate_highest_throughput_for_backend(backend, name, graph_spec, data, profiling_config)


def calculate_highest_throughput_for_backend(
    backend: Backend,
    name: str,
    graph_spec: GraphSpec,
    data: list[Sample],
    profiling_config: ProfilingConfig,
) -> tuple[int, float, dict[int, float]]:
    """Calculates highest throughput for a backend.

    Args:
        module: Model to calculate highest throughput for.
        name: Name of the model.
        graph_spec: Graph spec of the model.
        data: Data to profile.
        profiling_config: Profiling configuration.
        backend: Backend to use for the calculation.
        device: Device to use for the calculation.

    Returns:
        Tuple containing:
            - Batch size with highest throughput.
            - Throughput for the batch size.
            - Throughput of every finished batch size.
    """
    profiling_result = profile_backend(
        backend,
        name,
        graph_spec,
        data,
        profiling_config,
    )

    if profiling_result.status != ProfilingResult.Status.SUCCESS:
        logger.debug("Profiling failed for %s", backend.describe(), exc_info=profiling_result.error)
        raise profiling_result.error

    throughput_per_batch_size = profiling_result.throughput_per_batch_size
    if not throughput_per_batch_size:
        raise ValueError(f"No throughput data found for backend {backend.describe()}")

    batch_size, throughput = max(throughput_per_batch_size.items(), key=lambda item: item[1])
    return batch_size, throughput, throughput_per_batch_size


def _log_file(cache_dir: Path, filename: str) -> Path:
    secure_mkdir(cache_dir)
    log_file = cache_dir / filename
    return log_file
