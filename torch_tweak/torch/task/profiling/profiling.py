# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Profiling utilities for Torch Tweak models.

Measurements record the execution time of a single module call.
"""

import logging
from collections.abc import Callable, Generator
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.dataloader import DataLoaderFactory, DatasetLike, samples_generator
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.task.profiling.config import ProfilingConfig
from torch_tweak.torch.task.profiling.events import ProfilingResultEvent, get_inference_events
from torch_tweak.torch.task.profiling.metrics import get_throughput

logger = logging.getLogger(__name__)


@dataclass
class ProfilingResult:
    """Outcome of one profiling run."""

    class Status(Enum):
        """Whether the run finished or raised."""

        SUCCESS = "success"
        FAILURE = "failure"

    status: Status
    throughput_per_batch_size: dict[int, float] = field(default_factory=dict)
    error: Any | None = None


def profile(
    func: Callable,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    profiling_config: ProfilingConfig,
    model_name: str | None = None,
    backend_details: str | None = None,
) -> ProfilingResult:
    """Profile a callable with given dataset.

    Args:
        func: Function to profile.
        dataset: Dataset to profile.
        profiling_config: Configuration for profiling.
        model_name: Name of the model, optional.
        backend_details: Detailed information of the backend, optional.

    Returns:
        ProfilingResult: Outcome of the profiling.
    """
    return _profile(
        func,
        samples_generator(dataset, profiling_config.batch_sizes, max_num_batches_per_batch_size=1),
        profiling_config,
        model_name=model_name,
        backend_details=backend_details,
    )


def profile_backend(
    backend: Backend,
    name: str,
    graph_spec: GraphSpec,
    data: list[Sample],
    profiling_config: ProfilingConfig,
) -> ProfilingResult:
    """Profile a backend with graph spec and data.

    Args:
        backend: Backend to profile.
        name: Name of the model.
        graph_spec: Graph spec of the model. Used to generate batch samples.
        data: Data to profile. Only first sample is used to generate batch samples.
        profiling_config: Profiling configuration.

    Returns:
        ProfilingResult: Outcome of the profiling.
    """

    def generator():
        args, kwargs = data[0]
        if profiling_config.batching:
            for batch_size in profiling_config.batch_sizes:
                args, kwargs = graph_spec.input_spec.make_batch(args, kwargs, batch_size)
                yield batch_size, args, kwargs
        else:
            yield 1, args, kwargs

    return _profile(
        backend.infer,
        generator(),
        profiling_config,
        model_name=name,
        backend_details=backend.describe(),
    )


def _profile(
    func: Callable,
    samples: Generator[tuple[int, list, dict], None, None],
    profiling_config: ProfilingConfig,
    model_name: str | None = None,
    backend_details: str | None = None,
):
    """Profile a model."""
    profiling_stop_strategy = deepcopy(profiling_config.profiling_stop_strategy)
    throughput_per_batch_size: dict[int, float] = {}
    try:
        for batch_size, args, kwargs in samples:
            logger.debug("Profiling %s with backend %s and batch size %d", model_name, backend_details, batch_size)
            new_entries = _run_profiling_batch_size(
                batch_size,
                func,
                (args, kwargs),
                profiling_config=profiling_config,
                model_name=model_name or "model",
                backend_details=backend_details or "backend",
            )

            if new_entries is None:
                break

            # The first calls of a batch size can include compilation. Throughput is the measured
            # window only, and the stop decision uses that same number.
            measured = profiling_config.measurement_stop_strategy.get_events(get_inference_events(new_entries))
            throughput = get_throughput(measured, batch_size)
            throughput_per_batch_size[batch_size] = throughput
            if profiling_stop_strategy.update_and_should_stop(throughput):
                break

        return ProfilingResult(
            status=ProfilingResult.Status.SUCCESS, throughput_per_batch_size=throughput_per_batch_size
        )

    except Exception as e:
        return ProfilingResult(
            status=ProfilingResult.Status.FAILURE,
            error=e,
            throughput_per_batch_size=throughput_per_batch_size,
        )


def _run_profiling_batch_size(
    batch_size: int,
    model: Callable,
    sample: tuple[list, dict],
    profiling_config: ProfilingConfig,
    model_name: str,
    backend_details: str,
) -> list[ProfilingResultEvent] | None:
    entries = []
    measurement_stop_strategy = deepcopy(profiling_config.measurement_stop_strategy)
    try:
        while True:
            # make call idempotent (no side effects) since model can be stateful like LLM KV cache
            new_entries = profiling_config.measuring_strategy.do_measurement(
                batch_size, model, deepcopy(sample), backend_details=backend_details, model_name=model_name
            )
            entries += new_entries

            if measurement_stop_strategy.update_and_should_stop(get_inference_events(new_entries)):
                break
    except torch.OutOfMemoryError:
        logger.warning(
            "Out of memory error while profiling %s with backend %s and batch size %d",
            model_name,
            backend_details,
            batch_size,
        )
        return None
    except Exception as e:
        logger.warning(
            "Error while profiling %s with backend %s and batch size %d: %s",
            model_name,
            backend_details,
            batch_size,
            e,
        )
        return None

    # Note: keeping and returning all entries
    return entries
