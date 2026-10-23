# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Measuring strategies for profiling."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from torch_tweak.torch.task.profiling.events import ProfilingResultEvent
from torch_tweak.torch.utils.xpu import synchronize


class MeasuringStrategy(ABC):
    """Strategy how do the measurement and create a ProfilingResultEvent(s)."""

    @abstractmethod
    def do_measurement(
        self, batch_size: int, model: Callable, sample: tuple[list, dict], **kwargs
    ) -> list[ProfilingResultEvent]:
        """Do the measurement and create a ProfilingResultEvent(s).

        Args:
            batch_size: Batch size of the measurement.
            model: Model to measure.
            sample: Sample to measure.
            **kwargs: Additional keyword arguments.

        Returns:
            List of ProfilingResultEvent(s).
        """
        pass


class ModelExecutionTimeMeasuringStrategy(MeasuringStrategy):
    """Strategy to measure execution time."""

    counter: int = 0

    def __init__(self):
        """Initialize the measuring strategy."""

    def do_measurement(
        self, batch_size: int, model: Callable, sample: tuple[list, dict], **measurement_kwargs
    ) -> list[ProfilingResultEvent]:
        """Do the measurement and create a ProfilingResultEvent(s) for the model."""
        args, kwargs = sample

        synchronize()
        start = time.monotonic_ns()
        model(*args, **kwargs)
        # Replay returns after submit. The device may still be running.
        synchronize()
        end = time.monotonic_ns()

        ModelExecutionTimeMeasuringStrategy.counter += 1

        return [
            ProfilingResultEvent(
                measurement_id=ModelExecutionTimeMeasuringStrategy.counter,
                timestamp=start,
                batch_size=batch_size,
                # One module call. Token phases and events outside the module are not recorded.
                phase="inference",
                execution_time=end - start,
                backend_details=measurement_kwargs.get("backend_details", None),
                model_name=measurement_kwargs.get("model_name", None),
            )
        ]
