# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Task module."""

from torch_tweak.torch.task.profiling.config import ProfilingConfig
from torch_tweak.torch.task.profiling.events import ProfilingResultEvent, get_inference_events
from torch_tweak.torch.task.profiling.measuring_stop_strategy import (
    MeasuringStopStrategy,
    NumStepsMeasuringStopStrategy,
    StableWindowMeasuringStopStrategy,
)
from torch_tweak.torch.task.profiling.measuring_strategy import MeasuringStrategy, ModelExecutionTimeMeasuringStrategy
from torch_tweak.torch.task.profiling.profiling import ProfilingResult, profile

__all__ = [
    "MeasuringStrategy",
    "ModelExecutionTimeMeasuringStrategy",
    "MeasuringStopStrategy",
    "NumStepsMeasuringStopStrategy",
    "StableWindowMeasuringStopStrategy",
    "ProfilingConfig",
    "ProfilingResultEvent",
    "get_inference_events",
    "ProfilingResult",
    "profile",
]
