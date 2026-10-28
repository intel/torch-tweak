# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Default backends and strategy for a device."""

import torch

from torch_tweak.torch.backend import (
    Backend,
    OpenVINOBackend,
    TorchEagerBackend,
    TorchInductorBackend,
    XPUGraphBackend,
)
from torch_tweak.torch.tune_strategy.highest_throughput_strategy import HighestThroughputStrategy


def default_backends(device: str | torch.device) -> list[Backend]:
    """Every backend that runs on `device` and keeps the results of the model.

    TorchAO is left out because quantization changes the results.
    """
    device_type = torch.device(device).type
    if device_type == "xpu":
        return [TorchInductorBackend(), XPUGraphBackend(), OpenVINOBackend(), TorchEagerBackend()]
    if device_type == "cpu":
        return [TorchInductorBackend(), OpenVINOBackend(), TorchEagerBackend()]
    raise ValueError(f"No default backends for device {device_type}.")


def default_strategy(device: str | torch.device) -> HighestThroughputStrategy:
    """`HighestThroughputStrategy` over `default_backends(device)`: every backend is measured and the fastest wins."""
    return HighestThroughputStrategy(backends=default_backends(device))
