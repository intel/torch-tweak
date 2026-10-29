# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Default backends for a device."""

import torch

from torch_tweak.torch.backend import (
    Backend,
    OpenVINOBackend,
    TorchEagerBackend,
    TorchInductorBackend,
    XPUGraphBackend,
)


def default_backends(device: str | torch.device) -> list[Backend]:
    """Every backend that runs on `device` and keeps the results of the model.

    `FirstWinsStrategy` tries them in this order.
    TorchAO is left out because quantization changes the results.
    """
    device_type = torch.device(device).type
    if device_type == "xpu":
        return [XPUGraphBackend(), TorchInductorBackend(), OpenVINOBackend(), TorchEagerBackend()]
    if device_type == "cpu":
        return [TorchInductorBackend(), OpenVINOBackend(), TorchEagerBackend()]
    raise ValueError(f"No default backends for device {device_type}.")
