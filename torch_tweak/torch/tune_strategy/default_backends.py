# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Default backends for a device."""

import torch

from torch_tweak.torch.backend import (
    Backend,
    TorchEagerBackend,
    TorchInductorBackend,
    XPUGraphBackend,
)


def default_backends(device: str | torch.device) -> list[Backend]:
    """Every backend that runs on `device` and keeps the results of the model.

    `FirstWinsStrategy` tries them in this order.
    TorchAO is left out because quantization changes the results.
    OpenVINO is left out when the openvino package is not installed.
    """
    device_type = torch.device(device).type
    if device_type == "xpu":
        return [XPUGraphBackend(), TorchInductorBackend(), *_openvino(), TorchEagerBackend()]
    if device_type == "cpu":
        return [TorchInductorBackend(), *_openvino(), TorchEagerBackend()]
    raise ValueError(f"No default backends for device {device_type}.")


def _openvino() -> list[Backend]:
    try:
        from torch_tweak.torch.backend.openvino import OpenVINOBackend
    except ImportError:
        return []
    return [OpenVINOBackend()]
