# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Simple accelerator utilities (Intel XPU-first)."""

import re
from logging import getLogger
from pathlib import Path

import torch

from torch_tweak.troubleshooting import COMMON_ISSUES_URL, XPU_TORCH_INSTALL

logger = getLogger(__name__)

DRM_DIR = Path("/sys/class/drm")
_INTEL_PCI_VENDOR = "0x8086"
_RUN_ON_CPU = (
    'To tune on CPU instead, or if PyTorch XPU does not support this GPU, pass device="cpu" to tune() '
    "or set TORCH_TWEAK_DEVICE=cpu."
)


def has_intel_gpu() -> bool:
    """Check if the system has an Intel GPU, whether or not PyTorch can use it.

    Returns:
        bool: True if a DRM render node in `DRM_DIR` belongs to a device with the Intel PCI vendor id.
    """
    return any(
        vendor.read_text(encoding="utf-8").strip() == _INTEL_PCI_VENDOR
        for vendor in DRM_DIR.glob("renderD*/device/vendor")
    )


def auto_device() -> torch.device:
    """Pick the device for `device="auto"`.

    An XPU build of PyTorch or an Intel GPU in the system selects xpu:0. Otherwise the device is cpu.

    Returns:
        torch.device: xpu:0 or cpu.

    Raises:
        RuntimeError: If xpu:0 is selected but XPU is not available.
    """
    if torch.version.xpu is None and not has_intel_gpu():
        logger.info("No Intel GPU and no XPU build of PyTorch found. Tuning on CPU.")
        return torch.device("cpu")

    assert_is_available()
    return torch.device("xpu:0")


def assert_is_available():
    """Assert that Intel XPU is available.

    Code should be testable even if XPU is not available.

    By providing a assertion function, we allow mocking the XPU availability.

    Raises:
        RuntimeError: If XPU is not available.
    """
    if torch.version.xpu is None:
        raise RuntimeError(
            f"PyTorch {torch.__version__} has no XPU support, so Torch Tweak cannot use an Intel GPU.\n"
            f"Install the XPU build of PyTorch:\n    {XPU_TORCH_INSTALL}\n"
            f"{_RUN_ON_CPU}\n"
            f"See {COMMON_ISSUES_URL}"
        )
    if not torch.xpu.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} supports XPU, but no Intel GPU is visible.\n"
            "Check the GPU driver and membership in the render group.\n"
            f"{_RUN_ON_CPU}\n"
            f"See {COMMON_ISSUES_URL}"
        )


def is_available():
    """Check if Intel XPU is available.

    Returns:
        bool: True if XPU is available.
    """
    xpu = getattr(torch, "xpu", None)
    return bool(xpu is not None and xpu.is_available())


def synchronize():
    """Synchronize all XPU devices if available."""
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available() and hasattr(xpu, "synchronize"):
        xpu.synchronize()


def get_device(device: str | torch.device) -> str:
    """Normalize an XPU device string.

    Returns:
        str: The current XPU device in format 'xpu:<device_id>'.
    """
    if isinstance(device, torch.device):
        if device.type != "xpu":
            raise ValueError("Device must be 'xpu' type")
        return f"{device.type}:{device.index or 0}"
    elif isinstance(device, str):
        pattern = r"^xpu:(\d+)$"
        match = re.match(pattern, device)
        if match:
            return device

        pattern = r"^xpu$"
        match = re.match(pattern, device)
        if match:
            return "xpu:0"

    raise ValueError("device must be 'xpu' or in format 'xpu:<device_id>'")


def set_device(device: str | torch.device):
    """Set the current XPU device.

    Args:
        device: Device string or torch.device object. If string is 'xpu', it will be converted to 'xpu:0'.
    """
    device = get_device(device)
    xpu = getattr(torch, "xpu", None)
    if xpu is None or not hasattr(xpu, "set_device"):
        raise RuntimeError("torch.xpu.set_device is not available in this PyTorch build.")
    xpu.set_device(device)
