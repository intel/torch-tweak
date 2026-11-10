# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Simple accelerator utilities (Intel XPU-first)."""

import re

import torch


def assert_is_available():
    """Assert that Intel XPU is available.

    Code should be testable even if XPU is not available.

    By providing a assertion function, we allow mocking the XPU availability.

    Raises:
        RuntimeError: If XPU is not available.
    """
    if not torch.xpu._is_compiled():
        raise RuntimeError(
            "PyTorch was not built with XPU support. Check the Common Issues section in the installation guide: "
            "https://github.com/intel/torch-tweak/blob/main/docs/learn/install.md#common-issues"
        )
    if not torch.xpu.is_available():
        raise RuntimeError(
            "No Intel GPU is available. Check the Common Issues section in the installation guide: "
            "https://github.com/intel/torch-tweak/blob/main/docs/learn/install.md#common-issues"
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
