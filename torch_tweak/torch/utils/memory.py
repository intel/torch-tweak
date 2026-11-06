# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""PyTorch memory utilities for garbage collection and XPU cache cleanup."""

import gc
from collections.abc import Callable
from ctypes import CDLL, util

import torch

from torch_tweak.utils.logging import log


def free_device_memory(device: torch.device) -> None:
    """Drops the compile caches, collects garbage, and returns the unused cached XPU blocks to the driver."""
    torch._dynamo.reset()
    torch.compiler.reset()
    gc.collect()
    if device.type == "xpu":
        torch.xpu.synchronize(device)
        torch.xpu.empty_cache()


def release_device_memory(device: torch.device, sink: Callable) -> None:
    """Frees the memory nothing references anymore, so the next build starts with the same free memory.

    A failed build stays in `BackendState.INIT`, so `Backend.deactivate` never frees its memory.
    Call it after the `except` block, because the active exception holds the failed frames.
    The logged peak counts from the previous call.
    """
    free_device_memory(device)
    if device.type != "xpu":
        return
    free, total = torch.xpu.mem_get_info(device)
    log(
        "💾 device memory: %.2f GiB allocated, %.2f GiB reserved, %.2f GiB peak reserved, %.2f of %.2f GiB free",
        torch.xpu.memory_allocated(device) / 2**30,
        torch.xpu.memory_reserved(device) / 2**30,
        torch.xpu.max_memory_reserved(device) / 2**30,
        free / 2**30,
        total / 2**30,
        depth=2,
        sink=sink,
    )
    torch.xpu.reset_peak_memory_stats(device)


def cleanup_memory() -> None:
    """Perform garbage collection and XPU cache cleanup."""
    cpu_cleanup()
    gpu_cleanup()


def gpu_cleanup():
    """Perform XPU cache cleanup."""
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        if hasattr(xpu, "synchronize"):
            xpu.synchronize()
        if hasattr(xpu, "empty_cache"):
            xpu.empty_cache()


def cpu_cleanup():
    """Perform CPU memory cleanup."""
    gc.collect()
    cpu_cleanup_low_level()


def cpu_cleanup_low_level():
    """Perform low-level CPU memory cleanup."""
    import platform

    try:
        if platform.system() == "Linux":
            CDLL(util.find_library("c")).malloc_trim(0)
    except (OSError, AttributeError):
        # Silently ignore if platform-specific cleanup is not available
        pass
