# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""PyTorch module utilities for parameter counting, device management, and memory offloading."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
import torch.nn as nn

from torch_tweak.torch.utils.memory import cleanup_memory


def format_num_parameters(num: int) -> str:
    """Formats a number into human-readable format with appropriate suffixes.

    Args:
        num: The number to format.

    Returns:
        A string representation of the number in human-readable format
        (e.g., "1.2B", "500M", "100K", "50").

    Examples:
        >>> format_num_parameters(1_200_000_000)
        '1.2B'
        >>> format_num_parameters(500_000)
        '500.0K'
        >>> format_num_parameters(50)
        '50'
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num}"


FINGERPRINT_THREADS = 4


def _values_digest(tensor: torch.Tensor) -> bytes:
    values = tensor.detach().reshape(-1).contiguous().cpu()
    return hashlib.sha256(values.view(torch.uint8).numpy()).digest()


def weights_fingerprint(module: nn.Module) -> str:
    """Hashes the name, shape, dtype, and every value of every tensor in `module.state_dict()`.

    The values are hashed in logical order, so the memory layout and the device do not change the fingerprint.
    The tensors are hashed in name order, so the order in which the module registered them does not change it either.
    `FINGERPRINT_THREADS` threads copy and hash the tensors, since both release the GIL.

    >>> layer = nn.Linear(2, 2)
    >>> before = weights_fingerprint(layer)
    >>> before == weights_fingerprint(layer)
    True
    >>> with torch.no_grad():
    ...     _ = layer.weight.add_(1.0)
    >>> before == weights_fingerprint(layer)
    False
    """
    digest = hashlib.sha256()
    state = sorted(module.state_dict().items())
    with ThreadPoolExecutor(max_workers=FINGERPRINT_THREADS) as pool:
        digests = pool.map(_values_digest, (tensor for _, tensor in state))
        for (name, tensor), values_digest in zip(state, digests, strict=True):
            digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype};".encode())
            digest.update(values_digest)
    return digest.hexdigest()[:16]


def count_parameters(module: nn.Module) -> int:
    """Counts the total number of parameters and returns the count as an integer.

    Args:
        module: The PyTorch module to count parameters for.

    Returns:
       The total number of parameters as an integer (e.g., 1200000, 500000, 100).

    Examples:
        >>> import torch.nn as nn
        >>> module = nn.Linear(1000, 100)
        >>> count_parameters(module)
        100100
    """
    num_params = sum(p.numel() for p in module.parameters())
    return num_params


def get_module_device(module: nn.Module) -> Optional["torch.device"]:
    """Get the device of the given module.

    Args:
        module: Module to get the device of.

    Returns:
        The device of module based on parameters. If not parameters, returns None.
    """
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def offload(model: nn.Module, device: str | torch.device = "meta") -> None:
    """Offload model to meta device and freeing all memory.

    Args:
        model: Model to offload and destroy
        device: Device to offload to
    """
    # Step 1: Move model to meta device
    model.to(device)

    # Step 2: Memory cleanup
    cleanup_memory()
