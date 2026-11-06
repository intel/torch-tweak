# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""PyTorch module utilities for parameter counting, device management, and memory offloading."""

import hashlib
import inspect
from collections.abc import Callable, ValuesView
from inspect import Parameter
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


WEIGHT_SAMPLES_PER_TENSOR = 16


def weights_fingerprint(module: nn.Module) -> str:
    """Hashes the name, shape, dtype, and evenly spaced values of every tensor in `module.state_dict()`.

    The sampled values tell apart checkpoints of one architecture, such as two finetunes of one model.

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
    samples = []
    for name, tensor in module.state_dict().items():
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype};".encode())
        flat = tensor.detach().reshape(-1)
        if flat.numel():
            steps = torch.arange(WEIGHT_SAMPLES_PER_TENSOR, device=flat.device)
            index = steps * (flat.numel() - 1) // (WEIGHT_SAMPLES_PER_TENSOR - 1)
            samples.append(flat[index].float().cpu())
    if samples:
        digest.update(torch.cat(samples).numpy().tobytes())
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


def _callable_name(func: Callable) -> str:
    """Best-effort name for any callable.

    ``functools.partial`` objects and instances with ``__call__`` have no ``__name__``,
    so fall back to their type name.

    >>> import functools
    >>> _callable_name(functools.partial(print))
    'partial'
    """
    return getattr(func, "__name__", type(func).__name__)


def get_arguments_names(func: Callable) -> tuple[list[str], list[str]]:
    """Split callable parameter names by :mod:`inspect` parameter kind.

    The first list is ``POSITIONAL_ONLY`` names (before ``/``). The second list is
    ``POSITIONAL_OR_KEYWORD`` and ``KEYWORD_ONLY`` names (callable inputs for export).
    ``*args`` and ``**kwargs`` name no individual argument, so they appear in neither list.

    A signature yielding no name at all raises ``ValueError``: callers key ``torch.export``
    inputs by name, and an empty result would silently drop every input instead. This covers
    both an empty signature and one made up solely of ``*args``/``**kwargs`` -- the latter
    typically means a wrapper (``torch.compile``, ``nn.DataParallel``, a decorator without
    :func:`functools.wraps`) replaced ``forward`` and lost the original signature.

    Callers normally pass a *bound* ``forward`` (``my_module.forward``), where ``self`` is
    already consumed and therefore absent from the result. ``nn.Module`` allows any callable
    to be bound to ``forward`` -- ``wrapt`` proxies, ``functools.partial``, or objects with
    ``__call__`` -- and all of them work as long as :func:`inspect.signature` describes them.

    Args:
        func: Any callable, typically a bound ``forward`` such as ``my_module.forward``.

    Returns:
        ``(positional_only_names, or_keyword_and_keyword_only_names)``.

    Examples:
        >>> import torch.nn as nn
        >>> class _Lin(nn.Module):
        ...     def forward(self, x, y=1):
        ...         return x
        >>> get_arguments_names(_Lin().forward)
        ([], ['x', 'y'])
        >>> get_arguments_names(_Lin.forward)
        ([], ['self', 'x', 'y'])
        >>> def _positional_only(a, /, b, c=0):
        ...     pass
        >>> get_arguments_names(_positional_only)
        (['a'], ['b', 'c'])
        >>> class _KwOnly:
        ...     def __call__(self, x, *, mask=None):
        ...         return x
        >>> get_arguments_names(_KwOnly())
        ([], ['x', 'mask'])
        >>> def _varargs(x, *args, **kwargs):
        ...     pass
        >>> get_arguments_names(_varargs)
        ([], ['x'])
        >>> def _empty():
        ...     pass
        >>> get_arguments_names(_empty)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Callable '_empty()' has no named parameters!
        >>> def _only_varargs(*args, **kwargs):
        ...     pass
        >>> get_arguments_names(_only_varargs)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Callable '_only_varargs(*args, **kwargs)' has no named parameters!
    """
    signature: inspect.Signature = inspect.signature(func)
    params: ValuesView[Parameter] = signature.parameters.values()

    posonly_arg_names: list[str] = [p.name for p in params if p.kind is Parameter.POSITIONAL_ONLY]
    therest_arg_names: list[str] = [
        p.name for p in params if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    ]

    if not posonly_arg_names and not therest_arg_names:
        raise ValueError(f"Callable '{_callable_name(func)}{signature}' has no named parameters!")

    return posonly_arg_names, therest_arg_names


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
