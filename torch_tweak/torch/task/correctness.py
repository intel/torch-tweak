# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Correctness checking utilities."""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils._pytree import tree_flatten

from torch_tweak.torch.module.tensor_spec import TensorSpec


class CorrectnessValueError(ValueError):
    """Error raised when value is not finite i.e. NaN or infinity."""


class CorrectnessTensorShapeError(ValueError):
    """Error raised when tensor shapes do not match."""


class CorrectnessMismatchError(ValueError):
    """Error raised when outputs differ from the expected outputs."""


def check_output_correctness(output: Any, name: str = "output", depth: int = 0):  # noqa: C901
    """Check if model outputs contain any NaN or infinity values.

    Args:
        output: Model outputs. Dict, list, tensors or scalars. Strings are ignored.
        name: Name of the output. If not provided, the name will be inferred from the output type.
        depth: Depth of the output. Auxiliary variable for recursive calls.

    Note:
        Numpy arrays are converted to torch tensors.
        Objects raise an error.

    Raises:
        CorrectnessCheckError: If any output contains NaN or infinity values
        ValueError: If output does not contain tensors or scalars
    """
    if output is None:
        return

    if isinstance(output, (int, float)):
        if not math.isfinite(output):
            raise CorrectnessValueError(f"Output(int, float) {name} is not finite")
        return

    if isinstance(output, str):
        return

    if isinstance(output, Sequence):
        for i, value in enumerate(output):
            check_output_correctness(value, f"{name}[{i}]", depth + 1)
        return

    if isinstance(output, dict):
        for key, value in output.items():
            check_output_correctness(value, f"{name}['{key}']", depth + 1)
        return

    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)

    if isinstance(output, torch.Tensor):
        if torch.isinf(output).any():
            raise CorrectnessValueError(f"Output tensor {name} contains infinity values")

        if torch.isnan(output).any():
            raise CorrectnessValueError(f"Output tensor {name} contains NaN values")


def tolerance(dtype: torch.dtype) -> float:
    """Largest difference from a reference, relative to its largest magnitude, that a `dtype` output may show.

    >>> tolerance(torch.float16)
    0.03125
    """
    return math.sqrt(torch.finfo(dtype).eps)


def check_outputs_match(expected: Any, actual: Any, *, tolerance_floor: float = 0.0, name: str = "output"):
    """Check that the tensors of `actual` match those of `expected`.

    A floating point tensor may differ by the larger of `tolerance` of its dtype and `tolerance_floor`, times the
    largest magnitude in `expected`. Other tensors must be equal. Values that are not tensors are not compared.

    Raises:
        CorrectnessMismatchError: If the structure differs, or a tensor differs in shape, dtype, device, or value.
    """
    expected_leaves, expected_spec = tree_flatten(expected)
    actual_leaves, actual_spec = tree_flatten(actual)
    if actual_spec != expected_spec:
        raise CorrectnessMismatchError(f"{name} has structure {actual_spec}, expected {expected_spec}")
    for index, (reference, value) in enumerate(zip(expected_leaves, actual_leaves, strict=True)):
        if not isinstance(reference, torch.Tensor):
            continue
        allowed = 0.0
        if reference.is_floating_point() and reference.numel():
            allowed = max(tolerance(reference.dtype), tolerance_floor) * reference.abs().max().item()
        try:
            torch.testing.assert_close(value, reference, rtol=0, atol=allowed)
        except AssertionError as error:
            details = "; ".join(line for line in str(error).splitlines() if line)
            raise CorrectnessMismatchError(
                f"{name} tensor {index} differs from the expected output: {details}"
            ) from error


def check_output_tensor_shapes(expected_tensor_specs: list[TensorSpec], actual_tensor_specs: list[TensorSpec]):
    """Check if the output tensor shapes are the same as the original output tensor shapes."""
    errors = []
    for orig_tensor, tuned_tensor in zip(expected_tensor_specs, actual_tensor_specs, strict=True):
        if not orig_tensor.matches(tuned_tensor):
            errors.append(
                f"Expected tensor {orig_tensor.name} to have shape {orig_tensor.shape} but got {tuned_tensor.shape}"
            )

    if errors:
        raise CorrectnessTensorShapeError(
            f"{len(errors)} error(s) related to output tensor shapes:\n- " + "\n- ".join(errors)
        )
