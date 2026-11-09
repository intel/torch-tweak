# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Tests for correctness checking functionality."""

import numpy as np
import pytest
import torch

from torch_tweak.torch.module.tensor_spec import TensorSpec
from torch_tweak.torch.task.correctness import (
    CorrectnessMismatchError,
    CorrectnessTensorShapeError,
    check_output_correctness,
    check_output_tensor_shapes,
    check_outputs_match,
)


def test_check_output_correctness_valid():
    """Test check_output_correctness with valid outputs."""
    outputs = {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])}
    check_output_correctness(outputs)


def test_check_output_correctness_nan():
    """Test check_output_correctness with NaN values."""
    outputs = {"output1": torch.tensor([1.0, float("nan"), 3.0]), "output2": np.array([4.0, 5.0, 6.0])}
    with pytest.raises(ValueError, match="contains NaN values"):
        check_output_correctness(outputs)


def test_check_output_correctness_inf():
    """Test check_output_correctness with infinity values."""
    outputs = {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, float("inf"), 6.0])}
    with pytest.raises(ValueError, match="contains infinity values"):
        check_output_correctness(outputs)


def test_check_output_correctness_just_a_string():
    """Test check_output_correctness with just a string."""
    outputs = {
        "output1": "not a tensor",
    }
    check_output_correctness(outputs)


def test_check_output_correctness_list():
    """Test check_output_correctness with a list of tensors and numpy arrays."""
    outputs = [torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    check_output_correctness(outputs)


def test_check_output_correctness_sample():
    """Test check_output_correctness with a sample output."""
    outputs = (
        (torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])},
    )
    check_output_correctness(outputs)


def test_check_output_correctness_sample_scalar_nan():
    """Test check_output_correctness with a sample output."""
    outputs = (
        (float("nan"), torch.tensor([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
        {"output1": torch.tensor([1.0, 2.0, 3.0]), "output2": np.array([4.0, 5.0, 6.0])},
    )
    with pytest.raises(ValueError, match="is not finite"):
        check_output_correctness(outputs)


def test_check_output_tensor_shapes_matching_shapes():
    """Test check_output_tensor_shapes with matching tensor shapes."""
    expected_specs = [
        TensorSpec.from_tensor("output__0", torch.randn(2, 5), batch_size=2),
        TensorSpec.from_tensor("output__1", torch.randn(2, 10), batch_size=2),
    ]
    actual_specs = [
        TensorSpec.from_tensor("output__0", torch.randn(2, 5), batch_size=2),
        TensorSpec.from_tensor("output__1", torch.randn(2, 10), batch_size=2),
    ]

    # Should not raise any exception
    check_output_tensor_shapes(expected_specs, actual_specs)


def test_check_output_tensor_shapes_matching_with_symbolic_dimensions():
    """Test check_output_tensor_shapes with symbolic dimensions that should match."""
    # Create specs with symbolic dimensions that should match
    expected_specs = [
        TensorSpec(
            name="output__0",
            shape=[2, "dim1"],
            min_shape=[2, 5],
            max_shape=[2, 10],
            dtype=torch.float32,
            _bs_multipliers=[1.0, 2.5],
        ),
        TensorSpec(
            name="output__1",
            shape=[2, "batch1"],
            min_shape=[2, 10],
            max_shape=[2, 10],
            dtype=torch.float32,
            _bs_multipliers=[1.0, 5.0],
        ),
    ]
    actual_specs = [
        TensorSpec(
            name="output__0",
            shape=[2, 7],  # Concrete dimension that falls within symbolic range
            min_shape=[2, 7],
            max_shape=[2, 7],
            dtype=torch.float32,
            _bs_multipliers=[1.0, 3.5],
        ),
        TensorSpec(
            name="output__1",
            shape=[2, 10],  # Concrete dimension that matches batch dimension
            min_shape=[2, 10],
            max_shape=[2, 10],
            dtype=torch.float32,
            _bs_multipliers=[1.0, 5.0],
        ),
    ]

    # Should not raise any exception
    check_output_tensor_shapes(expected_specs, actual_specs)


def test_check_output_tensor_shapes_mismatched_shapes():
    """Test check_output_tensor_shapes with mismatched tensor shapes."""
    expected_specs = [
        TensorSpec.from_tensor("output__0", torch.randn(2, 5), batch_size=2),
        TensorSpec.from_tensor("output__1", torch.randn(2, 10), batch_size=2),
        TensorSpec.from_tensor("output__2", torch.randn(2, 10, 20), batch_size=2),
    ]
    actual_specs = [
        TensorSpec.from_tensor("output__0", torch.randn(1, 5), batch_size=1),  # Different batch size
        TensorSpec.from_tensor("output__1", torch.randn(2, 8), batch_size=2),  # Different feature size
        TensorSpec.from_tensor("output__2", torch.randn(2, 10), batch_size=2),
    ]

    with pytest.raises(CorrectnessTensorShapeError) as exc_info:
        check_output_tensor_shapes(expected_specs, actual_specs)

    error_message = str(exc_info.value)
    assert "Expected tensor output__0 to have shape [2, 5] but got [1, 5]" in error_message
    assert "Expected tensor output__1 to have shape [2, 10] but got [2, 8]" in error_message
    assert "Expected tensor output__2 to have shape [2, 10, 20] but got [2, 10]" in error_message
    assert "3 error(s) related to output tensor shapes" in error_message


def test_check_outputs_match_accepts_differences_within_the_dtype_tolerance():
    check_outputs_match(
        {"logits": torch.tensor([4.0, -2.0], dtype=torch.float16)},
        {"logits": torch.tensor([4.1, -2.0], dtype=torch.float16)},
    )


def test_check_outputs_match_rejects_differences_beyond_the_dtype_tolerance():
    with pytest.raises(CorrectnessMismatchError, match="Mismatched elements: 1 / 2"):
        check_outputs_match(torch.tensor([4.0, -2.0]), torch.tensor([4.1, -2.0]))


def test_check_outputs_match_accepts_differences_within_the_tolerance_floor():
    check_outputs_match(torch.tensor([4.0, -2.0]), torch.tensor([4.4, -2.0]), tolerance_floor=0.25)


def test_check_outputs_match_rejects_nan():
    with pytest.raises(CorrectnessMismatchError, match="Greatest absolute difference: nan"):
        check_outputs_match(torch.tensor([1.0]), torch.tensor([float("nan")]))


def test_check_outputs_match_requires_equal_integer_tensors():
    with pytest.raises(CorrectnessMismatchError, match="Mismatched elements: 1 / 2"):
        check_outputs_match(torch.tensor([1, 2]), torch.tensor([1, 3]))


def test_check_outputs_match_requires_equal_shapes():
    with pytest.raises(CorrectnessMismatchError, match="'shape' do not match"):
        check_outputs_match(torch.zeros(2), torch.zeros(1, 2))


def test_check_outputs_match_requires_equal_dtypes():
    with pytest.raises(CorrectnessMismatchError, match="'dtype' do not match"):
        check_outputs_match(torch.zeros(2), torch.zeros(2, dtype=torch.float16))


def test_check_outputs_match_requires_the_same_structure():
    with pytest.raises(CorrectnessMismatchError, match="has structure"):
        check_outputs_match((torch.zeros(2),), (torch.zeros(2), torch.zeros(2)))
