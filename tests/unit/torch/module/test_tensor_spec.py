# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Unit tests for tensor spec."""

import pytest
import torch

from torch_tweak.torch.module.tensor_spec import (
    InfoLevel,
    TensorSpec,
    _dtype_by_name,
    dtype_from_name,
    logical_batch_bounds,
)


def test_tensor_spec_from_tensor():
    spec = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    assert spec.name == "test"
    assert spec.shape == [1, 2, 3]
    assert spec.min_shape == [1, 2, 3]
    assert spec.max_shape == [1, 2, 3]
    assert spec.get_batch_axis_multipliers() == {}


def test_tensor_spec_hash():
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(3, 2, 1), batch_size=1)
    assert hash(spec1) == hash(spec2)
    assert spec1 == spec2
    spec3 = TensorSpec.from_tensor("test123", torch.randn(4, 2, 4), batch_size=1)
    assert hash(spec1) != hash(spec3)
    assert spec1 != spec3
    spec4 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    assert hash(spec1) != hash(spec4)
    assert spec1 != spec4


def test_tensor_spec_info():
    spec = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    assert spec.describe(InfoLevel.SHORT) == "test"
    assert spec.describe(InfoLevel.MEDIUM) == "test[1, 2, 3]"
    assert spec.describe(InfoLevel.FULL) == "test[1, 2, 3] min_shape=[1, 2, 3] max_shape=[1, 2, 3] dtype=torch.float32"
    assert str(spec) == "test"
    assert repr(spec) == "test[1, 2, 3]"


def test_tensor_spec_update_shapes_seen():
    # case 1
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(3, 2, 6), batch_size=3)
    spec1.update_shapes_seen(spec2)
    assert spec1.shape == ["batch0", 2, "dim2"]
    assert spec1.min_shape == [1, 2, 3]
    assert spec1.max_shape == [3, 2, 6]
    assert spec1.get_batch_axis_multipliers() == {0: 1}

    # case 2
    spec3 = TensorSpec.from_tensor("test", torch.randn(4, 2, 4), batch_size=1)
    spec1.update_shapes_seen(spec3)
    assert spec1.shape == ["dim0", 2, "dim2"]
    assert spec1.min_shape == [1, 2, 3]
    assert spec1.max_shape == [4, 2, 6]
    assert spec1.get_batch_axis_multipliers() == {}

    # case 3
    spec4 = TensorSpec.from_tensor("test", torch.randn(5, 4, 10), batch_size=5)
    spec2.update_shapes_seen(spec4)
    assert spec2.shape == ["batch0", "dim1", "batch2"]
    assert spec2.min_shape == [3, 2, 6]
    assert spec2.max_shape == [5, 4, 10]
    assert spec2.get_batch_axis_multipliers() == {0: 1, 2: 2}
    # case 4
    spec5 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    with pytest.raises(ValueError):
        spec1.update_shapes_seen(spec5)


def test_tensor_spec_to_dict_from_dict():
    spec = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    wrong_dict = spec.to_dict()
    assert wrong_dict["type"] == "TensorSpec"
    assert wrong_dict["name"] == "test"
    assert wrong_dict["shape"] == [1, 2, 3]
    assert wrong_dict["min_shape"] == [1, 2, 3]
    assert wrong_dict["max_shape"] == [1, 2, 3]
    spec_from_dict = TensorSpec.from_dict(wrong_dict)
    assert spec == spec_from_dict

    # wrong type
    wrong_dict = {"a": 1}
    with pytest.raises(ValueError):
        TensorSpec.from_dict(wrong_dict)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool, None])
def test_tensor_spec_dtype_round_trip(dtype):
    spec = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec.dtype = dtype

    data = spec.to_dict()
    assert data["dtype"] == (None if dtype is None else str(dtype))

    assert TensorSpec.from_dict(data).dtype == dtype


def test_tensor_spec_from_dict_rejects_unknown_dtype():
    data = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1).to_dict()
    data["dtype"] = "torch.load"
    with pytest.raises(ValueError, match="Unknown torch dtype"):
        TensorSpec.from_dict(data)


def test_dtype_registry_resolves_only_torch_dtypes():
    """The registry must never be able to hand back a non-dtype torch attribute."""
    registry = _dtype_by_name()

    assert registry, "dtype registry must not be empty"
    assert all(isinstance(value, torch.dtype) for value in registry.values())
    assert all(name == str(value) for name, value in registry.items())


@pytest.mark.parametrize("attribute", ["load", "save", "compile", "nn", "Tensor", "_C"])
def test_dtype_registry_excludes_non_dtype_attributes(attribute):
    registry = _dtype_by_name()

    assert attribute not in registry
    assert f"torch.{attribute}" not in registry


def test_dtype_registry_is_read_only():
    registry = _dtype_by_name()

    with pytest.raises(TypeError):
        registry["torch.injected"] = torch.float32

    assert "torch.injected" not in _dtype_by_name()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.bool, None])
def test_dtype_from_name_round_trip(dtype):
    assert dtype_from_name(None if dtype is None else str(dtype)) is dtype


@pytest.mark.parametrize("name", ["torch.load", "load", "not_a_dtype", "torch.nn"])
def test_dtype_from_name_rejects_unknown_name(name):
    with pytest.raises(ValueError, match="Unknown torch dtype"):
        dtype_from_name(name)


def test_has_batch_axis():
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(3, 2), batch_size=3)
    spec1.update_shapes_seen(spec2)
    assert spec1.has_batch_axis()


def test_logical_batch_bounds_are_physical_size_divided_by_multiplier():
    stacked = TensorSpec(
        name="stacked",
        shape=["batch0"],
        min_shape=[6],
        max_shape=[24],
        dtype=None,
        _bs_multipliers=[3.0],
    )
    plain = TensorSpec(
        name="plain",
        shape=[7, "batch1"],
        min_shape=[7, 2],
        max_shape=[7, 8],
        dtype=None,
        _bs_multipliers=[float("nan"), 1.0],
    )

    bounds = logical_batch_bounds([stacked, plain])

    assert bounds == (2, 8)


def test_logical_batch_bounds_reject_a_different_minimum():
    low = TensorSpec(
        name="low",
        shape=["batch0"],
        min_shape=[1],
        max_shape=[8],
        dtype=None,
        _bs_multipliers=[1.0],
    )
    high = TensorSpec(
        name="high",
        shape=["batch0"],
        min_shape=[4],
        max_shape=[8],
        dtype=None,
        _bs_multipliers=[1.0],
    )

    with pytest.raises(ValueError, match="Inconsistent logical minimum batch size"):
        logical_batch_bounds([low, high])


def test_logical_batch_bounds_reject_a_different_maximum():
    narrow = TensorSpec(
        name="narrow",
        shape=["batch0"],
        min_shape=[1],
        max_shape=[4],
        dtype=None,
        _bs_multipliers=[1.0],
    )
    wide = TensorSpec(
        name="wide",
        shape=["batch0"],
        min_shape=[1],
        max_shape=[8],
        dtype=None,
        _bs_multipliers=[1.0],
    )

    with pytest.raises(ValueError, match="Inconsistent logical maximum batch size"):
        logical_batch_bounds([narrow, wide])


def test_logical_batch_bounds_reject_specs_without_a_batch_axis():
    spec = TensorSpec(
        name="static",
        shape=[4, "dim1"],
        min_shape=[4, 2],
        max_shape=[4, 9],
        dtype=None,
        _bs_multipliers=[1.0, float("nan")],
    )

    with pytest.raises(ValueError, match="No batch axes found"):
        logical_batch_bounds([spec])


def test_has_dynamic_axis():
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(2, 2), batch_size=3)
    spec1.update_shapes_seen(spec2)
    assert spec1.has_dynamic_axis()


def test_matches():
    """Test the matches function with various scenarios."""
    # Test case 1: Same shapes with integer dimensions
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    assert spec1.matches(spec2)
    assert spec2.matches(spec1)

    # Test case 2: Same shapes with symbolic dimensions (dim1)
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2.shape[1] = "dim1"  # Replace second dimension with symbolic
    assert spec1.matches(spec2)
    assert spec2.matches(spec1)

    # Test case 3: Same shapes with symbolic dimensions (batch1)
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2.shape[1] = "batch1"  # Replace second dimension with batch symbolic
    assert spec1.matches(spec2)
    assert spec2.matches(spec1)

    # Test case 4: Different integer dimensions - should not match
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 3), batch_size=1)
    assert not spec1.matches(spec2)
    assert not spec2.matches(spec1)

    # Test case 5: Different ranks - should not match
    spec1 = TensorSpec.from_tensor("test", torch.randn(1, 2), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1, 2, 3), batch_size=1)
    assert not spec1.matches(spec2)
    assert not spec2.matches(spec1)

    # Test case 6: Scalars i.e. empty shapes
    spec1 = TensorSpec.from_tensor("test", torch.randn(1), batch_size=1)
    spec2 = TensorSpec.from_tensor("test", torch.randn(1), batch_size=1)
    spec1.shape = []
    spec2.shape = []
    assert spec1.matches(spec2)
    assert spec2.matches(spec1)
