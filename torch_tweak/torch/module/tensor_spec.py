# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Contains TensorSpec which represents metadata of a tensor."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum, auto
from functools import cache
from types import MappingProxyType
from typing import Any

import torch


class InfoLevel(Enum):
    """Enum representing different levels of information detail."""

    SHORT = auto()
    MEDIUM = auto()
    FULL = auto()


@cache
def _dtype_by_name() -> Mapping[str, torch.dtype]:
    """Map dtype names such as "torch.float32" to the matching torch dtype.

    The mapping is built by inspecting torch and keeping only real torch.dtype
    attributes, so resolving a name can never return an arbitrary torch object.
    The result is cached and shared by all callers, hence it is read only.
    """
    registry: dict[str, torch.dtype] = {}
    for attribute in dir(torch):
        value = getattr(torch, attribute, None)
        if isinstance(value, torch.dtype):
            registry[str(value)] = value
    return MappingProxyType(registry)


def dtype_from_name(name: str | None) -> torch.dtype | None:
    """Resolve a serialized dtype name back to a torch dtype.

    Raises:
        ValueError: If the name does not correspond to a known torch dtype.
    """
    if name is None:
        return None
    registry = _dtype_by_name()
    if name not in registry:
        raise ValueError(f"Unknown torch dtype: {name!r}")
    return registry[name]


@dataclass(slots=True)
class TensorSpec:
    """TensorSpec is used to describe tensor metadata.

    Attributes:
        name (str) - symbolic name of the tensor
        shape (list[Union[str, int]]) - shape of the tensor, int is a real dimension, str is a symbolic dimension
        min_shape (list[int]) - minimum dimensions seen so far
        max_shape (list[int]) - maximum dimensions seen so far
        dtype (torch.dtype) - dtype of the tensor
        _bs_multipliers (list[float]) - batch size multipliers for each axis
    """

    name: str
    shape: list[str | int]
    min_shape: list[int]
    max_shape: list[int]
    dtype: torch.dtype | None
    _bs_multipliers: list[float]

    @staticmethod
    def from_tensor(name: str, tensor: torch.Tensor, batch_size: int):
        """Create TensorSpec from tensor.

        Args:
            name: Name of the tensor
            tensor: Tensor to create TensorSpec from
            batch_size: Batch size
        """
        shape = list(tensor.shape)
        if math.isnan(batch_size):
            _bs_multipliers = [float("nan")] * len(shape)
        else:
            _bs_multipliers = [size / batch_size for size in shape]

        return TensorSpec(
            name=name,
            shape=shape,
            min_shape=shape[:],
            max_shape=shape[:],
            dtype=tensor.dtype,
            _bs_multipliers=_bs_multipliers,
        )

    def __repr__(self) -> str:
        """Get representation of TensorSpec."""
        return self.describe(InfoLevel.MEDIUM)

    def __str__(self) -> str:
        """Get string representation of TensorSpec."""
        return self.describe(InfoLevel.SHORT)

    def __eq__(self, other: object) -> bool:
        """Check if two TensorSpec are equal.

        Tensors of the same name and rank are considered equal.
        Particular dimensions sizes can be different as there can be dynamic dimensions.
        """
        if not isinstance(other, TensorSpec):
            return False
        return self.name == other.name and len(self.shape) == len(other.shape)

    def __hash__(self) -> int:
        """Hash of TensorSpec.

        Tensors of the same name and rank are considered equal.
        Particular dimensions sizes can be different as there can be dynamic dimensions.
        """
        return hash((self.name, len(self.shape)))

    def describe(self, info_level: InfoLevel = InfoLevel.FULL) -> str:
        """Get information describing TensorSpec."""
        if info_level == InfoLevel.SHORT:
            return self.name
        elif info_level == InfoLevel.MEDIUM:
            shapes = ", ".join(str(dim) for dim in self.shape)
            return f"{self.name}[{shapes}]"
        elif info_level == InfoLevel.FULL:
            shapes = ", ".join(str(dim) for dim in self.shape)
            min_shape = ", ".join(str(dim) for dim in self.min_shape)
            max_shape = ", ".join(str(dim) for dim in self.max_shape)
            return f"{self.name}[{shapes}] min_shape=[{min_shape}] max_shape=[{max_shape}] dtype={self.dtype}"  # type: ignore[bad-return-type]

    def update_shapes_seen(self, other: "TensorSpec"):
        """Update shapes seen from other TensorSpec.

        Tensor have to have same rank in order to update self.

        The algorithm for detecting batch dimension is the following: given two different batch sizes, if batch size
        multiplier is the same for same axis in both tensors and is an integer, then it is a batch dimension otherwise
        it is a dynamic dimension.

        The reason for using multipliers is that some models stack input tensor vertically and the resulting input has
        double of the batch size i.e. local batch size is 2x the global batch size.

        Example of the algorithm - let's assume we observed tensor[1, 2, 3, 4] given bs=1. We calculated multipliers
        to be [1, 2, 3, 4]. Now if we see tensor[2, 8, 6, 4] with bs=2, we calculate multipliers to be
        [1, 4, 3, 2]. We can make some conclusions:
        - 0th axis is batch axis, multiplier is 1
        - 1st axis is dynamic axis, multiplier is 2 and 4 - this could be for example length in LLMs
        - 2nd axis is batch axis, multiplier is 3 - the input tensor is v-stacked thus multiplier is 3
        - 3rd axis is static axis - never changes w.r.t. batch size

        This algorithm is not foolproof and can fail in some cases e.g. sequence length in LLMs is equal to batch size
        which is unlikely to happen in practice. However to mitigate this risk, there is additional check for batch axis
        multiplier which has to be an integer.
        """
        if len(self.shape) != len(other.shape):
            raise ValueError("Tensors must have the same rank")
        for i in range(len(self.shape)):
            if self.shape[i] != other.shape[i]:
                self.min_shape[i] = min(self.min_shape[i], other.min_shape[i])
                self.max_shape[i] = max(self.max_shape[i], other.max_shape[i])
                if self._bs_multipliers[i] == other._bs_multipliers[i] and TensorSpec.is_int(self._bs_multipliers[i]):
                    self.shape[i] = f"batch{i}"
                else:
                    self._bs_multipliers[i] = float("nan")
                    self.shape[i] = f"dim{i}"

    def to_dict(self) -> dict[str, Any]:
        """Convert TensorSpec to a dictionary built only from primitive values."""
        data = {"type": self.__class__.__name__} | asdict(self)
        data["dtype"] = None if self.dtype is None else str(self.dtype)
        return data

    @staticmethod
    def _is_batch_dim(dim: str | int) -> bool:
        return isinstance(dim, str) and dim.startswith("batch")

    def get_batch_axis_multipliers(self) -> dict[int, int]:
        """Return mapping for batch axis and its multiplier."""
        return {i: int(self._bs_multipliers[i]) for i, dim in enumerate(self.shape) if self._is_batch_dim(dim)}

    def get_max_batch_size(self) -> int:
        """Get max batch size from tensor spec."""
        batch_maxes = [self.max_shape[axis] for axis in self.get_batch_axis_multipliers()]
        return max(batch_maxes) if batch_maxes else 1

    def get_min_batch_size(self) -> int | None:
        """Get min batch size from tensor spec."""
        batch_mins = [self.min_shape[axis] for axis in self.get_batch_axis_multipliers()]
        return int(min(batch_mins)) if batch_mins else None

    def has_batch_axis(self) -> bool:
        """Check if tensor has batch axis."""
        return any(self._is_batch_dim(dim) for dim in self.shape)

    def has_dynamic_axis(self) -> bool:
        """Check if tensor has dynamic axis."""
        return any(isinstance(dim, str) and dim.startswith("dim") for dim in self.shape)

    def matches(self, other: "TensorSpec") -> bool:
        """Check if tensor spec matches other tensor spec.

        Tensor spec matches other if ranks are same and each ordinal dimension is the same.

        Example:
            tensor[1, 2] matches tensor[1, 2]
            tensor[1, 2] matches tensor[1, "dim1"]
            tensor[1, 2] matches tensor[1, "batch1"]
            tensor[1, 2] does not match tensor[1, 3]
            tensor[1, 2] does not match tensor[1, 2, 3]
        """
        if len(self.shape) != len(other.shape):
            return False
        for i in range(len(self.shape)):
            if isinstance(self.shape[i], int) and isinstance(other.shape[i], int) and self.shape[i] != other.shape[i]:
                return False
        return True

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TensorSpec":
        """Create TensorSpec from dictionary."""
        if data.get("type") != TensorSpec.__name__:
            raise ValueError(f"Invalid dictionary format for {TensorSpec.__name__}")
        fields = {key: value for key, value in data.items() if key != "type"}
        # Indexed, not .get(): a payload without a dtype must fail rather than default to None.
        fields["dtype"] = dtype_from_name(fields["dtype"])
        return TensorSpec(**fields)

    @staticmethod
    def is_int(value: float) -> bool:
        """Check if value is an integer by checking value not the type."""
        return not math.isnan(value) and int(value) == value


def logical_batch_bounds(tensor_specs: Sequence[TensorSpec]) -> tuple[int, int]:
    """Return the logical batch minimum and maximum shared by every batch axis.

    Each batch axis stores physical ``min_shape`` / ``max_shape`` as ``multiplier * batch``.
    All logical minimum values must agree, and all logical maximum values must agree.

    Raises:
        ValueError: No batch axes, or batch axes disagree on the bounds.
    """
    entries: list[tuple[str, int, int, int]] = []
    for tensor_spec in tensor_specs:
        for axis, multiplier in tensor_spec.get_batch_axis_multipliers().items():
            entries.append((
                tensor_spec.name,
                axis,
                tensor_spec.min_shape[axis] // multiplier,
                tensor_spec.max_shape[axis] // multiplier,
            ))

    if not entries:
        raise ValueError("No batch axes found in tensor specs; cannot determine logical batch bounds.")

    bounds: list[int] = []
    for kind, value_index in (("minimum", 2), ("maximum", 3)):
        values = {entry[value_index] for entry in entries}
        if len(values) != 1:
            details = ", ".join(f"{entry[0]} axis {entry[1]} -> {entry[value_index]}" for entry in entries)
            raise ValueError(f"Inconsistent logical {kind} batch size across inputs: {details}")
        bounds.append(values.pop())
    return bounds[0], bounds[1]
