# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Contains GraphSpec which represents a graph specification."""

from dataclasses import dataclass
from typing import Any

from torch_tweak.torch.module.sample_metadata import SampleMetadata


@dataclass
class GraphSpec:
    """GraphSpec used to describe a computational graph.

    Each torch module has its own specification of input and output variables. The input specification i.e. args and kwargs
    of the torch modules `forward` function is represented by SampleMetadata. Those inputs can change computational
    graph. Torch Tweak treats each unique input specification as a separate graph which is tuned separately. This object
    represents such a computational graph with a name and input_spec information.

    Attributes:
        name (str) - symbolic name of the graph
        input_spec (SampleMetadata) - spec which describes the graph input
        output_spec (SampleMetadata) - spec which describes the graph output
    """

    name: str
    input_spec: SampleMetadata
    output_spec: SampleMetadata

    def has_dynamic_axis(self) -> bool:
        """Check if any input or output tensor has a dynamic axis."""
        return self.input_spec.detected_dynamic_axis() or self.output_spec.detected_dynamic_axis()

    def update_shapes_seen(self, inputs_metadata: SampleMetadata, outputs_metadata: SampleMetadata):
        """Update input spec with other input spec."""
        self.input_spec.update_shapes_seen(inputs_metadata)
        self.output_spec.update_shapes_seen(outputs_metadata)

    def get_max_batch_size(self) -> int:
        """Get max batch size from input spec."""
        batch_maxes = [
            tensor_spec.max_shape[axis]
            for tensor_spec in self.input_spec.tensor_specs
            for axis in tensor_spec.get_batch_axis_multipliers()
        ]
        return max(batch_maxes) if batch_maxes else 1

    def get_min_batch_size(self) -> int | None:
        """Get min batch size from input spec."""
        batch_mins = [
            tensor_spec.min_shape[axis]
            for tensor_spec in self.input_spec.tensor_specs
            for axis in tensor_spec.get_batch_axis_multipliers()
        ]
        return int(min(batch_mins)) if batch_mins else None

    def to_dict(self) -> dict[str, Any]:
        """Convert GraphSpec to a dictionary built only from primitive values."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "input_spec": self.input_spec.to_dict(),
            "output_spec": self.output_spec.to_dict(),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GraphSpec":
        """Create GraphSpec from dictionary."""
        if data.get("type") != GraphSpec.__name__:
            raise ValueError(f"Invalid dictionary format for {GraphSpec.__name__}")
        return GraphSpec(
            name=data["name"],
            input_spec=SampleMetadata.from_dict(data["input_spec"]),
            output_spec=SampleMetadata.from_dict(data["output_spec"]),
        )

    def __str__(self) -> str:
        """Return string representation of GraphSpec."""
        return f"Name={self.name}\nInput_spec:\n{self.input_spec.describe()}Output_spec:\n{self.output_spec.describe()}"

    def __repr__(self) -> str:
        """Return representation of GraphSpec."""
        return self.__str__()
