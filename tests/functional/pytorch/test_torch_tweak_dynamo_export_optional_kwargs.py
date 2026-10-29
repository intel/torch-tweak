# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

from logging import getLogger
from pathlib import Path

import openvino as ov
import pytest
import torch
import torch.nn as nn

from torch_tweak.torch.backend.openvino.ov_model_converter import OpenVINOModelConverter
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.sample_metadata import SampleMetadata

logger = getLogger(Path(__file__).stem)


class ModelWithOptionalKwargs(nn.Module):
    """Test model with optional kwargs including None defaults."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x, optional_flag=None, optional_value=None):
        """Forward with optional kwargs.

        Args:
            x: Input tensor
            optional_flag: Optional flag (None or tensor)
            optional_value: Optional value (None or tensor)
        """
        out = self.linear(x)
        if optional_flag is not None:
            out = out * optional_flag
        if optional_value is not None:
            out = out + optional_value
        return out


def _input_names_from_ir(xml_path: Path) -> list[str]:
    model = ov.Core().read_model(xml_path)
    return [port.get_any_name() for port in model.inputs]


def _export_to_ir(module: nn.Module, sample: tuple, tmp_path: Path, subdir: str) -> Path:
    """Run the OpenVINO dynamo export flow for sample and return the resulting IR xml path."""
    args, kwargs = sample
    output = module(*args, **kwargs)

    input_metadata = SampleMetadata.from_inputs(args, kwargs, batch_size=2)
    output_metadata = SampleMetadata.from_outputs(output, batch_size=2)
    graph_spec = GraphSpec(name="test_graph", input_spec=input_metadata, output_spec=output_metadata)

    out_dir = tmp_path / subdir
    converter = OpenVINOModelConverter(output_dir=out_dir, use_dynamo=True)
    xml_path = converter.convert(module=module, sample=sample, graph_spec=graph_spec)
    return xml_path


_OPTIONAL_KWARGS_CASES = [
    # None-valued kwargs should be filtered out of the exported inputs.
    pytest.param({}, ["args_0"], "none_kwargs", id="none_kwargs"),
    # kwargs passed in reverse signature order should be reordered to match forward's signature.
    pytest.param(
        {"optional_value": 1.0, "optional_flag": 2.0},
        ["args_0", "kwargs_optional_flag", "kwargs_optional_value"],
        "wrong_order",
        id="wrong_order",
    ),
    # one non-None and one None kwarg: only the non-None one should remain.
    pytest.param(
        {"optional_flag": 2.0, "optional_value": None},
        ["args_0", "kwargs_optional_flag"],
        "mixed_kwargs",
        id="mixed_kwargs",
    ),
]


@pytest.mark.functional
@pytest.mark.parametrize("raw_kwargs, expected_names, subdir", _OPTIONAL_KWARGS_CASES)
def test_export_with_optional_kwargs(
    tmp_path: Path, torch_device: str, raw_kwargs: dict, expected_names: list[str], subdir: str
):
    """Test dynamo export for optional kwargs: None filtering, reordering, and mixed None/non-None cases."""
    logger.info("Export optional kwargs case '%s'", subdir)

    model = ModelWithOptionalKwargs().eval().to(torch_device)
    x = torch.randn(2, 10, device=torch_device)
    kwargs = {
        name: (torch.tensor(value, device=torch_device) if value is not None else None)
        for name, value in raw_kwargs.items()
    }
    sample = ((x,), kwargs)

    xml_path = _export_to_ir(model, sample, tmp_path, subdir)
    input_names = _input_names_from_ir(xml_path)
    assert len(input_names) == len(expected_names), f"Expected {len(expected_names)} inputs, got {len(input_names)}"
    assert input_names == expected_names

    logger.info("✓ Export optional kwargs case '%s' passed", subdir)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
