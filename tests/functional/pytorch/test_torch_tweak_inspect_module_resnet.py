# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

import pytest
import timm
import torch

from torch_tweak.torch import inspect


@pytest.mark.functional
def test_inspect_resnet50():
    # given
    model = timm.create_model("resnet50", pretrained=False)
    model.xpu()
    model.eval()
    data = torch.randn((3, 224, 224), device="xpu")

    # when
    modules_info = inspect(model, data)

    # then - verify inspection
    modules_info.describe()

    assert len(modules_info.get_modules()) == 1

    module_info = modules_info.get_modules()[0]
    assert module_info.name == model.__class__.__name__
    assert module_info.module_type == timm.models.resnet.ResNet
    assert module_info.total_execution_time > 0
    assert module_info.average_execution_time > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
