# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

import pytest
import timm
import torch

from torch_tweak.torch import Module, OneBackendStrategy
from torch_tweak.torch.backend import TorchInductorBackend


@pytest.mark.functional
def test_resnet50():
    # given
    device = torch.device("xpu")

    model = timm.create_model("resnet50", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((2, 3, 224, 224), device=device)

    with torch.no_grad():
        out = model(data)
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    # when
    module = Module(model, "functional-resnet50")

    # then - verify recording
    module(data)
    assert len(module.graph_specs) == 1

    # then - verify tuning
    strategy = OneBackendStrategy(TorchInductorBackend())
    module.tune(device=device, strategy=strategy, dry_run=False)
    out = module(data)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
