# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

import pytest
import timm
import torch

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch import tune
from torch_tweak.torch.backend.torch_inductor_backend import TorchInductorBackend, TorchInductorBackendConfig
from torch_tweak.torch.module.wrapper_module import Module
from torch_tweak.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

# max-autotune-no-cudagraphs exhausts XPU memory on current CI hardware.
# TORCH_COMPILE_MODES = ["default", "max-autotune-no-cudagraphs"]
TORCH_COMPILE_MODES = ["default"]


torch._inductor.config.autotune_in_subproc = True
torch._inductor.config.max_autotune_subproc_result_timeout_seconds = 10.0


@pytest.mark.functional
@requires_xpu
@pytest.mark.parametrize("mode", TORCH_COMPILE_MODES)
def test_tune_resnet_torch_inductor(mode: str, torch_device: torch.device):
    model = timm.create_model("resnet18").to(torch_device).eval()
    data = torch.randn((3, 224, 224), device=torch_device)

    with torch.no_grad():
        out = model(data.unsqueeze(0))
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    backend = TorchInductorBackend(config=TorchInductorBackendConfig(mode=mode))
    module = Module(model, "functional-resnet18", strategy=OneBackendStrategy(backend))
    tune(module, data, batch_sizes=[2, 1], device=torch_device)

    out = module(data.unsqueeze(0))
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
