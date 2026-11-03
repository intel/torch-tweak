# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import timm
import torch

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch import tune
from torch_tweak.torch.backend.xpu_graph_backend import XPUGraphBackend, XPUGraphBackendConfig
from torch_tweak.torch.module.tunable_module import TunableModule
from torch_tweak.torch.tune_strategy.one_backend_strategy import OneBackendStrategy


def reject_eager_fallback(self, args, kwargs):
    raise AssertionError("XPU graph replay fell back to the original module")


@pytest.mark.functional
@requires_xpu
@pytest.mark.parametrize(
    ("batch_size", "max_recorded_batch_size"),
    [(3, 8), (11, 4)],
    ids=["padded", "chunked"],
)
def test_resnet18_xpu_graph_matches_eager(batch_size, max_recorded_batch_size, torch_device, monkeypatch):
    model = timm.create_model("resnet18", pretrained=False).to(torch_device).eval()
    reference = copy.deepcopy(model)
    image = torch.linspace(-1, 1, 3 * 64 * 64, device=torch_device).reshape(3, 64, 64)
    strategy = OneBackendStrategy(
        XPUGraphBackend(XPUGraphBackendConfig(max_recorded_batch_size=max_recorded_batch_size))
    ).enable_find_max_batch_size(False)
    module = TunableModule(model, "functional-resnet18-xpu-graph", strategy=strategy)
    tune(
        module,
        image,
        batch_sizes=[1, max_recorded_batch_size],
        device=torch_device,
        ignore_failing_modules=False,
    )
    batch = torch.linspace(-1, 1, batch_size * 3 * 64 * 64, device=torch_device)
    batch = batch.reshape(batch_size, 3, 64, 64)

    monkeypatch.setattr(XPUGraphBackend, "_infer_eager", reject_eager_fallback)
    with torch.no_grad():
        actual = module(batch)
        expected = reference(batch)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
