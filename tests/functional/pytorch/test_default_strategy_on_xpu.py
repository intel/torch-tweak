# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.module.tunable_module import TunableModule

XPU = torch.device("xpu")


def mlp() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 256), torch.nn.ReLU(), torch.nn.Linear(256, 8)).eval().to(XPU)


def ramp(*shape: int) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, steps=torch.Size(shape).numel(), device=XPU).reshape(shape)


@pytest.mark.functional
@requires_xpu
def test_default_strategy_matches_eager_on_a_static_shape_module():
    model = mlp()
    sample = ramp(4, 64)
    with torch.no_grad():
        eager = model(sample)
    module = TunableModule(model, "static-mlp")
    module(sample)

    module.tune(device=XPU)

    torch.testing.assert_close(module(sample), eager, rtol=1e-4, atol=1e-5)


@pytest.mark.functional
@requires_xpu
def test_an_earlier_graph_matches_eager_after_a_later_graph_is_tuned():
    model = mlp()
    first = ramp(4, 64)
    second = ramp(4, 3, 64)
    with torch.no_grad():
        eager_first = model(first)
    module = TunableModule(model, "two-graph-mlp")
    module(first)
    module(second)

    module.tune(device=XPU)

    torch.testing.assert_close(module(first), eager_first, rtol=1e-4, atol=1e-5)
