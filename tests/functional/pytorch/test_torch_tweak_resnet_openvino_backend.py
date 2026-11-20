# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest
import timm
import torch

import torch_tweak.torch as tt
from tests.utilities.helpers import requires_leo
from torch_tweak.torch.backend import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.module import TunableModule
from torch_tweak.torch.tune_strategy import OneBackendStrategy


@pytest.mark.functional
@requires_leo
@pytest.mark.parametrize("use_dynamo", [False, True], ids=["ovc", "dynamo"])
def test_openvino(use_dynamo: bool):
    torch.manual_seed(0)
    model = timm.create_model("resnet18").xpu().eval()
    samples = torch.randn((3, 224, 224), device="xpu")
    batch_1 = samples.unsqueeze(0)
    batch_10 = batch_1.expand(10, -1, -1, -1)
    with torch.no_grad():
        eager_1 = model(batch_1)
        eager_10 = model(batch_10)

    backend = OpenVINOBackend(config=OpenVINOBackendConfig(use_dynamo=use_dynamo))
    module = TunableModule(model, "test-resnet18", strategy=OneBackendStrategy(backend))
    tt.tune(module, samples, batch_sizes=[1, 10])

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test-resnet18.pt"
        tt.save(module, output_path)
        loaded_model = tt.load(timm.create_model("resnet18").xpu().eval(), output_path)

        with torch.no_grad():
            torch.testing.assert_close(loaded_model(batch_1), eager_1, rtol=1e-3, atol=2e-3)
            torch.testing.assert_close(loaded_model(batch_10), eager_10, rtol=1e-3, atol=2e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
