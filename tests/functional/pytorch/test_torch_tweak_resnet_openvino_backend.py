# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest
import timm
import torch

import torch_tweak.torch as tt
from torch_tweak.torch.backend import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.module import Module
from torch_tweak.torch.tune_strategy import OneBackendStrategy


@pytest.mark.functional
@pytest.mark.parametrize("use_dynamo", [False, True], ids=["ovc", "dynamo"])
def test_openvino(use_dynamo: bool):
    model = timm.create_model("resnet18").xpu().eval()
    samples = torch.randn((3, 224, 224), device="xpu")

    backend = OpenVINOBackend(config=OpenVINOBackendConfig(use_dynamo=use_dynamo))
    strategy = OneBackendStrategy(backend)
    module = Module(model, "test-resnet18", strategy=strategy)

    batch_sizes = [1, 10]
    tt.tune(module, samples, batch_sizes=batch_sizes)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = Path(tmp_dir) / "test-resnet18.pt"
        tt.save(module, output_path)

        new_model = timm.create_model("resnet18").xpu().eval()
        loaded_model = tt.load(new_model, output_path)

        for batch_size in batch_sizes:
            batch = samples.unsqueeze(0).expand(batch_size, -1, -1, -1)
            with torch.no_grad():
                expected = module(batch)
                output = loaded_model(batch)

            torch.testing.assert_close(output, expected, rtol=1e-3, atol=2e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
