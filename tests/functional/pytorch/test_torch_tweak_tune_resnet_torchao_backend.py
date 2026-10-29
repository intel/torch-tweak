# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

from itertools import product
from logging import getLogger

import pytest
import timm
import torch

from torch_tweak.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig
from torch_tweak.torch.module.wrapper_module import Module
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from torch_tweak.torch.tuning import tune
from torch_tweak.utils import system_resource_monitor

logger = getLogger(__name__)


@system_resource_monitor(logger_func=logger.info)
def do_test(backend: TorchAOBackend, dtype: torch.dtype):
    # given
    model = timm.create_model("resnet18", pretrained=False)
    model.to("xpu", dtype=dtype)
    model.eval()

    data = torch.randn((3, 224, 224), device="xpu").to(dtype)
    sample = torch.randn((4, 3, 224, 224), device="xpu").to(dtype)

    with torch.no_grad():
        out = model(sample)

    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    module = Module(
        model, "functional-resnet18", strategy=OneBackendStrategy(backend).enable_find_max_batch_size(False)
    )
    # when
    tune(module, data, batch_sizes=[1, 2, 4], dry_run=False, disable_external_logging=False)
    # then - verify tuning
    out = module(sample)
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-2, atol=1e-2)


@pytest.mark.nightly
def test_tune_resnet_torchao():
    errors = []
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    quantizations = TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys()

    for dtype, quantization in product(dtypes, quantizations):
        try:
            if quantization in ["fp8wo", "fp8dq"]:
                continue  # fp8wo and fp8dq require CUDA fp8 kernels; not supported on XPU
            logger.info("Testing %s and %s", quantization, dtype)
            config = TorchAOBackendConfig(quantization=quantization)  # type: ignore
            do_test(TorchAOBackend(config=config), dtype)  # type: ignore
            logger.info("Successfully quantized %s and %s", quantization, dtype)
        except Exception as e:
            logger.error("Error with %s and %s: %s", quantization, dtype, e)
            errors.append(f"Error with {quantization} and {dtype}: {e}")
        finally:
            MODULE_REGISTRY.clear()
    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
