# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

from logging import getLogger

import pytest
import timm
import torch

from torch_tweak.torch import tune
from torch_tweak.torch.backend.torch_eager import TorchEagerBackend, TorchEagerBackendConfig
from torch_tweak.torch.module.wrapper_module import Module
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy.one_backend_strategy import OneBackendStrategy

logger = getLogger(__name__)


def do_test(backend: TorchEagerBackend):
    # given
    device = torch.device("xpu")

    model = timm.create_model("resnet18", pretrained=False)
    model.to(device)
    model.eval()
    data = torch.randn((3, 64, 64), device=device)

    with torch.no_grad():
        out = model(data.unsqueeze(0))
    expected_probs = torch.nn.functional.softmax(out[0], dim=0)

    module = Module(model, "functional-resnet18", strategy=OneBackendStrategy(backend))
    # when
    tune(module, data, batch_sizes=[2, 1], dry_run=False, disable_external_logging=False)
    # then - verify tuning
    out = module(data.unsqueeze(0))
    actual_probs = torch.nn.functional.softmax(out[0], dim=0)
    torch.testing.assert_close(actual_probs, expected_probs, rtol=1e-4, atol=1e-5)


@pytest.mark.functional
def test_tune_resnet_torch_eager():
    errors = []

    try:
        logger.info("Testing eager")
        config = TorchEagerBackendConfig()
        do_test(TorchEagerBackend(config=config))
    except Exception as e:
        logger.error("Error with eager: %s", e)
        errors.append(f"Error with eager: {e}")
    finally:
        MODULE_REGISTRY.clear()

    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


@pytest.mark.functional
def test_tune_resnet_torch_eager_autocast():
    errors = []

    dtypes = [torch.float16, torch.bfloat16]
    for dtype in dtypes:
        try:
            logger.info("Testing autocast dtype: %s", dtype)
            config = TorchEagerBackendConfig(autocast_enabled=True, autocast_dtype=dtype)
            do_test(TorchEagerBackend(config=config))
        except Exception as e:
            logger.error("Error with autocast and dtype %s: %s", dtype, e)
            errors.append(f"Error with autocast and dtype {dtype}: {e}")
        finally:
            MODULE_REGISTRY.clear()

    if errors:
        raise RuntimeError("There were some errors:\n" + "\n".join(errors))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
