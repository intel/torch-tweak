# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Unit tests for TorchInductorBackend."""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from tests.toy_models import ToyTorchModel
from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.backend.torch_inductor_backend import TorchInductorBackend, TorchInductorBackendConfig
from torch_tweak.torch.checkpoint.storage_tasks import TorchLoadTask, TorchSaveTask
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample

IN_FEATURES = 32
OUT_FEATURES = 5
BATCH_SIZE = 2


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(model, torch_device) -> list[Sample]:
    sample = model.sample()
    args = (sample.to(torch_device).unsqueeze(0),)
    kwargs = {}
    return [(args, kwargs)]


def move_to_dtype(sample_data, dtype):
    args, kwargs = sample_data[0]
    args = (args[0].to(dtype),)
    return [(args, kwargs)]


def backend_build(backend, dtype, model, sample_data, tmp_path, device="xpu"):
    """Build the model with the backend."""
    device = torch.device(device)
    model = model.to(device, dtype=dtype)
    data = move_to_dtype(sample_data, dtype)
    backend = backend.build(model, graph_spec=Mock(spec=GraphSpec), data=data, device=device, cache_dir=tmp_path)
    return backend


def do_test_backend(backend, dtype, model, sample_data, tmp_path):
    """Helper function to test backend with given dtype and device.

    Args:
        backend: The backend instance to test
        dtype: The dtype to use for the test
        device: The device to use for the test
    """
    backend = backend_build(backend, dtype, model, sample_data, tmp_path)

    sample_data = move_to_dtype(sample_data, dtype)
    args, kwargs = sample_data[0]

    # Verify the backend is properly initialized
    assert backend is not None
    assert hasattr(backend, "_compiled_module")

    # Test inference
    output = backend.infer(*args, **kwargs)
    assert output.shape[0] == 1
    assert output.dtype == dtype

    # Test deactivation
    backend.deactivate()


@requires_xpu
@pytest.mark.parametrize(
    "mode",
    ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
    ids=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
    ids=["float16", "bfloat16", "float32"],
)
def test_torch_inductor_backend_build(mode, dtype, model, sample_data, tmp_path):
    """Test backend build with different modes and dtypes."""
    config = TorchInductorBackendConfig(mode=mode)
    backend = TorchInductorBackend(config=config)
    do_test_backend(backend, dtype, model, sample_data, tmp_path)


@requires_xpu
def test_torch_inductor_backend_with_options(model, sample_data, tmp_path):
    """Test backend with custom options."""
    # Testing with some common inductor options
    options = {
        "max_autotune": True,
        "aggressive_fusion": True,
        "debug": True,
    }
    config = TorchInductorBackendConfig(options=options)
    backend = TorchInductorBackend(config=config)
    do_test_backend(backend, torch.float32, model, sample_data, tmp_path)


@requires_xpu
@pytest.mark.parametrize(
    "autocast_dtype",
    [torch.float16, torch.bfloat16],
    ids=["float16", "bfloat16"],
)
def test_torch_inductor_backend_with_autocast(autocast_dtype, model, sample_data, tmp_path):
    """Test backend with autocast enabled."""
    config = TorchInductorBackendConfig(autocast_enabled=True, autocast_dtype=autocast_dtype)
    backend = TorchInductorBackend(config=config)
    do_test_backend(backend, torch.float32, model, sample_data, tmp_path)


def test_torch_inductor_backend_with_mode_and_options():
    """Test backend with mode and options."""
    with pytest.raises(ValueError, match="Cannot specify both 'mode' and 'options' parameters in config. "):
        config = TorchInductorBackendConfig(mode="max-autotune", options={"max_autotune": True})
        TorchInductorBackend(config=config)


@requires_xpu
def test_serialization(model, sample_data, tmp_path):
    backend = backend_build(TorchInductorBackend(), torch.float16, model, sample_data, tmp_path)
    state_dict = backend.to_dict()  # type: ignore

    TorchSaveTask().save(tmp_path, state_dict)
    state_dict = TorchLoadTask().load(tmp_path)
    loaded_backend = TorchInductorBackend.from_dict(model, state_dict)

    loaded_backend.activate()
    sample_data = move_to_dtype(sample_data, torch.float16)
    args, kwargs = sample_data[0]
    torch.testing.assert_close(backend.infer(*args, **kwargs), loaded_backend.infer(*args, **kwargs))


class _Cache:
    pass


class _ModelWithCache(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(IN_FEATURES, OUT_FEATURES)

    def forward(self, x, cache=None):
        return self.linear(x)


@requires_xpu
def test_serialization_leaves_out_samples_a_checkpoint_cannot_hold(tmp_path):
    model = _ModelWithCache().xpu().eval()
    x = torch.randn(1, IN_FEATURES, device="xpu")
    data = [((x,), {"cache": _Cache()}), ((x,), {})]
    backend = TorchInductorBackend().build(
        model, graph_spec=Mock(spec=GraphSpec), data=data, device=torch.device("xpu"), cache_dir=tmp_path
    )

    TorchSaveTask().save(tmp_path, backend.to_dict())
    loaded_backend = TorchInductorBackend.from_dict(model, TorchLoadTask().load(tmp_path))
    loaded_backend.activate()

    torch.testing.assert_close(loaded_backend.infer(x, cache=_Cache()), backend.infer(x, cache=_Cache()))
