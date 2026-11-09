# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torchao.quantization import Int8WeightOnlyConfig, PerGroup
from torchao.utils import is_sm_at_least_89

from tests.toy_models.torch_models import ToyTorchModel
from tests.utilities.helpers import assert_only_primitives, requires_xpu, save_and_load_weights_only
from torch_tweak.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig
from torch_tweak.torch.checkpoint.storage_tasks import TorchLoadTask, TorchSaveTask
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.module.sample_metadata import SampleMetadata


@pytest.fixture
def model(torch_device) -> nn.Module:
    return ToyTorchModel().to(torch_device).eval()


@pytest.fixture
def sample_data(model, torch_device) -> list[Sample]:
    sample = model.sample()
    args = (sample.to(torch_device).unsqueeze(0).repeat(32, 1),)
    kwargs = {}
    return [(args, kwargs)]


def move_to_dtype(sample_data, dtype):
    args, kwargs = sample_data[0]
    args = (args[0].to(dtype),)
    return [(args, kwargs)]


def build_backend(backend, dtype, model, sample_data, torch_device, tmp_path):
    metadata = Mock(SampleMetadata)
    sample_data = move_to_dtype(sample_data, dtype)
    model.to(dtype)
    return backend.build(model, metadata, sample_data, device=torch_device, cache_dir=tmp_path)


def test_torchao_config_key():
    config = TorchAOBackendConfig(quantization="int8wo")
    key1 = config.key()
    key2 = config.key()

    assert key1 == key2


def test_torchao_config_describe():
    config = TorchAOBackendConfig(quantization="int8wo")
    describe = config.describe()
    assert describe == "quantization_config=Int8WeightOnlyConfig()"


def test_torchao_config_initialization():
    # Test valid initialization with quantization type
    for quantization in TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys():
        TorchAOBackendConfig(quantization=quantization)  # type: ignore

    # Test valid initialization with quantization config
    config = Int8WeightOnlyConfig()
    TorchAOBackendConfig(quantization_config=config)

    # Test invalid initialization with both parameters
    with pytest.raises(ValueError, match="Only one of quantization or quantization_config should be provided."):
        TorchAOBackendConfig(quantization="int8wo", quantization_config=config)

    # Test invalid initialization with neither parameter
    with pytest.raises(ValueError, match="Either quantization or quantization_config should be provided."):
        TorchAOBackendConfig()


def do_test_backend(backend, dtype, model, sample_data, torch_device, tmp_path):
    """Helper function to test backend with given dtype data.

    Args:
        backend: The backend instance to test
        dtype: The dtype to use for the test
    """
    backend = build_backend(backend, dtype, model, sample_data, torch_device, tmp_path)
    sample_data = move_to_dtype(sample_data, dtype)
    args, kwargs = sample_data[0]

    # then
    assert backend is not None
    try:
        backend.infer(*args, **kwargs)
    finally:
        backend.deactivate()


@requires_xpu
@pytest.mark.parametrize("quantization", TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys())
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=["bfloat16", "float16", "float32"],
)
def test_torchao_backend_build(quantization, dtype, model, sample_data, torch_device, tmp_path):
    if quantization in ["fp8wo", "fp8dq"] and not is_sm_at_least_89():
        pytest.skip("fp8wo and fp8dq are not supported on this device")

    config = TorchAOBackendConfig(quantization=quantization)
    backend = TorchAOBackend(config=config)
    do_test_backend(backend, dtype, model, sample_data, torch_device, tmp_path)


@requires_xpu
@pytest.mark.parametrize(
    "quantization_config",
    [Int8WeightOnlyConfig(granularity=PerGroup(16))],
    ids=["int8wo with different group size"],
)
def test_torchao_backend_build_with_user_config(quantization_config, model, sample_data, torch_device, tmp_path):
    config = TorchAOBackendConfig(quantization_config=quantization_config)
    backend = TorchAOBackend(config=config)
    do_test_backend(backend, torch.bfloat16, model, sample_data, torch_device, tmp_path)


def test_invalid_quantization_type():
    with pytest.raises(ValueError):
        TorchAOBackendConfig(quantization="invalid_type")  # type: ignore


def test_torchao_config_key_differs_per_quantization():
    keys = {q: TorchAOBackendConfig(quantization=q).key() for q in TorchAOBackendConfig._QUANTIZATION_CONFIGS}

    assert len(set(keys.values())) == len(keys)


@pytest.mark.parametrize("quantization", TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys())
def test_torchao_config_to_dict_holds_only_primitives(quantization):
    config = TorchAOBackendConfig(quantization=quantization)

    assert_only_primitives(config.to_dict())


@pytest.mark.parametrize("quantization", TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys())
def test_torchao_config_to_from_dict(quantization):
    config = TorchAOBackendConfig(quantization=quantization)

    restored = TorchAOBackendConfig.from_dict(config.to_dict())

    assert restored.quantization_config == config.quantization_config


@pytest.mark.parametrize("quantization", TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys())
def test_torchao_config_survives_weights_only_load(quantization, tmp_path):
    config = TorchAOBackendConfig(quantization=quantization)

    loaded = save_and_load_weights_only(config.to_dict(), tmp_path)

    assert TorchAOBackendConfig.from_dict(loaded).quantization_config == config.quantization_config


def test_torchao_config_to_from_dict_keeps_offload_module():
    config = TorchAOBackendConfig(quantization="int8wo", offload_module=False)

    restored = TorchAOBackendConfig.from_dict(config.to_dict())

    assert restored.offload_module is False


def test_torchao_config_from_dict_without_offload_module_offloads():
    payload = {"quantization_config": TorchAOBackendConfig(quantization="int8wo").to_dict()["quantization_config"]}

    restored = TorchAOBackendConfig.from_dict(payload)

    assert restored.offload_module is True


def test_torchao_config_describe_names_offload_module_false():
    config = TorchAOBackendConfig(quantization="int8wo", offload_module=False)

    assert config.describe() == "quantization_config=Int8WeightOnlyConfig(),offload_module=False"


@requires_xpu
def test_module_stays_on_xpu_when_offload_module_is_false(sample_data, tmp_path):
    model = ToyTorchModel().to("xpu").eval()
    backend = TorchAOBackend(TorchAOBackendConfig(quantization="int8wo", offload_module=False))

    build_backend(backend, torch.float32, model, sample_data, torch.device("xpu"), tmp_path)

    assert next(model.parameters()).device.type == "xpu"
    backend.deactivate()


def test_torchao_config_to_from_dict_keeps_user_config():
    config = TorchAOBackendConfig(quantization_config=Int8WeightOnlyConfig(granularity=PerGroup(16)))

    restored = TorchAOBackendConfig.from_dict(config.to_dict())

    assert restored.quantization_config == config.quantization_config


@pytest.mark.parametrize(
    "raw,match",
    [
        pytest.param("evil", "expected a serialized torchao config dict", id="not-a-dict"),
        pytest.param([], "expected a serialized torchao config dict", id="list"),
        pytest.param({"_data": {}}, "expected a serialized torchao config dict", id="missing-type"),
        # torchao resolves the stored name against its own modules, which export more than configs.
        pytest.param(
            {"_type": "torch.dtype", "_data": "float32"}, "is not a supported AOBaseConfig subclass", id="dtype"
        ),
        # quantize_ is a plain function exported by torchao.quantization.
        pytest.param({"_type": "quantize_", "_data": {}}, "not a class", id="callable"),
    ],
)
def test_torchao_config_from_dict_rejects_bad_top_level_type(raw, match):
    """A crafted payload must raise ValueError, not AttributeError or KeyError."""
    with pytest.raises(ValueError, match=match):
        TorchAOBackendConfig.from_dict({"quantization_config": raw})


@pytest.mark.parametrize(
    "nested,match",
    [
        # Without this, config_from_dict calls quantize_ with attacker-chosen arguments.
        pytest.param({"_type": "quantize_", "_data": {"model": 1, "config": 2}}, "not a class", id="callable"),
        pytest.param({"_type": "NoSuchThing", "_data": {}}, "not exported by any allowed", id="unknown"),
        # config_from_dict resolves a torch.dtype payload as getattr(torch, _data).
        pytest.param({"_type": "torch.dtype", "_data": "load"}, "does not name a torch dtype", id="fake-dtype"),
    ],
)
def test_torchao_config_from_dict_rejects_bad_nested_type(nested, match):
    """Nested types are validated too, not only the top-level one."""
    payload = {
        "quantization_config": {
            "_type": "Int8WeightOnlyConfig",
            "_version": 2,
            "_data": {"granularity": nested},
        }
    }

    with pytest.raises(ValueError, match=match):
        TorchAOBackendConfig.from_dict(payload)


@requires_xpu
@pytest.mark.parametrize("quantization", TorchAOBackendConfig._QUANTIZATION_CONFIGS.keys())
def test_serialization(quantization, tmp_path, model, sample_data, torch_device):
    dtype = torch.float16
    if quantization in ["fp8wo", "fp8dq"] and not is_sm_at_least_89():
        pytest.skip("fp8wo and fp8dq are not supported on this device")

    config = TorchAOBackendConfig(quantization=quantization)
    backend = build_backend(TorchAOBackend(config=config), dtype, model, sample_data, torch_device, tmp_path)
    sample_data = move_to_dtype(sample_data, dtype)
    args, kwargs = sample_data[0]
    loaded_backend = None
    try:
        expected = backend.infer(*args, **kwargs)
        model.xpu()

        state_dict = backend.to_dict()  # type: ignore

        # Go through the production save task so the weights-only guard is exercised too.
        TorchSaveTask().save(tmp_path, state_dict)
        backend.deactivate()
        backend = None

        loaded_backend = TorchAOBackend.from_dict(model, TorchLoadTask().load(tmp_path))
        loaded_backend.activate()
        actual = loaded_backend.infer(*args, **kwargs)
        torch.testing.assert_close(expected, actual)
        loaded_backend.deactivate()
        loaded_backend = None
    except Exception as e:
        # do cleanup in case of an exception, torchao backend is susceptible to memory issues when not deactivated
        if backend:
            backend.deactivate()
        if loaded_backend:
            loaded_backend.deactivate()
        raise e
