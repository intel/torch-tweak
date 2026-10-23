# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Tests for model state storage functionality."""

from unittest.mock import Mock

import pytest
import torch

from tests.toy_models.torch_models import ToyComplexPipeline, ToyPipeline, ToyTorchModel
from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.backend.torch_eager import TorchEagerBackend
from torch_tweak.torch.backend.torch_inductor_backend import TorchInductorBackend
from torch_tweak.torch.checkpoint.local_torch_storage import LocalTorchStorage
from torch_tweak.torch.checkpoint.torch_checkpoint import TorchCheckpoint
from torch_tweak.torch.module.wrapper_module import Module
from torch_tweak.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from torch_tweak.torch.tuning import tune

DEVICE_MAP_KEYS = [
    pytest.param([], id="no_device_map"),
    pytest.param(["linear1", "linear2"], id="full_device_map"),
    pytest.param(["linear1"], id="partial_device_map"),
]


def expect_device_map_not_found():
    return pytest.raises(ValueError, match="Some modules in the device_map were not found")


@pytest.fixture
def checkpoint(tmp_path):
    return TorchCheckpoint(
        LocalTorchStorage(
            tmp_path,
            remove_checkpoint_after_tune=True,  # True to check unzip functionality
        )
    )


@pytest.fixture
def model_factory():
    return lambda: ToyTorchModel().xpu()


@pytest.fixture
def pipeline_factory():
    return lambda: ToyPipeline().to("xpu")


@pytest.fixture
def complex_pipeline_factory():
    return lambda: ToyComplexPipeline().to("xpu")


@pytest.fixture
def sample(model_factory):
    yield model_factory().sample().xpu()


def _tune_save_load_helper(model_factory, samples, checkpoint_name, wrap_model_fn, checkpoint, device_map=None):
    """Tune a wrapped model, save it, load it into a fresh model and compare predictions.

    Args:
        model_factory: Function that creates a fresh model instance.
        samples (torch.Tensor): Input samples for model inference.
        checkpoint_name (str): File name of the checkpoint inside the storage root.
        wrap_model_fn: Function that takes a model and returns a wrapped version
            ready for tuning.
        checkpoint (TorchCheckpoint): Checkpoint used to save and load the model.
        device_map (dict | None): Device map passed to `checkpoint.load`.

    Raises:
        AssertionError: If model predictions after load don't match expected results.
    """
    model = model_factory()
    test_data = samples.repeat(10, 1)  # make a bs=10
    with torch.no_grad():
        expected = model(test_data)

    wrapped_model = wrap_model_fn(model)
    tune(wrapped_model, samples, batch_sizes=[1, 2])
    checkpoint.save(wrapped_model, checkpoint_name)

    # create a new model instance and load the tuned model
    model = model_factory()
    model_loaded = checkpoint.load(model, checkpoint_name, device_map=device_map)

    with torch.no_grad():
        preds = model_loaded(test_data)
    torch.testing.assert_close(preds, expected, rtol=1e-3, atol=1e-3)


def _wrap_whole_model(model, backend=None):
    return Module(model, "demo-simple", strategy=OneBackendStrategy(backend or TorchInductorBackend()))


def _wrap_linear_layers(model):
    model.linear1 = Module(model.linear1, "demo-simple1", strategy=OneBackendStrategy(TorchInductorBackend()))
    model.linear2 = Module(model.linear2, "demo-simple2", strategy=OneBackendStrategy(TorchInductorBackend()))
    return model


@requires_xpu
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(TorchInductorBackend(), id="torch_inductor"),
        pytest.param(TorchEagerBackend(), id="torch_eager"),
    ],
)  # cover both compiled and eager backends
def test_tune_save_load_whole_model(model_factory, sample, backend, checkpoint):
    """Test tuning, saving and loading a whole model wrapped in a single Module."""
    _tune_save_load_helper(
        model_factory, sample, "top_model_test.tt", lambda model: _wrap_whole_model(model, backend), checkpoint
    )


@requires_xpu
def test_tune_save_load_whole_model_with_device_map(model_factory, sample, torch_device, checkpoint):
    """Test loading a whole tuned model with a device_map entry for the top module."""
    _tune_save_load_helper(
        model_factory, sample, "top_model_test.tt", _wrap_whole_model, checkpoint, device_map={"": torch_device}
    )


@requires_xpu
def test_tune_save_load_whole_model_with_invalid_device_map(model_factory, sample, torch_device, checkpoint):
    """Test that loading a whole tuned model rejects an unknown device_map key."""
    with expect_device_map_not_found():
        _tune_save_load_helper(
            model_factory,
            sample,
            "top_model_test.tt",
            _wrap_whole_model,
            checkpoint,
            device_map={"linear123": torch_device},
        )


@requires_xpu
@pytest.mark.parametrize("device_map_keys", DEVICE_MAP_KEYS)
def test_tune_save_load_part_of_model(model_factory, sample, torch_device, checkpoint, device_map_keys):
    """Test tuning, saving and loading a model with individual layers wrapped in Modules."""
    _tune_save_load_helper(
        model_factory,
        sample,
        "partial_model_test.tt",
        _wrap_linear_layers,
        checkpoint,
        device_map=dict.fromkeys(device_map_keys, torch_device),
    )


@requires_xpu
def test_tune_save_load_part_of_model_with_invalid_device_map(model_factory, sample, torch_device, checkpoint):
    """Test that loading a model with tuned layers rejects an unknown device_map key."""
    with expect_device_map_not_found():
        _tune_save_load_helper(
            model_factory,
            sample,
            "partial_model_test.tt",
            _wrap_linear_layers,
            checkpoint,
            device_map={"linear123": torch_device},
        )


@requires_xpu
@pytest.mark.parametrize("device_map_keys", DEVICE_MAP_KEYS)
def test_tune_save_load_pipeline(pipeline_factory, sample, torch_device, checkpoint, device_map_keys):
    """Test tuning, saving and loading a pipeline whose attributes are tuned modules."""
    _tune_save_load_helper(
        pipeline_factory,
        sample,
        "pipeline_test.tt",
        _wrap_linear_layers,
        checkpoint,
        device_map=dict.fromkeys(device_map_keys, torch_device),
    )


@requires_xpu
def test_tune_save_load_pipeline_with_invalid_device_map(pipeline_factory, sample, torch_device, checkpoint):
    """Test that loading a pipeline rejects an unknown device_map key."""
    with expect_device_map_not_found():
        _tune_save_load_helper(
            pipeline_factory,
            sample,
            "pipeline_test.tt",
            _wrap_linear_layers,
            checkpoint,
            device_map={"linear123": torch_device},
        )


@requires_xpu
@pytest.mark.parametrize(
    "device_map_keys", [pytest.param([""], id="device_map"), pytest.param([], id="empty_device_map")]
)
def test_tune_save_load_complex_pipeline_layers(
    complex_pipeline_factory, sample, torch_device, checkpoint, device_map_keys
):
    """Test saving and loading each tuned layer of a pipeline component separately."""
    device_map = dict.fromkeys(device_map_keys, torch_device)
    pipeline = complex_pipeline_factory()
    test_data = sample.repeat(10, 1)  # make a bs=10
    with torch.no_grad():
        expected = pipeline(test_data)

    _wrap_linear_layers(pipeline.net)
    tune(pipeline, sample, batch_sizes=[1, 2])

    checkpoint.save(pipeline.net.linear1, "linear1_test.tt")
    checkpoint.save(pipeline.net.linear2, "linear2_test.tt")

    # create a new model instance and load the tuned model
    pipeline = complex_pipeline_factory()
    pipeline.net.linear1 = checkpoint.load(pipeline.net.linear1, "linear1_test.tt", device_map=device_map)
    pipeline.net.linear2 = checkpoint.load(pipeline.net.linear2, "linear2_test.tt", device_map=device_map)

    with torch.no_grad():
        preds = pipeline(test_data)
    torch.testing.assert_close(preds, expected, rtol=1e-4, atol=1e-4)


@requires_xpu
def test_get_pipeline_modules(pipeline_factory):
    pipeline = pipeline_factory()
    modules = TorchCheckpoint.get_pipeline_modules(pipeline)

    # Check that only torch modules are included
    assert set(modules.keys()) == {"linear1", "linear2"}
    assert len(modules) == 2
    assert isinstance(modules["linear1"], torch.nn.Linear)
    assert isinstance(modules["linear2"], torch.nn.Linear)


@requires_xpu
def test_get_pipeline_modules_empty():
    class EmptyPipeline:
        def __init__(self):
            self.non_module = "not a module"
            self.none_value = None

    modules = TorchCheckpoint.get_pipeline_modules(EmptyPipeline())

    assert len(modules) == 0
    assert isinstance(modules, dict)


@requires_xpu
def test_state_dict_from_pipeline(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.linear1 = Mock(spec=Module)
    # second module is still torch.nn.Module i.e. not tuned
    state_dict = TorchCheckpoint.state_dict_from_pipeline(pipeline)
    assert isinstance(state_dict, dict)
    assert len(state_dict) == 1
    assert "linear1" in state_dict


@requires_xpu
def test_load_state_dict_for_pipeline_unknown_module(pipeline_factory):
    with pytest.raises(ValueError, match="Some modules in the state_dict were not found in the pipeline"):
        TorchCheckpoint.load_state_dict_for_pipeline(pipeline_factory(), {"missing": {}}, device_map={})


@requires_xpu
def test_load_state_dict_for_pipeline_non_module_attribute(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.config = "not a module"
    with pytest.raises(ValueError, match="Module config is not a torch.nn.Module or WrapperModule"):
        TorchCheckpoint.load_state_dict_for_pipeline(pipeline, {"config": {}}, device_map={})


@requires_xpu
def test_state_dict_from_pipeline_no_tuned_modules(pipeline_factory):
    """Test that ValueError is raised when trying to get state dict from a pipeline with no tuned modules."""
    pipeline = pipeline_factory()
    # Both modules are regular torch.nn.Module, not wrapped in Module
    with pytest.raises(ValueError, match="No tuned modules found in the pipeline"):
        TorchCheckpoint.state_dict_from_pipeline(pipeline)
