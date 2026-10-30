# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Tests for model state storage functionality."""

import logging
from unittest.mock import Mock

import pytest
import torch

from tests.toy_models.torch_models import ToyComplexPipeline, ToyPipeline, ToyTorchModel
from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.backend.torch_eager import TorchEagerBackend
from torch_tweak.torch.backend.torch_inductor_backend import TorchInductorBackend
from torch_tweak.torch.checkpoint.local_torch_storage import LocalTorchStorage
from torch_tweak.torch.checkpoint.torch_checkpoint import TorchCheckpoint
from torch_tweak.torch.module.tunable_module import TunableModule
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


def _tune_save_load_helper(model_factory, samples, wrap_model_fn, checkpoint, device_map=None):
    """Tune a wrapped model, save it, load it into a fresh model and compare predictions.

    Args:
        model_factory: Function that creates a fresh model instance.
        samples (torch.Tensor): Input samples for model inference.
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
    checkpoint.save(wrapped_model, "checkpoint.tt")

    # create a new model instance and load the tuned model
    model = model_factory()
    model_loaded = checkpoint.load(model, "checkpoint.tt", device_map=device_map)

    with torch.no_grad():
        preds = model_loaded(test_data)
    torch.testing.assert_close(preds, expected, rtol=1e-3, atol=1e-3)


def _wrap_whole_model(model, backend=None):
    return TunableModule(model, "demo-simple", strategy=OneBackendStrategy(backend or TorchInductorBackend()))


def _wrap_linear_layers(model):
    model.linear1 = TunableModule(model.linear1, "demo-simple1", strategy=OneBackendStrategy(TorchInductorBackend()))
    model.linear2 = TunableModule(model.linear2, "demo-simple2", strategy=OneBackendStrategy(TorchInductorBackend()))
    return model


def _wrap_net_layers(pipeline):
    _wrap_linear_layers(pipeline.net)
    return pipeline


def _alias_net_linear1(pipeline):
    pipeline.linear1 = pipeline.net.linear1
    return pipeline


def _tune_aliased_pipeline(pipeline, sample):
    pipeline = _alias_net_linear1(_wrap_net_layers(pipeline))
    tune(pipeline, sample, batch_sizes=[1, 2])
    return pipeline


class _PipelineWithLayerList(ToyPipeline):
    def __init__(self):
        super().__init__()
        self.layers = [vars(self).pop("linear2")]

    def __call__(self, x):
        return self.layers[0](self.linear1(self.preprocessing(x)))


@requires_xpu
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(TorchInductorBackend(), id="torch_inductor"),
        pytest.param(TorchEagerBackend(), id="torch_eager"),
    ],
)  # cover both compiled and eager backends
def test_tune_save_load_whole_model(model_factory, sample, backend, checkpoint):
    """Test tuning, saving and loading a whole model wrapped in a single TunableModule."""
    _tune_save_load_helper(model_factory, sample, lambda model: _wrap_whole_model(model, backend), checkpoint)


@requires_xpu
def test_tune_save_load_whole_model_with_device_map(model_factory, sample, torch_device, checkpoint):
    """Test loading a whole tuned model with a device_map entry for the top module."""
    _tune_save_load_helper(model_factory, sample, _wrap_whole_model, checkpoint, device_map={"": torch_device})


@requires_xpu
def test_tune_save_load_whole_model_with_invalid_device_map(model_factory, sample, torch_device, checkpoint):
    """Test that loading a whole tuned model rejects an unknown device_map key."""
    with expect_device_map_not_found():
        _tune_save_load_helper(
            model_factory,
            sample,
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
@pytest.mark.parametrize(
    "device_map_keys",
    [
        pytest.param([], id="no_device_map"),
        pytest.param(["net.linear1", "net.linear2"], id="full_device_map"),
        pytest.param(["net.linear1"], id="partial_device_map"),
    ],
)
def test_tune_save_load_pipeline_with_nested_tuned_modules(
    complex_pipeline_factory, sample, torch_device, checkpoint, device_map_keys
):
    _tune_save_load_helper(
        complex_pipeline_factory,
        sample,
        _wrap_net_layers,
        checkpoint,
        device_map=dict.fromkeys(device_map_keys, torch_device),
    )


@requires_xpu
def test_tune_save_load_pipeline_with_nested_tuned_modules_rejects_leaf_name_in_device_map(
    complex_pipeline_factory, sample, torch_device, checkpoint
):
    with expect_device_map_not_found():
        _tune_save_load_helper(
            complex_pipeline_factory,
            sample,
            _wrap_net_layers,
            checkpoint,
            device_map={"linear1": torch_device},
        )


@requires_xpu
def test_state_dict_from_pipeline_stores_only_nested_tuned_modules(complex_pipeline_factory, sample):
    pipeline = complex_pipeline_factory()
    pipeline.net.linear1 = TunableModule(
        pipeline.net.linear1, "demo-simple1", strategy=OneBackendStrategy(TorchInductorBackend())
    )
    tune(pipeline, sample, batch_sizes=[1, 2])

    state_dict = TorchCheckpoint.state_dict_from_pipeline(pipeline)

    assert list(state_dict) == ["net.linear1"]


@requires_xpu
def test_state_dict_from_pipeline_stores_aliased_tuned_module_once(complex_pipeline_factory, sample):
    pipeline = _tune_aliased_pipeline(complex_pipeline_factory(), sample)

    state_dict = TorchCheckpoint.state_dict_from_pipeline(pipeline)

    assert list(state_dict) == ["net.linear1", "net.linear2"]


@requires_xpu
def test_tune_save_load_pipeline_restores_aliased_tuned_module_as_one_module(
    complex_pipeline_factory, sample, checkpoint
):
    checkpoint.save(_tune_aliased_pipeline(complex_pipeline_factory(), sample), "checkpoint.tt")

    restored = checkpoint.load(_alias_net_linear1(complex_pipeline_factory()), "checkpoint.tt")

    assert isinstance(restored.linear1, TunableModule)
    assert restored.linear1 is restored.net.linear1


@requires_xpu
def test_tune_save_load_pipeline_restores_tuned_module_aliased_by_its_parent(
    complex_pipeline_factory, sample, checkpoint
):
    pipeline = _wrap_net_layers(complex_pipeline_factory())
    pipeline.net.linear1_alias = pipeline.net.linear1
    tune(pipeline, sample, batch_sizes=[1, 2])
    checkpoint.save(pipeline, "checkpoint.tt")
    restored = complex_pipeline_factory()
    restored.net.linear1_alias = restored.net.linear1

    restored = checkpoint.load(restored, "checkpoint.tt")

    assert isinstance(restored.net.linear1_alias, TunableModule)
    assert restored.net.linear1_alias is restored.net.linear1


@requires_xpu
def test_load_state_dict_for_pipeline_restores_duplicate_alias_keys_as_one_module(complex_pipeline_factory, sample):
    state_dict = TorchCheckpoint.state_dict_from_pipeline(_tune_aliased_pipeline(complex_pipeline_factory(), sample))
    state_dict["linear1"] = state_dict["net.linear1"]
    restored = _alias_net_linear1(complex_pipeline_factory())

    TorchCheckpoint.load_state_dict_for_pipeline(restored, state_dict, device_map={})

    assert isinstance(restored.linear1, TunableModule)
    assert restored.linear1 is restored.net.linear1


@requires_xpu
def test_load_state_dict_for_pipeline_consumes_device_map_for_duplicate_alias_keys(
    complex_pipeline_factory, sample, torch_device
):
    state_dict = TorchCheckpoint.state_dict_from_pipeline(_tune_aliased_pipeline(complex_pipeline_factory(), sample))
    state_dict["linear1"] = state_dict["net.linear1"]
    device_map = {"linear1": torch_device, "net.linear1": torch_device}

    TorchCheckpoint.load_state_dict_for_pipeline(_alias_net_linear1(complex_pipeline_factory()), state_dict, device_map)

    assert device_map == {}


@requires_xpu
@pytest.mark.parametrize("key", ["missing", "net.missing", "missing.linear1", "net.linear1.missing"])
def test_load_state_dict_for_pipeline_rejects_missing_path(complex_pipeline_factory, key):
    with pytest.raises(ValueError, match="Some modules in the state_dict were not found in the pipeline"):
        TorchCheckpoint.load_state_dict_for_pipeline(complex_pipeline_factory(), {key: {}}, device_map={})


@requires_xpu
def test_state_dict_from_pipeline_warns_about_tuned_module_outside_module_attributes(sample, caplog):
    pipeline = _PipelineWithLayerList()
    pipeline.linear1 = TunableModule(
        pipeline.linear1, "demo-simple1", strategy=OneBackendStrategy(TorchInductorBackend())
    )
    pipeline.layers[0] = TunableModule(
        pipeline.layers[0], "demo-simple2", strategy=OneBackendStrategy(TorchInductorBackend())
    )
    tune(pipeline, sample, batch_sizes=[1, 2])
    # `tune` calls `setup_logging`, which removes the `caplog` handler from the root logger.
    logging.getLogger().addHandler(caplog.handler)
    caplog.clear()

    TorchCheckpoint.state_dict_from_pipeline(pipeline)

    assert caplog.messages == [
        "Tuned modules not saved, the pipeline does not hold them in module attributes: demo-simple2"
    ]


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


def test_get_pipeline_modules_does_not_descend_into_module_twice():
    pipeline = ToyComplexPipeline()
    pipeline.net.self_ref = pipeline.net

    modules = TorchCheckpoint.get_pipeline_modules(pipeline)

    assert list(modules) == ["net", "net.linear1", "net.linear2", "net.self_ref"]


@requires_xpu
def test_state_dict_from_pipeline(pipeline_factory):
    pipeline = pipeline_factory()
    pipeline.linear1 = Mock(spec=TunableModule)
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
    with pytest.raises(ValueError, match="Module config is not a torch.nn.Module or TunableModule"):
        TorchCheckpoint.load_state_dict_for_pipeline(pipeline, {"config": {}}, device_map={})


@requires_xpu
def test_state_dict_from_pipeline_no_tuned_modules(pipeline_factory):
    """Test that ValueError is raised when trying to get state dict from a pipeline with no tuned modules."""
    pipeline = pipeline_factory()
    # Both modules are regular torch.nn.Module, not wrapped in TunableModule
    with pytest.raises(ValueError, match="No tuned modules found in the pipeline"):
        TorchCheckpoint.state_dict_from_pipeline(pipeline)
