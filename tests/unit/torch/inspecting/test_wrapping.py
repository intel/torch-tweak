# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

import pytest
import torch
import torch.nn as nn

from tests.toy_models import ToyTorchModel
from torch_tweak.torch.backend import OpenVINOBackend, TorchEagerBackend, TorchInductorBackend
from torch_tweak.torch.inspecting.inspecting import inspect
from torch_tweak.torch.inspecting.module_info import ModuleInfo
from torch_tweak.torch.inspecting.module_inspector import DictOfModulesInfo, ListOfModulesInfo
from torch_tweak.torch.inspecting.wrapping import wrap
from torch_tweak.torch.module.sample_metadata import SampleMetadata
from torch_tweak.torch.module.wrapper_module import Module, ModuleState
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy import FirstWinsStrategy, OneBackendStrategy
from torch_tweak.torch.tuning import tune


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))


@pytest.fixture
def nested_model():
    """Create a nested model for testing."""

    class NestedModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    return NestedModule()


def test_wrap_root_module(simple_model):
    """Test wrapping the root module."""
    # When wrapping with root module
    wrapped_model = wrap(
        simple_model,
        [
            ModuleInfo(module=simple_model),
        ],
    )

    # Then the model should be wrapped in a Module
    assert isinstance(wrapped_model, Module)
    assert wrapped_model._self_name == "Sequential"


def test_wrap_submodule(nested_model):
    """Test wrapping a submodule."""
    parent = ModuleInfo(module=nested_model)
    # When wrapping a submodule
    wrapped_model = wrap(
        nested_model,
        [
            ModuleInfo(name="linear", module=nested_model.linear, parent=parent, object_path=".linear"),
        ],
    )

    # Then the submodule should be wrapped
    assert not isinstance(wrapped_model, Module)

    assert isinstance(wrapped_model.linear, Module)
    assert wrapped_model.linear._self_name == "linear"

    # And the original model should be returned
    assert wrapped_model is nested_model


def test_wrap_multiple_modules(nested_model):
    """Test wrapping multiple modules."""
    # When wrapping multiple modules
    parent = ModuleInfo(module=nested_model)

    # When wrapping a submodule
    wrapped_model = wrap(
        nested_model,
        [
            ModuleInfo(name="linear", module=nested_model.linear, parent=parent, object_path=".linear"),
            ModuleInfo(name="relu", module=nested_model.relu, parent=parent, object_path=".relu"),
        ],
    )

    # Then both submodules should be wrapped
    assert isinstance(nested_model.linear, Module)
    assert isinstance(nested_model.relu, Module)
    assert nested_model.linear._self_name == "linear"
    assert nested_model.relu._self_name == "relu"

    # And the original model should be returned
    assert wrapped_model is nested_model


def test_wrap_with_strategy(simple_model):
    """Test wrapping with a strategy."""
    strategy = FirstWinsStrategy(backends=[TorchInductorBackend(), TorchEagerBackend()])

    # When wrapping with a strategy
    wrapped_model = wrap(
        simple_model,
        [
            ModuleInfo(module=simple_model),
        ],
        strategy=strategy,
    )

    # Then the module should be wrapped with the strategy
    assert isinstance(wrapped_model, Module)
    assert wrapped_model._self_strategy.__class__ == strategy.__class__
    assert wrapped_model._self_strategy_list is None
    assert wrapped_model._self_strategy_map is None


def test_wrap_with_strategies_list(simple_model):
    """Test wrapping with a strategy."""
    strategies = [
        FirstWinsStrategy(backends=[OpenVINOBackend(), TorchInductorBackend()]),
        OneBackendStrategy(backend=TorchInductorBackend()),
    ]

    # When wrapping with a strategy
    wrapped_model = wrap(
        simple_model,
        [
            ModuleInfo(module=simple_model),
        ],
        strategies=strategies,
    )

    # Then the module should be wrapped with the strategy
    assert isinstance(wrapped_model, Module)
    assert wrapped_model._self_strategy is None
    assert wrapped_model._self_strategy_map is None


def test_wrap_with_strategies_map(simple_model):
    """Test wrapping with a strategy."""
    model = ToyTorchModel().to("cpu").eval()
    samples = model.inputs(device="cpu")
    sample_metadata1 = SampleMetadata.from_inputs(args=samples[0], kwargs={})
    sample_metadata2 = SampleMetadata.from_inputs(args=samples[0], kwargs={})

    strategies = {
        sample_metadata1: FirstWinsStrategy(backends=[TorchInductorBackend(), OpenVINOBackend()]),
        sample_metadata2: OneBackendStrategy(backend=TorchInductorBackend()),
    }

    # When wrapping with a strategy
    wrapped_model = wrap(
        simple_model,
        [
            ModuleInfo(module=simple_model),
        ],
        strategies=strategies,
    )

    # Then the module should be wrapped with the strategy
    assert isinstance(wrapped_model, Module)
    assert wrapped_model._self_strategy is None
    assert wrapped_model._self_strategy_list is None
    assert set(wrapped_model._self_strategy_map.keys()) == set(strategies.keys())


def test_wrap_empty_modules_list(simple_model):
    """Test wrapping with an empty modules list."""
    # When wrapping with an empty list
    wrapped_model = wrap(simple_model, [])

    # Then the original model should be returned unchanged
    assert not isinstance(wrapped_model, Module)
    assert wrapped_model is simple_model


def test_wrap_module_execution(simple_model):
    """Test that wrapped modules can still be executed."""
    # When wrapping modules
    wrapped_model = wrap(simple_model, [ModuleInfo(module=simple_model)])

    # Then the model should still work
    input_tensor = torch.randn(1, 10)
    output = wrapped_model(input_tensor)
    assert output.shape == (1, 5)


def test_wrap_module_state_preservation(simple_model):
    """Test that module state is preserved after wrapping."""
    # Set some state in the original model
    simple_model[0].weight.data.fill_(1.0)
    original_weight = simple_model[0].weight.clone()

    # When wrapping the module
    wrapped_model = wrap(simple_model, [ModuleInfo(module=simple_model)])

    # Then the state should be preserved
    assert torch.allclose(wrapped_model[0].weight, original_weight)


def test_wrap_custom_object():
    """Test wrapping a custom object with PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())

        def __call__(self, x):
            return self.model(x)

    custom_obj = CustomObject()
    parent = ModuleInfo(module=custom_obj)

    strategy = FirstWinsStrategy(backends=[TorchInductorBackend(), OpenVINOBackend()])

    module_info = [ModuleInfo(name="model", module=custom_obj.model, parent=parent, object_path=".model")]

    # When wrapping custom object
    wrapped_obj = wrap(custom_obj, module_info, strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert isinstance(wrapped_obj.model, Module)


def test_wrap_custom_object_nested():
    """Test wrapping a custom object with nested PyTorch modules."""

    class CustomLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)

        def forward(self, x):
            return self.linear(x)

    class CustomObject:
        def __init__(self):
            self.layer = CustomLayer()

        def __call__(self, x):
            return self.layer(x)

    custom_obj = CustomObject()
    parent = ModuleInfo(module=custom_obj)

    strategy = FirstWinsStrategy(backends=[TorchInductorBackend(), OpenVINOBackend()])

    module_info = [ModuleInfo(name="layer", module=custom_obj.layer, parent=parent, object_path=".layer")]

    # When wrapping custom object with nested module
    wrapped_obj = wrap(custom_obj, module_info, strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert isinstance(wrapped_obj.layer, Module)


def test_wrap_custom_object_multiple_modules():
    """Test wrapping a custom object with multiple PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.model1 = nn.Linear(10, 20)
            self.model2 = nn.Linear(20, 5)

        def __call__(self, x):
            x = self.model1(x)
            return self.model2(x)

    custom_obj = CustomObject()
    parent = ModuleInfo(module=custom_obj)

    strategy = FirstWinsStrategy(backends=[TorchInductorBackend(), OpenVINOBackend()])

    module_info = [
        ModuleInfo(name="model1", module=custom_obj.model1, parent=parent, object_path=".model1"),
        ModuleInfo(name="model2", module=custom_obj.model2, parent=parent, object_path=".model2"),
    ]

    # When wrapping multiple modules
    wrapped_obj = wrap(custom_obj, module_info, strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert isinstance(wrapped_obj.model1, Module)
    assert isinstance(wrapped_obj.model2, Module)


def test_wrap_custom_object_without_modules():
    """Test wrapping a custom object without PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.data = torch.randn(5, 5)

        def __call__(self, x):
            return x + self.data

    custom_obj = CustomObject()

    strategy = FirstWinsStrategy(backends=[TorchInductorBackend(), OpenVINOBackend()])

    # When wrapping custom object without modules
    wrapped_obj = wrap(custom_obj, [], strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert not isinstance(wrapped_obj.data, Module)


def test_wrap_custom_object_with_attributes():
    """Test wrapping a custom object with various attribute types."""

    class CustomObject:
        def __init__(self):
            self.model = nn.Linear(10, 20)
            self.data = torch.randn(5, 5)
            self.list_data = [1, 2, 3]
            self.dict_data = {"key": "value"}

        def __call__(self, x):
            return self.model(x)

    custom_obj = CustomObject()
    parent = ModuleInfo(module=custom_obj)

    strategy = FirstWinsStrategy(backends=[TorchInductorBackend(), OpenVINOBackend()])

    module_info = [ModuleInfo(name="model", module=custom_obj.model, parent=parent, object_path=".model")]

    # When wrapping custom object with various attributes
    wrapped_obj = wrap(custom_obj, module_info, strategy=strategy)

    # Then verify the results
    assert isinstance(wrapped_obj, CustomObject)
    assert isinstance(wrapped_obj.model, Module)

    assert not isinstance(wrapped_obj.data, Module)
    assert not isinstance(wrapped_obj.list_data, Module)
    assert not isinstance(wrapped_obj.dict_data, Module)


def test_wrap_module_in_list():
    """Test wrapping a module in a list."""
    model = [nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)]
    parent = ListOfModulesInfo(module=model)

    wrapped_model = wrap(model, [ModuleInfo(name="0", module=model[0], parent=parent, object_path=".0")])
    assert not isinstance(wrapped_model, Module)
    assert isinstance(wrapped_model[0], Module)
    assert not isinstance(wrapped_model[1], Module)
    assert not isinstance(wrapped_model[2], Module)


def test_wrap_module_in_dict():
    """Test wrapping a module in a dict."""
    model = {"linear": nn.Linear(10, 20), "relu": nn.ReLU(), "linear2": nn.Linear(20, 5)}
    parent = DictOfModulesInfo(module=model)

    wrapped_model = wrap(
        model, [ModuleInfo(name="linear", module=model["linear"], parent=parent, object_path=".linear")]
    )
    assert not isinstance(wrapped_model, Module)
    assert isinstance(wrapped_model["linear"], Module)
    assert not isinstance(wrapped_model["relu"], Module)
    assert not isinstance(wrapped_model["linear2"], Module)


def test_wrap_nested_diffrent_types():
    class NestedDifferentModules(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.dict = {"linear": nn.Linear(10, 20)}
            self.list = [nn.Linear(10, 20)]

        def forward(self, x):
            return self.relu(self.linear(x))

    model = NestedDifferentModules()
    parent = ModuleInfo(module=model)
    parent_dict = DictOfModulesInfo(module=model.dict, parent=parent)
    parent_list = ListOfModulesInfo(module=model.list, parent=parent)

    modules = [
        ModuleInfo(name="linear", module=model.linear, parent=parent, object_path=".linear"),
        ModuleInfo(name="linear", module=model.dict["linear"], parent=parent_dict, object_path=".dict.linear"),
        ModuleInfo(name="0", module=model.list[0], parent=parent_list, object_path=".list.0"),
    ]

    wrapped_model = wrap(model, modules)
    assert not isinstance(wrapped_model, Module)
    assert isinstance(wrapped_model.linear, Module)
    assert isinstance(wrapped_model.dict["linear"], Module)
    assert isinstance(wrapped_model.list[0], Module)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(8, 8)

    def forward(self, x):
        return self.mlp(x)


class _EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.ModuleList([_Block(), _Block()])
        self.dec = nn.ModuleList([_Block()])

    def forward(self, x):
        for block in [*self.enc, *self.dec]:
            x = block(x)
        return x


def test_wrap_names_modules_by_object_path():
    """Siblings with the same attribute name get distinct wrapper names."""
    model = _EncoderDecoder()
    parent = ModuleInfo(module=model)
    enc = ListOfModulesInfo(module=model.enc, parent=parent, object_path=".enc")
    dec = ListOfModulesInfo(module=model.dec, parent=parent, object_path=".dec")

    wrap(
        model,
        [
            ModuleInfo(name="0", module=model.enc[0], parent=enc, object_path=".enc.0"),
            ModuleInfo(name="1", module=model.enc[1], parent=enc, object_path=".enc.1"),
            ModuleInfo(name="0", module=model.dec[0], parent=dec, object_path=".dec.0"),
        ],
    )

    assert [model.enc[0].name, model.enc[1].name, model.dec[0].name] == ["enc.0", "enc.1", "dec.0"]
    assert set(MODULE_REGISTRY.modules) == {"enc.0", "enc.1", "dec.0"}


def test_tune_reaches_every_module_with_a_repeated_name(tmp_path, monkeypatch):
    """Every wrapped `mlp` is tuned, not only the last one registered."""
    monkeypatch.setenv("TORCH_TWEAK_CACHE_DIR", str(tmp_path))
    model = _EncoderDecoder().eval()
    sample = torch.randn(8)
    modules_info = inspect(model, sample, number_of_iterations=1, warmup_iterations=1, depth=2)
    strategy = FirstWinsStrategy(backends=[TorchEagerBackend()])

    wrap(model, modules_info.get_modules(), strategy=strategy)
    tune(model, sample, batch_sizes=[1, 2], device="cpu")

    blocks = [*model.enc, *model.dec]
    assert sorted(MODULE_REGISTRY.modules) == ["dec.0.mlp", "enc.0.mlp", "enc.1.mlp"]
    assert [block.mlp.state for block in blocks] == [ModuleState.TUNED] * 3
