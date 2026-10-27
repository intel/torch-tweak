# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

import pytest
import torch
import torch.nn as nn

from torch_tweak.torch.inspecting.module_inspector import ModuleInspector


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
            self.linear = nn.Linear(20, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    return nn.Sequential(nn.Linear(10, 20), NestedModule(), nn.Linear(10, 5))


@pytest.fixture
def model_with_dict():
    """Create a model with dictionary attributes for testing."""

    class ModelWithDict(nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction = {
                "dec_rnn": nn.Linear(10, 20),
                "dec_linear": nn.Linear(20, 5),
                "dec_activation": nn.ReLU(),
            }
            self.encoder = nn.Linear(5, 10)

        def forward(self, x):
            return self.prediction["dec_rnn"](x)

    return ModelWithDict()


@pytest.fixture
def model_with_list():
    """Create a model with list attributes for testing."""

    class ModelWithList(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)]
            self.encoder = nn.Linear(5, 10)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.encoder(x)

    return ModelWithList()


@pytest.fixture
def model_with_nested_containers():
    """Create a model with nested containers for testing."""

    class ModelWithNestedContainers(nn.Module):
        def __init__(self):
            super().__init__()
            self.complex_structure = {
                "encoder": nn.Linear(10, 20),
                "decoder": {"layers": [nn.Linear(20, 15), nn.ReLU(), nn.Linear(15, 10)], "output": nn.Linear(10, 5)},
                "auxiliary": [{"head1": nn.Linear(20, 5)}, {"head2": nn.Linear(20, 3)}],
            }

        def forward(self, x):
            x = self.complex_structure["encoder"](x)
            for layer in self.complex_structure["decoder"]["layers"]:
                x = layer(x)
            x = self.complex_structure["decoder"]["output"](x)
            return x

    return ModelWithNestedContainers()


def test_initialization():
    """Test ModuleInspector initialization."""
    inspector = ModuleInspector()

    assert inspector._module_info == {}
    assert inspector._original_forward == {}
    assert inspector._inspected_objects == set()
    assert inspector._depth == 0


def test_reset(simple_model):
    """Test reset functionality."""
    # First inspect a model
    inspector = ModuleInspector()
    inspector.inspect(simple_model)
    assert len(inspector._module_info) > 0
    assert len(inspector._original_forward) > 0
    assert len(inspector._inspected_objects) > 0

    # Then reset
    inspector.reset()
    assert inspector._module_info == {}
    assert inspector._original_forward == {}
    assert inspector._inspected_objects == set()


def test_inspect_simple_model(mocker, simple_model):
    """Test inspecting a simple model."""

    mocker.patch("torch_tweak.torch.inspecting.module_inspector.DEFAULT_INSPECT_DEBUG", True)

    inspector = ModuleInspector()
    inspector.inspect(simple_model)

    # Check that all modules were registered
    assert len(inspector._module_info) == 4  # Sequential, Linear, ReLU, Linear

    # Check module hierarchy
    root_module = next(iter(inspector._module_info.values()))
    assert root_module.name == "Sequential"


def test_inspect_nested_model(nested_model):
    """Test inspecting a nested model."""
    inspector = ModuleInspector()
    inspector.inspect(nested_model)

    # Check that all modules were registered
    assert len(inspector._module_info) == 6  # Sequential, Linear, NestedModule, Linear, Linear, ReLU

    # Check module hierarchy
    root_module = next(iter(inspector._module_info.values()))
    assert root_module.name == "Sequential"


def test_inspect_model_with_dict(model_with_dict):
    """Test inspecting a model with dictionary attributes."""
    inspector = ModuleInspector()
    inspector.inspect(model_with_dict)

    # Check that all modules were registered including those in dict
    assert len(inspector._module_info) == 5  # ModelWithDict, encoder, dec_rnn, dec_linear, dec_activation

    # Check that modules in dict are registered
    modules_set = {info.object_path for info in inspector._module_info.values()}
    assert {".prediction.dec_rnn", ".prediction.dec_linear", ".prediction.dec_activation"}.issubset(modules_set)


def test_inspect_model_with_list(model_with_list):
    """Test inspecting a model with list attributes."""
    inspector = ModuleInspector()
    inspector.inspect(model_with_list)

    # Check that all modules were registered including those in list
    assert len(inspector._module_info) == 5

    # Check that modules in list are registered
    modules_set = {info.object_path for info in inspector._module_info.values()}
    assert {".encoder", ".layers.0", ".layers.1", ".layers.2"}.issubset(modules_set)


def test_inspect_model_with_nested_containers(model_with_nested_containers):
    """Test inspecting a model with nested containers."""
    inspector = ModuleInspector()
    inspector.inspect(model_with_nested_containers)

    # Check that all modules were registered including those in nested containers
    assert len(inspector._module_info) == 8  # Multiple levels of nesting

    # Check that modules in nested dict are registered
    modules_set = {info.object_path for info in inspector._module_info.values()}

    assert {
        "",
        ".complex_structure.decoder.layers.1",
        ".complex_structure.encoder",
        ".complex_structure.auxiliary.1.head2",
        ".complex_structure.auxiliary.0.head1",
        ".complex_structure.decoder.output",
        ".complex_structure.decoder.layers.0",
        ".complex_structure.decoder.layers.2",
    } == modules_set


def test_inspect_dict_with_modules():
    """Test inspecting a dictionary containing modules."""
    inspector = ModuleInspector()
    module_dict = {"linear1": nn.Linear(10, 20), "activation": nn.ReLU(), "linear2": nn.Linear(20, 5)}

    inspector.inspect(module_dict)

    # Check that all modules in dict are registered
    assert len(inspector._module_info) == 3
    modules_set = {info.object_path for info in inspector._module_info.values()}
    assert {".linear1", ".activation", ".linear2"} == modules_set


def test_inspect_list_with_modules():
    """Test inspecting a list containing modules."""
    inspector = ModuleInspector()
    module_list = [nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5)]

    inspector.inspect(module_list)

    # Check that all modules in list are registered
    assert len(inspector._module_info) == 3
    modules_set = {info.object_path for info in inspector._module_info.values()}
    assert {".0", ".1", ".2"} == modules_set


def test_inspect_nested_dict_list():
    """Test inspecting nested dictionaries and lists."""
    nested_structure = {
        "encoder": nn.Linear(10, 20),
        "decoder": {"layers": [nn.Linear(20, 15), nn.ReLU(), nn.Linear(15, 10)], "output": nn.Linear(10, 5)},
    }

    inspector = ModuleInspector()
    inspector.inspect(nested_structure)

    # Check that all modules in nested structure are registered
    assert len(inspector._module_info) == 5  # encoder, 3 layers, output

    modules_set = {info.object_path for info in inspector._module_info.values()}
    assert {".encoder", ".decoder.layers.0", ".decoder.layers.1", ".decoder.layers.2", ".decoder.output"} == modules_set


def test_inspect_dict_with_non_module_values():
    """Test inspecting a dictionary with non-module values."""
    mixed_dict = {"module": nn.Linear(10, 20), "string": "not a module", "number": 42, "list": [1, 2, 3]}

    inspector = ModuleInspector()
    inspector.inspect(mixed_dict)

    # Only the module should be registered
    assert len(inspector._module_info) == 1
    assert next(iter(inspector._module_info.values())).name == "module"


def test_inspect_list_with_non_module_items():
    """Test inspecting a list with non-module items."""
    mixed_list = [nn.Linear(10, 20), "not a module", 42, [1, 2, 3]]

    inspector = ModuleInspector()
    inspector.inspect(mixed_list)

    # Only the module should be registered
    assert len(inspector._module_info) == 1


@pytest.fixture
def deep_linears():
    """Linears placed along one nested path."""
    m1 = nn.Linear(10, 10)
    m2 = nn.Linear(10, 10)
    m3 = nn.Linear(10, 10)
    m4 = nn.Linear(10, 10)
    m5 = nn.Linear(10, 10)
    m6 = nn.Linear(10, 10)
    return m1, m2, m3, m4, m5, m6


@pytest.fixture
def deep_dict(deep_linears):
    """One nested dict. Comments are `path_depth` of each linear."""
    m1, m2, m3, m4, m5, m6 = deep_linears
    return {
        "nested": {  # path_depth 1
            "module": m1,  # path_depth 2
            "nested": {  # path_depth 2
                "module": m2,  # path_depth 3
                "nested": {  # path_depth 3
                    "module": m3,  # path_depth 4
                    "nested": {  # path_depth 4
                        "module": m4,  # path_depth 5
                        "nested": {  # path_depth 5
                            "module": m5,  # path_depth 6
                            "nested": {  # path_depth 6
                                "module": m6,  # path_depth 7
                            },
                        },
                    },
                },
            },
        },
    }


def test_inspect_recursion_depth_limit(deep_dict, deep_linears):
    """Test that recursion depth is properly limited."""
    m1, m2, m3, m4, m5, _m6 = deep_linears
    inspector = ModuleInspector()
    inspector.inspect(deep_dict)

    assert inspector._max_path_depth == 6
    assert {id(module) for module in inspector._module_info} == {id(m1), id(m2), id(m3), id(m4), id(m5)}


def test_get_modules_is_empty_when_module_is_past_max_path_depth(deep_dict, deep_linears):
    """A module past the calculated path limit is not returned."""
    m6 = deep_linears[5]
    inspector = ModuleInspector()
    inspector.inspect(deep_dict)
    m6(torch.randn(1, 10))

    assert inspector.get_modules() == []


def test_inspect_duplicate_objects():
    """Test that duplicate objects are not inspected multiple times."""
    shared_module = nn.Linear(10, 20)
    structure = {
        "dict1": {"module": shared_module},
        "dict2": {"module": shared_module},
        "list1": [shared_module],
        "list2": [shared_module],
    }

    inspector = ModuleInspector()
    inspector.inspect(structure)

    # The shared module should only be registered once
    assert len(inspector._module_info) == 1
    assert len(inspector._original_forward) == 1


def test_get_modules(simple_model):
    """Test getting executed modules."""
    inspector = ModuleInspector()
    inspector.inspect(simple_model)

    # Initially no modules should be executed
    assert len(inspector.get_modules()) == 0

    # Execute the model
    input_tensor = torch.randn(1, 10)
    simple_model(input_tensor)

    # Now we should have executed modules
    executed_modules = inspector.get_modules()
    assert len(executed_modules) > 0
    assert all(module.forward_called for module in executed_modules)


def test_get_modules_with_dict_model(model_with_dict):
    """Test getting executed modules from model with dict."""
    inspector = ModuleInspector()
    inspector.inspect(model_with_dict)

    # Initially no modules should be executed
    assert len(inspector.get_modules()) == 0

    # Execute the model
    input_tensor = torch.randn(1, 10)
    model_with_dict(input_tensor)

    # Now we should have executed modules including those from dict
    executed_modules = inspector.get_modules()
    assert len(executed_modules) > 0
    assert all(module.forward_called for module in executed_modules)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(8, 8)

    def forward(self, x):
        return self.mlp(x)


class _EncoderWithHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(8, 8)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _executed_encoder_inspector(depth: int = 0) -> ModuleInspector:
    model = _EncoderWithHead()
    inspector = ModuleInspector(depth=depth)
    inspector.inspect(model)
    model(torch.randn(1, 8))
    return inspector


def test_depth_counts_executed_modules_only():
    """`nn.ModuleList` is never executed, so it does not add a depth level."""
    depths = {m.object_path: m.depth for m in _executed_encoder_inspector()._module_info.values() if m.forward_called}

    assert depths == {"": 0, ".layers.0": 1, ".layers.1": 1, ".head": 1, ".layers.0.mlp": 2, ".layers.1.mlp": 2}


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (0, [""]),
        (1, [".layers.0", ".layers.1", ".head"]),
        (2, [".layers.0.mlp", ".layers.1.mlp", ".head"]),
        (5, [".layers.0.mlp", ".layers.1.mlp", ".head"]),
    ],
)
def test_get_modules_keeps_shallower_leaves(depth, expected):
    """`head` has no executed children, so it stays in every cut below its own depth."""
    modules = _executed_encoder_inspector(depth).get_modules()

    assert sorted(m.object_path for m in modules) == sorted(expected)


def test_get_modules_can_be_called_twice():
    inspector = _executed_encoder_inspector(depth=2)
    first = inspector.get_modules()

    assert inspector.get_modules() == first


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (0, [".encoder", ".decoder"]),
        (1, [".encoder.mlp", ".decoder.mlp"]),
    ],
)
def test_get_modules_from_non_module_root(depth, expected):
    """Executed members of a plain object are at depth 0."""

    class Pipeline:
        def __init__(self):
            self.encoder = _Block()
            self.decoder = _Block()

        def __call__(self, x):
            return self.decoder(self.encoder(x))

    pipeline = Pipeline()
    inspector = ModuleInspector(depth=depth)
    inspector.inspect(pipeline)
    pipeline(torch.randn(1, 8))

    assert [m.object_path for m in inspector.get_modules()] == expected


def test_negative_depth():
    with pytest.raises(ValueError):
        ModuleInspector(depth=-1)


def test_wrap_forward_methods(simple_model):
    """Test wrapping forward methods."""
    inspector = ModuleInspector()
    inspector.inspect(simple_model)

    # Check that forward methods were wrapped
    assert len(inspector._original_forward) == 4  # One for each module

    # Check that the wrapped forward method tracks execution
    input_tensor = torch.randn(1, 10)
    simple_model(input_tensor)

    for module_info in inspector._module_info.values():
        assert module_info.forward_called
        assert module_info.execution_count == 1
        assert module_info.total_execution_time > 0


def test_wrap_forward_methods_with_dict_model(model_with_dict):
    """Test wrapping forward methods for model with dict."""
    inspector = ModuleInspector()
    inspector.inspect(model_with_dict)

    # Check that forward methods were wrapped including those in dict
    assert len(inspector._original_forward) == 5  # One for each module

    # Check that the wrapped forward method tracks execution
    input_tensor = torch.randn(1, 10)
    model_with_dict(input_tensor)

    # Find the dec_rnn module specifically
    dec_rnn_info = next(
        (info for info in inspector._module_info.values() if info.object_path == ".prediction.dec_rnn"), None
    )
    assert dec_rnn_info is not None
    assert dec_rnn_info.forward_called
    assert dec_rnn_info.execution_count == 1
    assert dec_rnn_info.total_execution_time > 0


def test_inspect_builtin_object():
    """Test inspecting a built-in object."""
    builtin_obj = [1, 2, 3]
    inspector = ModuleInspector()
    inspector.inspect(builtin_obj)

    # Built-in objects should be skipped
    assert len(inspector._module_info) == 0
    assert len(inspector._original_forward) == 0


def test_inspect_module_with_custom_forward():
    """Test inspecting a module with a custom forward method."""

    class CustomModule(nn.Module):
        def forward(self, x, extra_arg=None):
            return x + 1 if extra_arg else x

    model = CustomModule()
    inspector = ModuleInspector()
    inspector.inspect(model)

    # Execute with different arguments
    input_tensor = torch.randn(1, 10)
    model(input_tensor)
    model(input_tensor, extra_arg=True)

    # Check execution tracking
    module_info = next(iter(inspector._module_info.values()))
    assert module_info.execution_count == 2
    assert len(module_info.output_types) == 2


def test_inspect_module_with_errors():
    """Test inspecting a module that raises errors."""

    class ErrorModule(nn.Module):
        def forward(self, x):
            raise RuntimeError("Test error")

    model = ErrorModule()
    inspector = ModuleInspector()
    inspector.inspect(model)

    # The inspection should complete without errors
    assert len(inspector._module_info) == 1

    # The forward call should raise the error
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 10))


def test_inspect_module_with_xpu():
    """Test inspecting a module on XPU if available."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    model = nn.Linear(10, 20).xpu()
    inspector = ModuleInspector()
    inspector.inspect(model)

    # Execute on XPU
    input_tensor = torch.randn(1, 10).xpu()
    model(input_tensor)

    # Check execution tracking
    module_info = next(iter(inspector._module_info.values()))
    assert module_info.forward_called
    assert module_info.execution_count == 1
    assert module_info.total_execution_time > 0


def test_inspect_module_with_multiple_forward_calls(simple_model):
    """Test inspecting a module with multiple forward calls."""
    inspector = ModuleInspector()
    inspector.inspect(simple_model)

    # Execute multiple times
    input_tensor = torch.randn(1, 10)
    for _ in range(3):
        simple_model(input_tensor)

    # Check execution tracking
    for module_info in inspector._module_info.values():
        assert module_info.forward_called
        assert module_info.execution_count == 3
        assert module_info.total_execution_time > 0
        assert module_info.average_execution_time > 0


def test_inspect_module_with_complex_output():
    """Test inspecting a module with complex output types."""

    class ComplexOutputModule(nn.Module):
        def forward(self, x):
            return {"output1": x + 1, "output2": (x * 2, x * 3), "output3": [x, x + 1]}

    model = ComplexOutputModule()
    inspector = ModuleInspector()
    inspector.inspect(model)

    # Execute
    input_tensor = torch.randn(1, 10)
    model(input_tensor)

    # Check output type tracking
    module_info = next(iter(inspector._module_info.values()))
    assert len(module_info.output_types) == 1
    output_info = module_info.output_types[0]
    assert output_info["type"] == "dict"


@pytest.fixture
def parakeet_rnnt_like_model():
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))

        def forward(self, x):
            return self.layers(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.prediction = {
                "embed": nn.Linear(10, 20),
                "dec_rnn": nn.Sequential(nn.Linear(20, 5), nn.ReLU()),
            }

        def forward(self, x):
            return self.prediction["dec_rnn"](x)

    class Joint(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def enc(self, x):
            embedding = self.decoder.prediction["embed"](x)
            return self.decoder.prediction["dec_rnn"](embedding)

    class ParakeetRNNTLikeModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.encoder = Encoder()
            self.decoder = Decoder()
            self.joint = Joint(self.decoder)

        def forward(self, x):
            return self.joint.enc(self.encoder(x))

    return ParakeetRNNTLikeModel()


def test_get_modules_with_parakeet_rnnt_like_model(parakeet_rnnt_like_model):
    """Test getting executed modules from ParakeetRNNTLikeModel."""

    inspector = ModuleInspector()
    inspector.inspect(parakeet_rnnt_like_model)

    parakeet_rnnt_like_model(torch.randn(1, 10))

    module_info = {m.object_path: m for m in inspector.get_modules()}
    assert set(module_info.keys()) == {""}


def test_get_modules_with_parakeet_rnnt_like_model_depth_1(parakeet_rnnt_like_model):
    """Test getting executed modules from ParakeetRNNTLikeModel."""

    inspector = ModuleInspector(depth=1)
    inspector.inspect(parakeet_rnnt_like_model)

    parakeet_rnnt_like_model(torch.randn(1, 10))

    module_info = {m.object_path: m for m in inspector.get_modules()}
    assert set(module_info.keys()) == {
        ".decoder.prediction.embed",
        ".decoder.prediction.dec_rnn",
        ".encoder",
    }


class ModelWithModuleForwardNotCalled(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self._not_called = nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(x)


def test_forward_not_called():
    model = ModelWithModuleForwardNotCalled()
    inspector = ModuleInspector()

    inspector.inspect(model)
    assert len(inspector._module_info) == 3

    model(torch.randn(1, 10))

    module_info = {m.object_path: m for m in inspector.get_modules()}
    assert len(module_info.keys()) == 1

    assert "" in module_info
    assert module_info[""].name == "ModelWithModuleForwardNotCalled"
    assert module_info[""].forward_called
    assert module_info[""].execution_count == 1


class SecondReferencesToModuleCustomClassBase:
    def __init__(self):
        self.components = {}

    def add_component(self, name, component):
        self.components[name] = component
        return component


class SecondReferencesToModuleCustomClass(SecondReferencesToModuleCustomClassBase):
    def __init__(self):
        super().__init__()
        self.unet = self.add_component("unet", nn.Linear(10, 20))

    def run(self, x):
        return self.unet(x)


@pytest.mark.xfail(reason="Known issue: double references to the same module returned will be the first one inspected")
def test_second_references_to_module_custom_class():
    model = SecondReferencesToModuleCustomClass()

    inspector = ModuleInspector(depth=1)
    inspector.inspect(model)

    model.run(torch.randn(1, 10))

    module_info = {m.object_path: m for m in inspector.get_modules()}
    assert len(module_info.keys()) == 1
    # we want .unet but .components.unet is returned, because of the way we inspect
    # this will break the wrap and tune functions as wrong reference is used
    assert ".unet" in module_info
    assert module_info[".unet"].name == "unet"


class SecondReferencesToModule(SecondReferencesToModuleCustomClassBase, nn.Module):
    def __init__(self):
        SecondReferencesToModuleCustomClassBase.__init__(self)
        nn.Module.__init__(self)
        self.unet = self.add_component("unet", nn.Linear(10, 20))
        self.unet2_unused = nn.Linear(10, 20)

    def forward(self, x):
        return self.unet(x)


def test_second_references_to_module_in_module():
    model = SecondReferencesToModule()

    assert "unet" in dict(model.named_children())

    assert "unet2" not in set(vars(model).keys())
    assert "unet2_unused" not in set(vars(model).keys())

    inspector = ModuleInspector(depth=1)
    inspector.inspect(model)

    model(torch.randn(1, 10))

    module_info = {m.object_path: m for m in inspector.get_modules()}
    assert len(module_info.keys()) == 1

    # this will pass as we take named_children first
    assert ".unet" in module_info
    assert module_info[".unet"].name == "unet"
