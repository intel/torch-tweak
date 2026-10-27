# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

import pytest
import torch
import torch.nn as nn

from torch_tweak.torch.inspecting import inspect
from torch_tweak.torch.inspecting.module_info import InspectedModulesInfo, ModuleInfo

TEST_NUMBER_OF_ITERATIONS = 10


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
            self.linear = nn.Linear(10, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            return self.relu(self.linear(x))

    return NestedModule()


@pytest.fixture
def custom_object():
    """Create a custom object for testing."""

    class CustomObject:
        def __init__(self):
            self.num_iterations = 4
            self.linear1 = nn.Linear(10, 10)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(10, 10)

        def __call__(self, x):
            # Call linear1 multiple times
            for _ in range(self.num_iterations):
                x = self.linear1(x)
            # Then continue with the rest of the forward pass
            x = self.relu(x)
            x = self.linear2(x)
            return x

    return CustomObject()


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return torch.randn(10, 10)  # 10 samples of size 10


def test_inspect_simple_model(simple_model, sample_dataset):
    """Test inspecting a simple model."""
    # When inspecting the model
    modules_info = inspect(simple_model, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert all(isinstance(info, ModuleInfo) for info in modules)

    # Check that the model was executed
    assert any(info.forward_called for info in modules)
    assert any(info.execution_count > 0 for info in modules)
    assert any(info.total_execution_time > 0 for info in modules)


def test_inspect_nested_model(nested_model, sample_dataset):
    """Test inspecting a nested model."""
    # When inspecting the nested model
    modules_info = inspect(nested_model.linear, sample_dataset, number_of_iterations=TEST_NUMBER_OF_ITERATIONS)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0

    # Verify execution tracking
    assert modules[0].module == nested_model.linear
    assert modules[0].forward_called is True
    assert modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS


def test_inspect_nested_model_with_inference_function(nested_model, sample_dataset):
    """Test inspecting a nested model with an inference function."""

    def inference_function(x):
        return nested_model.forward(x)

    # When inspecting the nested model
    modules_info = inspect(
        nested_model,
        sample_dataset,
        inference_function=inference_function,
        number_of_iterations=TEST_NUMBER_OF_ITERATIONS,
    )

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0

    # Verify execution tracking
    assert modules[0].module == nested_model
    assert modules[0].forward_called is True
    assert modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS


def test_inspect_with_dataloader(simple_model):
    """Test inspecting with a DataLoader."""
    from torch_tweak.torch.dataloader import DataLoaderFactory

    # Create a simple dataloader
    dataset = torch.randn(10, 10)
    dataloader = DataLoaderFactory(dataset=dataset)

    # When inspecting with dataloader
    modules_info = inspect(simple_model, dataloader)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert any(info.forward_called for info in modules)


def test_inspect_with_complex_output(sample_dataset):
    """Test inspecting a model with complex output types."""

    class ComplexOutputModel(nn.Module):
        def forward(self, x):
            return {"output1": x + 1, "output2": (x * 2, x * 3), "output3": [x, x + 1]}

    model = ComplexOutputModel()

    # When inspecting the model
    modules_info = inspect(model, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0

    # Check output type tracking
    module_info = next(iter(modules))
    assert len(module_info.output_types) > 0
    assert module_info.output_types[0]["type"] == "dict"


def test_inspect_with_xpu(simple_model, sample_dataset):
    """Test inspecting a model on XPU if available."""
    if not torch.xpu.is_available():
        pytest.skip("XPU not available")

    # Move model and data to XPU
    model = simple_model.xpu()
    dataset = sample_dataset.xpu()

    # When inspecting on XPU
    modules_info = inspect(model, dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert any(info.forward_called for info in modules)
    assert any(info.total_execution_time > 0 for info in modules)


def test_inspect_with_mixed_precision(sample_dataset):
    """Test inspecting a model with mixed precision."""

    class MixedPrecisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            # First layer in float16
            self.linear1 = nn.Linear(10, 20).to(torch.float16)
            self.relu = nn.ReLU()
            # Second layer in float32
            self.linear2 = nn.Linear(20, 5).to(torch.float32)

        def forward(self, x):
            # Convert input to float16 for first layer
            x = x.to(torch.float16)
            x = self.linear1(x)
            x = self.relu(x)
            # Convert to float32 for second layer
            x = x.to(torch.float32)
            x = self.linear2(x)
            return x

    model = MixedPrecisionModel()

    # When inspecting the model
    modules_info = inspect(model, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0

    # Check precision tracking
    module_info = next(iter(modules))
    precisions = module_info.precisions
    assert torch.float16 in precisions
    assert torch.float32 in precisions


def test_inspect_with_invalid_input():
    """Test inspecting with invalid input."""
    # When inspecting with invalid input
    with pytest.raises(TypeError):
        inspect(None, torch.randn(10, 10))


def test_inspect_with_multiple_forward_calls(simple_model, sample_dataset):
    """Test inspecting with multiple forward calls."""
    # When inspecting with multiple forward calls
    modules_info = inspect(simple_model, sample_dataset)

    # Then verify execution tracking
    module_info = next(iter(modules_info.get_modules()))
    assert module_info.execution_count > 0
    assert module_info.total_execution_time > 0
    assert module_info.average_execution_time > 0


def test_inspect_with_custom_object(sample_dataset):
    """Test inspecting a custom object with PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
            self.extra_data = torch.randn(5, 5)

        def __call__(self, x):
            return self.model(x)

    custom_obj = CustomObject()

    # When inspecting custom object
    modules_info = inspect(custom_obj, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert any(info.forward_called for info in modules)


def test_inspect_with_custom_object_without_modules(sample_dataset):
    """Test inspecting a custom object without PyTorch modules."""

    class CustomObject:
        def __init__(self):
            self.data = torch.randn(10, 10)

        def __call__(self, x):
            return x + self.data

    custom_obj = CustomObject()

    # When inspecting custom object without modules
    modules_info = inspect(custom_obj, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)
    assert len(modules_info.get_modules()) == 0  # No modules to inspect


def test_inspect_with_custom_object_with_attributes(sample_dataset):
    """Test inspecting a custom object with various attribute types."""

    class CustomObject:
        def __init__(self):
            self.model = nn.Linear(10, 20)
            self.data = torch.randn(5, 5)
            self.list_data = [1, 2, 3]
            self.dict_data = {"key": "value"}

        def __call__(self, x):
            return self.model(x)

    custom_obj = CustomObject()

    # When inspecting custom object with various attributes
    modules_info = inspect(custom_obj, sample_dataset)

    # Then verify the results
    assert isinstance(modules_info, InspectedModulesInfo)

    modules = modules_info.get_modules()

    assert len(modules) > 0
    assert any(info.forward_called for info in modules)


def test_get_modules_after_inspection(custom_object, sample_dataset):
    """Test getting executed modules after inspection."""
    # When inspecting custom object with various attributes
    modules_info = inspect(custom_object, sample_dataset, number_of_iterations=TEST_NUMBER_OF_ITERATIONS)

    # Then verify the results
    executed_modules = modules_info.get_modules()

    assert len(executed_modules) == 3
    assert executed_modules[0].name == "linear1"
    assert executed_modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS * custom_object.num_iterations
    for idx in [1, 2]:
        assert executed_modules[idx].name in ["linear2", "relu"]
        assert executed_modules[idx].execution_count == TEST_NUMBER_OF_ITERATIONS


def test_get_modules_after_inspection_with_min_execution_percentage(custom_object, sample_dataset):
    """Test getting executed modules with a minimum execution percentage."""
    # When inspecting custom object with various attributes
    modules_info = inspect(custom_object, sample_dataset, number_of_iterations=TEST_NUMBER_OF_ITERATIONS)

    # Then verify the results
    executed_modules = modules_info.get_modules(min_execution_percentage=0.25)

    assert len(executed_modules) == 1
    assert executed_modules[0].name == "linear1"
    assert executed_modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS * custom_object.num_iterations


@pytest.fixture
def module_past_default_path_limit():
    """A linear at path depth 7. Default depth=0 stops at path depth 6."""
    return {
        "nested": {
            "nested": {
                "nested": {
                    "nested": {
                        "nested": {
                            "nested": {"module": nn.Linear(4, 4)},
                        },
                    },
                },
            },
        },
    }


def _echo(tensor):
    return tensor


def test_inspect_warns_when_the_walk_stops_before_any_module(module_past_default_path_limit, capsys):
    modules_info = inspect(module_past_default_path_limit, torch.zeros(1, 4), inference_function=_echo)

    assert modules_info.get_modules() == []
    assert (
        "No modules found. The walk stopped at path depth 6, calculated from depth=0. Increase depth to walk further."
    ) in capsys.readouterr().out


def test_inspect_warning_can_be_disabled(module_past_default_path_limit, capsys):
    modules_info = inspect(
        module_past_default_path_limit,
        torch.zeros(1, 4),
        inference_function=_echo,
        warn_if_no_modules=False,
    )

    assert modules_info.get_modules() == []
    assert "No modules found" not in capsys.readouterr().out


def test_inspect_does_not_warn_when_a_module_is_found(simple_model, sample_dataset, capsys):
    inspect(simple_model, sample_dataset, number_of_iterations=1, warmup_iterations=1)

    assert "No modules found" not in capsys.readouterr().out


def test_get_modules_after_inspection_with_limit(custom_object, sample_dataset):
    """Test getting executed modules with a limit."""
    # When inspecting custom object with various attributes
    modules_info = inspect(custom_object, sample_dataset, number_of_iterations=TEST_NUMBER_OF_ITERATIONS)

    # Then verify the results
    executed_modules = modules_info.get_modules(limit=2)
    assert len(executed_modules) == 2
    assert executed_modules[0].name == "linear1"
    assert executed_modules[0].execution_count == TEST_NUMBER_OF_ITERATIONS * custom_object.num_iterations

    assert executed_modules[1].name in ["linear2", "relu"]
    assert executed_modules[1].execution_count == TEST_NUMBER_OF_ITERATIONS
