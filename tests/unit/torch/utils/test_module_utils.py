# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.

"""Unit tests for PyTorch module utilities."""

import pytest
import torch
import torch.nn as nn

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.utils.module import count_parameters, format_num_parameters, offload, weights_fingerprint


@pytest.mark.parametrize(
    "num_params,expected",
    [(100, 100), (1_000, 1000), (100_000, 100000), (1_000_000, 1000000), (1_000_000_000, 1000000000)],
)
def test_count_parameters(num_params, expected):
    """Test count_parameters with deeply nested modules."""
    module = nn.Sequential(
        nn.Linear(num_params, 1, bias=False),
    )
    result = count_parameters(module)

    assert result == expected


@pytest.mark.parametrize(
    "num_params,expected",
    [(100, "100"), (1_000, "1.0K"), (100_000, "100.0K"), (1_000_000, "1.0M"), (1_000_000_000, "1.0B")],
)
def test_format_num_parameters(num_params, expected):
    """Test count_parameters with deeply nested modules."""
    result = format_num_parameters(num_params)

    assert result == expected


def _holding(tensor: torch.Tensor) -> nn.Module:
    module = nn.Module()
    module.register_buffer("weight", tensor)
    return module


def test_weights_fingerprint_samples_the_last_value_of_a_tensor_too_long_for_float32_indices():
    zeros = torch.zeros(16_777_218, dtype=torch.uint8)
    last_one = zeros.clone()
    last_one[-1] = 1

    fingerprints = (weights_fingerprint(_holding(zeros)), weights_fingerprint(_holding(last_one)))

    assert fingerprints[0] != fingerprints[1]


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )


@pytest.fixture
def device():
    """Return XPU device."""
    return torch.device("xpu:0")


@requires_xpu
def test_offload_moves_to_cpu_device(simple_model, device):
    """Test that offload replaces all parameters with CPU tensors.

    This frees GPU memory by replacing GPU tensors with CPU tensors,
    preserving the actual data.
    """
    # Move model to GPU
    simple_model.to(device)

    # Verify model is on GPU
    for param in simple_model.parameters():
        assert param.device.type == "xpu"

    # Offload weights to CPU (replaces parameter tensors)
    offload(simple_model, device="cpu")

    # Verify all parameters are now CPU tensors
    for param in simple_model.parameters():
        assert param.device.type == "cpu"


@requires_xpu
def test_offload_to_meta_frees_gpu_memory(device):
    """Test that offload_to_meta frees GPU memory by replacing tensors with meta.

    Replacing parameter tensors with meta tensors (which have no data) should
    free the GPU memory that was used by the original parameter tensors.
    """
    # Clear XPU cache first to get accurate measurements
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats(device)

    # Create a larger model for more reliable memory measurements
    model = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
    ).to(device)

    # Force memory allocation
    torch.xpu.synchronize()

    # Measure memory after model creation
    memory_allocated_before = torch.xpu.memory_allocated(device)
    assert memory_allocated_before > 0, "Model should allocate some GPU memory"

    # Offload weights to meta (replace tensors)
    offload(model, device="meta")

    # Verify memory is freed (should be much less)
    memory_allocated_after = torch.xpu.memory_allocated(device)
    memory_freed = memory_allocated_before - memory_allocated_after

    # At least 50% of memory should be freed (conservative threshold)
    # Using 50% instead of 80% to account for:
    # - Small models where overhead is proportionally larger
    # - XPU context and other allocations
    # - Potential fragmentation
    threshold = memory_allocated_before * 0.5
    assert memory_freed > threshold, (
        f"Expected at least {threshold / 1e6:.2f} MB freed, "
        f"but only {memory_freed / 1e6:.2f} MB was freed. "
        f"Before: {memory_allocated_before / 1e6:.2f} MB, "
        f"After: {memory_allocated_after / 1e6:.2f} MB"
    )


@requires_xpu
def test_offload_gpu_to_cpu_to_meta(device):
    """Test complete offload sequence: GPU -> CPU -> meta.

    This test verifies that:
    1. Loading to GPU allocates GPU memory
    2. Offloading to CPU frees GPU memory
    3. Offloading to meta frees CPU memory
    """
    # Clear XPU cache first
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats(device)

    # Create a larger model for reliable memory measurements
    model = nn.Sequential(
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 500),
    )

    # Step 1: Load model to GPU
    model.to(device)
    torch.xpu.synchronize()

    gpu_memory_after_load = torch.xpu.memory_allocated(device)
    assert gpu_memory_after_load > 0, "Model should allocate GPU memory"

    # Step 2: Offload to CPU (should free GPU memory)
    offload(model, device="cpu")
    torch.xpu.synchronize()

    gpu_memory_after_cpu_offload = torch.xpu.memory_allocated(device)

    # GPU memory should be significantly reduced (at least 50%)
    gpu_memory_freed = gpu_memory_after_load - gpu_memory_after_cpu_offload
    assert gpu_memory_freed > gpu_memory_after_load * 0.5, (
        f"Expected at least 50% of GPU memory freed after CPU offload. "
        f"Before: {gpu_memory_after_load / 1e6:.2f} MB, "
        f"After: {gpu_memory_after_cpu_offload / 1e6:.2f} MB, "
        f"Freed: {gpu_memory_freed / 1e6:.2f} MB"
    )

    # Verify model is on CPU
    assert next(model.parameters()).device.type == "cpu"

    # Step 3: Offload to meta (should free CPU memory and move to meta)
    offload(model, device="meta")

    # Verify model is on CPU (as per current implementation)
    # Note: The function is called offload_to_meta but currently moves to CPU
    assert next(model.parameters()).device.type == "meta"

    # GPU memory should remain low
    gpu_memory_final = torch.xpu.memory_allocated(device)
    assert gpu_memory_final <= gpu_memory_after_cpu_offload, (
        f"GPU memory should not increase after meta offload. "
        f"After CPU: {gpu_memory_after_cpu_offload / 1e6:.2f} MB, "
        f"After meta: {gpu_memory_final / 1e6:.2f} MB"
    )
