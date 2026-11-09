# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# NOTE: This file has been modified by Intel Corporation.
"""Unit tests for OneBackendStrategy."""

from functools import partial
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from tests.toy_backends import AllocateThenFailBackend, AllocatingBackend, MemoryProbeBackend
from tests.toy_models.torch_models import toy_on_xpu
from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.module.graph_spec import GraphSpec
from torch_tweak.torch.module.recording_module import Sample
from torch_tweak.torch.tune_strategy.extension import TuneStrategyFindMaxBatchSizeExtension
from torch_tweak.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from torch_tweak.torch.tune_strategy.tune_strategy import NoCorrectBackendError


@pytest.fixture
def mock_backend():
    """Create a mock backend for testing."""
    backend = MagicMock(spec=Backend)
    backend.name = "mock_backend"
    backend.describe.return_value = "mock_backend"
    return backend


@pytest.fixture
def mock_module():
    """Create a mock torch module for testing."""
    module = MagicMock(spec=nn.Module)
    module.to = MagicMock()
    return module


@pytest.fixture
def mock_graph_spec():
    """Create a mock graph spec for testing."""
    graph_spec = MagicMock(spec=GraphSpec)
    graph_spec.name = "mock_graph_spec"
    return graph_spec


@pytest.fixture
def mock_sample():
    """Create a mock sample for testing."""
    return MagicMock(spec=Sample)


def test_describe(mock_backend):
    """Test describe method returns correct string."""
    strategy = OneBackendStrategy(mock_backend)
    assert strategy.describe() == "name: One Backend Strategy\ndescription: Use only one backend\nbackend: mock_backend"


def test_tune_success(mock_backend, mock_module, mock_graph_spec, torch_device, mock_sample, tmp_path):
    """Test tune method when backend succeeds."""
    # Setup
    strategy = OneBackendStrategy(mock_backend)
    strategy._describe = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.__deepcopy__ = lambda _: mock_backend
    mock_backend.build.return_value = mock_backend

    # Execute
    result = strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    # Verify
    assert result == mock_backend
    mock_backend.build.assert_called_once()


def test_tune_backend_fails(mock_backend, mock_module, mock_graph_spec, torch_device, mock_sample, tmp_path):
    """Test tune method when backend fails."""
    # Setup
    strategy = OneBackendStrategy(mock_backend)
    strategy._describe = MagicMock()
    strategy.enable_find_max_batch_size(False)
    strategy.enable_correctness_check(False)
    mock_backend.__deepcopy__ = lambda _: mock_backend
    mock_backend.build.side_effect = Exception("Backend failed")

    # Execute and verify
    with pytest.raises(NoCorrectBackendError) as exc_info:
        strategy.tune(mock_module, "test_module", mock_graph_spec, [mock_sample], torch_device, tmp_path)

    assert str(exc_info.value.__cause__) == "Backend failed"
    mock_backend.build.assert_called_once()


@requires_xpu
def test_backend_builds_without_the_memory_find_max_batch_size_held(tmp_path):
    model, data, graph_spec = toy_on_xpu()
    find_bytes = 2**30
    reserved_at_build = []
    strategy = OneBackendStrategy(MemoryProbeBackend(on_build=reserved_at_build.append))
    strategy.set_find_max_batch_size_default_backend_class(partial(AllocatingBackend, find_bytes))
    strategy.set_find_max_batch_size_profiling_config(
        TuneStrategyFindMaxBatchSizeExtension.default_profiling_config(max_batch_size=2)
    )
    torch.xpu.empty_cache()
    reserved_before = torch.xpu.memory_reserved()

    strategy.tune(model, "test", graph_spec, data, torch.device("xpu"), tmp_path)

    assert reserved_at_build[0] < reserved_before + find_bytes


@requires_xpu
def test_tune_starts_without_the_memory_a_failed_tune_held(tmp_path):
    model, data, graph_spec = toy_on_xpu()
    failed_bytes = 2**30
    reserved_at_build = []
    failing = OneBackendStrategy(AllocateThenFailBackend(failed_bytes))
    failing.enable_find_max_batch_size(False)
    probing = OneBackendStrategy(MemoryProbeBackend(on_build=reserved_at_build.append))
    probing.enable_find_max_batch_size(False)
    torch.xpu.empty_cache()
    reserved_before = torch.xpu.memory_reserved()
    with pytest.raises(RuntimeError, match="Capture failed"):
        failing.tune(model, "test", graph_spec, data, torch.device("xpu"), tmp_path)

    probing.tune(model, "test", graph_spec, data, torch.device("xpu"), tmp_path)

    assert reserved_at_build[0] < reserved_before + failed_bytes
