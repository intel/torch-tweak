# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Test for tune function."""

import logging
from collections import Counter
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import Dataset

from tests.toy_backends import SleepBackend
from tests.utilities.helpers import requires_xpu
from torch_tweak.torch import tuning
from torch_tweak.torch.config import DEFAULT_DEVICE
from torch_tweak.torch.dataloader import DataLoaderFactory
from torch_tweak.torch.module.tunable_module import ModuleState, TunableModule
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy import OneBackendStrategy
from torch_tweak.torch.tuning import LOG_FORMAT, tune


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(3, 32, 32)


@requires_xpu
@pytest.mark.parametrize("dry_run", [True, False])
def test_tune(mocker, dry_run):
    # given
    mock_module = Mock()
    mock_module.backends = {}
    dataset = DummyDataset()
    batch_sizes = [1, 2, 4]
    max_batches = 2

    mocker.patch.dict(MODULE_REGISTRY.modules, {"test_module": mock_module}, clear=True)
    mocker.patch("torch_tweak.torch.tuning._describe_module")

    # when
    tune(
        mock_module,
        dataset,
        batch_sizes=batch_sizes,
        max_num_batches_per_batch_size=max_batches,
        dry_run=dry_run,
        disable_external_logging=False,
    )

    # then
    assert mock_module.call_count > 0
    mock_module.tune.assert_called_once_with(dry_run=dry_run, device=torch.device(DEFAULT_DEVICE))

    batch_size_counter = Counter()
    for args, _ in mock_module.call_args_list:
        bs = args[0].size(0)
        batch_size_counter[bs] += 1

    assert batch_size_counter == Counter({1: max_batches, 2: max_batches, 4: max_batches})


def test_tune_raises_when_torch_has_no_xpu_support(mocker):
    mocker.patch("torch.version.xpu", None)

    with pytest.raises(RuntimeError, match="has no XPU support"):
        tune(Mock(), [torch.tensor([1.0])], batch_sizes=[1], device="xpu")


@pytest.mark.parametrize("device", ["cpu", None])
def test_tune_without_xpu_device_runs_on_torch_without_xpu_support(mocker, device):
    mocker.patch("torch.version.xpu", None)
    model = torch.nn.Linear(2, 1, bias=False)
    sample = torch.tensor([[1.0, 2.0]])
    eager = model(sample)
    module = TunableModule(
        model,
        "cpu-only",
        strategy=OneBackendStrategy(SleepBackend()).enable_find_max_batch_size(False),
    )

    tune(module, [torch.tensor([1.0, 2.0])], batch_sizes=[1], device=device)

    torch.testing.assert_close(module(sample), eager, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("clear_cache", [True, False])
def test_tune_with_cache_clear(mocker, clear_cache):
    # given
    mock_module = Mock()
    mock_module.backends = {}
    dataset = DummyDataset()
    batch_sizes = [1, 2, 4]
    max_batches = 2

    mocker.patch.dict(MODULE_REGISTRY.modules, {"test_module": mock_module}, clear=True)
    mocker.patch("torch_tweak.torch.tuning._describe_module")
    spy_clear_cache = mocker.spy(tuning, "_clear_cache")

    # when
    tune(
        mock_module,
        dataset,
        batch_sizes=batch_sizes,
        max_num_batches_per_batch_size=max_batches,
        device="cpu",
        disable_external_logging=False,
        clear_cache=clear_cache,
    )

    # then
    assert mock_module.call_count > 0
    mock_module.tune.assert_called_once_with(dry_run=False, device=torch.device("cpu"))

    assert spy_clear_cache.call_count == (1 if clear_cache else 0)

    batch_size_counter = Counter()
    for args, _ in mock_module.call_args_list:
        bs = args[0].size(0)
        batch_size_counter[bs] += 1

    assert batch_size_counter == Counter({1: max_batches, 2: max_batches, 4: max_batches})


def test_tune_with_dataloader_factory(mocker, torch_device):
    # given
    mock_module = Mock()
    mock_module.backends = {}
    dataset = DummyDataset()
    factory = DataLoaderFactory(dataset)

    mocker.patch.dict(MODULE_REGISTRY.modules, {"test_module": mock_module}, clear=True)
    mocker.patch("torch_tweak.torch.tuning._describe_module")

    # when
    tune(mock_module, factory, batch_sizes=[2], device=torch_device, disable_external_logging=False)

    # then
    assert mock_module.call_count > 0


def test_tune_log_level(mocker, torch_device):
    """Test that the tune function correctly responds to global logging level."""
    # given
    mock_func = Mock()
    dataset = DummyDataset(size=1)  # Small dataset for quick test

    # Mock the enable_gpu_memory_logging function
    mock_enable_gpu_memory_logging = mocker.patch("torch_tweak.utils.logging.enable_gpu_memory_logging")
    spy_setup_logging = mocker.spy(tuning, "setup_logging")

    # Get the root logger to check its level
    root_logger = logging.getLogger()
    original_level = root_logger.level

    mocker.patch.dict(MODULE_REGISTRY.modules, {}, clear=True)

    try:
        # Test 1: Global logging level DEBUG - should enable GPU memory logging
        root_logger.setLevel(logging.DEBUG)
        tune(mock_func, dataset, batch_sizes=[1], device=torch_device)
        spy_setup_logging.assert_called_once_with(format_string=LOG_FORMAT)
        mock_enable_gpu_memory_logging.assert_called_once()

        # Reset
        mock_enable_gpu_memory_logging.reset_mock()
        spy_setup_logging.reset_mock()

        # Test 2: Global logging level WARNING - should NOT enable GPU memory logging
        root_logger.setLevel(logging.WARNING)
        tune(mock_func, dataset, batch_sizes=[1], device=torch_device)
        spy_setup_logging.assert_called_once_with(format_string=LOG_FORMAT)
        mock_enable_gpu_memory_logging.assert_not_called()

        # Reset
        mock_enable_gpu_memory_logging.reset_mock()
        spy_setup_logging.reset_mock()

        # Test 3: Global logging level INFO - should NOT enable GPU memory logging
        root_logger.setLevel(logging.INFO)
        tune(mock_func, dataset, batch_sizes=[1], device=torch_device)
        spy_setup_logging.assert_called_once_with(format_string=LOG_FORMAT)
        mock_enable_gpu_memory_logging.assert_not_called()

        # Reset
        mock_enable_gpu_memory_logging.reset_mock()
        spy_setup_logging.reset_mock()

        # Test 4: Global logging level ERROR - should NOT enable GPU memory logging
        root_logger.setLevel(logging.ERROR)
        tune(mock_func, dataset, batch_sizes=[1], device=torch_device)
        spy_setup_logging.assert_called_once_with(format_string=LOG_FORMAT)
        mock_enable_gpu_memory_logging.assert_not_called()

    finally:
        # Restore original logging level
        root_logger.setLevel(original_level)


def test_tune_disable_external_logging(mocker, torch_device):
    """Test that the tune function correctly responds to disable_external_logging parameter."""
    # given
    mock_func = Mock()
    dataset = DummyDataset(size=1)  # Small dataset for quick test

    spy_libraries_logging = mocker.spy(tuning, "libraries_logging")

    mocker.patch.dict(MODULE_REGISTRY.modules, {}, clear=True)

    # when
    tune(mock_func, dataset, batch_sizes=[1], device=torch_device, disable_external_logging=False)

    # then
    spy_libraries_logging.assert_called_once_with(False)

    # Reset spy for next test
    spy_libraries_logging.reset_mock()

    # when
    tune(mock_func, dataset, batch_sizes=[1], device=torch_device, disable_external_logging=True)

    # then
    spy_libraries_logging.assert_called_once_with(True)


class ReactivationFails(SleepBackend):
    """Build succeeds. Activation after a deactivate raises."""

    def __init__(self):
        super().__init__()
        self._deactivated = False

    def _deactivate(self):
        self._deactivated = True

    def _activate(self):
        if self._deactivated:
            raise RuntimeError("reactivation failed")


def test_failed_reactivation_lets_the_next_module_tune(torch_device):
    failing_model = torch.nn.Linear(2, 1, bias=False).to(torch_device)
    later_model = torch.nn.Linear(2, 1, bias=False).to(torch_device)
    sample = torch.tensor([[1.0, 2.0]], device=torch_device)
    sample_with_different_shape = torch.tensor([1.0, 2.0], device=torch_device)
    eager = later_model(sample)

    failing = TunableModule(
        failing_model,
        "fails-reactivation",
        strategy=OneBackendStrategy(ReactivationFails()).enable_find_max_batch_size(False),
    )
    later = TunableModule(
        later_model,
        "tunes-after-failure",
        strategy=OneBackendStrategy(SleepBackend()).enable_find_max_batch_size(False),
    )

    def pipeline(x):
        failing(x)
        later(x)

    tune(
        pipeline,
        [sample, sample_with_different_shape],
        batch_sizes=[1],
        device=torch_device,
        ignore_failing_modules=True,
    )

    assert failing.state == ModuleState.PASSTHROUGH
    assert later.state == ModuleState.TUNED
    torch.testing.assert_close(later(sample), eager, rtol=1e-4, atol=1e-5)
