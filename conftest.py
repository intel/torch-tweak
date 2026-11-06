# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Pytest fixtures both for unit and integration tests and docttest for production code."""

import atexit
import logging
import os

import pytest
import torch

from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.utils.xpu import is_available as is_xpu_available
from torch_tweak.utils.logging import setup_logging


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Schedule a hard exit after pytest finishes reporting.

    The callback is registered most recent, it runs before any older atexit handlers
    and C-extension teardowns (LIFO order). It still allows pytest to print summary.
    """
    atexit.register(os._exit, exitstatus)


@pytest.fixture(autouse=True)
def module_registry_cleanup():
    """Cleans up the module registry after each test."""
    try:
        yield
    finally:
        MODULE_REGISTRY.clear()


@pytest.fixture(autouse=True)
def torch_tweak_cache_dir(mocker, tmp_path):
    """Sets cache dir to temporary directory."""
    import sys

    # torch_tweak.torch.config resolves to the config object due to import in torch_tweak.torch.__init__
    # so we fetch the module directly from sys.modules
    config_module = sys.modules["torch_tweak.torch.config"]

    cache_dir = tmp_path / "torch_tweak_cache"
    mocker.patch.object(config_module, "DEFAULT_CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture
def torch_device():
    """Returns the device specified in TORCH_TWEAK_TESTS_USE_DEVICE environment variable if set,
    otherwise returns XPU if available, or CPU as fallback.
    """  # noqa: D205
    user_device = os.environ.get("TORCH_TWEAK_TESTS_USE_DEVICE")

    if user_device is not None:
        return torch.device(user_device)

    return torch.device("xpu" if is_xpu_available() else "cpu")


@pytest.fixture(autouse=True)
def torch_tweak_logging_setup():
    """Setup logging for torch_tweak."""
    setup_logging(level=logging.DEBUG if os.environ.get("TORCH_TWEAK_TESTS_LOG_LEVEL") == "DEBUG" else logging.INFO)
    yield
    logging.shutdown()
