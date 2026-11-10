# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Test for XPU utilities."""

import pytest
import torch

from torch_tweak.torch.utils.xpu import assert_is_available, is_available, synchronize

requires_xpu = pytest.mark.skipif(not is_available(), reason="XPU is not available")


def test_is_available():
    """Test the is_available function."""
    xpu = getattr(torch, "xpu", None)
    expected = bool(xpu is not None and xpu.is_available())
    assert is_available() == expected


def test_synchronize(mocker):
    """Test the synchronize function."""
    xpu_available = is_available()
    mocker.patch("torch.xpu.synchronize")
    mocker.patch("torch.xpu.is_available", return_value=xpu_available)

    synchronize()

    torch.xpu.is_available.assert_called_once()
    if xpu_available:
        torch.xpu.synchronize.assert_called_once()


def test_assert_is_available_when_torch_has_no_xpu_support(mocker):
    mocker.patch("torch.xpu._is_compiled", return_value=False)

    with pytest.raises(RuntimeError, match="PyTorch was not built with XPU support"):
        assert_is_available()


def test_assert_is_available_when_no_intel_gpu_is_visible(mocker):
    mocker.patch("torch.xpu._is_compiled", return_value=True)
    mocker.patch("torch.xpu.is_available", return_value=False)

    with pytest.raises(RuntimeError, match="No Intel GPU is available"):
        assert_is_available()


def test_assert_is_available_when_intel_gpu_is_visible(mocker):
    mocker.patch("torch.xpu._is_compiled", return_value=True)
    mocker.patch("torch.xpu.is_available", return_value=True)

    assert_is_available()
