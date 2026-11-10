# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Test for XPU utilities."""

import re

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


@pytest.mark.parametrize(
    "message",
    [
        "has no XPU support",
        "pip install --force-reinstall --index-url https://download.pytorch.org/whl/xpu torch",
        'pass device="cpu"',
    ],
)
def test_torch_without_xpu_support_is_rejected(mocker, message):
    mocker.patch("torch.version.xpu", None)

    with pytest.raises(RuntimeError, match=re.escape(message)):
        assert_is_available()


@pytest.mark.parametrize("message", ["no Intel GPU is visible", "render group", 'pass device="cpu"'])
def test_xpu_build_without_a_visible_gpu_is_rejected(mocker, message):
    mocker.patch("torch.version.xpu", "20260101")
    mocker.patch("torch.xpu.is_available", return_value=False)

    with pytest.raises(RuntimeError, match=re.escape(message)):
        assert_is_available()


def test_xpu_build_with_a_visible_gpu_is_accepted(mocker):
    mocker.patch("torch.version.xpu", "20260101")
    mocker.patch("torch.xpu.is_available", return_value=True)

    assert_is_available()
