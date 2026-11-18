# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Helper functions and classes for testing."""

import importlib.util

import pytest
import torch

from torch_tweak.torch.module.sample_metadata import CHECKPOINTABLE_VALUE_TYPES
from torch_tweak.torch.utils.xpu import is_available as is_xpu_available

requires_xpu = pytest.mark.skipif(not is_xpu_available(), reason="XPU is not available")


def _has_level_zero_interop() -> bool:
    if not is_xpu_available() or importlib.util.find_spec("openvino") is None:
        return False
    from torch_tweak.torch.backend.openvino.xpu_remote import has_level_zero_interop

    return has_level_zero_interop(torch.device("xpu"))


has_leo = _has_level_zero_interop()
LEO_MISSING = "The Intel GPU driver cannot run OpenCL on Level Zero (EnableLEO=1 on Battlemage)"
requires_leo = pytest.mark.skipif(not has_leo, reason=LEO_MISSING)


def assert_only_primitives(value):
    """Assert value is built exclusively from types that need no custom unpickling.

    Types are matched exactly rather than with isinstance, so that a subclass which
    would need its own class at load time (a namedtuple, or a dict subclass such as the
    transformers ModelOutput family) fails here just as it does in production code.

    Args:
        value: Serialized payload to inspect recursively.
    """
    if type(value) is dict:
        for key, item in value.items():
            assert_only_primitives(key)
            assert_only_primitives(item)
    elif type(value) in (list, tuple):
        for item in value:
            assert_only_primitives(item)
    else:
        assert type(value) in CHECKPOINTABLE_VALUE_TYPES, f"unexpected {type(value).__name__} in serialized data"


def save_and_load_weights_only(payload, tmp_path, name="payload.pth"):
    """Round trip payload through torch.save and a weights only torch.load.

    A weights only load accepts tensors and plain Python values only, so this raises
    for any payload which needs an arbitrary class to be reconstructed.

    Args:
        payload: Serialized payload to round trip.
        tmp_path: Directory to store the temporary checkpoint in.
        name: File name to use inside tmp_path.

    Returns:
        The payload as loaded back by torch.
    """
    path = tmp_path / name
    torch.save(payload, path)
    return torch.load(path, weights_only=True)
