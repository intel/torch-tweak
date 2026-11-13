# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for DeviceLoadProbe."""

import torch

from tests.utilities.helpers import requires_xpu
from torch_tweak.torch.utils.device_load import DeviceLoadProbe


def test_cpu_probe_takes_no_time():
    probe = DeviceLoadProbe()

    seconds = probe.seconds(torch.device("cpu"))

    assert seconds == 0.0


@requires_xpu
def test_xpu_probe_takes_time():
    probe = DeviceLoadProbe()

    seconds = probe.seconds(torch.device("xpu"))

    assert seconds > 0.0
