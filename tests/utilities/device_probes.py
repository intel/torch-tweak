# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Device load probes with fixed timings."""

from collections.abc import Iterable

import torch

from torch_tweak.torch.utils.device_load import DeviceLoadProbe


class ScriptedProbe(DeviceLoadProbe):
    """Returns `seconds` one by one and counts its calls."""

    def __init__(self, seconds: Iterable[float]):
        super().__init__()
        self._seconds = list(seconds)
        self.calls = 0

    def seconds(self, device: torch.device) -> float:
        self.calls += 1
        return self._seconds.pop(0)
