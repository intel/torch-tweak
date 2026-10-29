# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Default strategy for a device."""

import torch

from torch_tweak.torch.tune_strategy.default_backends import default_backends
from torch_tweak.torch.tune_strategy.highest_throughput_strategy import HighestThroughputStrategy


def default_strategy(device: str | torch.device) -> HighestThroughputStrategy:
    """`HighestThroughputStrategy` over `default_backends(device)`: every backend is measured and the fastest wins."""
    return HighestThroughputStrategy(backends=default_backends(device))
