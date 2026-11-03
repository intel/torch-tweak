# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Torch tweak strategy module."""

from torch_tweak.torch.tune_strategy.default_backends import default_backends
from torch_tweak.torch.tune_strategy.default_strategy import default_strategy
from torch_tweak.torch.tune_strategy.first_wins_strategy import FirstWinsStrategy
from torch_tweak.torch.tune_strategy.highest_throughput_strategy import HighestThroughputStrategy
from torch_tweak.torch.tune_strategy.one_backend_strategy import OneBackendStrategy
from torch_tweak.torch.tune_strategy.tune_strategy import TuneStrategy

__all__ = [
    "FirstWinsStrategy",
    "OneBackendStrategy",
    "HighestThroughputStrategy",
    "TuneStrategy",
    "default_backends",
    "default_strategy",
]
