# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the default backends and strategy."""

import pytest

from torch_tweak.torch.backend import (
    OpenVINOBackend,
    TorchEagerBackend,
    TorchInductorBackend,
    XPUGraphBackend,
)
from torch_tweak.torch.tune_strategy import default_backends, default_strategy


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("xpu:0", [XPUGraphBackend, TorchInductorBackend, OpenVINOBackend, TorchEagerBackend]),
        ("cpu", [TorchInductorBackend, OpenVINOBackend, TorchEagerBackend]),
    ],
)
def test_default_backends_are_every_backend_that_runs_on_the_device(device, expected):
    assert [type(backend) for backend in default_backends(device)] == expected


def test_default_backends_reject_an_unknown_device():
    with pytest.raises(ValueError, match="No default backends"):
        default_backends("meta")


def test_default_strategy_on_cpu_measures_inductor_openvino_and_eager():
    strategy = default_strategy("cpu")

    assert strategy.describe().splitlines() == [
        "name: Highest Throughput Strategy",
        "description: evaluate all backends, return backend with highest throughput",
        "backends:",
        "  TorchInductorBackend()",
        "  OpenVINOBackend()",
        "  TorchEagerBackend()",
    ]
