# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Torch Tweak tuning strategies for the Stable Diffusion example."""

from __future__ import annotations

import torch

from torch_tweak.torch.backend import (
    Backend,
    OpenVINOBackend,
    OpenVINOBackendConfig,
    TorchEagerBackend,
    TorchInductorBackend,
    TorchInductorBackendConfig,
)
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy, OneBackendStrategy


def _openvino_backend(use_dynamo: bool) -> OpenVINOBackend:
    return OpenVINOBackend(OpenVINOBackendConfig(use_dynamo=use_dynamo))


def _inductor_backend() -> TorchInductorBackend:
    return TorchInductorBackend(
        TorchInductorBackendConfig(
            mode="default",
            dynamic=True,
            autocast_enabled=True,
            autocast_dtype=torch.float16,
        )
    )


def _backend_catalog() -> dict[str, Backend]:
    return {
        "inductor": _inductor_backend(),
        "openvino": _openvino_backend(use_dynamo=False),
        "openvino_dynamo": _openvino_backend(use_dynamo=True),
        "eager": TorchEagerBackend(),
    }


def hybrid_strategy() -> HighestThroughputStrategy:
    """Benchmark backends per module/graph and pick highest throughput."""
    return HighestThroughputStrategy(
        backends=_backend_catalog().values(),
    ).enable_find_max_batch_size(False)


def single_backend_strategy(backend_name: str) -> OneBackendStrategy:
    """Force a single backend for every wrapped module (benchmark baseline)."""
    catalog = _backend_catalog()
    if backend_name not in catalog:
        raise ValueError(f"Unknown backend: {backend_name}")
    return OneBackendStrategy(catalog[backend_name]).enable_find_max_batch_size(False)
