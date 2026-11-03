# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Torch backend module.

OpenVINO and TorchAO backends import on first use, so the openvino and torchao packages are optional.
"""

import importlib
from typing import TYPE_CHECKING, Any

from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.backend.batch_size_router_backend import BatchSizeRouterBackend
from torch_tweak.torch.backend.torch_eager import TorchEagerBackend, TorchEagerBackendConfig
from torch_tweak.torch.backend.torch_inductor_backend import TorchInductorBackend, TorchInductorBackendConfig
from torch_tweak.torch.backend.xpu_graph_backend import XPUGraphBackend, XPUGraphBackendConfig

if TYPE_CHECKING:
    from torch_tweak.torch.backend.openvino import OpenVINOBackend, OpenVINOBackendConfig
    from torch_tweak.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig

_OPTIONAL_BACKENDS = {
    "OpenVINOBackend": "torch_tweak.torch.backend.openvino",
    "OpenVINOBackendConfig": "torch_tweak.torch.backend.openvino",
    "TorchAOBackend": "torch_tweak.torch.backend.torchao_backend",
    "TorchAOBackendConfig": "torch_tweak.torch.backend.torchao_backend",
}


def __getattr__(name: str) -> Any:
    if name in _OPTIONAL_BACKENDS:
        return getattr(importlib.import_module(_OPTIONAL_BACKENDS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def import_optional_backends() -> None:
    """Imports every optional backend whose package is installed, so Backend.__subclasses__ lists it."""
    for module in set(_OPTIONAL_BACKENDS.values()):
        try:
            importlib.import_module(module)
        except ImportError:
            pass


__all__ = [
    "Backend",
    "BatchSizeRouterBackend",
    "OpenVINOBackend",
    "OpenVINOBackendConfig",
    "TorchInductorBackend",
    "TorchInductorBackendConfig",
    "TorchAOBackend",
    "TorchAOBackendConfig",
    "TorchEagerBackend",
    "TorchEagerBackendConfig",
    "XPUGraphBackend",
    "XPUGraphBackendConfig",
    "import_optional_backends",
]
