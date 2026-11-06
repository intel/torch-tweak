# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Torch backend module.

The TorchAO backend imports on first use, so the torchao package is optional.
"""

import importlib
from typing import TYPE_CHECKING, Any

from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.backend.openvino import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.backend.torch_eager import TorchEagerBackend, TorchEagerBackendConfig
from torch_tweak.torch.backend.torch_inductor_backend import TorchInductorBackend, TorchInductorBackendConfig
from torch_tweak.torch.backend.xpu_graph_backend import XPUGraphBackend, XPUGraphBackendConfig

if TYPE_CHECKING:
    from torch_tweak.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig

_OPTIONAL_BACKENDS = {
    "TorchAOBackend": "torch_tweak.torch.backend.torchao_backend",
    "TorchAOBackendConfig": "torch_tweak.torch.backend.torchao_backend",
}


def __getattr__(name: str) -> Any:
    if name in _OPTIONAL_BACKENDS:
        return getattr(importlib.import_module(_OPTIONAL_BACKENDS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def import_optional_backend(name: str) -> None:
    """Imports the optional backend `name`. A missing package raises ImportError."""
    if name not in _OPTIONAL_BACKENDS:
        return
    importlib.import_module(_OPTIONAL_BACKENDS[name])


__all__ = [
    "Backend",
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
    "import_optional_backend",
]
