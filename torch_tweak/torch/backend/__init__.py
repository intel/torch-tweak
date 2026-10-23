# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Torch backend module."""

from torch_tweak.torch.backend.backend import Backend
from torch_tweak.torch.backend.openvino import OpenVINOBackend, OpenVINOBackendConfig
from torch_tweak.torch.backend.torch_eager import TorchEagerBackend, TorchEagerBackendConfig
from torch_tweak.torch.backend.torch_inductor_backend import TorchInductorBackend, TorchInductorBackendConfig
from torch_tweak.torch.backend.torchao_backend import TorchAOBackend, TorchAOBackendConfig
from torch_tweak.torch.backend.xpu_graph_backend import XPUGraphBackend, XPUGraphBackendConfig

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
]
