# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Install guidance shown in error messages."""

# This module must not import torch: torch_tweak.torch reports a missing torch with it.

COMMON_ISSUES_URL = "https://github.com/intel/torch-tweak/blob/main/docs/learn/install.md#common-issues"
XPU_TORCH_INSTALL = "pip install --force-reinstall --index-url https://download.pytorch.org/whl/xpu torch"
CPU_TORCH_INSTALL = "pip install --index-url https://download.pytorch.org/whl/cpu torch"
