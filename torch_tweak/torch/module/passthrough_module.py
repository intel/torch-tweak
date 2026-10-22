# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Passthrough module."""

from typing import Any

from torch_tweak.torch.module.locator import Locator


class PassthroughModule:
    """Module that passes through the original module.

    When a device is given, inputs are moved to it before the call.
    When no device is given, inputs stay where they are.
    """

    def __init__(
        self,
        module,
        device: str | None,
    ) -> None:
        """Initializes module.

        Args:
                module: module to be tuned.
                device: Device on which tuned module has to be executed. None leaves the module and its
                    inputs where they are, for modules without parameters.
        """
        super().__init__()
        self._forward_call = module.__call__
        self._device = device
        if self._device is not None:
            module.to(self._device)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Native inference on wrapped module."""
        if self._device is not None:
            for locator, tensor in Locator.find_leaves(args, only_tensors=True):
                args = locator.set_value(args, tensor.to(self._device))
            for locator, tensor in Locator.find_leaves(kwargs, only_tensors=True):
                kwargs = locator.set_value(kwargs, tensor.to(self._device))
        return self._forward_call(*args, **kwargs)
