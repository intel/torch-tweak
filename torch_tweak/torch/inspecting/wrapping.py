# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Inspecting and patching modules."""

from logging import getLogger

from torch_tweak.torch.inspecting.module_info import ModuleInfo
from torch_tweak.torch.module.wrapper_module import Module, StrategyList, StrategyMap
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.tune_strategy.tune_strategy import TuneStrategy
from torch_tweak.utils.logging import setup_logging

logger = getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


def wrap(
    obj: object,
    modules: list[ModuleInfo],
    strategy: TuneStrategy | None = None,
    strategies: StrategyList | StrategyMap | None = None,
) -> object:
    """Wrap provided modules with inspection logic.

    Args:
        obj: Callable object to wrap.
        modules: Dictionary of module names and their corresponding ModuleInfo objects.
        strategy: Strategy to use for patching.
        strategies: Strategies to use for patching.

    Returns:
        Wrapped callable object.
    """
    setup_logging(format_string=LOG_FORMAT)
    MODULE_REGISTRY.clear()
    for module_info in modules:
        logger.info("Wrapping module: %s", module_info.object_path or module_info.name or obj.__class__.__name__)
        if module_info.parent is None:
            return Module(obj, name=obj.__class__.__name__, strategy=strategy, strategies=strategies)

        # `module_info.name` repeats across siblings (`layers.0.mlp`, `layers.1.mlp`); the path does not.
        name = module_info.object_path.lstrip(".")
        wrapped = Module(module_info.module, name=name, strategy=strategy, strategies=strategies)
        module_info.parent.set_wrapped(module_info.name, wrapped)

    return obj
