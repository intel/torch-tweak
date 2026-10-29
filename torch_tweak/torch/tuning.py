# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# NOTE: This file has been modified by Intel Corporation.
"""Main tune function."""

import shutil
from collections.abc import Callable
from logging import getLogger
from pathlib import Path

import torch

from torch_tweak.global_context import BATCH_SIZE_KEY, global_context
from torch_tweak.torch.checkpoint.local_torch_storage import LocalTorchStorage
from torch_tweak.torch.checkpoint.storage import Storage
from torch_tweak.torch.checkpoint.torch_checkpoint import TorchCheckpoint
from torch_tweak.torch.config import DEFAULT_DEVICE, torch_tweak_cache_dir
from torch_tweak.torch.dataloader import DataLoaderFactory, DatasetLike, samples_generator
from torch_tweak.torch.module.tensor_spec import InfoLevel
from torch_tweak.torch.module.wrapper_module import Module, ModuleState
from torch_tweak.torch.module_registry import MODULE_REGISTRY
from torch_tweak.torch.utils.device import get_device
from torch_tweak.torch.utils.xpu import synchronize
from torch_tweak.utils.logging import libraries_logging, setup_logging
from torch_tweak.utils.timer import Timer
from torch_tweak.utils.trace import annotate

logger = getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


@annotate(domain="Torch Tweak")
def tune(
    func: Callable,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    batch_sizes: list[int] | None = None,
    max_num_batches_per_batch_size: int | None = None,
    device: str | torch.device | None = DEFAULT_DEVICE,
    dry_run: bool = False,
    disable_external_logging: bool = False,
    clear_cache: bool = False,
    ignore_failing_modules: bool = True,
) -> None:
    """Tune a callable which runs inference on a pipeline or a model.

    Tunes the modules wrapped with `wrap` or `Module`. `tweak` picks the modules itself and is slower.

    Args:
        func: The function to tune.
        dataset: The dataset to tune on. It can be DataLoaderFactory or any dataset/iterable and even torch.Tensor.
            Tensor will be treated as a single sample dataset.
        batch_sizes: The batch sizes to use for tuning. At least 2 different batch sizes are required for determining
            batch axis.
        max_num_batches_per_batch_size: The maximum number of batches to use for tuning per batch size.
        device: The device to use for tuning.
        dry_run: If True, only dry run the tuning.
        disable_external_logging: If True, libraries logging will be suppressed.
        clear_cache: If True, the cache will be cleared before tuning.
        ignore_failing_modules: If True, failing modules will be ignored and tuning will continue.

    Note:
        Max batch size is limited by specified batch_size.
    """
    # Setup logging
    setup_logging(format_string=LOG_FORMAT)
    if clear_cache:
        _clear_cache()

    with libraries_logging(disable_external_logging):
        _record_and_tune_registered_modules(
            func,
            dataset,
            _validate_and_normalize_batch_sizes(batch_sizes),
            max_num_batches_per_batch_size,
            device,
            dry_run=dry_run,
            ignore_failing_modules=ignore_failing_modules,
            offload=True,
        )


def _record_and_tune_registered_modules(
    func: Callable,
    dataset: DatasetLike | DataLoaderFactory | torch.Tensor,
    batch_sizes: list[int],
    max_num_batches_per_batch_size: int | None,
    device: str | torch.device | None,
    *,
    dry_run: bool,
    ignore_failing_modules: bool,
    offload: bool,
) -> None:
    """Record samples by calling `func`, then tune and activate every module in MODULE_REGISTRY."""
    # Convert device to torch.device
    if device is not None:
        device = get_device(device)

    for batch_size, args, kwargs in samples_generator(dataset, batch_sizes, max_num_batches_per_batch_size):
        with global_context:
            global_context.set(BATCH_SIZE_KEY, batch_size)
            with torch.no_grad():
                func(*args, **kwargs)

    for module in MODULE_REGISTRY.modules.values():
        # Before tuning, deactivate backends in other modules
        # If any backend is activated by Torch Tweak or by the user it will affect available memory during tuning
        for other_module in MODULE_REGISTRY.modules.values():
            if other_module != module:
                other_module.deactivate()

        logger.info("════════════════════════════════════════════════════════════════")
        logger.info("🎯 Tuning module: `%s` (all graphs)", module.name)
        try:
            module.tune(device=device, dry_run=dry_run, offload=offload)
        except Exception:
            # If ignore_failing_modules is False, we will raise the error and stop tuning.
            if not ignore_failing_modules:
                raise

            # If ignore_failing_modules is True, we use original forward for this module and continue tuning the next module.
            logger.info("⚠️ Tuning module: `%s` failed", module.name)
            module.enable_passthrough()
            continue

        logger.info("✅ Tuning module: `%s` (all graphs) completed.", module.name)

    # Activate the backends after tuning for inference
    if not dry_run:
        _activate_tuned_modules()


def save(
    module: torch.nn.Module,
    path: str | Path,
    storage: Storage | None = None,
) -> None:
    """Save the tuned module to a file.

    Args:
        module: The module to save.
        path: The path to save the module to.
        storage: The storage to use to save the module. If not provided, a local storage will be used.

    If storage is not provided, a default storage will be used.

    If you would like to save the module to a different folder, you can use the storage parameter.
    For example, if you want to save the module to the `ckpt` folder, you can do the following:

    Example:
        >>> import doctest
        >>> doctest.ELLIPSIS_MARKER = "***" # using *** instead of ... to avoid doctest failure
        >>> import torch_tweak.torch as tt
        >>> from torch.nn import Linear
        >>> model = tt.Module(Linear(10, 10), "model", strategy=tt.FirstWinsStrategy([tt.backend.TorchEagerBackend()]))
        >>> dataset = torch.randn(10, 10)
        >>> tt.tune(model, dataset, batch_sizes=[1, 2], device="cpu") # doctest: +ELLIPSIS
        ***🎯 Tuning module: `model` (all graphs)***
        >>> storage = tt.LocalTorchStorage(base_folder="ckpt")
        >>> tt.save(model, "tuned_model.tt", storage=storage)
        ***✅ Checkpoint compressed and saved to***
        >>> loaded_model = tt.load(model, "tuned_model.tt", storage=storage)
        ***

    """
    checkpoint = TorchCheckpoint(storage or LocalTorchStorage())
    checkpoint.save(module, path)


def load(
    module: torch.nn.Module,
    path: str | Path,
    storage: Storage | None = None,
    device_map: dict[str, torch.device] | None = None,
    disable_external_logging: bool = True,
) -> torch.nn.Module | Module:
    """Load the tuned module from a file.

    Args:
        module: The module to load.
        path: The path to load the module from.
        storage: The storage to use to load the module. If not provided, a local storage will be used.
        device_map: The device map to load modules to. Overrides the device of stored in state dict.
        disable_external_logging: If True, libraries logging will be suppressed.

    If storage is not provided, a default storage will be used. Check save function for more details.
    """
    # Setup logging
    setup_logging(format_string=LOG_FORMAT)

    with libraries_logging(disable_external_logging):
        checkpoint = TorchCheckpoint(storage or LocalTorchStorage())

        # Measure loading time
        with Timer("Load checkpoint", silent=True) as timer:
            module = checkpoint.load(module, path=path, device_map=device_map)
            synchronize()

        logger.info("✅ Checkpoint loaded from: %s in %.2f seconds", path, timer.elapsed)

        return module


def _validate_and_normalize_batch_sizes(batch_sizes: list[int] | None) -> list[int]:
    """Validate and normalize batch sizes for tuning."""
    if not batch_sizes:
        logger.info("No batch sizes provided, using default batch sizes [1, 2].")
        return [1, 2]

    if len(set(batch_sizes)) == 1:
        logger.warning("Only 1 batch size provided, static shapes will be assumed.")

    return batch_sizes


def _activate_tuned_modules() -> None:
    """Activate all tuned modules after tuning is complete."""
    logger.info("════════════════════════════════════════════════════════════════")
    logger.info("✅ Completed tuning all modules.")
    logger.info("🎯 Activating tuned modules for inference:")
    for module in MODULE_REGISTRY.modules.values():
        module.activate()
        _describe_module(module)
    logger.info("════════════════════════════════════════════════════════════════")


def _describe_module(module: Module):
    logger.info("📦 module: %s", module.name)
    if module.state == ModuleState.TUNED:
        for metadata, backend in module.module.backends.items():
            logger.info(" ➤ graph: %s", metadata.describe(InfoLevel.MEDIUM))
            logger.info("   backend: %s", backend.describe())
    else:
        logger.info(" ➤ tuning failed: using original module")


def _clear_cache():
    """Clear the cache."""
    cache_dir = torch_tweak_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logger.info("🧹 Cleared cache directory: %s", cache_dir)
