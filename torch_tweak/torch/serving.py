# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Record, tune, and serve modules inside a framework that owns the request loop.

A serving framework cannot hand Torch Tweak a dataset and a callable to drive.
It loads the model and runs requests itself. ServedModule fits that order:

* record: calls run the original module and the inputs of every new graph are written to disk.
* tune: the first call tunes the recorded graphs, writes the choice, and serves it.
* serve: the first call loads the choice and builds its backends from the recorded samples.
* auto: serve when a choice exists, tune when only a recording exists, and record otherwise.

Tuning and loading run inside the first forward call, so they see the same device, streams,
and framework context as inference. Every tensor shape is its own graph. A graph that was not
recorded runs the fallback. Weights are not written: the framework loads them before Torch Tweak
sees the module.
"""

import copy
import hashlib
import logging
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torch_tweak.torch.backend import (
    Backend,
    TorchEagerBackend,
    TorchInductorBackend,
    XPUGraphBackend,
    XPUGraphBackendConfig,
)
from torch_tweak.torch.backend.backend import checkpoint_without_weights
from torch_tweak.torch.checkpoint.storage_tasks import STATE_DICT_FILE, TorchLoadTask, TorchSaveTask
from torch_tweak.torch.config import config as global_config
from torch_tweak.torch.module.recording_module import RecordingModule
from torch_tweak.torch.module.tunable_module import ModuleState, TunableModule, sanitize_model_name
from torch_tweak.torch.module.tuned_module import TunedModule
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy
from torch_tweak.torch.utils.path_utils import MAX_FILENAME_BYTES, sanitize_filename

logger = logging.getLogger(__name__)

TUNED_DIR = "tuned"
RECORDING_DIR = "recording"
TUNE_LOG = "tune.log"


def _xpu_graph_inductor() -> Backend:
    return XPUGraphBackend(XPUGraphBackendConfig(compile_with_inductor=True))


BACKEND_FACTORIES: dict[str, Callable[[], Backend]] = {
    "xpu_graph": XPUGraphBackend,
    "xpu_graph_inductor": _xpu_graph_inductor,
    "inductor": TorchInductorBackend,
    "eager": TorchEagerBackend,
}


class ServingFallback(str, Enum):
    """What runs a graph that was not recorded."""

    INDUCTOR = "inductor"
    EAGER = "eager"


class ServingMode(str, Enum):
    """What ServedModule does with the calls it sees."""

    RECORD = "record"
    TUNE = "tune"
    SERVE = "serve"
    AUTO = "auto"


@dataclass
class ServingConfig:
    """Configuration of ServedModule.

    Args:
        workdir: Holds one directory per module name with the recording and the tuned choice.
        mode: See ServingMode.
        backends: Names from BACKEND_FACTORIES. HighestThroughputStrategy measures each and keeps the fastest.
        fallback: Runs graphs that were not recorded. INDUCTOR compiles the module with dynamic shapes
            on the first such call. EAGER runs the original module.
    """

    workdir: Path
    mode: ServingMode = ServingMode.AUTO
    backends: Sequence[str] = tuple(BACKEND_FACTORIES)
    fallback: ServingFallback = ServingFallback.INDUCTOR

    def __post_init__(self) -> None:
        """Normalizes workdir and mode, and rejects unknown or missing backends."""
        self.workdir = Path(self.workdir)
        self.mode = ServingMode(self.mode)
        self.fallback = ServingFallback(self.fallback)
        self.backends = tuple(self.backends)
        if not self.backends:
            raise ValueError("backends is empty")
        unknown = [name for name in self.backends if name not in BACKEND_FACTORIES]
        if unknown:
            raise ValueError(f"Unknown backends {unknown}; known: {list(BACKEND_FACTORIES)}")


class ServedModule(TunableModule):
    """TunableModule driven by the calls of a serving framework instead of a dataset."""

    def __init__(self, module: nn.Module, name: str, config: ServingConfig, *, device: torch.device | None = None):
        """Wraps module in place. Calls to module go through this wrapper.

        Args:
            module: Module with its weights loaded, on the device it serves from.
            name: Unique among the modules served from config.workdir.
            config: Serving configuration.
            device: Device to tune and serve on. Defaults to the device of the module parameters.

        Raises:
            FileNotFoundError: Mode tune finds no recording, or mode serve finds no tuned choice.
        """
        strategy = HighestThroughputStrategy(backends=[BACKEND_FACTORIES[name]() for name in config.backends])
        strategy.enable_find_max_batch_size(False)
        super().__init__(module, name, strategy=strategy)
        self._self_serving_device = device or self._self_device
        self._self_dir = config.workdir / _module_dir_name(name)
        self._self_recording_dir = self._self_dir / RECORDING_DIR
        self._self_tuned_dir = self._self_dir / TUNED_DIR
        self._self_tuned_path = self._self_tuned_dir / STATE_DICT_FILE
        self._self_module_config = copy.copy(global_config)
        self._self_module_config.exact_shapes = True
        self._self_fallback = config.fallback
        self._self_mode = self._resolve_mode(config.mode)
        self._self_pending = self._self_mode != ServingMode.RECORD
        if self._self_mode == ServingMode.RECORD:
            self._start_recording()
        logger.info("Torch Tweak serves `%s` in %s mode from %s", self.name, self._self_mode.value, self._self_dir)

    @property
    def serving_mode(self) -> ServingMode:
        """The mode resolved from the configuration and the files in the work directory."""
        return self._self_mode

    def _resolve_mode(self, mode: ServingMode) -> ServingMode:
        has_recording = RecordingModule.has_recording(self._self_recording_dir)
        if mode == ServingMode.AUTO:
            if self._self_tuned_path.is_file():
                return ServingMode.SERVE
            return ServingMode.TUNE if has_recording else ServingMode.RECORD
        if mode == ServingMode.TUNE and not has_recording:
            raise FileNotFoundError(f"No recording of `{self.name}` in {self._self_recording_dir}")
        if mode == ServingMode.SERVE and not self._self_tuned_path.is_file():
            raise FileNotFoundError(f"No tuned choice of `{self.name}` at {self._self_tuned_path}")
        return mode

    def _forward(self, wrapped, instance, args, kwargs) -> Any:
        """Tunes or loads on the first call, then dispatches like TunableModule."""
        if self._self_pending:
            self._self_pending = False
            if self._self_mode == ServingMode.TUNE:
                self._tune_recording()
            else:
                self._load_tuned()
        return super()._forward(wrapped, instance, args, kwargs)

    def _load_recording(self) -> RecordingModule:
        return RecordingModule.load(
            self.__wrapped__, self.name, self._self_recording_dir, config=self._self_module_config
        )

    def _start_recording(self) -> None:
        if RecordingModule.has_recording(self._self_recording_dir):
            self._self_wrapper = self._load_recording()
        else:
            self._self_wrapper = RecordingModule(
                self.__wrapped__, self.name, config=self._self_module_config, samples_dir=self._self_recording_dir
            )
        self._self_state = ModuleState.RECORDING

    def _tune_recording(self) -> None:
        self._self_wrapper = self._load_recording()
        self._self_state = ModuleState.RECORDING
        with _log_to(self._self_dir / TUNE_LOG):
            logger.info("Torch Tweak tunes `%s` on %d recorded graphs.", self.name, len(self.graph_specs))
            self.tune(device=self._self_serving_device)
            tuned = self._tuned_with_fallback()
            with checkpoint_without_weights():
                state = tuned.to_dict()
            self._self_tuned_dir.mkdir(exist_ok=True)
            TorchSaveTask().save(self._self_tuned_dir, state)
            self._describe_choice(tuned)

    def _load_tuned(self) -> None:
        state = TorchLoadTask().load(self._self_tuned_dir)
        self._self_wrapper = TunedModule.from_dict(self.__wrapped__, state, config=self._self_module_config)
        self._self_state = ModuleState.TUNED
        self._deploy_wrapper(self._self_serving_device)
        self._describe_choice(self._tuned_with_fallback())

    def _tuned_with_fallback(self) -> TunedModule:
        tuned = self._self_wrapper
        assert isinstance(tuned, TunedModule)
        if self._self_fallback == ServingFallback.INDUCTOR:
            tuned.fallback = torch.compile(self.__wrapped__, dynamic=True)
        else:
            tuned.fallback = self.__wrapped__
        return tuned

    def _describe_choice(self, tuned: TunedModule) -> None:
        for metadata, backend in tuned.backends.items():
            shapes = ", ".join(f"{spec.name}{list(spec.shape)}" for spec in metadata.tensor_specs)
            logger.info("Torch Tweak `%s`: %s for %s", self.name, backend.describe(), shapes)


@contextmanager
def _log_to(path: Path) -> Generator[None]:
    """Writes INFO records of the torch_tweak loggers to path. Serving frameworks often drop them otherwise."""
    package_logger = logging.getLogger("torch_tweak")
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    previous_level = package_logger.level
    package_logger.addHandler(handler)
    package_logger.setLevel(min(package_logger.getEffectiveLevel(), logging.INFO))
    try:
        yield
    finally:
        package_logger.setLevel(previous_level)
        package_logger.removeHandler(handler)
        handler.close()


def _module_dir_name(name: str) -> str:
    """Keeps the end of a long name, where the distinguishing parts are, plus a hash of the whole name."""
    safe = sanitize_filename(sanitize_model_name(name) or name, max_bytes=1 << 16)
    if len(safe) <= MAX_FILENAME_BYTES:
        return safe
    digest = hashlib.sha1(name.encode()).hexdigest()[:10]
    return f"{safe[-(MAX_FILENAME_BYTES - len(digest) - 1) :].lstrip('._-')}-{digest}"
