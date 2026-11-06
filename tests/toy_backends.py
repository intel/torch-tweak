# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# NOTE: This file has been modified by Intel Corporation.
"""Toy backends for unit tests."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from torch_tweak.torch.backend.backend import Backend


class SleepBackend(Backend):
    """Backend that simulate an inference using sleep()."""

    # Inference calls the original module, so its weights must stay in place.
    requires_original_module = True

    def __init__(self, sleep_time: float = 0.001):
        super().__init__()
        self.sleep_time = sleep_time

    def key(self) -> str:
        return f"{self.__class__.__name__}"

    def describe(self) -> str:
        return "SleepBackend - test only"

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        self._activate()
        self._module = module
        return self

    def _activate(self):
        pass

    def _infer(self, *args, **kwargs):
        time.sleep(self.sleep_time)
        return self._module(*args, **kwargs)

    def _deactivate(self):
        pass

    def _deploy(self):
        pass

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__} with sleep time {self.sleep_time}"

    def to_dict(self):
        return {"sleep_time": self.sleep_time}

    @classmethod
    def from_dict(cls, state_dict: dict):
        return cls(sleep_time=state_dict["sleep_time"])


class BuildFailsBackend(SleepBackend):
    """Backend that fails on build with a given exception.

    Args:
        exception_class: The exception class to raise on build.
    """

    def __init__(self, exception_class: type[Exception]):
        super().__init__()
        self.exception_class = exception_class

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        raise self.exception_class("Build failed")


class AllocatingBackend(SleepBackend):
    """Allocates `nbytes` of device memory and keeps it while it is alive."""

    def __init__(self, nbytes: int):
        super().__init__()
        self.nbytes = nbytes

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        self._scratch = torch.empty(self.nbytes, dtype=torch.uint8, device=self._device)
        return super()._build(module, graph_spec, data, cache_dir)


class AllocateThenFailBackend(AllocatingBackend):
    """Allocates `nbytes` of device memory and fails its build, like an XPU graph capture that the driver rejects."""

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        super()._build(module, graph_spec, data, cache_dir)
        raise RuntimeError("Capture failed")


class MemoryProbeBackend(SleepBackend):
    """Passes the device memory reserved when it builds to `on_build`.

    `copy.deepcopy` keeps a bound `list.append` as the same object, so the copy a strategy builds appends to the
    caller's list.
    """

    def __init__(self, on_build: Callable[[int], None]):
        super().__init__()
        self.on_build = on_build

    def _build(self, module: Any, graph_spec: Any, data: list[Any], cache_dir: Path) -> Backend:
        self.on_build(torch.xpu.memory_reserved(self._device))
        return super()._build(module, graph_spec, data, cache_dir)
