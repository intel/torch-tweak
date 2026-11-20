# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tells whether another process used the device while a backend was measured."""

import time

import torch

PROBE_MATRIX_SIZE = 2048
PROBE_MATMULS = 16
PROBE_REPEATS = 3
SHARED_DEVICE_SLOWDOWN = 2.0


class DeviceLoadProbe:
    """Times a fixed matmul on a device.

    Another process computing on the device slows the probe two to five times, or stalls it for about a second.
    After the warm-up run, probes on a quiet device stay within 1.5 times of each other.
    """

    def __init__(self):
        """Initializes the probe. It allocates its matrices on the first `seconds` call for each device."""
        self._matrices: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    def seconds(self, device: torch.device) -> float:
        """The total time of `PROBE_REPEATS` runs of `PROBE_MATMULS` matmuls on `device`.

        CPU probes return 0.0, since the probe would cost as much as the measurement it checks.
        """
        if device.type == "cpu":
            return 0.0
        if device not in self._matrices:
            matrix = torch.ones(PROBE_MATRIX_SIZE, PROBE_MATRIX_SIZE, dtype=torch.float16, device=device)
            self._matrices[device] = (matrix, torch.empty_like(matrix))
            self._run(device)
        torch.accelerator.synchronize(device)
        start = time.perf_counter()
        for _ in range(PROBE_REPEATS):
            self._run(device)
        return time.perf_counter() - start

    def release(self) -> None:
        """Frees the probe matrices."""
        self._matrices.clear()

    def _run(self, device: torch.device) -> None:
        matrix, product = self._matrices[device]
        for _ in range(PROBE_MATMULS):
            torch.mm(matrix, matrix, out=product)
        torch.accelerator.synchronize(device)


def is_quiet(window: tuple[float, float], fastest: float) -> bool:
    """Whether both probes around a measurement stay within `SHARED_DEVICE_SLOWDOWN` of the fastest probe.

    >>> is_quiet((1.0, 1.9), fastest=1.0)
    True
    >>> is_quiet((1.0, 2.1), fastest=1.0)
    False
    """
    return max(window) <= fastest * SHARED_DEVICE_SLOWDOWN
