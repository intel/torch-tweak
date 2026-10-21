# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tune a small MLP and keep the faster backend."""

import os
import sys

import torch
import torch.nn as nn

import torch_tweak.torch as tt
from torch_tweak.torch.backend import (
    TorchEagerBackend,
    TorchInductorBackend,
    XPUGraphBackend,
)


def main() -> None:
    """Tune a small MLP, print the selected backend, and exit the process."""
    xpu = getattr(torch, "xpu", None)
    device = "xpu" if xpu is not None and xpu.is_available() else "cpu"
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 8)).eval().to(device)
    sample = torch.randn(32, device=device)

    # One batch size keeps every tensor shape static, so the XPU graph can be recorded.
    strategy = tt.HighestThroughputStrategy(
        backends=[
            TorchInductorBackend(),
            XPUGraphBackend(),
            TorchEagerBackend(),
        ],
    ).enable_find_max_batch_size(False)

    module = tt.Module(model, name="mlp", strategy=strategy)
    tt.tune(module, sample, batch_sizes=[4], device=device)

    results = strategy.results[-1].highest_throughput_results if strategy.results else []
    if not results:
        print("Tuning did not select a backend.", file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)

    best = max(results, key=lambda item: item.throughput)
    print(f"{best.backend_details}: {best.throughput:.1f} samples/s")
    sys.stdout.flush()
    # Level Zero can SIGSEGV (exit 139) while the interpreter unloads XPU state.
    os._exit(0)


if __name__ == "__main__":
    main()
