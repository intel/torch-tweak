# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Find the fastest cut depth of a transformer encoder with tweak, then save and load it."""

import os
import sys
import tempfile

import torch
import torch.nn as nn

import torch_tweak.torch as tt


def make_model(device: str) -> nn.Module:
    """Builds the same encoder on every call."""
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, batch_first=True)
    return nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False).eval().to(device)


def main() -> None:
    """Runs a dry run, tunes, prints the result, and checks the loaded model."""
    # The fused fast path runs a whole layer without calling its submodules, so deeper cuts would not exist.
    torch.backends.mha.set_fastpath_enabled(False)
    xpu = getattr(torch, "xpu", None)
    device = "xpu" if xpu is not None and xpu.is_available() else "cpu"
    model = make_model(device)
    sample = torch.randn(64, 256, device=device)
    batch = sample.unsqueeze(0).repeat(8, 1, 1)
    with torch.no_grad():
        expected = model(batch)

    plan = tt.tweak(model, sample, batch_sizes=[8], device=device, max_depth=3, dry_run=True)
    for depth in plan.depths:
        print(f"Depth {depth.depth} wraps {len(depth.module_paths)} modules")

    result = tt.tweak(model, sample, batch_sizes=[8], device=device, max_depth=3, time_budget=600)
    print(result)
    if result.best_depth is None:
        print("No depth is faster than the untuned model.")
        sys.stdout.flush()
        os._exit(0)

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "encoder.tt")
        tt.save(result.obj, path)
        loaded = tt.load(make_model(device), path)
        with torch.no_grad():
            torch.testing.assert_close(loaded(batch), expected, rtol=1e-2, atol=1e-2)
    print(f"Depth {result.best_depth} saved and loaded, outputs match.")
    sys.stdout.flush()
    # Level Zero can SIGSEGV (exit 139) while the interpreter unloads XPU state.
    os._exit(0)


if __name__ == "__main__":
    main()
