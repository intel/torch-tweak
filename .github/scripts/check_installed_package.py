# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check that the installed torch-tweak release can run on an Intel XPU.

Run it with the interpreter of the environment that holds the installed wheel,
outside the repository root, so torch_tweak is not imported from the sources.
"""

import argparse
import logging
import os
import sys
from importlib.metadata import distribution
from pathlib import Path

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _problems(expected_version: str) -> list[str]:
    """Report every reason why the installed package cannot be used on an XPU."""
    import torch

    import torch_tweak

    dist = distribution("torch-tweak")
    installed = Path(dist.locate_file("torch_tweak/__init__.py")).resolve()
    imported = Path(torch_tweak.__file__).resolve()

    logger.info("torch_tweak %s from %s", dist.version, imported)
    logger.info("torch %s, xpu build %s", torch.__version__, torch.version.xpu)

    problems = []
    if dist.version != expected_version:
        problems.append(f"installed torch-tweak {dist.version} is not {expected_version}")
    # An in-tree torch_tweak.egg-info on sys.path is found too, but has no INSTALLER.
    if dist.read_text("INSTALLER") is None:
        problems.append(f"torch-tweak metadata at {dist.locate_file('')} does not come from an installer")
    if imported != installed:
        problems.append(f"torch_tweak is imported from {imported}, not from the installed {installed}")
    if torch.version.xpu is None:
        problems.append("installed torch is not an XPU build")
        return problems

    # is_available() can report a device that fails to initialize, e.g. when
    # ONEAPI_DEVICE_SELECTOR filters it out, so run a computation on it.
    selector = os.environ.get("ONEAPI_DEVICE_SELECTOR", "<unset>")
    try:
        logger.info("XPU device: %s", torch.xpu.get_device_name(0))
        expected_result = 12
        result = (torch.arange(4, device="xpu") * 2).sum().item()
    except RuntimeError as error:
        problems.append(f"XPU device cannot be used ({error}, ONEAPI_DEVICE_SELECTOR={selector})")
    else:
        if result != expected_result:
            problems.append(f"XPU computation returned {result} instead of {expected_result}")
    return problems


def main() -> int:
    """Check the installed package against the released version."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="released torch-tweak version, e.g. 0.5.0rc1")
    args = parser.parse_args()

    problems = _problems(args.version)
    for problem in problems:
        logger.error("FAIL  %s", problem)
    if not problems:
        logger.info("OK    torch-tweak %s with XPU support", args.version)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
