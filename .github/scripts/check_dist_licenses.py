# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check that built distributions ship the required license files.

Build backends stay silent when those files go missing, so the artifacts are
inspected here instead of trusting the build.
"""

import argparse
import logging
import sys
import tarfile
import zipfile
from pathlib import Path

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig(level=logging.INFO, format="%(message)s")

REQUIRED_FILES = ("LICENSE", "third-party-programs.txt")


def _members(dist: Path) -> list[str]:
    """Return the member names stored inside a wheel or an sdist."""
    if dist.suffix == ".whl":
        with zipfile.ZipFile(dist) as archive:
            return archive.namelist()
    with tarfile.open(dist) as archive:
        return archive.getnames()


def _problems(dist: Path) -> list[str]:
    """Report every required file that is missing from dist or stored in the wrong place."""
    members = _members(dist)

    if dist.suffix != ".whl":
        basenames = {name.rsplit("/", 1)[-1] for name in members}
        return [f"missing {filename}" for filename in REQUIRED_FILES if filename not in basenames]

    # A wheel carries exactly one .dist-info directory. A second one means some
    # force-include target has gone stale and no longer spells out the current
    # name and version.
    dist_infos = {name.split("/", 1)[0] for name in members if name.split("/", 1)[0].endswith(".dist-info")}
    if len(dist_infos) != 1:
        return [f"expected one .dist-info directory, found {sorted(dist_infos)}"]

    # Directory that PEP 639 reserves for license files inside a wheel.
    licenses = f"{dist_infos.pop()}/licenses/"
    return [f"missing {licenses}{filename}" for filename in REQUIRED_FILES if f"{licenses}{filename}" not in members]


def main() -> int:
    """Inspect every distribution given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dists", nargs="+", type=Path, help="wheel or sdist files to inspect")
    args = parser.parse_args()

    failed = False
    for dist in sorted(args.dists):
        problems = _problems(dist)
        failed = failed or bool(problems)
        for problem in problems:
            logger.error("FAIL  %s: %s", dist.name, problem)
        if not problems:
            logger.info("OK    %s", dist.name)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
