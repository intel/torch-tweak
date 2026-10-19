# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Check that built distributions ship the required license files.

Build backends stay silent when those files go missing, so we inspect the build
artifacts. Each packaged file must also be identical to the canonical copy at
the repository root - check if e.g. symlinks haven't been changed or replaced by
regular files that have drifted out of date.
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
REPO_ROOT = Path(__file__).resolve().parents[2]


def _member_names(dist: Path) -> set[str]:
    """Return the regular files stored inside a wheel or an sdist."""
    if dist.suffix == ".whl":
        with zipfile.ZipFile(dist) as archive:
            return {name for name in archive.namelist() if not name.endswith("/")}
    with tarfile.open(dist) as archive:
        return {member.name for member in archive.getmembers() if member.isfile()}


def _read_file(dist: Path, name: str) -> bytes:
    """Return the content of a single member of a wheel or an sdist."""
    if dist.suffix == ".whl":
        with zipfile.ZipFile(dist) as archive:
            return archive.read(name)
    with tarfile.open(dist) as archive:
        extracted = archive.extractfile(name)
        if extracted is None:
            raise KeyError(name)
        return extracted.read()


def _license_dir(dist: Path, members: set[str]) -> tuple[str | None, str | None]:
    """Return the directory expected to hold the license files, or an error message."""
    top_dirs = {name.split("/", 1)[0] for name in members if "/" in name}

    if dist.suffix != ".whl":
        # An sdist unpacks into a single <name>-<version>/ directory.
        if len(top_dirs) != 1:
            return None, f"expected one top-level directory, found {sorted(top_dirs)}"
        return f"{top_dirs.pop()}/", None

    # A valid wheel carries exactly one .dist-info directory.
    dist_infos = {name for name in top_dirs if name.endswith(".dist-info")}
    if len(dist_infos) != 1:
        return None, f"expected one .dist-info directory, found {sorted(dist_infos)}"
    # Directory that PEP 639 reserves for license files inside a wheel.
    return f"{dist_infos.pop()}/licenses/", None


def _problems(dist: Path, reference: dict[str, bytes]) -> list[str]:
    """Report every required file that is missing from dist or differs from the reference copy."""
    members = _member_names(dist)
    license_dir, error = _license_dir(dist, members)
    if error:
        return [error]

    problems = []
    for filename in REQUIRED_FILES:
        name = f"{license_dir}{filename}"
        if name not in members:
            problems.append(f"missing {name}")
        elif _read_file(dist, name) != reference[filename]:
            problems.append(f"{name} differs from {filename} at the repository root")
    return problems


def main() -> int:
    """Inspect every distribution given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dists", nargs="+", type=Path, help="wheel or sdist files to inspect")
    args = parser.parse_args()

    reference = {filename: (REPO_ROOT / filename).read_bytes() for filename in REQUIRED_FILES}

    failed = False
    for dist in sorted(args.dists):
        problems = _problems(dist, reference)
        failed = failed or bool(problems)
        for problem in problems:
            logger.error("FAIL  %s: %s", dist.name, problem)
        if not problems:
            logger.info("OK    %s", dist.name)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
