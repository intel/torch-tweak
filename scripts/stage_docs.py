# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""ProperDocs hook: copy root files into docs/ before every build.

A handful of files live outside docs/ but are referenced in the nav:
  ChangeLog.md, LICENSE, examples/*/README.md

This hook copies them into docs/ so properdocs can find them.  The copies are
gitignored and never committed; the originals at the repo root are the source
of truth.  Runs automatically on every build, including live-reload rebuilds.
"""

import logging
import pathlib
import shutil

logger = logging.getLogger("scripts.stage_docs")

_REPO_ROOT = pathlib.Path(__file__).parent.parent
_DOCS = _REPO_ROOT / "docs"


def on_pre_build(config, **kwargs):
    """Copy root files that the nav references into docs/."""
    shutil.copy(_REPO_ROOT / "ChangeLog.md", _DOCS / "ChangeLog.md")
    shutil.copy(_REPO_ROOT / "LICENSE", _DOCS / "LICENSE.md")

    examples_dst = _DOCS / "examples"
    examples_dst.mkdir(exist_ok=True)
    shutil.copy(_REPO_ROOT / "examples" / "README.md", examples_dst / "README.md")

    for readme in sorted((_REPO_ROOT / "examples").glob("*/README.md")):
        dst_dir = examples_dst / readme.parent.name
        dst_dir.mkdir(exist_ok=True)
        shutil.copy(readme, dst_dir / "README.md")

    logger.info("Staged root files into docs/")
