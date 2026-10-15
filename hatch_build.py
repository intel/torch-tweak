# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Keep the sdist limited to files that git tracks.

Hatchling otherwise walks the worktree and packs every file that is not
gitignored, including untracked files.
"""

from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404 - subprocess is used only with fixed git argv and shell=False.
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _tracked_files(root: str) -> list[str] | None:
    """Return paths tracked by git, or None when this tree has no git metadata.

    An unpacked sdist has no .git directory. File selection then stays with
    Hatchling, which sees the set already stored in that sdist.
    """
    git = shutil.which("git")
    if git is None:
        return None

    completed = subprocess.run(  # nosec B603
        [git, "-C", root, "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return None

    tracked: list[str] = []
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        candidate = os.path.join(root, relative)
        # Skip gitlinks. A directory path would make Hatchling walk untracked children.
        if os.path.isdir(candidate) or not os.path.isfile(candidate):
            continue
        tracked.append(relative)
    return tracked


class SdistGitFilesHook(BuildHookInterface):
    """Select sdist members from ``git ls-files``."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002
        """Restrict sdist members to the git index when git metadata is present."""
        if self.target_name != "sdist":
            return

        tracked = _tracked_files(self.root)
        if not tracked:
            return

        self.build_config.target_config["only-include"] = tracked
        # only_include is a cached_property. Drop a value computed before this assignment.
        self.build_config.__dict__.pop("only_include", None)
