# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""commit-msg hook: enforce the length limits from COMMIT_CONVENTION.md.

Rules:
  - the title (first line) has at most 72 characters,
  - the second line is blank when a body is present,
  - every body and footer line has at most 72 characters.

Lines starting with "#" (git comments), URLs and indented lines (code blocks) are exempt
from the body limit. Merge, fixup, squash and revert commits are skipped.
"""

import re
import sys
from pathlib import Path

MAX_LINE = 72
SKIPPED_PREFIXES = ("Merge ", "fixup! ", "squash! ", "amend! ", "Revert ")
URL_RE = re.compile(r"https?://\S+")


def check_message(text: str) -> list[str]:
    """Return a list of problems found in the commit message, empty when it is valid."""
    lines = [line for line in text.splitlines() if not line.startswith("#")]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[0].startswith(SKIPPED_PREFIXES):
        return []

    problems = []
    if len(lines[0]) > MAX_LINE:
        problems.append(f"title has {len(lines[0])} characters, the limit is {MAX_LINE}")
    if len(lines) > 1 and lines[1].strip():
        problems.append("the second line must be blank to separate the title from the body")
    for number, line in enumerate(lines[1:], start=2):
        if len(line) <= MAX_LINE or line.startswith((" ", "\t")) or URL_RE.search(line):
            continue
        problems.append(f"line {number} has {len(line)} characters, the limit is {MAX_LINE}; wrap the body")
    return problems


def main(argv: list[str]) -> int:
    """Check the message file passed by git as the first argument."""
    if len(argv) != 2:
        sys.stderr.write("usage: check_commit_message.py <commit-msg-file>\n")
        return 2
    problems = check_message(Path(argv[1]).read_text(encoding="utf-8"))
    if problems:
        sys.stderr.write("Commit message violates COMMIT_CONVENTION.md:\n")
        for problem in problems:
            sys.stderr.write(f"  - {problem}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
