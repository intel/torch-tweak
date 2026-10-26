# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""License compliance checker for pull requests and pre-commit header auto-fix.

Two distinct git references are used for **check** mode (CI):

  UPSTREAM_BASELINE_SHA  - fixed commit that represents the original upstream-authored
                         codebase.  Used to determine what upstream copyright lines
                         existed before Intel forked the project.

  origin/<base_ref>    - HEAD of the PR target branch (e.g. origin/main).
                         Used only to determine which files this PR actually
                         changed (three-dot diff).  Kept separate so that files
                         modified on main in earlier commits - but not touched by
                         the current PR - do not generate false positives.

Rules enforced (check mode)
---------------------------
Rule 1 - Original Header Retention
    For every modified file: if the upstream baseline version contained an upstream
    copyright line, the PR version must keep that exact line unchanged.

Rule 2 - Intel Copyright & Modification Notice
    Every modified file that carried an upstream copyright in the baseline must
    also contain an Intel copyright line AND the modification notice:
        Copyright (c) <year> Intel Corporation
        ...
        NOTE: This file has been modified by Intel Corporation.
    (Use the appropriate comment syntax for the file type, e.g. ``#`` for Python
    or ``<!-- ... -->`` for Markdown/HTML.)

Rule 3 - Intel-Only Header (brand-new files)
    Every newly added file must carry an Intel copyright line.

Rule 4 - Root LICENSE file integrity
    The root LICENSE file must not be modified or deleted in the PR.

Notebooks (``.ipynb``) keep the headers from these rules in the first Markdown cell.

**Fix mode** (``--fix``, pre-commit) writes the same header normalization that **check**
compares against: if normalize would change a ``*.py`` file, check reports a violation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - subprocess is used only with fixed git argv and shell=False.
import sys
from datetime import date
from pathlib import Path

logger = logging.getLogger(Path(__file__).stem)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------------------------------------------------------------------------
# Compiled patterns & header text
# ---------------------------------------------------------------------------

# "upstream" means NVIDIA AITune, the project Torch Tweak was forked from; its
# headers carry an NVIDIA copyright, hence the literal "NVIDIA" below.
_UPSTREAM_COPYRIGHT_RE = re.compile(
    r"SPDX-FileCopyrightText:\s*Copyright\s*\(c\)\s*\d+\s*NVIDIA",
)

# Rule 1 also covers this tag: accidental deletion/edit must be caught, same as the
# NVIDIA copyright line above.
_SPDX_LICENSE_IDENTIFIER_RE = re.compile(r"SPDX-License-Identifier:\s*\S+")

_INTEL_COPYRIGHT_RE = re.compile(
    r"Copyright\s*\(c\)\s*\d+\s*Intel Corporation",
)

_INTEL_MODIFICATION_RE = re.compile(
    r"(?:<!--\s*|#\s*)?NOTE:\s*This file has been modified by Intel Corporation",
)

# PEP 263: optional ``# coding: …`` (or ``# -*- coding: … -*-``) source encoding declaration.
_PEP263_ENCODING_RE = re.compile(r"^#.*coding[:=]\s*[-\w.]+", re.IGNORECASE)

_INTEL_COPYRIGHT_BODY = f"Copyright (c) {date.today().year} Intel Corporation"
_INTEL_MODIFICATION_BODY = "NOTE: This file has been modified by Intel Corporation."

_INTEL_COPYRIGHT_LINE_PY = f"# {_INTEL_COPYRIGHT_BODY}"
_INTEL_MODIFICATION_LINE_PY = f"# {_INTEL_MODIFICATION_BODY}"

# Fixed SHA representing the original upstream-authored state of the repository.
# Rules 1 and 2 are checked against this commit, not against origin/main, so
# that the baseline cannot be eroded by later commits on main.
UPSTREAM_BASELINE_SHA = "31ae3e0cd0b5ed15c4018cbddd4854e1f06cd6bb"

# First _CPY_HEAD_BYTES octets at file start (ruff CPY001 / Python header auto-fix window).
_CPY_HEAD_BYTES = 4096

# Suffixes excluded from all license checks.
# Notebooks are not listed: the copyright header is the first Markdown cell
# and is checked by _check_notebook.
_SKIP_SUFFIXES: frozenset[str] = frozenset({".png", ".webp"})

# Files under these paths are never checked or auto-fixed.
_SKIP_PATH_PREFIXES: frozenset[str] = frozenset({
    ".github/workflows/",
})

# Exact file paths that are exempt from all license checks.
_SKIP_EXACT_PATHS: frozenset[str] = frozenset({
    ".github/dependabot.yml",
})

# Suffixes whose headers live inside HTML comments (<!-- -->) rather than
# #-prefixed lines.
_HTML_COMMENT_SUFFIXES: frozenset[str] = frozenset({".md", ".html"})

# Non-Python files that use ``#``-prefixed comment headers (YAML, TOML, INI, …).
_HASH_COMMENT_SUFFIXES: frozenset[str] = frozenset({".yaml", ".yml", ".toml", ".ini", ".cfg"})


def _is_skipped(filepath: str) -> bool:
    if Path(filepath).suffix.lower() in _SKIP_SUFFIXES:
        return True
    normalized = filepath.replace("\\", "/")
    if normalized in _SKIP_EXACT_PATHS:
        return True
    return any(normalized.startswith(p) for p in _SKIP_PATH_PREFIXES)


def _uses_html_comments(filepath: str) -> bool:
    return Path(filepath).suffix.lower() in _HTML_COMMENT_SUFFIXES


def _uses_hash_comments(filepath: str) -> bool:
    """True for non-Python files that use ``#`` comment syntax (YAML, TOML, INI, …)."""
    p = Path(filepath)
    return p.suffix.lower() in _HASH_COMMENT_SUFFIXES


def _is_python(filepath: str) -> bool:
    return Path(filepath).suffix.lower() == ".py"


def _python_header_autofix_applies(filepath: str, *, require_file: bool = False) -> bool:
    path = Path(filepath)
    if not _is_python(filepath) or _is_skipped(filepath) or _uses_html_comments(filepath):
        return False
    return path.is_file() if require_file else True


def _comment_prefix(filepath: str) -> str:
    return "" if _uses_html_comments(filepath) else "# "


def _intel_copyright_hint(filepath: str) -> str:
    return f"{_comment_prefix(filepath)}{_INTEL_COPYRIGHT_BODY}"


def _intel_modification_hint(filepath: str) -> str:
    return f"{_comment_prefix(filepath)}{_INTEL_MODIFICATION_BODY}"


def _upstream_copyright_lines(text: str) -> list[str]:
    return [ln.rstrip("\n") for ln in text.splitlines() if _UPSTREAM_COPYRIGHT_RE.search(ln)]


def _spdx_license_lines(text: str) -> list[str]:
    return [ln.rstrip("\n") for ln in text.splitlines() if _SPDX_LICENSE_IDENTIFIER_RE.search(ln)]


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


_SHA_RE = re.compile(r"^[0-9a-f]{4,40}$", re.IGNORECASE)

_GIT_BIN = shutil.which("git")
if _GIT_BIN is None:
    raise RuntimeError("git executable was not found in PATH")


def _run_git(
    args: list[str],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run git with a fully-qualified executable path and no shell expansion."""
    return subprocess.run(
        [_GIT_BIN, *args],
        capture_output=True,
        text=True,
        errors="replace",
        check=check,
    )  # nosec B603


def _resolve_ref(ref: str) -> str:
    """Return the git ref string to pass to git commands.

    A branch/tag name is prefixed with ``origin/`` so it resolves against the
    remote.  A commit SHA (4-40 hex chars) is returned as-is - git can resolve
    it directly without a remote prefix.
    """
    return ref if _SHA_RE.match(ref) else f"origin/{ref}"


def _git_show(ref: str, filepath: str) -> str | None:
    """Return file content at *ref*, or None if the path does not exist there."""
    result = _run_git(["show", f"{ref}:{filepath}"])
    return result.stdout if result.returncode == 0 else None


def _head_content(filepath: str) -> str | None:
    path = Path(filepath)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _changed_files(base_ref: str) -> list[tuple[str, str, str]]:
    """Return [(status, source_path, dest_path), …] for changed files.

    Status is the single-character git diff status: A, M, D, R, C, …
    For non-rename/copy entries, source_path == dest_path.
    """
    result = _run_git(["diff", "--name-status", f"{_resolve_ref(base_ref)}...HEAD"], check=True)
    entries: list[tuple[str, str, str]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]  # first char: A / M / D / R / C …
        if status in ("R", "C") and len(parts) >= 3:
            entries.append((status, parts[1], parts[2]))
        else:
            entries.append((status, parts[1], parts[1]))
    return entries


def _git_show_at_baseline(filepath: str) -> str | None:
    """Return file content at UPSTREAM_BASELINE_SHA, following directory renames.

    Tries the current path first.  When not found (e.g. a parent directory was
    renamed), computes the git blob hash of the working-tree file and looks for
    that exact blob in the baseline tree. A match means the file is a pure rename,
    with no content changes.

    Returns ``None`` when the file was modified or is brand-new.
    """
    content = _git_show(UPSTREAM_BASELINE_SHA, filepath)
    if content is not None:
        return content

    # Compute the git blob hash of the current file (same as baseline when
    # content is identical, i.e. a pure rename).
    hash_result = _run_git(["hash-object", filepath])
    if hash_result.returncode != 0 or not hash_result.stdout.strip():
        return None
    blob_hash = hash_result.stdout.strip()

    # Walk the baseline tree looking for this blob.
    tree_result = _run_git(["ls-tree", "-r", UPSTREAM_BASELINE_SHA])
    if tree_result.returncode != 0:
        return None
    for line in tree_result.stdout.splitlines():
        # Format: "<mode> blob <hash>\t<path>"
        tab_parts = line.split("\t", 1)
        if len(tab_parts) < 2:
            continue
        obj_info = tab_parts[0].split()
        if len(obj_info) >= 3 and obj_info[2] == blob_hash:
            return _git_show(UPSTREAM_BASELINE_SHA, tab_parts[1])
    return None


def _baseline_upstream_lines(filepath: str) -> list[str]:
    return _upstream_copyright_lines(_git_show_at_baseline(filepath) or "")


def _file_is_upstream_derived(filepath: str, head: str) -> bool:
    """True when *head* is an Intel-modified derivative of an upstream-authored file.

    A file is "derived" only when Intel actually changed its content relative to
    the upstream baseline.  Pure renames and verbatim upstream copies return False
    so that ``--all-files`` pre-commit runs do not add incorrect Intel attribution.
    """
    baseline = _git_show_at_baseline(filepath)
    if not baseline:
        return bool(_UPSTREAM_COPYRIGHT_RE.search(head))
    if not _upstream_copyright_lines(baseline):
        return False
    return baseline.splitlines() != head.splitlines()


# ---------------------------------------------------------------------------
# Python header normalization (shared by check and --fix)
# ---------------------------------------------------------------------------


def _cpy_head_prefix(content: str) -> str:
    """Leading text within the first ``_CPY_HEAD_BYTES`` UTF-8 octets (matches ruff CPY001 scope)."""
    return content.encode("utf-8")[:_CPY_HEAD_BYTES].decode("utf-8", errors="ignore")


def _cpy_head_line_prefix(lines: list[str]) -> list[str]:
    """Leading lines of ``lines`` that lie inside the CPY head byte window."""
    return _cpy_head_prefix("".join(lines)).splitlines(keepends=True)


def _insert_lines(lines: list[str], index: int, to_insert: list[str]) -> list[str]:
    block = [ln if ln.endswith("\n") else f"{ln}\n" for ln in to_insert]
    return lines[:index] + block + lines[index:]


def _python_header_insert_index(lines: list[str]) -> int:
    """Index after shebang and optional PEP 263 encoding cookie (headers follow)."""
    index = 0
    if index < len(lines) and lines[index].startswith("#!"):
        index += 1
    if index < len(lines) and _PEP263_ENCODING_RE.match(lines[index].strip()):
        index += 1
    return index


def _derived_intel_insert_index(lines: list[str]) -> int:
    start = _python_header_insert_index(lines)
    insert_after_upstream = start
    for i, line in enumerate(_cpy_head_line_prefix(lines)):
        if i < start:
            continue
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") and _UPSTREAM_COPYRIGHT_RE.search(line):
            insert_after_upstream = i + 1
            continue
        # First non-empty line after upstream block (comment or docstring): insert Intel here.
        return i
    return insert_after_upstream


def _add_intel_copyright_line(lines: list[str], head: str, *, derived: bool) -> list[str]:
    if _INTEL_COPYRIGHT_RE.search(head):
        return lines
    if derived:
        return _insert_lines(lines, _derived_intel_insert_index(lines), [_INTEL_COPYRIGHT_LINE_PY])
    return _insert_lines(lines, _python_header_insert_index(lines), [_INTEL_COPYRIGHT_LINE_PY])


def _insert_intel_modification_notice(lines: list[str]) -> list[str]:
    scan = _cpy_head_line_prefix(lines)
    if _INTEL_MODIFICATION_RE.search("".join(scan)):
        return lines
    for i, line in enumerate(scan):
        if "SPDX-License-Identifier:" not in line:
            continue
        insert_at = i + 1
        if insert_at < len(lines) and lines[insert_at].strip() == "":
            insert_at += 1
        return _insert_lines(lines, insert_at, [_INTEL_MODIFICATION_LINE_PY])
    return _insert_lines(lines, _python_header_insert_index(lines), [_INTEL_MODIFICATION_LINE_PY, ""])


_HEADER_FIX_MSG = (
    "  License header is not canonical.\n"
    "  Run: python3 .github/scripts/check_licenses.py --fix <file>\n"
    "  Or commit again after pre-commit (fix-copyright-headers hook)."
)


def _normalize_python_headers(filepath: str, content: str) -> str:
    """Return *content* after applying all automatic Python header fixes (Rules 2-3 / CPY001)."""
    if not _python_header_autofix_applies(filepath):
        return content

    lines = content.splitlines(keepends=True)
    upstream_from_baseline = _baseline_upstream_lines(filepath)
    derived = _file_is_upstream_derived(filepath, content)

    head_stripped = {ln.strip() for ln in _cpy_head_line_prefix(lines)}
    missing_upstream = [ln for ln in upstream_from_baseline if ln.strip() not in head_stripped]
    # Restore baseline upstream lines when the working tree lost them (Rule 1 retention).
    if missing_upstream:
        lines = _insert_lines(lines, _python_header_insert_index(lines), missing_upstream)

    head = _cpy_head_prefix("".join(lines))

    has_upstream_copyright = bool(_UPSTREAM_COPYRIGHT_RE.search(head))
    if derived or not has_upstream_copyright:
        lines = _add_intel_copyright_line(lines, head, derived=derived)
        if derived:
            lines = _insert_intel_modification_notice(lines)

    return "".join(lines)


def _check_python_header_via_fixup(filepath: str) -> list[str]:
    content = _head_content(filepath)
    if content is None or _normalize_python_headers(filepath, content) == content:
        return []
    return [_HEADER_FIX_MSG]


def fix_python_copyright_header(filepath: str) -> bool:
    """Rewrite Python SPDX copyright headers in *filepath* in place when needed.

    Applies ``_normalize_python_headers`` and writes UTF-8 with LF newlines.
    Returns True when the file on disk was updated.
    """
    if not _python_header_autofix_applies(filepath, require_file=True):
        return False

    path = Path(filepath)
    content = path.read_text(encoding="utf-8", errors="replace")
    new_content = _normalize_python_headers(filepath, content)
    if new_content == content:
        return False
    path.write_text(new_content, encoding="utf-8", newline="\n")
    return True


def _html_new_file_header(content: str, has_intel_copyright: bool) -> str:
    """Return *content* with a full Intel header prepended for brand-new files."""
    if has_intel_copyright:
        return content
    lines = content.splitlines(keepends=True)
    header = [
        "<!--\n",
        f"{_INTEL_COPYRIGHT_BODY}\n",
        "SPDX-License-Identifier: Apache-2.0\n",
        "-->\n",
    ]
    return "".join(header + lines)


def _html_locate_comment_block(lines: list[str]) -> tuple[int, int] | tuple[None, None]:
    """Return ``(block_start, block_end)`` of the first multi-line ``<!-- … -->`` block."""
    block_start: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if block_start is None and stripped.startswith("<!--") and "-->" not in stripped[4:]:
            block_start = i
        elif block_start is not None and "-->" in line:
            return block_start, i
    return None, None


def _html_insert_standalone(
    lines: list[str],
    has_intel_copyright: bool,
    has_intel_modification: bool,
) -> str:
    """Fallback: insert missing Intel lines as standalone ``<!-- … -->`` tags."""
    insert_at = next((i + 1 for i, ln in enumerate(lines) if "-->" in ln), 0)
    to_insert: list[str] = []
    if not has_intel_copyright:
        to_insert.append(f"<!-- {_INTEL_COPYRIGHT_BODY} -->\n")
    if not has_intel_modification:
        to_insert.append(f"<!-- {_INTEL_MODIFICATION_BODY} -->\n")
    return "".join(lines[:insert_at] + to_insert + lines[insert_at:])


def _html_insert_in_block(
    lines: list[str],
    has_intel_copyright: bool,
    has_intel_modification: bool,
    block_start: int,
    block_end: int,
) -> str:
    """Insert missing Intel attribution lines inside an existing ``<!-- … -->`` block."""
    if not has_intel_copyright:
        last_upstream = max(
            (i for i in range(block_start, block_end) if _UPSTREAM_COPYRIGHT_RE.search(lines[i])),
            default=block_start,
        )
        lines = lines[: last_upstream + 1] + [f"{_INTEL_COPYRIGHT_BODY}\n"] + lines[last_upstream + 1 :]
        block_end += 1

    if not has_intel_modification:
        insert_after = next(
            (i for i in range(block_start, block_end) if "SPDX-License-Identifier:" in lines[i]),
            block_end - 1,
        )
        lines = lines[: insert_after + 1] + [f"{_INTEL_MODIFICATION_BODY}\n"] + lines[insert_after + 1 :]

    return "".join(lines)


def _normalize_html_headers(filepath: str, content: str) -> str:
    """Normalize SPDX copyright headers in HTML-comment files (.md, .html)."""
    if not _uses_html_comments(filepath) or _is_skipped(filepath):
        return content

    has_upstream = bool(_UPSTREAM_COPYRIGHT_RE.search(content))
    has_intel_copyright = bool(_INTEL_COPYRIGHT_RE.search(content))
    has_intel_modification = bool(_INTEL_MODIFICATION_RE.search(content))

    if not has_upstream:
        return _html_new_file_header(content, has_intel_copyright)

    if has_intel_copyright and has_intel_modification:
        return content

    # Only add Intel attribution when Intel actually modified this file;
    # verbatim upstream copies must keep their upstream-only header intact.
    if not _file_is_upstream_derived(filepath, content):
        return content

    lines = content.splitlines(keepends=True)
    block_start, block_end = _html_locate_comment_block(lines)

    if block_start is None:
        return _html_insert_standalone(lines, has_intel_copyright, has_intel_modification)

    return _html_insert_in_block(lines, has_intel_copyright, has_intel_modification, block_start, block_end)


def _normalize_hash_headers(filepath: str, content: str) -> str:
    """Normalize ``#``-comment SPDX headers for YAML/TOML/INI files.

    New Intel files:   prepends copyright + ``SPDX-License-Identifier`` lines.
    Upstream-derived:  inserts Intel copyright after the last upstream line and
                       the modification notice after ``SPDX-License-Identifier``.
    """
    if not _uses_hash_comments(filepath) or _is_skipped(filepath):
        return content

    has_upstream = bool(_UPSTREAM_COPYRIGHT_RE.search(content))
    has_intel_copyright = bool(_INTEL_COPYRIGHT_RE.search(content))
    has_intel_modification = bool(_INTEL_MODIFICATION_RE.search(content))

    lines = content.splitlines(keepends=True)

    if not has_upstream:
        if has_intel_copyright:
            return content
        return "".join([f"# {_INTEL_COPYRIGHT_BODY}\n", "# SPDX-License-Identifier: Apache-2.0\n"] + lines)

    if has_intel_copyright and has_intel_modification:
        return content

    # Only add Intel attribution when Intel actually modified this file;
    # verbatim upstream copies must keep their upstream-only header intact.
    if not _file_is_upstream_derived(filepath, content):
        return content

    if not has_intel_copyright:
        last_upstream = max(
            (i for i, ln in enumerate(lines) if _UPSTREAM_COPYRIGHT_RE.search(ln)),
            default=0,
        )
        lines = lines[: last_upstream + 1] + [f"# {_INTEL_COPYRIGHT_BODY}\n"] + lines[last_upstream + 1 :]

    if not has_intel_modification:
        insert_after = next(
            (i for i, ln in enumerate(lines) if "SPDX-License-Identifier:" in ln),
            len(lines) - 1,
        )
        lines = lines[: insert_after + 1] + [f"# {_INTEL_MODIFICATION_BODY}\n"] + lines[insert_after + 1 :]

    return "".join(lines)


def _check_hash_header_via_fixup(filepath: str) -> list[str]:
    content = _head_content(filepath)
    if content is None or _normalize_hash_headers(filepath, content) == content:
        return []
    return [_HEADER_FIX_MSG]


def fix_hash_comment_copyright_header(filepath: str) -> bool:
    """Rewrite ``#``-comment SPDX headers in *filepath* in place when needed."""
    path = Path(filepath)
    if not _uses_hash_comments(filepath) or _is_skipped(filepath) or not path.is_file():
        return False
    content = path.read_text(encoding="utf-8", errors="replace")
    new_content = _normalize_hash_headers(filepath, content)
    if new_content == content:
        return False
    path.write_text(new_content, encoding="utf-8", newline="\n")
    return True


def _check_nonpython_header_via_fixup(filepath: str) -> list[str]:
    content = _head_content(filepath)
    if content is None or _normalize_html_headers(filepath, content) == content:
        return []
    return [_HEADER_FIX_MSG]


def fix_nonpython_copyright_header(filepath: str) -> bool:
    """Rewrite HTML-comment SPDX copyright headers in *filepath* in place when needed."""
    path = Path(filepath)
    if not _uses_html_comments(filepath) or _is_skipped(filepath) or not path.is_file():
        return False

    content = path.read_text(encoding="utf-8", errors="replace")
    new_content = _normalize_html_headers(filepath, content)
    if new_content == content:
        return False
    path.write_text(new_content, encoding="utf-8", newline="\n")
    return True


# ---------------------------------------------------------------------------
# Per-rule checks
# ---------------------------------------------------------------------------


def _check_non_python_upstream_attribution(filepath: str, head: str) -> list[str]:
    errors: list[str] = []
    if not _INTEL_COPYRIGHT_RE.search(head):
        errors.append(f"  Missing Intel copyright line.\n  Add: {_intel_copyright_hint(filepath)}")
    if not _INTEL_MODIFICATION_RE.search(head):
        errors.append(f"  Missing Intel modification notice.\n  Add: {_intel_modification_hint(filepath)}")
    return errors


def _original_content_before_pr(
    filepath: str,
    base_ref: str,
    original_filepath: str | None,
) -> str | None:
    """Content of *filepath* before this PR: fixed baseline first, else tip of base_ref."""
    baseline_path = original_filepath or filepath
    return _git_show(UPSTREAM_BASELINE_SHA, baseline_path) or _git_show(_resolve_ref(base_ref), baseline_path)


def _check_spdx_license_identifier_retention(original: str, head: str) -> list[str]:
    """Rule 1, extended to SPDX-License-Identifier: it must survive verbatim too."""
    head_lines_stripped = {ln.strip() for ln in head.splitlines()}
    return [
        f"  Original SPDX-License-Identifier line missing or altered.\n  Expected verbatim: {orig_line.strip()!r}"
        for orig_line in _spdx_license_lines(original)
        if orig_line.strip() not in head_lines_stripped
    ]


def _check_upstream_copyright_retention(upstream_in_original: list[str], head: str) -> list[str]:
    """Rule 1: every original upstream copyright line must survive verbatim."""
    head_lines_stripped = {ln.strip() for ln in head.splitlines()}
    return [
        f"  Original upstream copyright line missing or altered.\n  Expected verbatim: {orig_line.strip()!r}"
        for orig_line in upstream_in_original
        if orig_line.strip() not in head_lines_stripped
    ]


def _is_notebook(filepath: str) -> bool:
    return Path(filepath).suffix.lower() == ".ipynb"


def _notebook_preamble(content: str) -> str:
    """Return the first Markdown cell, where a notebook copyright header lives."""
    try:
        notebook = json.loads(content)
    except json.JSONDecodeError:
        return ""
    for cell in notebook.get("cells") or []:
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source") or ""
        if isinstance(source, list):
            return "".join(source)
        return str(source)
    return ""


def _notebook_intel_errors(preamble: str, *, modification: bool) -> list[str]:
    errors: list[str] = []
    if not _INTEL_COPYRIGHT_RE.search(preamble):
        errors.append(f"  Missing Intel copyright in the first Markdown cell.\n  Add: {_INTEL_COPYRIGHT_BODY}")
    if modification and not _INTEL_MODIFICATION_RE.search(preamble):
        errors.append(
            f"  Missing Intel modification notice in the first Markdown cell.\n  Add: {_INTEL_MODIFICATION_BODY}"
        )
    return errors


def _check_notebook(
    filepath: str,
    *,
    added: bool,
    base_ref: str = "",
    original_filepath: str | None = None,
) -> list[str]:
    """Check the copyright header in the first Markdown cell of a notebook."""
    head = _head_content(filepath)
    if head is None:
        return []
    preamble = _notebook_preamble(head)
    if added:
        has_upstream = _UPSTREAM_COPYRIGHT_RE.search(preamble) is not None
        return _notebook_intel_errors(preamble, modification=has_upstream)

    original = _git_show(UPSTREAM_BASELINE_SHA, original_filepath or filepath) or _git_show(
        _resolve_ref(base_ref), filepath
    )
    if original is None:
        return []

    baseline_lines = _upstream_copyright_lines(_notebook_preamble(original))
    preamble_lines = {line.strip() for line in preamble.splitlines()}
    errors: list[str] = []
    for orig_line in baseline_lines:
        if orig_line.strip() not in preamble_lines:
            errors.append(
                "  Original upstream copyright line missing from the first Markdown cell.\n"
                f"  Expected verbatim: {orig_line.strip()!r}"
            )
    if baseline_lines or _UPSTREAM_COPYRIGHT_RE.search(preamble):
        errors.extend(_notebook_intel_errors(preamble, modification=True))
    return errors


def _check_modified(
    filepath: str,
    base_ref: str,
    original_filepath: str | None = None,
) -> list[str]:
    """Check original upstream header retention and Intel attribution."""
    if _is_notebook(filepath):
        return _check_notebook(
            filepath,
            added=False,
            base_ref=base_ref,
            original_filepath=original_filepath,
        )
    if _is_skipped(filepath):
        return []

    head = _head_content(filepath)
    if head is None:
        return []

    original = _original_content_before_pr(filepath, base_ref, original_filepath)
    if original is None:
        return []

    upstream_in_original = _upstream_copyright_lines(original)
    errors: list[str] = _check_spdx_license_identifier_retention(original, head)

    # No upstream copyright in original - not an upstream file, nothing to enforce.
    if not upstream_in_original:
        return errors + (_check_python_header_via_fixup(filepath) if _is_python(filepath) else [])

    errors.extend(_check_upstream_copyright_retention(upstream_in_original, head))

    if _is_python(filepath):
        errors.extend(_check_python_header_via_fixup(filepath))
        return errors

    return errors + _check_non_python_upstream_attribution(filepath, head)


def _check_added(filepath: str) -> list[str]:
    """Check a newly added file for correct copyright headers.

    Enforces the Intel-only header (Rule 3), unless the file already
    carries an upstream copyright line in which case it is treated as a derived
    upstream file needing Intel attribution (Rule 2).
    """
    if _is_notebook(filepath):
        return _check_notebook(filepath, added=True)

    if _is_skipped(filepath):
        return []

    if _is_python(filepath):
        return _check_python_header_via_fixup(filepath)

    if _uses_html_comments(filepath):
        return _check_nonpython_header_via_fixup(filepath)

    if _uses_hash_comments(filepath):
        return _check_hash_header_via_fixup(filepath)

    return []


def _check_license_file(changed_files: list[tuple[str, str, str]]) -> list[str]:
    """Check that the root LICENSE file has not been modified or deleted."""
    errors: list[str] = []

    if not Path("LICENSE").exists():
        errors.append("  Root LICENSE file is missing from the repository.")
        return errors

    for status, source_path, filepath in changed_files:
        if status in ("M", "D") and filepath == "LICENSE":
            verb = "modified" if status == "M" else "deleted"
            errors.append(f"  Root LICENSE file has been {verb} in this PR. It must remain intact.")
        elif status in ("R", "C") and source_path == "LICENSE":
            errors.append("  Root LICENSE file has been renamed in this PR. It must remain intact.")

    return errors


# ---------------------------------------------------------------------------
# GHA annotation helpers
# ---------------------------------------------------------------------------


def _annotate_error(filepath: str, message: str) -> None:
    """Emit a GitHub Actions error annotation for *filepath* when running in CI."""
    # Strip leading whitespace / rule tags for the inline annotation title.
    title = message.strip().splitlines()[0].lstrip()
    sys.stdout.write(f"::error file={filepath}::{title}\n")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _collect_violations(
    changed: list[tuple[str, str, str]],
    base_ref: str,
) -> dict[str, list[str]]:
    """Return a mapping of filepath → error list for every file that fails checks."""
    violations: dict[str, list[str]] = {}
    for status, source_path, filepath in changed:
        if status == "M":
            errs = _check_modified(filepath, base_ref)
        elif status == "A":
            errs = _check_added(filepath)
        elif status in ("R", "C"):
            errs = _check_modified(filepath, base_ref, original_filepath=source_path)
        else:
            continue
        if errs:
            violations[filepath] = errs
    return violations


def _report_violations(violations: dict[str, list[str]]) -> None:
    """Log all violations and emit GHA annotations."""
    logger.error("FAILED - license violations found:\n")
    for filepath, errs in violations.items():
        logger.error("  %s", filepath)
        for err in errs:
            logger.error("%s", err)
            _annotate_error(filepath, err)
        logger.error("")
    logger.error("Files with violations: %d", len(violations))


def run_check() -> int:
    """Run license compliance checks and return an exit code."""
    base_ref = os.environ.get("TORCH_TWEAK_LICENSE_CHECKER_BASE_REF", "main")

    logger.info("License compliance check  |  PR diff base: %s", _resolve_ref(base_ref))
    logger.info("                          |  upstream baseline: %s", UPSTREAM_BASELINE_SHA)
    logger.info("=" * 60)

    changed = _changed_files(base_ref)

    if not changed:
        logger.info("No files to check.")
        return 0

    violations = _collect_violations(changed, base_ref)

    license_errs = _check_license_file(changed)
    if license_errs:
        violations.setdefault("LICENSE", []).extend(license_errs)

    if violations:
        _report_violations(violations)
        return 1

    logger.info("OK - %d file(s) passed all license checks.", len(changed))
    return 0


def run_fix(filenames: list[str]) -> int:
    """Fix copyright headers for pre-commit. Exit 1 when any file was modified."""
    if not filenames:
        return 0
    changed = False
    for filepath in filenames:
        fixed = (
            fix_python_copyright_header(filepath)
            or fix_nonpython_copyright_header(filepath)
            or fix_hash_comment_copyright_header(filepath)
        )
        if fixed:
            sys.stderr.write(f"Fixed copyright header: {filepath}\n")
            changed = True
    return 1 if changed else 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="License compliance checker (CI) and Python copyright header auto-fix (pre-commit).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Normalize SPDX copyright headers in the given files (exit 1 if any file changed).",
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="Files to fix when --fix is set (pre-commit passes staged paths).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to check (CI) or ``--fix`` (pre-commit) mode."""
    args = _build_arg_parser().parse_args(argv)
    if args.fix:
        return run_fix(args.files)
    return run_check()


if __name__ == "__main__":
    sys.exit(main())
