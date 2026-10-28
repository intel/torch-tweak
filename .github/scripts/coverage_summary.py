# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Compare line coverage against a baseline and render a GitHub Actions job summary.

Inputs are ``coverage json`` reports (current and optional baseline) and an optional
``diff-cover --format json`` report with coverage of the lines changed by a PR.

The script appends Markdown to ``$GITHUB_STEP_SUMMARY`` (or stdout), prints a ``::notice``
annotation with the headline numbers, and fails on exit when total line coverage dropped
below ``--fail-threshold`` percentage compared to the baseline.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def emit(message: str) -> None:
    """Write a line to stdout, where GitHub Actions picks up workflow commands."""
    sys.stdout.write(message + "\n")


def load_json(path: Path | None) -> dict | None:
    """Return parsed JSON from ``path`` or None when the file is not available."""
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def line_stats(summary: dict) -> tuple[int, int]:
    """Return ``(covered_lines, num_statements)`` from a coverage.py summary block."""
    return summary["covered_lines"], summary["num_statements"]


def percent(covered: int, total: int) -> float:
    """Return coverage percentage; empty code counts as fully covered."""
    return 100.0 * covered / total if total else 100.0


def file_percents(report: dict) -> dict[str, tuple[float, int, int]]:
    """Map each file to ``(percent, covered, statements)``."""
    result = {}
    for name, data in report["files"].items():
        covered, total = line_stats(data["summary"])
        result[name] = (percent(covered, total), covered, total)
    return result


def fmt_delta(delta: float) -> str:
    """Format a percentage-point delta with an explicit sign."""
    if abs(delta) < 0.005:
        return "±0.00 pp"
    return f"{delta:+.2f} pp"


def changed_files_table(current: dict, baseline: dict) -> list[str]:
    """Build a Markdown table of files whose line coverage differs from the baseline."""
    cur = file_percents(current)
    base = file_percents(baseline)
    rows = []
    for name in sorted(cur.keys() | base.keys()):
        c, b = cur.get(name), base.get(name)
        if c and b and abs(c[0] - b[0]) < 0.005:
            continue
        base_txt = f"{b[0]:.2f}% ({b[1]}/{b[2]})" if b else "—"
        cur_txt = f"{c[0]:.2f}% ({c[1]}/{c[2]})" if c else "removed"
        delta_txt = fmt_delta(c[0] - b[0]) if c and b else ("new" if c else "removed")
        rows.append(f"| `{name}` | {base_txt} | {cur_txt} | {delta_txt} |")
    if not rows:
        return ["No per-file coverage changes."]
    return [
        "| File | Baseline | Current | Δ |",
        "|------|---------:|--------:|--:|",
        *rows,
    ]


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--current", type=Path, required=True, help="coverage json report of this run")
    parser.add_argument("--baseline", type=Path, help="coverage json report of the baseline (latest main)")
    parser.add_argument("--baseline-label", default="main", help="label of the baseline, e.g. 'main @ abc1234'")
    parser.add_argument("--diff-cover", type=Path, help="diff-cover json report of changed lines")
    parser.add_argument("--diff-cover-markdown", type=Path, help="diff-cover markdown report to embed")
    parser.add_argument(
        "--fail-threshold", type=float, help="fail when total coverage drops by more then this percentage"
    )
    args = parser.parse_args()

    current = load_json(args.current)
    if current is None:
        emit(f"::error title=Coverage::Coverage report {args.current} not found")
        return 1
    baseline = load_json(args.baseline)
    diff = load_json(args.diff_cover)

    cur_cov, cur_total = line_stats(current["totals"])
    cur_pct = percent(cur_cov, cur_total)

    lines = ["## Line coverage", ""]
    lines += ["| | Baseline | Current | Δ |", "|---|---:|---:|--:|"]
    notice = [f"Total line coverage: {cur_pct:.2f}%"]
    delta = None
    if baseline is not None:
        base_cov, base_total = line_stats(baseline["totals"])
        base_pct = percent(base_cov, base_total)
        delta = cur_pct - base_pct
        lines.append(
            f"| **Total** ({args.baseline_label}) | {base_pct:.2f}% ({base_cov}/{base_total}) "
            f"| **{cur_pct:.2f}%** ({cur_cov}/{cur_total}) | **{fmt_delta(delta)}** |"
        )
        notice[0] += f" ({fmt_delta(delta)} vs {args.baseline_label})"
    else:
        lines.append(f"| **Total** | no baseline | **{cur_pct:.2f}%** ({cur_cov}/{cur_total}) | — |")
        notice[0] += " (no baseline available)"

    if diff is not None:
        diff_total = diff.get("total_num_lines", 0)
        if diff_total:
            diff_missed = diff.get("total_num_violations", 0)
            diff_pct = float(diff.get("total_percent_covered", percent(diff_total - diff_missed, diff_total)))
            lines.append(
                f"| **Changed lines** (diff-cover) | — | **{diff_pct:.2f}%** "
                f"({diff_total - diff_missed}/{diff_total}) | — |"
            )
            notice.append(f"changed lines: {diff_pct:.2f}% ({diff_total - diff_missed}/{diff_total})")
        else:
            lines.append("| **Changed lines** (diff-cover) | — | no measured lines changed | — |")
            notice.append("changed lines: none measured")

    fail_threshold = args.fail_threshold
    failed = delta is not None and fail_threshold is not None and delta < -fail_threshold
    if failed:
        lines += [
            "",
            f"> [!CAUTION]\n> Total line coverage dropped by {-delta:.2f} pp, "
            f"which exceeds the allowed {fail_threshold:.2f} pp.",
        ]

    if baseline is not None:
        lines += ["", "<details><summary>Per-file changes vs baseline</summary>", ""]
        lines += changed_files_table(current, baseline)
        lines += ["", "</details>"]

    diff_md = args.diff_cover_markdown
    if diff_md is not None and diff_md.is_file():
        lines += ["", "<details><summary>diff-cover report</summary>", ""]
        lines += [diff_md.read_text(encoding="utf-8").strip(), "", "</details>"]

    lines.append("")
    lines.append("**HTML coverage report** is available in the `coverage-report` artifact.")
    lines.append("Download it down below and open `htmlcov\\index.html` to view the report.")
    lines.append("")

    output = "\n".join(lines) + "\n"
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(output)
    else:
        emit(output)

    emit(f"::notice title=Coverage::{', '.join(notice)}")
    if failed:
        emit(f"::error title=Coverage::Total line coverage dropped by {-delta:.2f}% (threshold: {fail_threshold:.2f}%)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
