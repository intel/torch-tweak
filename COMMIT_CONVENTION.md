<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Commit Message Convention

Commit titles in this repository follow a lightweight variant of
[Conventional Commits](https://www.conventionalcommits.org/). Make sure commit
message explains problem you're solving and/or fully describes enhancements, with reasoning.

## Format

```
<type>(<scope>): <short description>
```

- `<type>` is **required**.
- `(<scope>)` is optional but strongly encouraged when the change targets a
  specific area of the codebase.
- `<short description>` is a concise, lowercase imperative phrase (no trailing
  period). Aim for 72 characters or fewer for the whole title.

---

## Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or user-visible capability |
| `fix` | Bug fix |
| `refactor` | Code restructuring with no behaviour change |
| `perf` | Performance improvement |
| `tests` | Test additions or changes |
| `docs` | Documentation-only changes |
| `chore` | Maintenance that does not fit any other type (cleanup, config, tooling, etc.) |
| `security` | Security-related changes |

---

## Scopes

Scopes map to the main logical areas of the repository. Pick the narrowest
scope that accurately describes the change.

### Core library - `torch_tweak/`

| Scope | Directory / area |
|-------|-----------------|
| `backend` | `torch_tweak/torch/backend/` - shared backend interface |
| `openvino` | `torch_tweak/torch/backend/openvino/` |
| `torchao` | `torch_tweak/torch/backend/torchao_backend.py` |
| `inductor` | `torch_tweak/torch/backend/torch_inductor_backend.py` |
| `tune` | Tuning API (`tuning.py`, `config.py`, top-level `tt.*`) |
| `inspect` | `torch_tweak/torch/inspecting/` |
| `module` | `torch_tweak/torch/module/` |
| `tune-strategy` | `torch_tweak/torch/tune_strategy/` |
| `checkpoint` | `torch_tweak/torch/checkpoint/` - save/load |
| `task` | `torch_tweak/torch/task/` - profiling, correctness, batch-size tasks |
| `integrations` | `torch_tweak/torch/integrations/` - HuggingFace and similar |
| `utils` | `torch_tweak/torch/utils/` or `torch_tweak/utils/` |
| `dataloader` | `torch_tweak/torch/dataloader.py` |

### Examples - `examples/`

| Scope | Example |
|-------|---------|
| `examples` | Changes that affect one or moreexamples |

### Tests - `tests/`

| Scope | Area |
|-------|------|
| `tests` | General or cross-cutting test changes |

### Infrastructure and tooling

| Scope | Area |
|-------|------|
| `ci` | `.github/workflows/` |
| `deps` | `pyproject.toml`, `uv.lock`, Dependabot config |
| `docs` | `docs/`, `notebooks/`, `*.rst` |
| `scripts` | `scripts/` |
| `license` | SPDX headers and `check_licenses.py` |

---

## Body and footer (optional)

When the title alone is not enough, add a blank line followed by a longer
description. Close issues or reference PRs in the footer:

```
fix(checkpoint): check zip file paths before extracting

ZIP entries with path-traversal sequences (../) could be silently
extracted outside the intended target directory. Validate each member
path against the destination root before extraction.

Closes #130
```
