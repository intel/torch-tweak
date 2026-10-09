<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Contributing to Torch Tweak

Thank you for your interest in contributing to Torch Tweak. This document
describes how to set up a development environment, run tests and linters,
and submit changes.

Torch Tweak is in pre-release (pre-1.0). For any larger changes, please
open an issue to discuss it before starting work on it.

## Ways to contribute

- Report bugs and request features via
  [GitHub Issues](https://github.com/intel/torch-tweak/issues).
- Work on any existing feature requests, or fix bugs via pull requests.
- Improve documentation in `docs/`, `notebooks/` or `examples/`.

When reporting a bug, please include:

- Torch Tweak commit hash, Python, PyTorch and OpenVINO versions, if applicable.
- GPU model and driver version
- A minimal snippet reproducing the problem and the full error message.

Security vulnerabilities must **not** be reported as public issues. Follow
the process described in [SECURITY.md](SECURITY.md).

## Development environment

Torch Tweak targets Linux x86_64 with Intel GPUs (XPU). The project uses
[uv](https://docs.astral.sh/uv/) to manage dependencies and virtual
environments.

### Stable PyTorch XPU (default)

From the repository root:

```bash
uv sync --extra dev
```

This creates `.venv` with stable PyTorch XPU wheels and all development
dependencies (for tests, docs, linters). Prefer `uv run ...` over activating the
virtual environment manually.

### Nightly PyTorch XPU (optional)

Use nightly only when a change requires PyTorch XPU nightly wheels. Keep it
in a separate virtual environment, and set `UV_PROJECT_ENVIRONMENT` on every
`uv sync` and `uv run` call:

```bash
UV_PROJECT_ENVIRONMENT=.venv-nightly uv sync --no-default-groups --group xpu-nightly --extra dev
UV_PROJECT_ENVIRONMENT=.venv-nightly uv run pytest
```

Do not combine the `xpu-nightly` and `xpu-stable` groups in a single sync;
they conflict by design.

### Adding dependencies

If you add or change a dependency, update `pyproject.toml`, then regenerate
the lock file and re-sync:

```bash
uv lock
uv sync --extra dev
```

Commit `pyproject.toml` and `uv.lock` together.

## Running tests

```bash
# Unit tests and library doctests
uv run pytest -m "not functional and not nightly"

# Functional tests
uv run pytest -m functional

# A single test file
uv run pytest tests/unit/torch/backend/test_openvino_backend.py -v
```

Test markers:

- `functional` - slower integration tests that use real models and third-party
  libraries (timm, transformers, diffusers) and may download pretrained models.
- `nightly` - long-running tests that run only in the nightly workflow.

Most tests require an Intel GPU. Use `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
to select the device, as CI does.

New features and bug fixes should come with tests. Place them under
'test/unit' or 'test/functional', respectively.

## Code style

- Formatting and linting are done with [Ruff](https://docs.astral.sh/ruff/).
- Docstrings follow the Google convention.
- Use plain ASCII in code and comments (for example, use `-` instead of
  typographic dashes, and avoid arrows or other special symbols).

Run the linters before pushing:

```bash
uv run ruff check --fix && uv run ruff format
uv run pre-commit run --all-files
```


Installing the hooks lets them run automatically on every commit:

```bash
uv run pre-commit install
```

The `ruff` version pinned in `pyproject.toml` must match the
`ruff-pre-commit` revision in `.pre-commit-config.yaml`. CI fails if they
differ. Bare that in mind when updating ruff checker.

## License headers

Every new source file must start with an SPDX license header, for example:

```python
# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
```

Parts of this project were forked from
[ai-dynamo/aitune](https://github.com/ai-dynamo/aitune). When modifying a file
that carries an NVIDIA copyright line, keep that line unchanged, and add the
Intel copyright and the `NOTE: This file has been modified by Intel
Corporation.` notice.

The `fix-copyright-headers` pre-commit hook adds missing headers, and the
License Check workflow (`.github/scripts/check_licenses.py`) enforces these
rules on every pull request.

## Commit messages

Commit titles follow the convention described in
[COMMIT_CONVENTION.md](COMMIT_CONVENTION.md):

```
<type>(<scope>): <short description>
```

For example: `chore(ci): fix nightly workflow`.

### Sign your work

All commits must be signed off to certify that you wrote the change or
otherwise have the right to submit it under the project license, as stated in
the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO). Add a sign-off line using `git commit -s`:

```
Signed-off-by: Jane Doe <jane.doe@example.com>
```

Use your real name and an email address that matches your Git configuration.

## Pull requests

1. Fork the repository and create a topic branch from `main`. Direct commits
   to `main` are blocked by a pre-commit hook.
2. Keep pull requests focused on a single change. Split unrelated changes into
   separate pull requests.
3. Make sure tests, linters and the license check pass locally.
4. Update documentation in `docs/` and `ChangeLog.md` when the change affects
   users.
5. Open a pull request against `main` with a clear description of the change
   and links to related issues (for example, `Closes #123`), if applicable.

CI runs linting and the license check on every pull request. GPU tests run
only on the upstream repository, since forks do not have access to the
required hardware.

A maintainer will review your pull request and may ask for changes before
merging it.

## License

By contributing to Torch Tweak, you agree that your contributions will be
licensed under the [Apache License 2.0](LICENSE).
