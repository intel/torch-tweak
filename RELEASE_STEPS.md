<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
## Torch Tweak Release Steps

This document contains all the steps required to make a new release of Torch Tweak.

As a helper, we use in this guide this variable:
```bash
  VERSION=0.1.0rc1   # New version, including an optional rc suffix (set here as an example)
```

**Note:**
> Before doing the final release, it's recommended to prepare a pre-release version - a "release candidate"
> (or "rc" in short). This requires adding, e.g., `rc1` to the VERSION string. When all tests and checks
> end properly, you can follow up with the final release. If any fix is required, it should be included in
> another rc version (e.g., `0.1.0rc2`).

1. Make sure remotes are up-to-date on your machine (`git remote update`).
1. Make changes for the release.
1. Create a new tag based on the latest commit - it should follow the format:
  `v<major>.<minor>.<patch>` (e.g., `v0.1.0`).
1. Push the tag and branch to the repository.
1. Create a new GitHub release using the tag created in the previous step.

## Make a release locally

Prepare changes for the release:
- Update remotes
  - `git remote update`
- Add a new entry to the `ChangeLog.md`
  - For major releases mention API compatibility with the previous releases
- Update `fallback_version` in `[tool.hatch.version] raw-options` in all `pyproject.toml` files,
  including top-level and examples, and refresh the `uv.lock` files
  - Run commands like:
    ```bash
    sed -i -E "s/(fallback_version = \")[^\"]*\"/\1${VERSION}\"/" pyproject.toml examples/*/pyproject.toml
    git diff --stat   # expect one changed line in every pyproject.toml
    uv lock
    for example in examples/*/pyproject.toml; do uv lock --directory "$(dirname "${example}")"; done
    ```
  - The fallback is the last released version, for a build on a machine without access to github's tags
- Update `version` and `date-released` in `CITATION.cff`
- Once all changes are done, [build locally](CONTRIBUTING.md#building-packages) from a clean worktree
  (and/or verify changes on CI), including:
  - `git status --porcelain` prints nothing. The packaging workflow fails when it does not, so a dirty tree is not packed
  - Verify if scanners/linters/checkers passed
  - Make sure packaging workflow builds properly (you can check that on your fork)
  - Verify if version is set properly in the package
- Commit these changes and tag the release:
  - `git add uv.lock examples CITATION.cff pyproject.toml ChangeLog.md`
  - `git commit -S -m "chore: release v$VERSION"`
  - `git tag -a -s -m "Version $VERSION" v$VERSION`
- Verify if commit and tag are properly signed:
  - `git verify-commit <commit's sha>`
  - `git verify-tag v$VERSION`

## Publish changes

- Push `main` branch and the new tag
  - `git push origin HEAD:main v$VERSION --dry-run`
  - Note `--dry-run` in the above command - before actual push, make sure you're using the right remote!

## Announce release

To make the release official:
- Go to [GitHub's releases tab](https://github.com/intel/torch-tweak/releases/new):
  - Tag version: `v$VERSION`
  - Release title: `Torch Tweak $VERSION`
  - Description: copy entry from `ChangeLog.md` and format it (e.g., no characters limit in line)
- Announce the release in all appropriate channels

## More information

To assure the community that the release is a valid package from Torch Tweak maintainers, it's recommended to sign the release
commit and the tag (`-S`/`-s` parameters in commands above). If you require to generate a GPG key follow
[these steps](https://docs.github.com/en/authentication/managing-commit-signature-verification/generating-a-new-gpg-key).
After that you'd also have to add this new key to your GitHub account - please do the steps in
[this guide](https://docs.github.com/en/authentication/managing-commit-signature-verification/telling-git-about-your-signing-key).
