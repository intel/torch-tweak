<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## Unreleased

- feat (breaking): `inspect` measures module depth in executed modules.
  Containers such as `nn.ModuleList` and modules that never run no longer add a level.
  `min_depth` is renamed to `depth`.
  `depth` keeps shallower executed modules without executed children, so a cut covers the whole model.
  `max_depth` is removed, the traversal limit follows from `depth`.
  A diffusers pipeline that used `min_depth=1` to reach its components now uses `depth=0`.
  `inspect` warns when the walk stops before any module is registered.
  `warn_if_no_modules=False` turns that warning off.

## 0.4.0

This is the first release of **Torch Tweak**. Torch Tweak is focused on Intel GPU (XPU) support.
This release introduces i.a. new backend, tighter dependency and security updates,
code cleanups, and new continuous integration system.

This is a pre-release (pre-1.0) version and not yet production-quality.

Please note, this project was forked from repository https://github.com/ai-dynamo/aitune,
from version 0.3.0. To see its changelog go to the upstream repository.

Details:
- feat: add XPU support with Intel GPU-focused installation requirements and device handling throughout the toolkit
- feat: add OpenVINO backend with tests
- feat: improve tuning flow for XPU workloads, including hybrid XPU benchmark support
- fix: improve robustness for safety-critical file handling and I/O operations
- fix: update dependency constraints and security patches for vulnerable packages and compatibility issues
- misc: streamline docs, CI, and environment setup for XPU-based development and deployment
- misc: change default save extension from .ait to .tt
- misc: unblock latest Python 3.13+
