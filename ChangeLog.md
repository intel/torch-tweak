<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Changelog

## Unreleased

- feat (breaking): a module without a strategy is tuned with `tt.default_strategy(device)`.
  It measures every backend that runs on the device and keeps the fastest (#55)
- fix (breaking): `profile` returns `ProfilingResult` with measured throughput per batch size (#64)
- fix(openvino): inputs that compilation renames are fed again.
  A diffusers UNet ran with an unset timestep and returned wrong outputs without an error.
- fix: bf16 and f8 weights on XPU, and non-contiguous ones on CPU, keep their values in the OpenVINO IR (#103)
- feat (breaking): OpenVINO on XPU binds PyTorch tensors as GPU remote tensors without copying them through the host (#102)
- feat: `OpenVINOBackendConfig(offload_module=False)` leaves the module on its device after conversion,
  for a module that a framework keeps calling. The weights then exist twice on the device (#117)
- feat: `OnlineTunedModule` tunes every new input shape on its first call,
  for a module that a framework owns and keeps calling.
  Only the backend of the last call stays active, so tuned shapes do not each keep a copy of the weights (#87)
- feat (breaking): every strategy raises `NoCorrectBackendError`, a `RuntimeError`, when none of its backends works
  for a graph (#97).
  `OneBackendStrategy` no longer raises the error of its backend. Catch `NoCorrectBackendError` instead.
  `__cause__` is a `RuntimeError` with the type and message of the last backend failure.
  `OnlineTunedModule` runs a shape without a backend only on this error. Any other tuning error reaches the caller.
## 0.4.1

- feat: add an XPU graph backend (#33)
- feat (breaking): `inspect` measures module depth in executed modules (#36)
  Containers such as `nn.ModuleList` and modules that never run no longer add a level.
  `min_depth` is renamed to `depth`.
  `depth` keeps shallower executed modules without executed children, so a cut covers the whole model.
  `max_depth` is removed, the traversal limit follows from `depth`.
  A diffusers pipeline that used `min_depth=1` to reach its components now uses `depth=0`.
  `inspect` warns when the walk stops before any module is registered.
  `warn_if_no_modules=False` turns that warning off.
- feat: `save` warns about tuned modules it does not save, e.g. a module the pipeline holds in a plain list (#42)
- fix: a passthrough module without parameters keeps its inputs on their device instead of moving them to CPU (#43)
- fix: `save` stores tuned modules nested in pipeline components under their dotted path, e.g. `vae.decoder`.
  Modules that are not tuned are not saved (#42)
- fix: a tuned module that a pipeline holds under several paths is saved once and restored as one module (#42)
- fix(openvino): save outputs of transformers models (#38)
- fix(openvino): run on the GPU that PyTorch uses (#39)
- fix(checkpoint): leave out samples a checkpoint cannot hold (#37)
- chore: change the build backend to Hatchling (#21)

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
