<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0

NOTE: This file has been modified by Intel Corporation.
-->

# Torch Tweak

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-blue.svg)](https://pytorch.org/)
[![Tests](https://github.com/intel/torch-tweak/actions/workflows/tests.yml/badge.svg)](https://github.com/intel/torch-tweak/actions/workflows/tests.yml)
[![Nightly Tests](https://github.com/intel/torch-tweak/actions/workflows/nightly.yml/badge.svg?event=schedule)](https://github.com/intel/torch-tweak/actions/workflows/nightly.yml)

**Torch Tweak** is an inference toolkit designed for tuning and deploying Deep Learning models with a focus on Intel XPU (GPU) accelerators.
It provides model tuning capabilities through compilation and conversion paths that can significantly improve inference speed and efficiency
across various AI workloads including Computer Vision, Natural Language Processing, Speech Recognition, and Generative AI.

The toolkit tunes PyTorch models and pipelines through one Python API. Four backends ship with the package:

* **TorchAO** quantizes weights through PyTorch AO
* **Torch Inductor** compiles with `torch.compile`
* **OpenVINO** converts a module to an OpenVINO model and runs it on an Intel CPU or GPU
* **Torch Eager** runs the module with no compilation, as the baseline other backends are compared against

The guides under [`docs/learn/`](docs/learn/overview.md) are the maintained documentation.
This page is the landing summary. The same file is the PyPI long description.

**Note**: This toolkit is currently in pre-release (pre-1.0) and not yet production-quality. The API is subject to change in future updates.

## Features at Glance

| Feature                     | Description                                                                                                               |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------|
| Ease-of-use                 | Single line of code to run all possible tuning paths directly from your source code                                       |
| Wide Backend Support        | Compatible with TorchAO, Torch Inductor, XPU graph, OpenVINO, and eager PyTorch                                           |
| Model Tuning                | Enhance the performance of models such as ResNET and BERT for efficient inference deployment                              |
| Pipeline Tuning             | Streamline Python code pipelines for models such as Stable Diffusion and Flux using seamless model wrapping and tuning    |
| Correctness Testing         | Ensures tuned models produce correct outputs by validating on provided data samples                                       |
| Performance Profiling       | Profiles models to select the optimal backend based on performance metrics such as latency and throughput                 |
| Model Persistence           | Save and load tuned models for production deployment with flexible storage options                                        |

## When to Use Torch Tweak

Torch Tweak provides compute graph optimizations for PyTorch models at the `nn.Module` level. Use Torch Tweak when you want automated inference optimization with minimal code changes.

If your model is supported by a dedicated serving framework and benefits from runtime optimizations (e.g. continuous batching, speculative decoding),
use frameworks like vLLM or SGLang for best performance. Use Torch Tweak for general PyTorch models and pipelines that lack such specialized tooling.

## Backends

```python
from torch_tweak.torch.backend import (
    TorchAOBackend,
    TorchInductorBackend,
    XPUGraphBackend,
    OpenVINOBackend,
    TorchEagerBackend,
)
from torch_tweak.torch.tune_strategy import HighestThroughputStrategy

# The strategy measures every backend and keeps the fastest.
strategy = HighestThroughputStrategy(
    backends=[
        TorchAOBackend(),
        TorchInductorBackend(),
        XPUGraphBackend(),
        OpenVINOBackend(),
        TorchEagerBackend(),
    ]
)
```

TorchAO, Torch Inductor, XPU graph, and OpenVINO configuration is in the [backends guide](docs/learn/backends.md).
Torch Eager accepts an optional autocast dtype and otherwise runs the module unchanged.
Tune strategies are in [core functionalities](docs/learn/core_functionalities.md).

## Install

PyTorch XPU wheels are not on PyPI. Stable is the default stack. Nightly is uv-only.
The full rules for switching stacks are in the [install guide](docs/learn/install.md).

```bash
# uv, stable (recommended)
uv sync --extra dev

# pip, editable install with the XPU extra
pip install \
    --index-url https://download.pytorch.org/whl/xpu \
    --extra-index-url https://pypi.org/simple \
    -e ".[xpu,dev]"
```

## Guides

* [Overview](docs/learn/overview.md)
* [Install](docs/learn/install.md)
* [Quick start](docs/learn/quick_start.md)
* [Core functionalities](docs/learn/core_functionalities.md): inspect, tune, save, load, and tune strategies
* [Backends](docs/learn/backends.md)
* [Examples](examples/README.md)

## Useful Links

* [Changelog](ChangeLog.md)
* [Contributing](CONTRIBUTING.md)
* [License](LICENSE)
* [Documentation](https://intel.github.io/torch-tweak/)
* [GitHub Issues](https://github.com/intel/torch-tweak/issues)

## General Misuse Statement

Intel is committed to respecting human rights and avoiding complicity in human rights abuses. See Intel's [Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Intel's products and software are intended only to be used in applications that do not cause or contribute to a violation of an internationally recognized human right.

Please be aware that using this toolkit for very large model optimizations (greater than 10^23 FLOPs) may require additional compliance obligations under the EU AI Act (https://ec.europa.eu/newsroom/dae/redirection/document/118340), depending on usecase and setting.

## Acknowledgments

This project is a port of [NVIDIA's AITune library](https://github.com/ai-dynamo/aitune),
adapted for Intel hardware. Most of the core logic still belongs to the original authors.
Huge credit to the NVIDIA team for their incredible work.
