<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
# Walkthroughs
This directory contains several notebooks that walk through the use of the Torch Tweak package.

In order to run each notebook, you need to have *jupyter* installed and run the following command:
```bash
jupyter notebook
```

Then, open the notebook you want to run by opening the notebook file in your browser.

## Policy

These notebooks are developer walkthroughs. They are not published in the docs site and CI does not execute them.

- Do not commit cell outputs. The `nbstripout` pre-commit hook removes them, including local paths that show up in tracebacks and cache directories.
- Put the copyright header in the first Markdown cell. The license check reads that cell when a notebook is added or edited.
- Ruff does not lint `*.ipynb` files. Notebooks use IPython magics, which are not Python modules.

## 1. Inspect module demo
This notebook demonstrates how to inspect modules in a pipeline, pick modules for tuning based on gathered data, and tune them.

## 2. SampleMetadata Walkthrough
This notebook provides a comprehensive tutorial on the `SampleMetadata` class, which is a core component of the Torch Tweak library for tracking and managing tensor shapes in PyTorch models.

## 3. Tunable Module Demo
This notebook demonstrates how to wrap a module in `TunableModule` and tune it.

## 4. Storage Demo
This notebook demonstrates how to use the `Storage` class to save and load checkpoints.

## 5. Saving and Loading Checkpoints
This notebook demonstrates how to save and load checkpoints.
