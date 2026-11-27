<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
---
name: test-validation
description: >-
  Selects and runs proportional validation for Torch Tweak changes. Use after
  implementing or reviewing behavior changes, when deciding which unit,
  functional, nightly, or CI tests to run, and when reporting validation.
---

# Validate a change

Select tests by the behavior and boundary that changed, not only by changed file names.
Use the environment and commands from `.cursor/rules/local-env.mdc`.

## Select tests

Use this ladder in order:

1. Run the exact regression test for each changed behavior.
2. Run the test files found by searching for the behavior, changed symbol, owning class, and module.
3. Run the affected unit package when shared production code has callers that the selected files do not cover.
4. Run focused functional tests when the change crosses a real boundary such as export, compilation, device
   execution, checkpoint loading, serialization, or an external library.
5. Run the full unit suite once when the change has broad callers or no smaller selection covers them.
6. Run the full functional suite once when several functional paths can change, selection remains uncertain,
   or the corresponding CI job requires it before a final verdict.
7. Run nightly tests only when the change reaches behavior that those tests alone cover.

Stop after the narrowest level that covers every changed contract.
Prefer a file or node ID selected by behavior search over a `-k` expression.
A test name does not prove which behavior the test exercises.

## Bound the cost

Do not rerun a wider green level after an amendment unless the amendment can affect code that level exercises.
Estimate unusually expensive validation before starting it.
Run accelerator tests serially unless each process has its own device.

If a needed run is impractical on the available machine, name the command, why it is needed, and why it was not run.
Do not silently replace it with tests that cannot observe the behavior.

## Judge the result

A passing unit suite does not replace a functional test when the defect exists only across a real boundary.
A pending CI job proves nothing until it completes.
Completed CI can supply a wide level; report the level still missing while it is pending.

Report the commands that ran, their results, and relevant tests that did not run, with the reason.
