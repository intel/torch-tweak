<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes

## After you change code

Before you treat the work as done, review the diff you just wrote.
Apply the fixes. Then read the diff again.
Stop when a second pass finds nothing new.

For each change, answer four questions:

1. **Correct.** The behavior matches the task, including aliases, errors, and edge values.
2. **Necessary.** Every new name, branch, and helper earns its place. Delete the rest.
3. **Idiomatic.** For Python, follow `.cursor/skills/python-idioms/SKILL.md`.
4. **Simpler.** If a smaller form keeps the same results, keeps the same cost on the hot path, and reads more clearly, use it.

The checklist and what to skip because Ruff already checks it are in
`.cursor/skills/python-review/SKILL.md`.

Editing tests also follows `.cursor/rules/unit-tests.mdc`.
