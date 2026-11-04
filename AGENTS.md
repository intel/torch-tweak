<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes

## Upstream AITune

Before you implement, fix, or improve a behavior, read the current
[AITune](https://github.com/ai-dynamo/aitune) checkout.
See whether that feature or fix exists there, and how it is written.
Compare it with the approach you would take.
Keep the one that is right for this repository.

This repository does not merge with AITune.
It does not keep AITune's API.
When an upstream idea fits, rewrite it so the names, the structure, and
the call shapes match this code.

Use a local clone. Do not download a second one.
Look in this order:

1. The path a machine rule already names.
2. `$HOME/src/aitune`, when `origin` is `https://github.com/ai-dynamo/aitune.git`.
3. The absolute path on the first line of `tmp/aitune-checkout.txt`.

Confirm `origin` before you trust the tree.
If none of those checkouts exist, clone that repository once into a
directory git does not track, and write the absolute path into
`tmp/aitune-checkout.txt`.
Fetch when you need the current upstream.
Fast-forward a clean checkout.
Leave its branch as it is when the update is not a fast-forward.

## After you change code

Before you treat the work as done, review the diff you just wrote.
Apply the fixes. Then read the diff again.
Stop when a second pass finds nothing new.

For each change, answer five questions:

1. **Correct.** The behavior matches the task, including aliases, errors, and edge values.
2. **Necessary.** Every new name, branch, and helper earns its place. Delete the rest.
3. **Duplicated.** Search this repository for the same job.
   Call the function that already does it.
   If two places you wrote carry the same logic, and one helper would be shorter and easier to read, make that helper.
   Leave a single use where it is.
4. **Idiomatic.** For Python, follow `.cursor/skills/python-idioms/SKILL.md`.
5. **Simpler.** If a smaller form keeps the same results, keeps the same cost on the hot path, and reads more clearly, use it.

The checklist and what to skip because Ruff already checks it are in
`.cursor/skills/python-review/SKILL.md`.

Editing tests also follows `.cursor/rules/unit-tests.mdc`.
