<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes

## Upstream AITune

Read the current [AITune](https://github.com/ai-dynamo/aitune) checkout
before a large change.
A large change adds behavior, fixes a bug that may already be fixed
there, or edits a module ported from AITune.
Skip the lookup for a rename, a comment, a docstring, or a change that
does not alter behavior.

See whether that feature or fix exists there, and how it is written.
Treat that implementation and your own analysis as two inputs.
Weigh the approach, the corner cases, and the holes in each.
Build the strongest solution for this repository from those parts.
Their approach can supply one case that improves yours.
Yours can close a hole in theirs.
Name things the way this repository names them.
Put the code where this repository puts that kind of code.
Call it the way callers here already call this kind of function.

Do not paste AITune source into this repository.
This repository does not merge with AITune.
It does not keep AITune's API.

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

For each change, answer the five questions in `.cursor/skills/python-review/SKILL.md` and apply the checklist in that same file.

Editing tests also follows `.cursor/rules/unit-tests.mdc`.
