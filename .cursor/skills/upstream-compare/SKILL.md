<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
---
name: upstream-compare
description: >-
  Compare a large Torch Tweak change with the current AITune checkout and
  build the stronger solution for this repository. Use when implementing or
  reviewing a change that alters behavior, including a bug fix. Skip a
  rename, formatting, a comment, a docstring, a test-only change, or a
  refactor that keeps behavior.
---

# Compare with AITune

Use this when you implement a large change and when you review one.
`AGENTS.md` and the review skill say which changes are large.

Read the current [AITune](https://github.com/ai-dynamo/aitune) checkout.
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
