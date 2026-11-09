<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Agent notes

A large change alters what the code does: a result, an error, saved data, file or device access, or how modules call each other.
A bug fix is large.
Judge the behavior, not the line count.
When unsure, ask what could break outside the lines you edited.
A rename, formatting, a comment, a docstring, a test-only change, or a refactor that keeps behavior is not large.
When you implement a large change, and when you review one, follow `.cursor/skills/upstream-compare/SKILL.md`.

## After you change code

Before you treat the work as done, review the diff you just wrote.
Base that review on the five questions in `.cursor/skills/python-review/SKILL.md`.
For each change, answer those questions and apply the checklist in that same file.
When the change is large, also follow `.cursor/skills/upstream-compare/SKILL.md`.
Apply the fixes. Then read the diff again.
Stop when a second pass finds nothing new.

Editing tests also follows `.cursor/rules/unit-tests.mdc`.
