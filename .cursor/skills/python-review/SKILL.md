<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
---
name: python-review
description: >-
  Review Python diffs in Torch Tweak for correctness, necessity, duplicated
  logic, idiom, and a simpler form, then apply the fixes on a self-review.
  Use after writing
  Python, before treating the work as done, and when reviewing a change or
  pull request. Skips anything Ruff already enforces. Flags comments that
  restate the code or describe removed code.
---

# Review Python in Torch Tweak

Judge the diff. Idiom details live in [python-idioms](../python-idioms/SKILL.md).
Test style lives in `.cursor/rules/unit-tests.mdc`.
Do not repeat either one in the review.

A large change alters what the code does: a result, an error, saved data, file or device access, or how modules call each other.
A bug fix is large.
Judge the behavior, not the line count.
When unsure, ask what could break outside the lines you edited.
A rename, formatting, a comment, a docstring, a test-only change, or a refactor that keeps behavior is not large.
For a large change, also follow `.cursor/skills/upstream-compare/SKILL.md`.

## Self-review

Run this before you treat your own change as done.

1. Read the diff you just wrote.
2. Take each change through the five questions below. When the change is large, also follow `.cursor/skills/upstream-compare/SKILL.md`.
3. Edit the code. A review note left in a comment is not a fix.
4. When the change alters behavior, read the CI diff-cover report and cover the lines it names.
5. Read the diff again.
6. Stop when the new pass finds nothing the previous pass missed.

A simpler form keeps the same results and the same cost on the hot path,
and it reads more clearly. It is not simpler if it removes a check, a
narrow `except`, or the `*` that forces a keyword argument.

## Review for someone else

Same five questions. When the change is large, also follow `.cursor/skills/upstream-compare/SKILL.md`.
Report. Edit only if the user asked you to change the code.

Start with the bug that would reach users.
One bullet per finding: the file, what goes wrong, and what to change.
Skip a remark that would not change behavior or a decision.
When the change alters behavior, name the changed lines the CI diff-cover report leaves uncovered.
If the author is about to open a pull request, end with the behavior
questions a reviewer will ask. Those belong in the pull request description.

## The five questions

### Correct

- The caller still holds a list, dict, module, or class-level container that this code mutates.
- `None` means both "argument omitted" and a real value, and there is no sentinel.
- A tensor or ndarray used in `if`, `and`, `or`, or compared with `==` as a condition.
- A write through a view while another name still needs the old values. A `.clone()` in timed code when the copy is not what is being measured.
- A generator or iterator stored and then looped over twice.
- A function inside a loop that reads the loop variable and therefore sees its final value.
- `setdefault` called with a default expression that is expensive to build. A public `**kwargs` that ignores unknown names.
- An `except` wider than the error, a subclass handler placed after its base class, or `raise NewError` that drops the original error.
- `for`/`else` or `try`/`else` that does not follow the rule in the idioms skill.
- Branches that return different kinds of values.
- Device access, a download, compilation, or a lock while the module is imported.
- Cleanup outside `finally`, or a resource released in `__del__`.
- A `dict`, `list`, or `tuple` subclass that overrides a method a C implementation will not call.
- A key that is mutated after insertion, or a tensor used as a key.
- `is` used to compare values, or `==` used for `None` or a sentinel.
- A secret, token, or credential in the diff.

### Necessary

- A helper used once, whose name does not make the call easier to read.
- A `found` flag where a `return` or `break` in the loop, plus `else` for the miss, already says the same thing.
- A wrapper that only calls one other function.
- A comment that repeats the next line.
- A comment about deleted code, the previous behavior, or why an older version differed.
- A keyword argument that only repeats the default, where seeing it does not help the reader.
- A `_name` read from outside the class or module that owns it.
- A class hierarchy standing in for one function. A decorator with a single call site.
- Threads, `asyncio`, a cache, or `slots` on a path that is not hot.

### Duplicated

- A new function that repeats a job an existing function in this repository already does. Call that function.
- The same logic at two call sites, when one helper would be shorter and easier to read.

### Idiomatic

Check the change against `python-idioms`, including the comments section.
Match the surrounding module: dataclass records, keyword-only options,
`pathlib`, an encoding on `open`, and logger arguments.
A public function that returns several values uses a named type.
A strategy that stores no fields is a function.

### Simpler

Fewer names and less nesting, with the same behavior and the same cost.
Prefer that edit over a new abstraction.

### Leave these alone

- A `for`/`else` search that returns the match from the loop. The `else` is the miss.
- A result that already has one shape, including a stand-in string for a miss. Do not invent an exception for it.
- A broken idiom where following it would make that code clearly worse.
- Anything in the Ruff list below.

## Leave these to Ruff

Do not comment on them. The `select` list in `pyproject.toml` already
includes layout, imports, naming, pydocstyle, `print`, debugger calls,
logging f-strings, bugbear, pyupgrade, comprehensions, copyright headers,
and McCabe complexity. Quotes are double. The line length is 120.
Pre-commit runs Ruff. A review that restates Ruff is noise.
