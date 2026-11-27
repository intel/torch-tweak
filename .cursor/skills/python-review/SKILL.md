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
  restate the code or describe removed code. Finds observable contracts and
  the tests and callers that depend on them.
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

1. Read the complete diff for the change. After an amendment, do not inspect only the latest delta.
2. Follow [Behavior contracts](#behavior-contracts). Account for every removed or replaced behavior.
3. Take each change through the five questions below. When the change is large, also follow `.cursor/skills/upstream-compare/SKILL.md`.
4. Edit the code. A review note left in a comment is not a fix.
5. Run the proportional validation ladder in `.cursor/rules/unit-tests.mdc`.
6. When the change alters behavior, read the CI diff-cover report and cover the lines it names.
7. Read the diff, affected documentation, and pull request description again.
8. Stop when the new pass finds nothing the previous pass missed.

A simpler form keeps the same results and the same cost on the hot path,
and it reads more clearly. It is not simpler if it removes a check, a
narrow `except`, or the `*` that forces a keyword argument.

## Review for someone else

Same five questions. When the change is large, also follow `.cursor/skills/upstream-compare/SKILL.md`.
Follow [Behavior contracts](#behavior-contracts) and the proportional validation ladder in
`.cursor/rules/unit-tests.mdc`.
Report. Edit only if the user asked you to change the code.

Start with the bug that would reach users.
One bullet per finding: the file, what goes wrong, and what to change.
Skip a remark that would not change behavior or a decision.
When the change alters behavior, name the changed lines the CI diff-cover report leaves uncovered.
End with the tests that ran and the relevant tests that did not run, with the reason.
Treat a no-findings verdict as provisional while a required validation tier is pending.
If the author is about to open a pull request, end with the behavior
questions a reviewer will ask. Those belong in the pull request description.

## Behavior contracts

A diff can break a promise that none of its changed lines states.
Find the promise, then find who relies on it.

1. List each observable fact the change alters: a result, error type or message, default, order, name,
   saved key, file, device access, dtype, shape, or how modules call each other.
2. For every removed or replaced branch, helper, validation, and test assertion, state the behavior it provided.
   Deleting an assertion does not prove that its contract should disappear.
3. Search the whole repository for readers of that behavior. Search for the changed symbol, its class and module,
   the attribute a caller reads, and words that describe the behavior. Include unchanged unit and functional tests,
   examples, and documentation.
4. Trace the changed value from the public input through normalization, export or storage, and execution to the
   user-visible result. A test can observe a contract without naming the helper that implemented it.
5. Map every changed contract to a test. Decide whether a failing test protects public behavior or only an
   implementation detail before changing either the code or the test.

Do not carry an architectural assumption from an earlier review round.
After an amendment, repeat this inventory against the complete change.

Do not use a `-k` expression as the only way to select relevant tests.
It matches test names, not the behavior they exercise.
Use the files and node IDs found by the contract search.

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
- A variable used once, whose value is a short, simple expression. Inline it at its one use.
  Keep the variable when its name says what the expression cannot, or when inlining breaks the line.
- A `found` flag where a `return` or `break` in the loop, plus `else` for the miss, already says the same thing.
- A wrapper that only calls one other function.
- A comment that repeats the next line.
- A comment about deleted code, the previous behavior, or why an older version differed.
- Code that loads a `.tt` checkpoint of an older format: a default for a missing key, a migration, a second reader.
  See [Checkpoint format](#checkpoint-format).
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

## Checkpoint format

Torch Tweak never loads a `.tt` checkpoint of an older format.
Simple loading code is worth more than an old checkpoint. The user tunes the model again.
Do not add a default for a missing key, a migration, or a second reader.

A change can break loading between the last release and this code.
Examples are a new field in a saved backend config, a renamed key, or a removed key.
Then compare `TT_FORMAT_VERSION` in `torch_tweak/torch/checkpoint/checkpoint_header.py` with the last release.
If the release has the same version, bump it in this change.
If this code already has a higher version, an unreleased change bumped it. Leave it.
A release without `checkpoint_header.py` wrote no format version, and `read_header` already rejects its checkpoints.

## Leave these to Ruff

Do not comment on them. The `select` list in `pyproject.toml` already
includes layout, imports, naming, pydocstyle, `print`, debugger calls,
logging f-strings, bugbear, pyupgrade, comprehensions, copyright headers,
and McCabe complexity. Quotes are double. The line length is 120.
Pre-commit runs Ruff. A review that restates Ruff is noise.
