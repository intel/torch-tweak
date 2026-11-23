<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
---
name: pr-description
description: >-
  Write the title and description of a Torch Tweak pull request so a reviewer
  understands the change without reading the diff first, and a maintainer can
  still follow it a year later. Use when opening a pull request, drafting or
  revising a PR title or description, or summarizing a branch for review.
---

# Write a pull request description

A description answers two questions: what changes, and why.
The diff shows how.

Two people read it.
The reviewer reads it today, before the diff, with none of your context.
A maintainer reads it later, after `git blame` or a search led them to this change.
They remember the topic, not the details.
They need to know why the code is the way it is before they dare to change it.

Write for the second reader too.
A description that only makes sense during this week's review is not finished.

## How it reaches history

This repository merges by squash only.
The squash commit takes its title from the pull request, or from the only commit when there is one.
Its body is the commit messages.
The description body stays on the pull request page.

- The title follows `COMMIT_CONVENTION.md`. It becomes the commit title on `main`.
  With a single commit, keep the commit title and the pull request title the same.
- A fact the maintainer needs later goes into a commit message as well as the description.
  Wrap commit body lines at 72 characters. The `commit-msg` hook rejects longer lines.
- The description may add what only matters for review: where to look, what to skip, what is still open.

## Workflow

1. Read every commit on the branch against `main`, not only the last one.
   Read the full diff, not your memory of it. That is where an accidental change shows up.
2. Collect the facts: the problem, the cause, the fix, what you ran, and what you did not do.
3. Write the title, then the opening sentence, then the body.
4. Check the draft against the checklist at the end.
5. Re-read it before merge. Review rounds change the code. The description must match the final diff.

## Title

Use `<type>(<scope>): <short description>` from `COMMIT_CONVENTION.md`.
Lowercase, imperative, no trailing period. Aim for 50 characters or fewer.
Use up to 72 only when a shorter title cannot say what the change does.

The title alone tells a reader of `git log --oneline` what this change did.
It must be specific enough to tell this change apart from its neighbors.
A good test: it completes the sentence "Merging this will ...".

```
Bad:  fix(openvino): fix bug
Bad:  fix(openvino): bf16 weights
Bad:  fix(openvino): fixed garbage weights          (past tense)
Good: fix(openvino): copy bf16 and f8 weights of XPU modules into the IR
```

If the title needs "and" to cover the change, the pull request probably does two things.
Consider splitting it.

## Opening sentence

The first line of the body states the result in one sentence.
A reader who stops there knows what is different for a user of Torch Tweak.
Lead with the effect, not with the story of how you found it.

```
Bad:  While profiling SD1.5 I noticed that some outputs looked strange, so I dug into the OpenVINO backend.
Good: After this change, the OpenVINO backend feeds an input that compilation gave an extra name.
```

## Anchor every claim in time

A pull request describes a difference between two states of the code.
Every sentence about behavior must say which state it describes.

Words like "now", "currently", "at the moment", "the new behavior", and "the old code" have no fixed reference.
In review they are ambiguous: "the backend now copies the weight" may describe the code before your change or after it.
In history they are wrong: a year later, "now" means the reader's present, not the merge date.

Name the state explicitly:

- "Before this change, ..." and "After this change, ..."
- "On `main`, ..." and "With this change, ..."
- An imperative verb for what the change does: "Copy the data", "Drop the check".

```
Bad:  Transposed bf16 weights now keep their values.
Bad:  The workaround now shares memory only for contiguous CPU tensors.

Good: Before this change, the workaround covered only weights on XPU.
      A transposed bf16 weight on CPU still ended up as garbage in the IR.
      After this change, a constant shares memory only with a contiguous CPU tensor.
```

When the change extends earlier work, say so in those words: "The workaround needs to handle one more case."

## Body

Write the body in this order.
Skip a part that has nothing to say.

1. **Problem.** What a user saw, or what was wrong. Use the real symptom: the error message, the wrong number, the crash.
2. **Cause.** Why it happened. This is the part a reader cannot recover from the diff.
3. **Fix.** What the change does about it, at the level of behavior. Name functions and classes exactly as the code does.
4. **Decisions.** Choices the code does not show: the alternative you rejected and why, a trade-off you accepted.
5. **Limits.** What this change does not fix, what stays slow, which case is still open. Name the follow-up issue.
6. **Evidence.** What you ran and what it showed. See below.

After the opening sentence and the links, add the section `## Why it matters`, described below.

Be concrete.
"Improves accuracy" says nothing.
"The UNet was off by 1.97 at an output scale of 4.2; after this change the outputs match eager" says what happened.

Name identifiers exactly: `HighestThroughputStrategy`, not "the strategy"; `torch.export`, not "the exporter".
Spell out a term the first time if a newcomer to the module would not know it.
Avoid "this", "it", and "the issue" when the referent is more than one sentence away.

## Why it matters

This section is for a big-picture reader.
The reader knows the goal of the project and its architecture at the level of a diagram.
The reader does not read code.

- Start with the goal of the project in one or two sentences, and the condition it puts on this change.
- List the scenarios this change improves, at most five, most important first.
- Number each scenario and open it with a bold headline that names what a user saw.
- Then say what happened and why it hurt the user, and what happens after this change.
- Use plain words: an image, a model, the card, the disk. No identifiers, no file names, no backend names.
- Close with one or two sentences that compare before and after.
- Claim only what this pull request delivers. Name the pull request that fixes a related scenario.

When the change has no visible effect, write that in one sentence and stop.
A refactor, a test, or an internal cleanup often has none.
Do not invent a scenario to fill the section.

```
## Why it matters

This change has no effect a user can see.
It moves the export code into its own module so #120 can reuse it.
```

## Evidence

State what you actually ran.
Never claim a test, a benchmark, or a device you did not use.

- Say whether a new test fails before the change and passes after it. That is the strongest evidence a fix has.
- Name the device and stack when it matters: CPU or XPU, stable or nightly PyTorch, OpenVINO version.
- Give numbers for a performance or accuracy change: model, batch size, before, after, unit.
- Quote the one line of a log that matters. Do not paste a whole log.
- When there is no test, say why in one sentence. "A test would only assert that a deleted line is gone" is a reason.

Do not add a checklist of routine steps that CI already enforces, such as lint and formatting.
Do not add a test-plan section unless the repository asks for one.

## Guide the reviewer

Tell the reviewer where to spend attention.

- Point to the file or function that carries the change when the diff touches many files.
- Give a review order when commits build on each other.
- Say which part is mechanical and can be skimmed.
- Ask the question you want answered: "Is a process-wide patch acceptable here, or should this wait for the upstream fix?"

## Mechanical and generated changes

When a tool produced part of the diff, name the tool and the command.
The reviewer can then rerun it instead of reading every line.

```
`uv lock` regenerated `uv.lock` after the `openvino` bound changed in `pyproject.toml`.
`ruff format` reformatted the two files under `torch_tweak/torch/backend/openvino/`.
```

Generated text, including a description an agent drafts, follows every rule here.
The author reads it, removes what is false or vague, and adds what only the author knows.

## Links and context

Link the issue, the earlier pull request, and the upstream bug.
Use closing keywords such as `Closes #82` in the description when the merge should close an issue.

A link is not an explanation.
Internal trackers, chat threads, and private documents are not visible to every reader, and links rot.
Write the key fact into the description and keep the link as a reference.

```
Bad:  Fixes the problem from the meeting, see the internal ticket.
Good: OpenVINO frees the temporary CPU copy of a bf16 tensor while the constant still points at it.
      Upstream fix: openvinotoolkit/openvino#39510.
```

Use the relations a reader needs:

- "Split out of #77."
- "Follows #81, which does the same for `Backend.build`."
- "#82 tracks the remaining case."
- "#47 needs a small rebase after this lands."

## Breaking changes

A change that breaks a caller says so in the description and in `ChangeLog.md` with `(breaking)`.
Say what breaks, who is affected, and how to migrate.

```
`inspect` renames `min_depth` to `depth`.
A diffusers pipeline that used `min_depth=1` to reach its components uses `depth=0` after this change.
```

## One change per pull request

A description that needs several unrelated paragraphs describes several pull requests.
A cleanup found along the way, a rename, or a dependency bump goes into its own pull request.
A small pull request gets a better review, and its description stays short.

A refactor that only makes the feature possible can stay in the same pull request.
Put it in its own commit and say so: "The first commit moves `_port_names` out of the class without changing behavior."

## Descriptions that fail

Each of these leaves the reader with a question the author could have answered.

- "Fix bug." Which bug? What changed?
- "Fix tests." Which tests, and was the test wrong or the code?
- "Address review comments." That is a history of the review, not a description of the change.
- "Phase 1." Of what? What does this phase deliver on its own?
- "Refactor backend." Which part, and why now?
- "See ticket." The ticket may not open for the reader.
- A list of touched files. The diff already has it.

## Example

```
fix(openvino): copy bf16 and f8 weights of transposed CPU tensors

The bf16 and f8 workaround needs to handle one more case: a transposed weight on CPU.

Before this change, the workaround covered only weights on XPU.
A transposed bf16 or f8 weight on CPU still ended up as garbage in the OpenVINO IR.

The cause is the same as on XPU.
The OpenVINO frontend first makes the weight contiguous, and that copy is temporary.
The constant points at the copy and does not keep it alive.

After this change, a constant shares memory only with a contiguous CPU tensor.
Every other tensor, on XPU or transposed on CPU, is copied into the constant.

A new CPU test exports a module with a transposed bf16 weight.
It fails before this change and passes after it.
It needs no XPU, so CI covers the workaround on any machine.

The upstream fix is openvinotoolkit/openvino#39510.
Remove the workaround once the minimum `openvino` version includes it.
```

## Checklist

- [ ] The title follows `COMMIT_CONVENTION.md` and says specifically what the change does.
- [ ] The first body sentence states the result for a user.
- [ ] `## Why it matters` explains the change to a big-picture reader, or says in one sentence that it has no visible effect.
- [ ] Every claim about behavior says whether it holds before or after the change. No bare "now".
- [ ] The cause is explained, not only the fix.
- [ ] Decisions and limits the code does not show are written down.
- [ ] Evidence is real: what ran, on which device, with which result.
- [ ] Key facts from links are written into the text.
- [ ] Facts a maintainer needs later are also in a commit message.
- [ ] The description matches the final diff, after review changes.
