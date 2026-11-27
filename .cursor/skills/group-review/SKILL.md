<!--
Copyright (c) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->
---
name: group-review
description: >-
  Run a group review: the main agent codes and fixes, and several reviewer
  subagents on different models review the whole change in rounds until no
  one finds anything new. Use when the user asks for a group review
  ("grupowe review"). The main agent follows this skill; reviewers follow
  python-review only.
---

# Group review

One agent writes the code.
Several agents on other models review it.
They argue until the code is right.

## Roles

**Main agent.**
Runs on the strongest model available, for example Claude Opus 5.5.
It is the only agent that edits files, commits, or pushes.
It follows `python-review` for its own diff, and this skill for the rounds.
It runs the expensive checks: the full unit suite, the docs build, coverage.

**Reviewers.**
Three subagents by default, for example GPT-5.6 Sol, Grok 4.7, and Claude Fable 5.2.
Every agent in the group runs on a different model, the main agent included.
Pick the latest version of each model family.
Reviewers follow the "Review for someone else" section of `python-review`, and `upstream-compare` when the change is large.
They read and run code. They never edit the repository.
They run only the tests their finding needs. They do not run the full suite.

The user may name other models or another number of reviewers. Their choice wins.

## Rounds

1. **Collect.** Gather every finding: earlier reviews when the user points to them, then the reviewers' reports.
2. **Judge.** For each finding, read the code it names and decide.
   - Agree: fix it, and add a test that fails without the fix. Revert the fix once to see the test fail.
   - Disagree: write the reason. Evidence beats argument: a reproduction, the line that already handles it, or the same behavior on `main`.
   - Out of scope: behavior that `main` already has belongs in its own pull request. Say which.
3. **Self-review.** Read the whole diff with `python-review`. Fix. Read it again. Stop when a pass finds nothing new.
4. **Test.** Run the targeted tests, then the full unit suite once per round. Run the docs build when docs changed.
5. **Ask.** Send every reviewer the decisions and ask for a review of the whole change, not only the delta.
6. **Repeat** from step 1.

Stop when, in one round, every reviewer reports no new finding and the main agent's self-review finds nothing.

Start the reviewers once, in the first round.
In every later round, resume the same reviewers so they keep their context.
Do not start new ones.

Wait until every reviewer of the round has reported before you edit code.
A reviewer that reads a tree that changes under it reports on a moving target.

## Disagreement

When the main agent rejects a finding, it tells that reviewer why.
When a reviewer pushes back, the main agent checks the new argument against the code.
Look for a precedent in the repository: the same problem solved elsewhere decides most disputes.
Change the decision when the argument holds. Say so plainly.
When two exchanges do not settle it, ask the user. Give both positions in two sentences each.

## The reviewer prompt

A reviewer does not see the conversation. The first prompt carries everything:

- The repository path, the branch, and the pull request link.
- What to review: `git diff origin/main`, including uncommitted changes.
- The rules to read: `AGENTS.md`, `python-review`, `python-idioms`, `upstream-compare`, `.cursor/rules/unit-tests.mdc`, `.cursor/rules/local-env.mdc`.
- What the change does, in a few sentences.
- Every decision so far: fixed, rejected with the reason, out of scope.
  A reviewer raises a decided point again only with new evidence.
- The constraints:
  - Do not edit, create, or delete files in the repository. Do not commit, stash, check out, or push.
  - Put scratch files under a private directory in `/tmp`.
  - Set an environment variable such as `TORCH_TWEAK_CACHE_DIR` only inline, for one command. Other agents run tests at the same time.
  - Do not run the full unit suite. Keep GPU experiments short.
- The answer format: findings, most severe first, each with severity, `file:line`, what a user sees, a reproduction or exact reasoning, and the proposed fix.
  With nothing to report, the answer is exactly `No new findings.`

A later round sends only what changed: the decision on each of that reviewer's findings, the fixes made for the others, and the request to review the whole change again.

## Report to the user

After each round, tell the user in a few lines:

- the round number and which reviewers reported,
- what was fixed, and the test for each fix,
- what was rejected, and why,
- the state of the full test suite.

Push or update a pull request only when the user asks.
