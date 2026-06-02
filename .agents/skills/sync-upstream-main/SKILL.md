---
name: sync-upstream-main
description: Sync upstream/main into backend/uipc. Use when asked to update the backend/uipc branch from upstream main, fetch and merge upstream changes, or reconcile merge results without auto-resolving conflicts.
---

# Sync Upstream Main

## Overview

Sync `upstream/main` into `backend/uipc` with a conservative merge flow. Preserve the user's control over conflicts: never auto-resolve, never discard local changes, and stop as soon as manual intervention is needed.

## Workflow

1. Confirm the repo is on `backend/uipc`.
2. Require a clean worktree before merging.
3. Fetch `upstream/main`.
4. Merge `upstream/main` into `backend/uipc` with `git merge --no-edit`.
5. If the merge succeeds, report the result and the project impact (see below).
6. If Git reports conflicts:
   - stop immediately
   - do not stage, edit, or resolve conflicted files
   - list the conflicted paths
   - hand the decision to the user

## Impact analysis (clean merges)

A clean merge means git found no textual conflict -- it does **not** mean the
upstream changes are safe for this fork. The fork's value lives in the UIPC
backend (`newton/_src/solvers/uipc/`), which imports from upstream modules. The
helper script prints an impact report after a successful merge so silent
semantic drift surfaces before the result is pushed. Relay it and flag the
two high-risk categories:

- **Files changed on both sides** -- git auto-merged these, so they are the
  most likely place for upstream and local logic to disagree without a
  conflict marker. Review each one.
- **Upstream changes on the UIPC-backend interface surface** -- edits under
  `newton/_src/sim/`, `newton/_src/geometry/`, `newton/_src/solvers/{solver,flags,__init__}.py`,
  `newton/_src/utils/import_usd.py`, or the public `newton/{solvers,geometry,__init__}.py`
  modules. These are what the UIPC backend calls into; an upstream signature
  or behavior change here can break the integration even with a clean merge.

Inspect a flagged file with `git show <upstream_sha> -- <file>`, and build/run
the UIPC backend or its examples before pushing when the interface surface
moved. If nothing landed in either category, say so plainly -- the sync is low
risk.

## Commands

Use the helper script when you want the exact flow encoded. On a clean merge it
also prints the impact report described above; on conflicts it stops without
resolving.

```bash
.agents/skills/sync-upstream-main/scripts/sync_upstream_main.sh
```

Arguments:
- `upstream_ref` defaults to `upstream/main`
- `target_branch` defaults to `backend/uipc`

Exit codes: `0` clean (or already up to date), `1` precondition failed
(wrong branch / dirty worktree), `2` merge conflict awaiting the user.

## Conflict handling

When merge conflicts appear, treat them as a user decision point:

- report the conflicted files
- explain that the merge is paused
- ask the user how to proceed

Do not attempt automatic conflict resolution in this skill.
