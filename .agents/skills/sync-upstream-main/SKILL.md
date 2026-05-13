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
5. If the merge succeeds, report the result and any changed files.
6. If Git reports conflicts:
   - stop immediately
   - do not stage, edit, or resolve conflicted files
   - list the conflicted paths
   - hand the decision to the user

## Commands

Use the helper script when you want the exact flow encoded:

```bash
.agents/skills/sync-upstream-main/scripts/sync_upstream_main.sh
```

Arguments:
- `upstream_ref` defaults to `upstream/main`
- `target_branch` defaults to `backend/uipc`

## Conflict handling

When merge conflicts appear, treat them as a user decision point:

- report the conflicted files
- explain that the merge is paused
- ask the user how to proceed

Do not attempt automatic conflict resolution in this skill.
