#!/usr/bin/env bash
set -euo pipefail

upstream_ref="${1:-upstream/main}"
target_branch="${2:-backend/uipc}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository." >&2
  exit 1
fi

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$target_branch" ]]; then
  echo "Switch to $target_branch first. Current branch: ${current_branch:-detached}" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Worktree is dirty. Stop and ask the user whether to stash or commit first." >&2
  git status --short --branch >&2
  exit 1
fi

git fetch upstream main

if git merge --no-edit "$upstream_ref"; then
  echo "Merge completed cleanly." 
  exit 0
fi

conflicts="$(git diff --name-only --diff-filter=U || true)"
echo "Merge conflict detected. Do not resolve automatically." >&2
if [[ -n "$conflicts" ]]; then
  echo "Conflicted files:" >&2
  printf '%s\n' "$conflicts" >&2
fi
echo "Hand the conflict resolution decision to the user." >&2
exit 2
