#!/usr/bin/env bash
set -euo pipefail

upstream_ref="${1:-upstream/main}"
target_branch="${2:-backend/uipc}"

# Paths the UIPC backend (newton/_src/solvers/uipc/) imports from upstream.
# Upstream edits here are the most likely to break the UIPC integration even
# when git reports no textual conflict.
IFACE_RE='^newton/_src/sim/|^newton/_src/geometry/|^newton/_src/solvers/(solver|flags|__init__)\.py$|^newton/_src/utils/import_usd\.py$|^newton/(solvers|geometry|__init__)\.py$'

report_impact() {
  local pre="$1" up="$2"
  local base base_short pre_short up_short up_files local_files both iface

  base="$(git merge-base "$pre" "$up")"
  base_short="$(git rev-parse --short "$base")"
  pre_short="$(git rev-parse --short "$pre")"
  up_short="$(git rev-parse --short "$up")"

  echo ""
  echo "=== Sync impact analysis ==="
  echo "Merge base:        $base_short"
  echo "Local (pre-merge): $pre_short"
  echo "Upstream merged:   $up_short"
  echo ""
  echo "Upstream diffstat (vs merge base):"
  git diff --shortstat "$base" "$up" | sed 's/^ */  /'
  echo ""

  up_files="$(git diff --name-only "$base" "$up")"
  local_files="$(git diff --name-only "$base" "$pre")"

  # Files both upstream and this fork changed since the merge base: git
  # auto-merged them, so they carry the highest risk of silent semantic drift.
  both="$(comm -12 <(printf '%s\n' "$up_files" | sort) <(printf '%s\n' "$local_files" | sort) || true)"

  echo "[!] Files changed on BOTH sides (git auto-merged -- review for semantic drift):"
  if [[ -n "$both" ]]; then
    while IFS= read -r f; do
      [[ -z "$f" ]] && continue
      if printf '%s' "$f" | grep -qE "$IFACE_RE"; then
        echo "  $f   <-- UIPC interface"
      else
        echo "  $f"
      fi
    done <<< "$both"
  else
    echo "  (none)"
  fi
  echo ""

  # Upstream changes landing on the UIPC-backend interface surface, whether or
  # not this fork also touched them.
  iface="$(printf '%s\n' "$up_files" | grep -E "$IFACE_RE" || true)"
  echo "[*] Upstream changes touching the UIPC-backend interface surface:"
  if [[ -n "$iface" ]]; then
    printf '%s\n' "$iface" | sed 's/^/  /'
  else
    echo "  (none)"
  fi
  echo ""

  echo "Review tips:"
  echo "  - Inspect a UIPC-interface change: git show $up_short -- <file>"
  echo "  - Build and run the UIPC backend/examples before pushing."
}

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

pre_merge_head="$(git rev-parse HEAD)"
git fetch upstream main
upstream_sha="$(git rev-parse "$upstream_ref")"

if git merge --no-edit "$upstream_ref"; then
  if [[ "$(git rev-parse HEAD)" == "$pre_merge_head" ]]; then
    echo "Already up to date with $upstream_ref. No changes merged."
    exit 0
  fi
  echo "Merge completed cleanly."
  report_impact "$pre_merge_head" "$upstream_sha"
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
