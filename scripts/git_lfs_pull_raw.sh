#!/usr/bin/env bash
# Run `git lfs pull` in every git clone under data/raw (default) or the path you pass.
set -euo pipefail
ROOT="${1:-./data/raw}"
if [[ ! -d "$ROOT" ]]; then
  echo "Directory not found: $ROOT" >&2
  exit 1
fi
shopt -s nullglob
for d in "$ROOT"/*; do
  [[ -d "$d/.git" ]] || continue
  echo "==> $d"
  git -C "$d" lfs pull
done
