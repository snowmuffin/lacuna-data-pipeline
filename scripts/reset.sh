#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "This will remove generated files under data/raw, data/refined, data/labeled, data/export."
read -r -p "Continue? [y/N] " ans
case "${ans:-N}" in
  y|Y) ;;
  *) echo "Aborted."; exit 1 ;;
esac

find "$ROOT/data/raw" "$ROOT/data/refined" "$ROOT/data/labeled" "$ROOT/data/export" \
  -mindepth 1 -delete 2>/dev/null || true
touch "$ROOT/data/raw/.gitkeep" "$ROOT/data/refined/.gitkeep" \
  "$ROOT/data/labeled/.gitkeep" "$ROOT/data/export/.gitkeep"
echo "Reset complete."
