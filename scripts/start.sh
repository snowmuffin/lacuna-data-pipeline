#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# TODO: start Prefect server / agent and Label Studio (e.g. docker compose or label-studio CLI).
echo "Stub: start Prefect + Label Studio from $ROOT"
