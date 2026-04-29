"""Fetch Label Studio annotations into JSONL (delegates to export flow)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """CLI mirror of ``python -m pipeline.flows.export`` (see PIPELINE.md)."""
    p = argparse.ArgumentParser(
        description="Export Label Studio JSON to messages JSONL (wrapper around export flow).",
    )
    p.add_argument("-i", "--input", type=Path, required=True, help="Label Studio export.json")
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for messages.jsonl",
    )
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from pipeline.flows.export import export_flow

    export_flow(args.input, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
