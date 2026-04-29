#!/usr/bin/env python3
"""Run ``convmerge fetch`` using ``config/convmerge_manifest.yaml`` (notebook download.ipynb equivalent)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    try:
        from convmerge.fetch import load_manifest, run_manifest
    except ImportError:
        print(
            "Missing package: pip install 'convmerge[preset,fetch-all,parquet]'",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--manifest",
        type=Path,
        default=None,
        help="Path to convmerge manifest YAML (default: config/convmerge_manifest.yaml)",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=None,
        help="Override manifest output_root (default: manifest value under repo root)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Fetch only these dataset `name` values",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = repo / "config" / "convmerge_manifest.yaml"
    else:
        manifest_path = manifest_path if manifest_path.is_absolute() else (repo / manifest_path).resolve()
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_manifest(manifest_path)
    out = args.output_root
    if out is not None:
        out = out if out.is_absolute() else (repo / out).resolve()
    else:
        raw = manifest.defaults.output_root
        p = Path(raw)
        out = p if p.is_absolute() else (repo / p).resolve()

    result = run_manifest(manifest, output_root=out, only=args.only)
    print(
        f"succeeded={len(result.succeeded)} skipped={len(result.skipped)} failed={len(result.failed)}"
    )
    if result.failed:
        print("failed:", [n for n, _ in result.failed], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
