"""Collect raw data via convmerge fetch (HF hub snapshots + GitHub trees)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from prefect import flow, get_run_logger

from pipeline.sources_config import REPO_ROOT


@flow(name="collect-convmerge", log_prints=True)
def collect_convmerge_flow(
    manifest_path: str | Path | None = None,
    output_root: str | Path | None = None,
    only: list[str] | None = None,
) -> dict[str, Any]:
    """Run ``convmerge.fetch.run_manifest`` (HF hub + GitHub trees), writing under ``output_root``.

    Uses ``HF_TOKEN`` / ``GITHUB_TOKEN`` from the environment or repo root ``.env``.
    """
    load_dotenv(REPO_ROOT / ".env")
    try:
        from convmerge.fetch import load_manifest, run_manifest
    except ImportError as e:
        raise RuntimeError(
            "convmerge is required. Install: pip install 'convmerge[preset,fetch-all,parquet]'"
        ) from e

    repo = REPO_ROOT
    mp = Path(manifest_path) if manifest_path is not None else repo / "config" / "convmerge_manifest.yaml"
    mp = mp if mp.is_absolute() else (repo / mp).resolve()
    if not mp.is_file():
        raise FileNotFoundError(f"convmerge manifest not found: {mp}")

    manifest = load_manifest(mp)
    if output_root is not None:
        out_p = Path(output_root)
        out_p = out_p if out_p.is_absolute() else (repo / out_p).resolve()
    else:
        # convmerge ``Manifest.defaults`` is a ``Defaults`` dataclass, not a dict.
        raw = manifest.defaults.output_root
        p = Path(raw)
        out_p = p if p.is_absolute() else (repo / p).resolve()

    log = get_run_logger()
    log.info("convmerge fetch manifest=%s output_root=%s", mp, out_p)
    result = run_manifest(manifest, output_root=out_p, only=only)
    log.info(
        "convmerge fetch done: succeeded=%s skipped=%s failed=%s",
        len(result.succeeded),
        len(result.skipped),
        len(result.failed),
    )
    if result.failed:
        names = [n for n, _ in result.failed]
        raise RuntimeError(f"convmerge fetch failed for: {names}")
    return {
        "manifest": str(mp),
        "output_root": str(out_p),
        "succeeded": len(result.succeeded),
        "skipped": len(result.skipped),
        "failed": len(result.failed),
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--manifest", type=Path, default=None)
    p.add_argument("-o", "--output-root", type=Path, default=None)
    p.add_argument("--only", nargs="*", default=None)
    args = p.parse_args()
    out = collect_convmerge_flow(
        manifest_path=args.manifest,
        output_root=args.output_root,
        only=(args.only or None),
    )
    print(out)
