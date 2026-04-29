"""Collect stage: clone or download sources from sources.yaml."""

from __future__ import annotations

from pathlib import Path

from prefect import flow

from pipeline.sources_config import REPO_ROOT, load_sources, token_from_sources
from pipeline.tasks.crawlers.fetch import (
    assert_git_lfs_for_collect,
    fetch_dataset_task,
    resolve_output_root,
)


@flow(name="collect", log_prints=True)
def collect_flow(
    sources_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> None:
    """Load ``sources.yaml`` (or ``sources_path``), write datasets under ``output_root`` or defaults."""
    sp = Path(sources_path) if sources_path is not None else None
    data = load_sources(sp)
    hf_token = token_from_sources(data, "hf_token_env")
    github_token = token_from_sources(data, "github_token_env")

    defaults = data.get("defaults") or {}
    resume = bool(defaults.get("resume", True))
    on_error = str(defaults.get("on_error", "continue")).strip().lower()
    git_lfs_default = bool(defaults.get("git_lfs", True))
    assert_git_lfs_for_collect(data)
    if output_root is not None:
        out = Path(output_root)
        output_root_resolved = out if out.is_absolute() else (REPO_ROOT / out).resolve()
    else:
        output_root_resolved = resolve_output_root(data)

    for ds in data.get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        git_lfs = bool(ds["git_lfs"]) if "git_lfs" in ds else git_lfs_default
        fetch_dataset_task(
            ds,
            output_root_resolved,
            hf_token=hf_token,
            github_token=github_token,
            resume=resume,
            on_error=on_error,
            git_lfs=git_lfs,
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Collect datasets from sources.yaml.")
    p.add_argument(
        "-i",
        "--sources",
        type=Path,
        default=None,
        help="Path to sources.yaml (default: repo root sources.yaml)",
    )
    p.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=None,
        help="Directory to clone/download into (overrides sources.yaml defaults.output_root)",
    )
    args = p.parse_args()
    collect_flow(sources_path=args.sources, output_root=args.output_root)
