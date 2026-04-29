"""Refine stage: clone cruft cleanup, filter stages [1]–[9], HF staging (no upload).

HF staging repo name: pass ``hf_repo_name=...``, or set ``HF_DATASET_REPO_NAME`` in ``.env``
(loaded from the repo root). Explicit ``hf_repo_name=""`` skips staging even if the env var is set.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from prefect import flow

from pipeline.sources_config import REPO_ROOT
from pipeline.tasks.refine.post_processing import (
    cleanup_raw_repo,
    prepare_hf_dataset_dir_task,
    refine_filter_stages_task,
)


@flow(name="refine", log_prints=True)
def refine_flow(
    raw_root: str | Path | None = None,
    *,
    refined_root: str | Path | None = None,
    cleanup_clone_dirs: bool = True,
    run_filter_stages: bool = True,
    hf_repo_name: str | None = None,
    hf_namespace: str | None = None,
    hf_remote_git_url: str | None = None,
    hf_classify_state_path: str | None = None,
    hf_classify_output_jsonl: str | None = None,
    hf_classify_category_dir: str | None = None,
) -> tuple[str | None, str | None]:
    """Per-dataset clone cleanup under ``raw``, then filter stages [1]–[9], then optional HF folder prep.

    Returns ``(train_jsonl, test_jsonl)`` paths when filter stages ran, else ``(None, None)``.
    """
    load_dotenv(REPO_ROOT / ".env")

    def _hf_repo_resolved() -> str | None:
        if hf_repo_name is not None:
            s = str(hf_repo_name).strip()
            return s or None
        return os.environ.get("HF_DATASET_REPO_NAME", "").strip() or None

    hf_repo_effective = _hf_repo_resolved()

    root = Path(raw_root) if raw_root is not None else REPO_ROOT / "data" / "raw"
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(str(root))

    if cleanup_clone_dirs:
        children = [str(p) for p in sorted(root.iterdir()) if p.is_dir()]
        print("Dataset dirs under raw:", len(children))
        if children:
            cleanup_raw_repo.map(target_dir=children)

    train_path: str | None = None
    test_path: str | None = None
    refined_arg: str | None = None
    if refined_root is not None:
        rr = Path(refined_root)
        refined_arg = str(rr if rr.is_absolute() else (REPO_ROOT / rr).resolve())
    if run_filter_stages:
        train_path, test_path = refine_filter_stages_task(str(root), refined_root=refined_arg)

    if hf_repo_effective and train_path and test_path:
        prepare_hf_dataset_dir_task(
            train_path,
            test_path,
            hf_repo_effective,
            namespace=hf_namespace,
            remote_git_url=hf_remote_git_url,
            classify_state_path=hf_classify_state_path,
            classify_output_jsonl=hf_classify_output_jsonl,
            classify_category_dir=hf_classify_category_dir,
        )

    return train_path, test_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Refine: cleanup raw clones, filter stages, optional HF staging.")
    p.add_argument(
        "-i",
        "--raw-root",
        type=Path,
        default=None,
        help="Directory containing per-dataset raw clones (default: ./data/raw under repo root)",
    )
    p.add_argument(
        "--refined-root",
        type=Path,
        default=None,
        help="Parent dir for refined jsonl/multi_turn/single_turn/final (default: ./data/refined)",
    )
    p.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip deleting non-data files under each dataset dir",
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip filter stages [1]–[9]",
    )
    p.add_argument(
        "--hf-repo-name",
        default=None,
        help="HF dataset repo name for staging (default: HF_DATASET_REPO_NAME from .env)",
    )
    p.add_argument(
        "--skip-hf-staging",
        action="store_true",
        help="Do not prepare HF dataset dir even if HF_DATASET_REPO_NAME is set",
    )
    p.add_argument("--hf-namespace", default=None)
    p.add_argument("--hf-remote-git-url", default=None)
    p.add_argument("--hf-classify-state-path", type=Path, default=None)
    p.add_argument("--hf-classify-output-jsonl", type=Path, default=None)
    p.add_argument("--hf-classify-category-dir", type=Path, default=None)
    args = p.parse_args()
    hf_name: str | None
    if args.skip_hf_staging:
        hf_name = ""
    else:
        hf_name = args.hf_repo_name
    refined_resolved: Path | None = None
    if args.refined_root is not None:
        rr = args.refined_root
        refined_resolved = rr if rr.is_absolute() else (REPO_ROOT / rr).resolve()
    refine_flow(
        args.raw_root,
        refined_root=refined_resolved,
        cleanup_clone_dirs=not args.no_cleanup,
        run_filter_stages=not args.no_filter,
        hf_repo_name=hf_name,
        hf_namespace=args.hf_namespace,
        hf_remote_git_url=args.hf_remote_git_url,
        hf_classify_state_path=str(args.hf_classify_state_path)
        if args.hf_classify_state_path
        else None,
        hf_classify_output_jsonl=str(args.hf_classify_output_jsonl)
        if args.hf_classify_output_jsonl
        else None,
        hf_classify_category_dir=str(args.hf_classify_category_dir)
        if args.hf_classify_category_dir
        else None,
    )
