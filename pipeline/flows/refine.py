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
    cleanup_clone_dirs: bool = True,
    run_filter_stages: bool = True,
    hf_repo_name: str | None = None,
    hf_namespace: str | None = None,
    hf_remote_git_url: str | None = None,
    hf_classify_state_path: str | None = None,
    hf_classify_output_jsonl: str | None = None,
    hf_classify_category_dir: str | None = None,
) -> None:
    """Per-dataset clone cleanup under ``raw``, then filter stages [1]–[9], then optional HF folder prep."""
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
    if run_filter_stages:
        train_path, test_path = refine_filter_stages_task(str(root))

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


if __name__ == "__main__":
    refine_flow()
