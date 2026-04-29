"""Post-processing for the refine stage (cleanup, format unify, etc.)."""


from prefect import task
import os
import subprocess
import sys
from pathlib import Path

from convmerge.normalize.jsonl import detect_jsonl_shape


def is_single_line_jsonl(file_path: str) -> str:
    """Return 'jsonl' | 'single_line' | 'json_array' | 'invalid' via convmerge."""
    shape = detect_jsonl_shape(Path(file_path))
    return {
        "jsonl": "jsonl",
        "single_line": "single_line",
        "json_array": "json_array",
    }.get(shape, "invalid")


def jsonl_path_looks_like_root_json_array(file_path: str) -> bool:
    """True when the file's first non-whitespace token is ``[`` (top-level JSON array)."""
    return detect_jsonl_shape(Path(file_path)) == "json_array"


def find_single_line_jsonl(dir_path: str) -> list[str]:
    """List files whose first line packs multiple JSON records (concat or array)."""
    hits: list[str] = []
    for root, _dirs, files in os.walk(dir_path):
        for fname in files:
            if not (fname.endswith(".json") or fname.endswith(".jsonl")):
                continue
            fp = os.path.join(root, fname)
            if is_single_line_jsonl(fp) in ("single_line", "json_array"):
                hits.append(fp)
    return hits


def filter_data(dir_path: str, out_dir: str) -> None:
    """Mirror ``dir_path`` into ``out_dir`` as .jsonl via ``python -m convmerge normalize``."""
    subprocess.run(
        [sys.executable, "-m", "convmerge", "normalize", "-i", dir_path, "-o", out_dir],
        check=False,
    )


@task(name="cleanup-raw-repo", log_prints=True)
def cleanup_raw_repo(
    target_dir: str,
    exception_formats: list[str] | None = None,
) -> None:
    """Remove files under ``target_dir`` that do not match kept extensions (clone cruft)."""
    if exception_formats is None:
        exception_formats = [".jsonl", ".json", ".csv", ".parquet"]
    for root, _dirs, files in os.walk(target_dir):
        for file in files:
            if not any(file.endswith(ext) for ext in exception_formats):
                os.remove(os.path.join(root, file))


@task(name="unify-file-formats", log_prints=True)
def unify_file_formats(target_dir: str, target_format: str = ".jsonl") -> None:
    """Placeholder: optional per-file unify (main path uses ``run_refine_filter_stages``)."""
    for root, _dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(target_format):
                pass  # TODO: optional extra normalizers


@task(name="refine-filter-stages", log_prints=True)
def refine_filter_stages_task(
    raw_root: str,
    refined_root: str | Path | None = None,
) -> tuple[str, str]:
    """Stages [1]–[9]: raw→jsonl→multi/single-turn→collect→dedup→split."""
    from pipeline.tasks.refine.refine_stages import default_refine_paths, run_refine_filter_stages

    paths = default_refine_paths(raw_root, refined_root=refined_root)
    return run_refine_filter_stages(paths)


@task(name="prepare-hf-dataset-dir", log_prints=True)
def prepare_hf_dataset_dir_task(
    final_train_path: str,
    final_test_path: str,
    repo_name: str,
    hf_datasets_root: str | None = None,
    namespace: str | None = None,
    remote_git_url: str | None = None,
    classify_state_path: str | None = None,
    classify_output_jsonl: str | None = None,
    classify_category_dir: str | None = None,
) -> str:
    """Stage train/test (+ optional classify artifacts); ``remote_git_url`` triggers clone/pull first."""
    from pipeline.sources_config import REPO_ROOT
    from pipeline.tasks.refine.hf_prepare import prepare_hf_dataset_repo

    root = Path(hf_datasets_root) if hf_datasets_root else REPO_ROOT / "data" / "refined" / "hf_datasets"
    return prepare_hf_dataset_repo(
        final_train_path,
        final_test_path,
        repo_name,
        root,
        namespace=namespace,
        remote_git_url=remote_git_url,
        classify_state_path=classify_state_path,
        classify_output_jsonl=classify_output_jsonl,
        classify_category_dir=classify_category_dir,
    )
