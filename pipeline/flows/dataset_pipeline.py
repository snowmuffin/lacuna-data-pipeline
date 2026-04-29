"""End-to-end dataset flow: collect → refine → optional RunPod classify."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from prefect import flow

from pipeline.sources_config import REPO_ROOT

CollectVia = Literal["git", "convmerge"]


def _resolve_raw(
    collect_output_root: str | Path | None,
    raw_root: str | Path | None,
) -> Path:
    if raw_root is not None:
        p = Path(raw_root)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    if collect_output_root is not None:
        p = Path(collect_output_root)
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    return (REPO_ROOT / "data" / "raw").resolve()


def _default_train_path(refined_root: str | Path | None, train_path: str | None) -> Path | None:
    if train_path:
        return Path(train_path).resolve()
    if refined_root is not None:
        p = Path(refined_root)
        base = p if p.is_absolute() else (REPO_ROOT / p).resolve()
    else:
        base = (REPO_ROOT / "data" / "refined").resolve()
    cand = base / "final" / "train.jsonl"
    return cand if cand.is_file() else None


@flow(name="dataset-pipeline", log_prints=True)
def dataset_pipeline_flow(
    *,
    sources_path: str | Path | None = None,
    collect_output_root: str | Path | None = None,
    collect_via: CollectVia = "git",
    convmerge_manifest: str | Path | None = None,
    convmerge_only: list[str] | None = None,
    raw_root: str | Path | None = None,
    refined_root: str | Path | None = None,
    run_collect: bool = True,
    run_refine: bool = True,
    run_classify: bool = False,
    classify_input_jsonl: str | Path | None = None,
    classify_output_jsonl: str | Path | None = None,
    classify_category_dir: str | Path | None = None,
    deploy_classify_workers: bool = False,
    classify_pod_count: int | None = None,
    refine_cleanup_clone_dirs: bool = True,
    refine_skip_hf_staging: bool = False,
    hf_repo_name: str | None = None,
    hf_namespace: str | None = None,
    hf_remote_git_url: str | None = None,
    hf_classify_state_path: str | None = None,
    hf_classify_output_jsonl: str | None = None,
    hf_classify_category_dir: str | None = None,
) -> dict[str, str | None]:
    """Run collect and/or refine and/or classify in order.

    Returns paths ``{"raw_root", "train_path", "test_path", "classify_output"}`` (strings or None).
    """
    load_dotenv(REPO_ROOT / ".env")

    if run_collect:
        if collect_via == "convmerge":
            from pipeline.flows.collect_convmerge import collect_convmerge_flow

            collect_convmerge_flow(
                manifest_path=convmerge_manifest,
                output_root=collect_output_root,
                only=convmerge_only,
            )
        else:
            from pipeline.flows.collect import collect_flow

            collect_flow(sources_path=sources_path, output_root=collect_output_root)

    raw_resolved = _resolve_raw(collect_output_root, raw_root)
    train_path: str | None = None
    test_path: str | None = None

    if run_refine:
        from pipeline.flows.refine import refine_flow

        hf_name = "" if refine_skip_hf_staging else hf_repo_name
        train_path, test_path = refine_flow(
            raw_resolved,
            refined_root=refined_root,
            cleanup_clone_dirs=refine_cleanup_clone_dirs,
            run_filter_stages=True,
            hf_repo_name=hf_name,
            hf_namespace=hf_namespace,
            hf_remote_git_url=hf_remote_git_url,
            hf_classify_state_path=hf_classify_state_path,
            hf_classify_output_jsonl=hf_classify_output_jsonl,
            hf_classify_category_dir=hf_classify_category_dir,
        )

    classify_out: str | None = None
    if run_classify:
        from pipeline.flows.classify import classify_flow

        cin = _default_train_path(refined_root, train_path)
        if classify_input_jsonl is not None:
            cin = Path(classify_input_jsonl).resolve()
        if cin is None or not cin.is_file():
            raise FileNotFoundError(
                "classify requires an existing JSONL; set classify_input_jsonl or run refine first."
            )
        if classify_output_jsonl is None:
            raise ValueError("classify_output_jsonl is required when run_classify=True")
        classify_out = classify_flow(
            cin,
            Path(classify_output_jsonl).resolve(),
            category_dir=classify_category_dir,
            deploy_workers=deploy_classify_workers,
            classify_pod_count=classify_pod_count,
        )

    return {
        "raw_root": str(raw_resolved),
        "train_path": train_path,
        "test_path": test_path,
        "classify_output": classify_out,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, default=None)
    p.add_argument(
        "--collect-via",
        choices=("git", "convmerge"),
        default="git",
        help="git=sources.yaml git+LFS; convmerge=HF hub fetch via convmerge manifest",
    )
    p.add_argument(
        "--convmerge-manifest",
        type=Path,
        default=None,
        help="Manifest YAML when --collect-via convmerge (default: config/convmerge_manifest.yaml)",
    )
    p.add_argument(
        "--convmerge-only",
        nargs="*",
        default=None,
        help="Only these dataset names from the convmerge manifest",
    )
    p.add_argument("--collect-output", type=Path, default=None, help="Raw download root")
    p.add_argument("--raw-root", type=Path, default=None, help="Override raw dir for refine")
    p.add_argument("--refined-root", type=Path, default=None)
    p.add_argument("--no-collect", action="store_true")
    p.add_argument("--no-refine", action="store_true")
    p.add_argument("--classify", action="store_true")
    p.add_argument("--classify-input", type=Path, default=None)
    p.add_argument("--classify-output", type=Path, default=None)
    p.add_argument("--classify-category-dir", type=Path, default=None)
    p.add_argument("--deploy-classify-workers", action="store_true")
    p.add_argument("--no-refine-cleanup", action="store_true")
    p.add_argument("--skip-hf-staging", action="store_true")
    args = p.parse_args()
    out = dataset_pipeline_flow(
        sources_path=args.sources,
        collect_output_root=args.collect_output,
        collect_via=args.collect_via,
        convmerge_manifest=args.convmerge_manifest,
        convmerge_only=(args.convmerge_only or None),
        raw_root=args.raw_root,
        refined_root=args.refined_root,
        run_collect=not args.no_collect,
        run_refine=not args.no_refine,
        run_classify=args.classify,
        classify_input_jsonl=args.classify_input,
        classify_output_jsonl=args.classify_output,
        classify_category_dir=args.classify_category_dir,
        deploy_classify_workers=args.deploy_classify_workers,
        refine_cleanup_clone_dirs=not args.no_refine_cleanup,
        refine_skip_hf_staging=args.skip_hf_staging,
    )
    print(out)
