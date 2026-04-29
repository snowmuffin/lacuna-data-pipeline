"""Classify flow: RunPod HTTP worker batch over JSONL (optional multi-pod deploy)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from prefect import flow, task

from pipeline.sources_config import REPO_ROOT


@task(name="ensure-classify-workers", log_prints=True)
def ensure_classify_workers_task(
    *,
    pod_count: int,
    stem: str | None,
    deploy: bool,
    use_graphql: bool,
) -> list[str]:
    from pipeline.tasks.classifiers.runpod_batch_classify import ensure_classify_runpod_bases

    return ensure_classify_runpod_bases(
        pod_count=pod_count,
        stem=stem,
        deploy=deploy,
        use_graphql=use_graphql,
    )


@task(name="batch-classify-runpod", log_prints=True)
def batch_classify_runpod_task(
    input_jsonl: str,
    output_jsonl: str,
    *,
    category_dir: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    state_path: str | None = None,
    batch_size: int | None = None,
    max_samples: int | None = None,
    stream: bool | None = None,
    parallel_workers: int | None = None,
) -> str:
    from pipeline.tasks.classifiers.runpod_batch_classify import batch_classify_runpod

    kw: dict = {}
    if category_dir is not None:
        kw["category_dir"] = category_dir
    if base_url is not None:
        kw["base_url"] = base_url
    if api_key is not None:
        kw["api_key"] = api_key
    if state_path is not None:
        kw["state_path"] = state_path
    if max_samples is not None:
        kw["max_samples"] = max_samples
    if stream is not None:
        kw["stream"] = stream
    if parallel_workers is not None:
        kw["parallel_workers"] = parallel_workers
    if batch_size is not None:
        kw["batch_size"] = batch_size
    return batch_classify_runpod(input_jsonl, output_jsonl, **kw)


@flow(name="classify", log_prints=True)
def classify_flow(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    category_dir: str | Path | None = None,
    state_path: str | Path | None = None,
    deploy_workers: bool = False,
    classify_pod_count: int | None = None,
    pod_name_stem: str | None = None,
    use_graphql_discovery: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
    batch_size: int | None = None,
    max_samples: int | None = None,
    stream: bool | None = None,
    parallel_workers: int | None = None,
) -> str:
    """Classify JSONL lines via GPU worker(s); set ``LACUNA_DATA_API_BASE`` in ``.env`` or pass ``base_url``.

    When ``deploy_workers`` is True, calls :func:`ensure_classify_runpod_bases` (needs ``RUNPOD_API_KEY``).
    """
    load_dotenv(REPO_ROOT / ".env")
    inp = str(Path(input_jsonl).resolve())
    outp = str(Path(output_jsonl).resolve())
    cat = str(Path(category_dir).resolve()) if category_dir is not None else None
    st = str(Path(state_path).resolve()) if state_path is not None else None

    n = classify_pod_count
    if n is None:
        raw = (os.environ.get("LACUNA_DATA_CLASSIFY_POD_COUNT") or "").strip()
        try:
            n = int(raw, 10) if raw else 1
        except ValueError:
            n = 1

    if deploy_workers:
        ensure_classify_workers_task(
            pod_count=max(1, n),
            stem=pod_name_stem,
            deploy=True,
            use_graphql=use_graphql_discovery,
        )

    return batch_classify_runpod_task(
        inp,
        outp,
        category_dir=cat,
        state_path=st,
        base_url=base_url,
        api_key=api_key,
        batch_size=batch_size,
        max_samples=max_samples,
        stream=stream,
        parallel_workers=parallel_workers,
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RunPod JSONL classification (see classify_flow).")
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input JSONL path",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path (used for resume state path unless --state-path is set)",
    )
    p.add_argument(
        "--category-dir",
        type=Path,
        default=None,
        help="Write per-label shards under this directory instead of merged output",
    )
    p.add_argument(
        "--state-path",
        type=Path,
        default=None,
        help="Resume state JSON path (default: <output>.state.json)",
    )
    p.add_argument("--base-url", default=None, help="Override LACUNA_DATA_API_BASE for this run")
    p.add_argument("--api-key", default=None, help="Override LACUNA_DATA_API_KEY for this run")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stream large JSONL (default: auto by file size)",
    )
    p.add_argument("--parallel-workers", type=int, default=None)
    p.add_argument(
        "--deploy-workers",
        action="store_true",
        help="Deploy or discover RunPod workers before classifying",
    )
    p.add_argument(
        "--classify-pod-count",
        type=int,
        default=None,
        help="Number of classify pods when --deploy-workers (default: env or 1)",
    )
    p.add_argument("--pod-name-stem", default=None, help="RunPod pod name stem (default: POD_NAME env)")
    p.add_argument(
        "--no-graphql-discovery",
        action="store_true",
        help="Do not refresh worker URLs from RunPod GraphQL when deploying",
    )
    args = p.parse_args()
    classify_flow(
        args.input,
        args.output,
        category_dir=args.category_dir,
        state_path=str(args.state_path) if args.state_path is not None else None,
        deploy_workers=args.deploy_workers,
        classify_pod_count=args.classify_pod_count,
        pod_name_stem=args.pod_name_stem,
        use_graphql_discovery=not args.no_graphql_discovery,
        base_url=args.base_url,
        api_key=args.api_key,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        stream=args.stream,
        parallel_workers=args.parallel_workers,
    )
