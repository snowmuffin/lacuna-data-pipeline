"""Optional Anthropic LLM steps (identity batch eval, assistant rewrite) after JSONL refine."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from prefect import flow, task

from pipeline.sources_config import REPO_ROOT


@task(name="identity-eval-chunked", log_prints=True)
def identity_eval_chunked_task(
    input_jsonl: str,
    work_dir: str,
    *,
    include_yes_only: bool = True,
    include_reason: bool = False,
    model: str = "claude-haiku-4-5",
    chunk_size: int = 200,
    max_chunks: int | None = None,
) -> None:
    from pipeline.tasks.refine.identity_eval_anthropic import run_identity_eval_chunked

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    run_identity_eval_chunked(
        input_jsonl,
        progress_jsonl=str(wd / "identity_eval_progress.jsonl"),
        output_scored_jsonl=str(wd / "identity_eval_scored.jsonl"),
        batch_id_path=str(wd / "identity_eval_batch_id.txt"),
        yes_only_jsonl=str(wd / "identity_yes.jsonl") if include_yes_only else None,
        include_reason=include_reason,
        model=model,
        chunk_size=chunk_size,
        max_chunks=max_chunks,
    )


@task(name="assistant-rewrite-jsonl", log_prints=True)
def assistant_rewrite_jsonl_task(
    input_jsonl: str,
    output_jsonl: str,
    *,
    max_rows: int | None = None,
    primary_language_hint: str | None = None,
) -> int:
    """Rewrite+translate each JSONL row that has a ``messages`` list (API-heavy)."""
    import json

    from pipeline.tasks.refine.assistant_sft_rewrite import request_rewrite_translate

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with (
        open(input_jsonl, encoding="utf-8") as rf,
        out_path.open("w", encoding="utf-8") as wf,
    ):
        for i, line in enumerate(rf):
            if max_rows is not None and n >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            merged = request_rewrite_translate(
                messages,
                primary_language_hint=primary_language_hint,
            )
            out = {**row, **merged}
            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


@flow(name="refine-llm-supplemental", log_prints=True)
def refine_llm_flow(
    *,
    run_identity_eval: bool = False,
    identity_input_jsonl: str | Path | None = None,
    identity_work_dir: str | Path | None = None,
    identity_include_reason: bool = False,
    identity_model: str = "claude-haiku-4-5",
    run_assistant_rewrite: bool = False,
    rewrite_input_jsonl: str | Path | None = None,
    rewrite_output_jsonl: str | Path | None = None,
    rewrite_max_rows: int | None = None,
) -> dict[str, int | None]:
    """Run optional Anthropic jobs; each branch is skipped unless its flag is true."""
    load_dotenv(REPO_ROOT / ".env")
    stats: dict[str, int | None] = {"identity_eval": None, "assistant_rewrite_rows": None}

    if run_identity_eval:
        if not identity_input_jsonl or not identity_work_dir:
            raise ValueError("identity_input_jsonl and identity_work_dir are required when run_identity_eval=True")
        identity_eval_chunked_task(
            str(Path(identity_input_jsonl).resolve()),
            str(Path(identity_work_dir).resolve()),
            include_reason=identity_include_reason,
            model=identity_model,
        )
        stats["identity_eval"] = 1

    if run_assistant_rewrite:
        if not rewrite_input_jsonl or not rewrite_output_jsonl:
            raise ValueError(
                "rewrite_input_jsonl and rewrite_output_jsonl are required when run_assistant_rewrite=True"
            )
        stats["assistant_rewrite_rows"] = assistant_rewrite_jsonl_task(
            str(Path(rewrite_input_jsonl).resolve()),
            str(Path(rewrite_output_jsonl).resolve()),
            max_rows=rewrite_max_rows,
        )

    return stats


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity", action="store_true")
    p.add_argument("--identity-input", type=Path, default=None)
    p.add_argument("--identity-work-dir", type=Path, default=None)
    p.add_argument("--identity-include-reason", action="store_true")
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--rewrite-input", type=Path, default=None)
    p.add_argument("--rewrite-output", type=Path, default=None)
    p.add_argument("--rewrite-max-rows", type=int, default=None)
    args = p.parse_args()
    if not args.identity and not args.rewrite:
        p.error("Specify --identity and/or --rewrite")
    refine_llm_flow(
        run_identity_eval=args.identity,
        identity_input_jsonl=args.identity_input,
        identity_work_dir=args.identity_work_dir,
        identity_include_reason=args.identity_include_reason,
        run_assistant_rewrite=args.rewrite,
        rewrite_input_jsonl=args.rewrite_input,
        rewrite_output_jsonl=args.rewrite_output,
        rewrite_max_rows=args.rewrite_max_rows,
    )
