"""Identity-label evaluation via Anthropic batch API (ported from refine notebook).

Loads ``ANTHROPIC_API_KEY`` from the repository root ``.env`` (see :func:`_load_pipeline_dotenv`).
"""

import html
import json
import os
import re
import shutil
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

from pipeline.sources_config import REPO_ROOT


def _load_pipeline_dotenv() -> None:
    """Load repository root ``.env`` (``ANTHROPIC_API_KEY``, etc.)."""
    p = REPO_ROOT / ".env"
    if p.is_file():
        load_dotenv(p, override=False)
    else:
        load_dotenv(override=False)


_load_pipeline_dotenv()

client = anthropic.Anthropic()

IDENTITY_EVAL_INSTRUCTIONS = """You are evaluating whether a chat sample genuinely belongs to the "identity" category.

"identity" means: The USER is probing the ASSISTANT's own nature — its name, whether it is an AI,
feelings/consciousness, who built it, or how it defines itself. Jailbreak attempts that try to
redefine the assistant's identity also count. Questions specifically asking whether the assistant
is based on or related to Qwen also count.

NOT identity:
- User asks about AI as a topic or technology in general
- User asks the assistant to perform tasks involving other AI tools (e.g. Midjourney, DALL-E)
- The assistant briefly mentions being an AI while answering an unrelated question
- Philosophical or academic discussion where AI-related words appear incidentally
"""


def build_identity_eval_prompt(sample_text: str, *, include_reason: bool) -> str:
    if include_reason:
        return (
            IDENTITY_EVAL_INSTRUCTIONS
            + """

Respond with ONLY a single JSON object (no markdown code fences). Keys:
- "verdict": one of "yes", "no", "borderline" (same meaning as below)
- "reason": 1–3 short sentences in English explaining the verdict

Verdict meanings:
- "yes" — clearly probing this assistant's identity
- "no" — not actually about this assistant's identity
- "borderline" — genuinely ambiguous

Sample:
"""
            + sample_text
        )
    return (
        IDENTITY_EVAL_INSTRUCTIONS
        + """

Reply with ONLY one word on a single line: yes, no, or borderline (nothing else).

Sample:
"""
        + sample_text
    )


def _strip_json_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t, count=1)
    return t.strip()


def parse_identity_eval_response(text: str, *, include_reason: bool) -> tuple[str, str | None]:
    """Return (verdict, reason_or_none). Verdict is yes|no|borderline|error."""
    raw = (text or "").strip()
    if not raw:
        return "error", None

    if include_reason:
        try:
            obj = json.loads(_strip_json_fence(raw))
            v = str(obj.get("verdict", "")).strip().lower()
            r = obj.get("reason")
            reason = str(r).strip() if r is not None else None
            if v in ("yes", "no", "borderline"):
                return v, reason
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        low = raw.lower()
        if "borderline" in low:
            return "borderline", None
        if re.match(r"^yes\b", low):
            return "yes", None
        if re.match(r"^no\b", low):
            return "no", None
        return "borderline", None

    low = raw.lower()
    if "borderline" in low:
        return "borderline", None
    if low.startswith("yes"):
        return "yes", None
    if low.startswith("no"):
        return "no", None
    return "borderline", None


def _stringify_content(content) -> str:
    """identity.jsonl uses string content; support list blocks (multimodal) if present."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, dict):
                parts.append(str(block.get("text", block)))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def format_sample_for_eval(sample: dict) -> str:
    """Format one JSONL row: expects `messages` (chat turns), as in category shards."""
    messages = sample.get("messages")
    if messages is None:
        raise ValueError("Sample has no 'messages' field (expected identity.jsonl chat format).")
    lines = []
    for m in messages:
        role = m.get("role", "")
        content = _stringify_content(m.get("content", ""))
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


def _eval_meta_path(batch_id_path: str) -> Path:
    p = Path(batch_id_path)
    return p.parent / f"{p.stem}_meta.json"


def submit_identity_eval_batch(
    input_jsonl: str,
    batch_id_path: str = "batch_id.txt",
    *,
    include_reason: bool = False,
    max_tokens: int | None = None,
    model: str = "claude-haiku-4-5",
) -> str:
    """Submit all samples to Batch API and save batch_id (+ eval meta) to disk."""
    if max_tokens is None:
        # Output cap (generation), not input context. JSON + short reason rarely needs more.
        max_tokens = 2048 if include_reason else 128

    meta = {
        "include_reason": include_reason,
        "max_tokens": max_tokens,
        "model": model,
    }
    meta_path = _eval_meta_path(batch_id_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    requests = []
    with open(input_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            sample_text = format_sample_for_eval(sample)
            user_content = build_identity_eval_prompt(sample_text, include_reason=include_reason)
            requests.append(
                {
                    "custom_id": f"sample-{i}",
                    "params": {
                        "model": model,
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": user_content}],
                    },
                }
            )

    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted {len(requests)} samples. batch_id: {batch.id}")
    print(f"include_reason={include_reason}, max_tokens={max_tokens} (meta → {meta_path})")
    Path(batch_id_path).write_text(batch.id, encoding="utf-8")
    return batch.id


def poll_batch(batch_id: str, poll_interval: int = 60):
    """Poll until batch processing ends. Retries on empty/non-JSON responses (transient proxies/gateways)."""
    retry_backoff = 1.0
    max_retry_backoff = min(120.0, float(max(poll_interval, 5)))
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
        except json.JSONDecodeError as e:
            print(
                f"[poll_batch] empty or invalid JSON from API ({e!r}); retry in {retry_backoff:.0f}s "
                "(often transient network/proxy; if this persists, check API status and VPN/firewall)."
            )
            time.sleep(retry_backoff)
            retry_backoff = min(max_retry_backoff, retry_backoff * 2)
            continue
        except httpx.HTTPError as e:
            print(
                f"[poll_batch] HTTP error while retrieving batch: {e!r}; retry in {retry_backoff:.0f}s"
            )
            time.sleep(retry_backoff)
            retry_backoff = min(max_retry_backoff, retry_backoff * 2)
            continue
        retry_backoff = 1.0
        counts = batch.request_counts
        print(
            f"[{batch.processing_status}] processing={counts.processing} / "
            f"succeeded={counts.succeeded} / errored={counts.errored}"
        )
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_interval)


def save_batch_results(
    batch_id: str,
    input_jsonl: str,
    output_jsonl: str,
    *,
    include_reason: bool | None = None,
    batch_meta_path: str | None = None,
) -> dict:
    """Merge eval verdicts into original samples and write to output JSONL."""
    originals = {}
    with open(input_jsonl, encoding="utf-8") as f:
        for i, line in enumerate(f):
            originals[f"sample-{i}"] = json.loads(line)

    if include_reason is None:
        if batch_meta_path and Path(batch_meta_path).is_file():
            meta = json.loads(Path(batch_meta_path).read_text(encoding="utf-8"))
            include_reason = bool(meta.get("include_reason", False))
        else:
            include_reason = False

    results: dict[str, int] = {"yes": 0, "no": 0, "borderline": 0, "error": 0}

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            sample = originals.get(custom_id, {})

            reason: str | None = None
            if result.result.type == "succeeded":
                raw_text = result.result.message.content[0].text
                verdict, reason = parse_identity_eval_response(
                    raw_text, include_reason=include_reason
                )
            else:
                verdict, reason = "error", None

            results[verdict] = results.get(verdict, 0) + 1
            row = {**sample, "_identity_eval": verdict}
            if include_reason:
                row["_identity_eval_reason"] = reason if reason is not None else ""
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(results.values())
    print(f"\nResults summary (total {total})")
    for k in ("yes", "no", "borderline", "error"):
        v = results.get(k, 0)
        pct = v / total * 100 if total else 0.0
        print(f"  {k}: {v} ({pct:.1f}%)")

    return results


def write_identity_eval_reports(
    scored_jsonl: str,
    *,
    report_md: str | None = None,
    report_html: str | None = None,
    max_preview_chars: int = 600,
) -> None:
    """Write human-readable Markdown + HTML reports from scored JSONL."""
    rows = []
    with open(scored_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    counts = Counter(r.get("_identity_eval", "") for r in rows)
    n = len(rows)

    md_lines = [
        "# Identity evaluation results",
        "",
        f"- **Samples:** {n}",
        "",
        "## Summary",
        "",
        "| Verdict | Count | % |",
        "|---------|-------|---|",
    ]
    for key in ("yes", "no", "borderline", "error"):
        c = counts.get(key, 0)
        pct = 100.0 * c / n if n else 0.0
        md_lines.append(f"| `{key}` | {c} | {pct:.1f}% |")
    md_lines.extend(["", "## Samples", ""])

    for i, r in enumerate(rows):
        v = r.get("_identity_eval", "")
        reason = r.get("_identity_eval_reason", "")
        try:
            preview = format_sample_for_eval(r)
        except Exception as ex:
            preview = f"(preview error: {ex})"
        if len(preview) > max_preview_chars:
            preview = preview[:max_preview_chars] + "…"
        md_lines.append(f"### #{i} — `{v}`")
        md_lines.append("")
        if reason:
            md_lines.append(f"**Reason:** {reason}")
            md_lines.append("")
        md_lines.append("```text")
        md_lines.append(preview)
        md_lines.append("```")
        md_lines.append("")

    md_text = "\n".join(md_lines)
    if report_md:
        Path(report_md).write_text(md_text, encoding="utf-8")
        print(f"Wrote Markdown report → {report_md}")

    if report_html:
        body_parts = [
            "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Identity eval</title>",
            "<style>body{font-family:system-ui,Segoe UI,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;}",
            "table{border-collapse:collapse;} th,td{border:1px solid #ccc;padding:8px;text-align:left;}",
            "pre{white-space:pre-wrap;background:#f6f8fa;padding:12px;border-radius:8px;font-size:13px;}",
            "h3{margin-top:1.6em;}</style></head><body>",
            "<h1>Identity evaluation results</h1>",
            f"<p><strong>Samples:</strong> {n}</p>",
            "<h2>Summary</h2><table><tr><th>Verdict</th><th>Count</th><th>%</th></tr>",
        ]
        for key in ("yes", "no", "borderline", "error"):
            c = counts.get(key, 0)
            pct = 100.0 * c / n if n else 0.0
            body_parts.append(
                f"<tr><td><code>{html.escape(key)}</code></td><td>{c}</td><td>{pct:.1f}%</td></tr>"
            )
        body_parts.append("</table><h2>Samples</h2>")
        for i, r in enumerate(rows):
            v = r.get("_identity_eval", "")
            reason = r.get("_identity_eval_reason", "")
            try:
                preview = format_sample_for_eval(r)
            except Exception as ex:
                preview = f"(preview error: {ex})"
            if len(preview) > max_preview_chars:
                preview = preview[:max_preview_chars] + "…"
            body_parts.append(
                f"<h3>#{html.escape(str(i))} — <code>{html.escape(str(v))}</code></h3>"
            )
            if reason:
                body_parts.append(f"<p><strong>Reason:</strong> {html.escape(reason)}</p>")
            body_parts.append(f"<pre>{html.escape(preview)}</pre>")
        body_parts.append("</body></html>")
        Path(report_html).write_text("".join(body_parts), encoding="utf-8")
        print(f"Wrote HTML report → {report_html}")


def export_yes_only_identity_jsonl(
    scored_jsonl: str,
    identity_jsonl_path: str | None = None,
    *,
    backup: bool = True,
) -> tuple[str, int]:
    """Replace ``identity.jsonl`` with only ``_identity_eval == \"yes\"`` rows from a scored JSONL.

    - Moves the existing ``identity.jsonl`` into ``categories/backups/`` (timestamped) when
      ``backup`` is True, then writes the filtered data to the original path.
    - Drops keys starting with ``_identity_eval`` so lines match the original shard format.
    """
    identity_jsonl_path = identity_jsonl_path or os.path.join(CATEGORY_DIR, "identity.jsonl")
    if not os.path.isfile(scored_jsonl):
        raise FileNotFoundError(f"Scored JSONL not found: {scored_jsonl!r}")

    ts = time.strftime("%Y%m%dT%H%M%S")
    backup_dir = os.path.join(os.path.dirname(identity_jsonl_path), "backups")
    if backup and os.path.isfile(identity_jsonl_path):
        os.makedirs(backup_dir, exist_ok=True)
        base = os.path.basename(identity_jsonl_path)
        backup_path = os.path.join(backup_dir, f"{base}.pre_yes_filter.{ts}.bak")
        shutil.move(identity_jsonl_path, backup_path)
        print(f"Moved original → {backup_path}")

    kept: list[str] = []
    with open(scored_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("_identity_eval") != "yes":
                continue
            clean = {k: v for k, v in obj.items() if not str(k).startswith("_identity_eval")}
            kept.append(json.dumps(clean, ensure_ascii=False) + "\n")

    with open(identity_jsonl_path, "w", encoding="utf-8") as out:
        out.writelines(kept)

    print(f"Wrote {len(kept):,} yes-only lines → {identity_jsonl_path}")
    if not kept:
        print("WARNING: no 'yes' rows — identity.jsonl is empty.")
    return identity_jsonl_path, len(kept)


def load_identity_eval_done_indices(progress_jsonl: str) -> set[int]:
    """Indices already recorded (batch finished and verdict is not `error`)."""
    done: set[int] = set()
    if not os.path.isfile(progress_jsonl):
        return done
    with open(progress_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                idx = o.get("index")
                if idx is None:
                    continue
                done.add(int(idx))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return done


def identity_eval_progress_status(
    source_jsonl: str,
    progress_jsonl: str,
) -> dict:
    """Compare ``source_jsonl`` line indices to ``progress_jsonl`` (non-empty lines must be evaluated).

    Returns keys: total_lines, non_empty_count, done_count, pending_count, is_complete,
    pending_sample (first up to 32 indices), progress_path, source_path.
    """
    with open(source_jsonl, encoding="utf-8") as f:
        lines = f.readlines()
    n = len(lines)
    nonempty_indices = [i for i in range(n) if (lines[i] or "").strip()]
    done = load_identity_eval_done_indices(progress_jsonl)
    nonempty_set = set(nonempty_indices)
    done_nonempty = done & nonempty_set
    pending = sorted(nonempty_set - done)
    return {
        "source_path": source_jsonl,
        "progress_path": progress_jsonl,
        "total_lines": n,
        "non_empty_count": len(nonempty_indices),
        "done_count": len(done_nonempty),
        "pending_count": len(pending),
        "is_complete": len(pending) == 0 and len(nonempty_indices) > 0,
        "pending_sample": pending[:32],
    }


def print_identity_eval_progress_report(source_jsonl: str, progress_jsonl: str) -> dict:
    """Print a short report; returns the same dict as ``identity_eval_progress_status``."""
    st = identity_eval_progress_status(source_jsonl, progress_jsonl)
    for k, v in st.items():
        if k == "pending_sample":
            print(f"  {k} (first {len(v)}): {v}")
        else:
            print(f"  {k}: {v}")
    return st


def replace_identity_with_yes_only_when_complete(
    *,
    source_jsonl: str,
    progress_jsonl: str,
    scored_jsonl: str,
    identity_out_path: str | None = None,
    backup: bool = True,
) -> tuple[str, int]:
    """Replace ``identity.jsonl`` with yes-only rows from ``scored_jsonl`` **only** if progress is complete.

    Raises ``RuntimeError`` if any non-empty source line is missing from progress.
    """
    st = identity_eval_progress_status(source_jsonl, progress_jsonl)
    if st["non_empty_count"] == 0:
        raise RuntimeError("Source has no non-empty lines; nothing to do.")
    if not st["is_complete"]:
        raise RuntimeError(
            f"Incomplete: {st['pending_count']:,} non-empty line(s) not in progress. "
            f"Pending sample indices: {st['pending_sample'][:10]}{'...' if st['pending_count'] > 10 else ''}"
        )
    if not os.path.isfile(scored_jsonl):
        raise FileNotFoundError(f"Scored JSONL missing: {scored_jsonl}")
    return export_yes_only_identity_jsonl(
        scored_jsonl,
        identity_jsonl_path=identity_out_path,
        backup=backup,
    )


def _flush_batch_to_progress_and_rebuild_scored(
    batch_id: str,
    input_lines: list[str],
    *,
    progress_jsonl: str,
    output_scored_jsonl: str,
    include_reason: bool | None,
    batch_meta_path: str | None,
) -> tuple[int, int]:
    """After a batch has ended: append non-error rows to progress, rebuild scored JSONL.

    Progress is only appended for succeeded API results with verdict != ``error``.
    Returns (n_appended_to_progress, n_lines_in_scored).
    """
    if include_reason is not None:
        ir = bool(include_reason)
    elif batch_meta_path and Path(batch_meta_path).is_file():
        meta = json.loads(Path(batch_meta_path).read_text(encoding="utf-8"))
        ir = bool(meta.get("include_reason", False))
    else:
        ir = False

    appended = 0
    with open(progress_jsonl, "a", encoding="utf-8") as prog:
        for result in client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                continue
            raw_text = result.result.message.content[0].text
            verdict, reason = parse_identity_eval_response(raw_text, include_reason=ir)
            if verdict == "error":
                continue
            try:
                idx = int(str(result.custom_id).replace("sample-", "", 1))
            except ValueError:
                continue
            rec = {
                "index": idx,
                "verdict": verdict,
                "batch_id": batch_id,
            }
            if ir and reason is not None:
                rec["reason"] = reason
            prog.write(json.dumps(rec, ensure_ascii=False) + "\n")
            appended += 1

    n_scored = rebuild_scored_jsonl_from_progress(
        input_lines,
        progress_jsonl,
        output_scored_jsonl,
        include_reason=ir,
    )
    return appended, n_scored


def rebuild_scored_jsonl_from_progress(
    input_lines: list[str],
    progress_jsonl: str,
    output_scored_jsonl: str,
    *,
    include_reason: bool,
) -> int:
    """Write ``output_scored_jsonl``: one line per index present in ``progress_jsonl`` (sorted by index)."""
    by_idx: dict[int, dict] = {}
    if os.path.isfile(progress_jsonl):
        with open(progress_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                idx = o.get("index")
                if idx is None:
                    continue
                by_idx[int(idx)] = o

    n = 0
    os.makedirs(os.path.dirname(output_scored_jsonl) or ".", exist_ok=True)
    with open(output_scored_jsonl, "w", encoding="utf-8") as out:
        for idx in sorted(by_idx.keys()):
            if idx < 0 or idx >= len(input_lines):
                continue
            raw = input_lines[idx].strip()
            if not raw:
                continue
            p = by_idx[idx]
            sample = json.loads(raw)
            row = {**sample, "_identity_eval": p["verdict"]}
            if include_reason:
                row["_identity_eval_reason"] = str(p.get("reason", "") or "")
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Rebuilt scored JSONL → {output_scored_jsonl} ({n:,} rows from progress)")
    return n


def rebuild_yes_only_jsonl_from_progress(
    input_lines: list[str],
    progress_jsonl: str,
    yes_jsonl: str,
    *,
    strip_eval_keys: bool = False,
) -> int:
    """Write ``yes_jsonl``: one line per index with verdict ``yes`` in progress (overwrites file)."""
    by_idx: dict[int, dict] = {}
    if os.path.isfile(progress_jsonl):
        with open(progress_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                idx = o.get("index")
                if idx is None:
                    continue
                if o.get("verdict") != "yes":
                    continue
                by_idx[int(idx)] = o

    n = 0
    os.makedirs(os.path.dirname(yes_jsonl) or ".", exist_ok=True)
    with open(yes_jsonl, "w", encoding="utf-8") as out:
        for idx in sorted(by_idx.keys()):
            if idx < 0 or idx >= len(input_lines):
                continue
            raw = input_lines[idx].strip()
            if not raw:
                continue
            p = by_idx[idx]
            sample = json.loads(raw)
            if strip_eval_keys:
                row = {k: v for k, v in sample.items() if not str(k).startswith("_")}
            else:
                row = {**sample, "_identity_eval": "yes"}
                r = p.get("reason")
                if r:
                    row["_identity_eval_reason"] = str(r)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"Rebuilt yes-only JSONL → {yes_jsonl} ({n:,} rows)")
    return n


def run_identity_eval_chunked(
    input_jsonl: str,
    *,
    progress_jsonl: str,
    output_scored_jsonl: str,
    batch_id_path: str,
    yes_only_jsonl: str | None = None,
    strip_yes_only_meta: bool = False,
    include_reason: bool = False,
    max_tokens: int | None = None,
    model: str = "claude-haiku-4-5",
    chunk_size: int = 200,
    max_chunks: int | None = None,
    poll_interval: int = 60,
) -> None:
    """Submit evaluation in chunks; skip indices already in ``progress_jsonl``.

    Progress is appended only after each batch **ends** and only for rows with a non-error verdict.
    ``output_scored_jsonl`` is fully rewritten from ``progress_jsonl`` after each chunk.
    """
    with open(input_jsonl, encoding="utf-8") as f:
        input_lines = f.readlines()

    n_lines = len(input_lines)
    done = load_identity_eval_done_indices(progress_jsonl)
    pending = [
        i
        for i in range(n_lines)
        if i not in done and (input_lines[i] or "").strip()
    ]
    print(
        f"Total non-empty lines: {len(pending) + len(done):,} | "
        f"already done: {len(done):,} | pending: {len(pending):,}"
    )
    if not pending:
        print("Nothing to submit — all indices recorded in progress (non-error).")
        rebuild_scored_jsonl_from_progress(
            input_lines,
            progress_jsonl,
            output_scored_jsonl,
            include_reason=include_reason,
        )
        if yes_only_jsonl:
            rebuild_yes_only_jsonl_from_progress(
                input_lines,
                progress_jsonl,
                yes_only_jsonl,
                strip_eval_keys=strip_yes_only_meta,
            )
        return

    meta_path = _eval_meta_path(batch_id_path)
    mt = max_tokens if max_tokens is not None else (2048 if include_reason else 128)
    meta_path.write_text(
        json.dumps(
            {"include_reason": include_reason, "max_tokens": mt, "model": model},
            indent=2,
        ),
        encoding="utf-8",
    )

    chunk_i = 0
    while pending:
        if max_chunks is not None and chunk_i >= max_chunks:
            print(f"Stopped after max_chunks={max_chunks} (more pending: {len(pending):,}).")
            break

        chunk_indices = pending[:chunk_size]
        requests = []
        for i in chunk_indices:
            sample = json.loads(input_lines[i].strip())
            sample_text = format_sample_for_eval(sample)
            user_content = build_identity_eval_prompt(
                sample_text, include_reason=include_reason
            )
            requests.append(
                {
                    "custom_id": f"sample-{i}",
                    "params": {
                        "model": model,
                        "max_tokens": mt,
                        "messages": [{"role": "user", "content": user_content}],
                    },
                }
            )

        batch = client.messages.batches.create(requests=requests)
        Path(batch_id_path).write_text(batch.id, encoding="utf-8")
        print(
            f"[chunk {chunk_i + 1}] Submitted {len(requests)} requests. batch_id={batch.id} "
            f"(indices {chunk_indices[0]}…{chunk_indices[-1]})"
        )

        poll_batch(batch.id, poll_interval=poll_interval)

        inc, n_scored = _flush_batch_to_progress_and_rebuild_scored(
            batch.id,
            input_lines,
            progress_jsonl=progress_jsonl,
            output_scored_jsonl=output_scored_jsonl,
            include_reason=include_reason,
            batch_meta_path=str(meta_path),
        )
        if yes_only_jsonl:
            rebuild_yes_only_jsonl_from_progress(
                input_lines,
                progress_jsonl,
                yes_only_jsonl,
                strip_eval_keys=strip_yes_only_meta,
            )
        print(
            f"[chunk {chunk_i + 1}] Recorded {inc} new verdict(s) to progress; "
            f"scored file has {n_scored:,} row(s)."
        )

        done = load_identity_eval_done_indices(progress_jsonl)
        pending = [
            i
            for i in range(n_lines)
            if i not in done and (input_lines[i] or "").strip()
        ]
        chunk_i += 1
        print(f"Remaining pending: {len(pending):,}")

    if not pending:
        print("All pending indices processed (or only errors left to retry).")


DIRECT_EVAL_BATCH_TAG = "messages-api"


def _anthropic_messages_create_text(
    *,
    model: str,
    max_tokens: int,
    user_content: str,
) -> str:
    """Single Messages API call; retries on rate limits / transient errors."""
    delay = 1.0
    max_attempts = 12
    for attempt in range(max_attempts):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": user_content}],
            )
            parts: list[str] = []
            for block in msg.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                else:
                    parts.append(str(block))
            return "".join(parts).strip()
        except Exception as e:
            sc = getattr(e, "status_code", None)
            err = str(e).lower()
            retriable = sc in (429, 500, 502, 503, 529) or any(
                x in err for x in ("rate", "overloaded", "timeout", "connection", "temporar")
            )
            if attempt < max_attempts - 1 and retriable:
                print(
                    f"[direct] retry in {delay:.1f}s after {e!r} (attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(delay)
                delay = min(120.0, delay * 1.6)
                continue
            raise


def run_identity_eval_chunked_direct(
    input_jsonl: str,
    *,
    progress_jsonl: str,
    output_scored_jsonl: str,
    batch_id_path: str,
    yes_only_jsonl: str | None = None,
    strip_yes_only_meta: bool = False,
    include_reason: bool = False,
    max_tokens: int | None = None,
    model: str = "claude-haiku-4-5",
    chunk_size: int = 1000,
    max_chunks: int | None = None,
    max_workers: int = 8,
) -> None:
    """Same progress/scored layout as ``run_identity_eval_chunked``, but uses Messages API (no Batch).

    Higher cost than Batch; use when batch queue latency is unacceptable. Concurrent workers
    speed up throughput; tune ``max_workers`` to stay under org rate limits.
    """
    with open(input_jsonl, encoding="utf-8") as f:
        input_lines = f.readlines()

    n_lines = len(input_lines)
    done = load_identity_eval_done_indices(progress_jsonl)
    pending = [
        i
        for i in range(n_lines)
        if i not in done and (input_lines[i] or "").strip()
    ]
    print(
        f"[direct] Total non-empty lines: {len(pending) + len(done):,} | "
        f"already done: {len(done):,} | pending: {len(pending):,} | workers={max_workers}"
    )
    if not pending:
        print("Nothing to run — all indices recorded in progress (non-error).")
        rebuild_scored_jsonl_from_progress(
            input_lines,
            progress_jsonl,
            output_scored_jsonl,
            include_reason=include_reason,
        )
        if yes_only_jsonl:
            rebuild_yes_only_jsonl_from_progress(
                input_lines,
                progress_jsonl,
                yes_only_jsonl,
                strip_eval_keys=strip_yes_only_meta,
            )
        return

    meta_path = _eval_meta_path(batch_id_path)
    mt = max_tokens if max_tokens is not None else (2048 if include_reason else 128)
    meta_path.write_text(
        json.dumps(
            {"include_reason": include_reason, "max_tokens": mt, "model": model, "mode": "direct"},
            indent=2,
        ),
        encoding="utf-8",
    )
    Path(batch_id_path).write_text(DIRECT_EVAL_BATCH_TAG, encoding="utf-8")

    def eval_index(idx: int) -> tuple[int, str, str | None]:
        try:
            sample = json.loads(input_lines[idx].strip())
            sample_text = format_sample_for_eval(sample)
            user_content = build_identity_eval_prompt(
                sample_text, include_reason=include_reason
            )
            raw_text = _anthropic_messages_create_text(
                model=model, max_tokens=mt, user_content=user_content
            )
            verdict, reason = parse_identity_eval_response(
                raw_text, include_reason=include_reason
            )
            return idx, verdict, reason
        except Exception as e:
            print(f"[direct] index {idx} error: {e!r}")
            return idx, "error", None

    progress_lock = threading.Lock()
    chunk_i = 0
    while pending:
        if max_chunks is not None and chunk_i >= max_chunks:
            print(f"Stopped after max_chunks={max_chunks} (more pending: {len(pending):,}).")
            break

        chunk_indices = pending[:chunk_size]
        print(
            f"[direct chunk {chunk_i + 1}] {len(chunk_indices)} request(s) "
            f"(indices {chunk_indices[0]}…{chunk_indices[-1]})"
        )

        appended = 0
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
            futures = {ex.submit(eval_index, i): i for i in chunk_indices}
            for fut in as_completed(futures):
                idx, verdict, reason = fut.result()
                if verdict == "error":
                    continue
                rec = {
                    "index": idx,
                    "verdict": verdict,
                    "batch_id": DIRECT_EVAL_BATCH_TAG,
                }
                if include_reason and reason is not None:
                    rec["reason"] = reason
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                with progress_lock:
                    with open(progress_jsonl, "a", encoding="utf-8") as prog:
                        prog.write(line)
                appended += 1

        n_scored = rebuild_scored_jsonl_from_progress(
            input_lines,
            progress_jsonl,
            output_scored_jsonl,
            include_reason=include_reason,
        )
        if yes_only_jsonl:
            rebuild_yes_only_jsonl_from_progress(
                input_lines,
                progress_jsonl,
                yes_only_jsonl,
                strip_eval_keys=strip_yes_only_meta,
            )
        print(
            f"[direct chunk {chunk_i + 1}] Appended {appended} verdict(s); "
            f"scored file has {n_scored:,} row(s)."
        )

        done = load_identity_eval_done_indices(progress_jsonl)
        pending = [
            i
            for i in range(n_lines)
            if i not in done and (input_lines[i] or "").strip()
        ]
        chunk_i += 1
        print(f"Remaining pending: {len(pending):,}")

    if not pending:
        print("All pending indices processed (or only errors left to retry).")


TARGET_FILE = os.path.join(CATEGORY_DIR, "identity.jsonl")

print(TARGET_FILE)
