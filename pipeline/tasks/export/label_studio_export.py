"""Convert Label Studio JSON exports into training JSONL (messages-first)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _messages_from_task_data(data: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return OpenAI-style messages if present or derivable from ``data``."""
    if not isinstance(data, dict):
        return None
    m = data.get("messages")
    if isinstance(m, list) and m and all(isinstance(x, dict) for x in m):
        out: list[dict[str, Any]] = []
        for x in m:
            role = str(x.get("role", "")).strip()
            content = x.get("content", "")
            if not role:
                continue
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            out.append({"role": role, "content": content})
        return out or None
    conv = data.get("conversation") or data.get("dialogue")
    if isinstance(conv, list) and conv:
        return _messages_from_task_data({"messages": conv})
    text = data.get("text") or data.get("utterance")
    if isinstance(text, str) and text.strip():
        return [{"role": "user", "content": text.strip()}]
    return None


def export_label_studio_tasks_to_jsonl(
    input_path: Path,
    output_jsonl: Path,
    *,
    field: str = "messages",
) -> int:
    """Read a Label Studio ``export.json`` (list of tasks) and write one JSON object per line.

    Each output row is ``{field: [...messages...], "meta": {...}}`` when messages exist.
    Rows that cannot be converted are skipped with a log line.

    Returns the number of lines written.
    """
    raw = input_path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of tasks, got {type(data).__name__}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_jsonl.open("w", encoding="utf-8") as wf:
        for idx, task in enumerate(data):
            if not isinstance(task, dict):
                continue
            td = task.get("data")
            if not isinstance(td, dict):
                logger.warning("skip task %s: missing data object", idx)
                continue
            messages = _messages_from_task_data(td)
            if not messages:
                logger.warning("skip task %s: no messages derived from data keys %s", idx, list(td.keys())[:12])
                continue
            row: dict[str, Any] = {
                field: messages,
                "meta": {
                    "ls_task_id": task.get("id"),
                    "annotations": len(task.get("annotations") or []),
                },
            }
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %s rows -> %s", n, output_jsonl)
    return n


def export_path_to_dir(input_path: Path, output_dir: Path) -> list[Path]:
    """If ``input_path`` is a JSON array export, write ``output_dir/messages.jsonl``.

    If ``input_path`` is already ``.jsonl``, copy lines to ``output_dir/messages.jsonl``
    (same schema when rows contain ``messages``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "messages.jsonl"
    if input_path.suffix.lower() == ".jsonl":
        text = input_path.read_text(encoding="utf-8")
        out_file.write_text(text, encoding="utf-8")
        return [out_file]
    n = export_label_studio_tasks_to_jsonl(input_path, out_file)
    if n == 0:
        logger.warning("No rows exported from %s", input_path)
    return [out_file]
