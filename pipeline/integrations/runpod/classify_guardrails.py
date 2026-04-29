"""Post-process classifier labels (e.g. drop false ``identity`` from document-QA turns)."""

from __future__ import annotations

import re

# Last user turn must look addressed to the assistant…
_BOT_ADDRESS = re.compile(
    r"<\|bot\|>|<\|assistant\|>|\b(너|당신|너희|챗봇|어시스턴트|assistant|chatbot)\b",
    re.IGNORECASE,
)
# …and probe the assistant’s nature / builder / name (not “who is X” in a reading passage).
_IDENTITY_INTENT = re.compile(
    r"\b(ai|gpt|llm|인공지능)\b|정체성|의식|감정(\s*이|\s*을)?\s*있|이름(\s*이|\s*은)?\s*뭐|"
    r"너\s*(gpt|ai)|누가\s*만들|만든\s*사람|제작(사|자)|개발사|who\s+are\s+you|your\s+name|"
    r"are\s+you\s+an?\s*ai|conscious|sentience",
    re.IGNORECASE,
)


def refine_categories_remove_false_identity(
    labs: list[str],
    *,
    sample: dict,
) -> list[str]:
    """If ``identity`` is present, keep it only when the last user turn targets the bot’s identity."""
    if "identity" not in labs:
        return labs

    messages = sample.get("messages") or []
    last_user = ""
    for m in messages:
        if (m.get("role") or "").strip() != "user":
            continue
        raw = m.get("content", "")
        last_user = raw if isinstance(raw, str) else str(raw)
    text = last_user.strip()
    if not text:
        return _drop_identity(labs)

    if _BOT_ADDRESS.search(text) and _IDENTITY_INTENT.search(text):
        return labs
    return _drop_identity(labs)


def _drop_identity(labs: list[str]) -> list[str]:
    order = ("identity", "values", "persona", "general")
    seen: set[str] = set()
    out: list[str] = []
    for c in order:
        if c in labs and c != "identity" and c not in seen:
            seen.add(c)
            out.append(c)
    return out if out else ["general"]
