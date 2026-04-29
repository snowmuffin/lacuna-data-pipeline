"""
GPU data worker API (RunPod): conversation sample classification.

Translation (en↔ko) is not loaded on this worker; /translate returns 501.

Env:
  LACUNA_DATA_API_KEY   If set, require X-API-Key header on mutating routes.
  CLASSIFY_MODEL       Default: Qwen/Qwen2.5-7B-Instruct
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN  Optional; passed to Hugging Face Hub downloads.
  PRELOAD_CLASSIFY     If "1", load classify model at startup (else lazy on first /classify).
  LACUNA_DATA_VERBOSE_ERRORS  If "1"/"true", include traceback in JSON ``detail`` on HTTP 500.
  CUDA_LAUNCH_BLOCKING  If unset, defaults to ``1`` (sync CUDA errors; clearer stacks, slower).
                        Set ``0`` in ``.env`` or the shell before starting uvicorn to disable.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]


def _load_repo_dotenv() -> None:
    """Load repository root ``.env`` so HF_TOKEN etc. apply when uvicorn cwd is ``scripts/``."""
    if load_dotenv is None:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


_load_repo_dotenv()
# Must run before ``import torch`` so CUDA sees synchronous launch (better assert stacks).
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

VALID_CATEGORIES = frozenset({"identity", "values", "persona", "general"})
# Stable order for serialization and merging multi-label outputs
_CATEGORY_ORDER = ("identity", "values", "persona", "general")

_lock = threading.Lock()
_classify_tokenizer: Any = None
_classify_model: Any = None

CLASSIFY_SYSTEM_PROMPT = (
    "You label SFT chat samples for downstream dataset mixing. "
    "Use the user–assistant turns (and system prompt if present). Korean and English both appear.\n\n"
    "Each label means:\n"
    "- identity: The assistant model itself — its name, whether it is an AI, feelings or "
    "consciousness claims, self-intro, or who built it. NOT the human user's personal life story "
    "unless the user is clearly probing the bot's identity.\n"
    "- values: Ethics, safety boundaries, refusal, disagreement, persuasion to do something "
    "harmful, or strong normative judgment.\n"
    "- persona: Assistant or user speaking style, tone, register, role-play character, "
    "emotional warmth, small talk, banter, venting, or human-centric chit-chat / personal narrative "
    "framing (how things are said or related, not factual task content).\n"
    "- general: Factual Q&A, analysis, coding, math, translation, tools, structured tasks "
    "without emphasis on the axes above.\n\n"
    "A single sample MAY match more than one label. Output ALL that clearly apply, as a "
    "comma-separated list with no spaces, using only these tokens: identity,values,persona,general\n"
    "Order does not matter; omit labels that do not apply. If nothing fits, output general.\n\n"
    "Examples:\n"
    "- \"What's your name?\" → identity\n"
    "- \"Refuse to help with hacking\" → values\n"
    "- \"Cheer me up, I'm sad\" → persona\n"
    "- \"Implement quicksort in Python\" → general\n"
    "- \"너 GPT야? 오늘 기분 안 좋아\" → identity,persona\n"
    "- \"Illegal request + explain why in a firm tone\" → values,persona"
)

logger = logging.getLogger(__name__)


def _verbose_http_errors() -> bool:
    return os.environ.get("LACUNA_DATA_VERBOSE_ERRORS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _classify_http_exception(exc: BaseException, *, item_index: int | None = None) -> HTTPException:
    """Build HTTP 500 with JSON ``detail`` so clients (and logs) see the real failure."""
    suffix = f" at batch item_index={item_index}" if item_index is not None else ""
    logger.error("Classification failed%s: %s", suffix, exc, exc_info=exc)
    detail: dict[str, Any] = {
        "error": type(exc).__name__,
        "message": str(exc),
    }
    if item_index is not None:
        detail["item_index"] = item_index
    if _verbose_http_errors():
        detail["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
    return HTTPException(status_code=500, detail=detail)


def _hf_hub_token() -> str | bool | None:
    """Token for from_pretrained; None lets the hub library use its defaults."""
    t = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    ).strip()
    return t if t else None


def _load_classify_model():
    global _classify_tokenizer, _classify_model
    with _lock:
        if _classify_model is not None:
            return
        mid = (os.environ.get("CLASSIFY_MODEL") or "").strip()
        if not mid:
            mid = "Qwen/Qwen2.5-7B-Instruct"
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        tok = _hf_hub_token()
        _classify_tokenizer = AutoTokenizer.from_pretrained(
            mid, trust_remote_code=True, token=tok
        )
        _classify_model = AutoModelForCausalLM.from_pretrained(
            mid,
            torch_dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
            token=tok,
        )
        if not use_cuda:
            _classify_model = _classify_model.to("cpu")


def _map_legacy_label(t: str) -> str:
    if t == "style":
        return "persona"
    return t


def _parse_categories(text: str) -> list[str]:
    """Parse model output into an ordered list of unique valid categories."""
    if not text:
        return ["general"]
    line = text.strip().splitlines()[0].strip().lower()
    picked: list[str] = []
    for cat in _CATEGORY_ORDER:
        if re.search(rf"\b{re.escape(cat)}\b", line):
            picked.append(cat)
    if picked:
        return picked
    line = re.sub(r"^[^a-z]*", "", line)
    for sep in (",", ";"):
        if sep in line:
            parts = [_map_legacy_label(p.strip()) for p in line.split(sep) if p.strip()]
            out = [p for p in parts if p in VALID_CATEGORIES]
            if out:
                return _dedupe_ordered(out)
    word = re.split(r"[\s\.,;:]+", line)[0] if line else ""
    word = _map_legacy_label(word)
    if word in VALID_CATEGORIES:
        return [word]
    for cat in VALID_CATEGORIES:
        if re.search(rf"\b{re.escape(cat)}\b", text.lower()):
            return [cat]
    return ["general"]


def _dedupe_ordered(labels: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in _CATEGORY_ORDER:
        if c in labels and c not in seen:
            seen.add(c)
            out.append(c)
    return out if out else ["general"]


def _classify_one(user_text: str) -> list[str]:
    _load_classify_model()
    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt = _classify_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _classify_tokenizer(prompt, return_tensors="pt")
    dev = next(_classify_model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.inference_mode():
        out = _classify_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=_classify_tokenizer.eos_token_id,
        )
    start = inputs["input_ids"].shape[1]
    gen = out[0, start:]
    raw = _classify_tokenizer.decode(gen, skip_special_tokens=True)
    return _parse_categories(raw)


def _optional_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")):
    expected = os.environ.get("LACUNA_DATA_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("PRELOAD_CLASSIFY", "").strip() in ("1", "true", "yes"):
        _load_classify_model()
    yield


app = FastAPI(title="Lacuna Data Worker", lifespan=lifespan)


class TranslateItem(BaseModel):
    text: str = Field(..., min_length=1)
    src_lang: str = Field(..., pattern="^(en|ko)$")
    tgt_lang: str = Field(..., pattern="^(en|ko)$")


class TranslateSingleRequest(BaseModel):
    text: str = Field(..., min_length=1)
    src_lang: str = Field(..., pattern="^(en|ko)$")
    tgt_lang: str = Field(..., pattern="^(en|ko)$")


class TranslateBatchRequest(BaseModel):
    items: list[TranslateItem] = Field(..., min_length=1)


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ClassifyItem(BaseModel):
    text: str = Field(..., min_length=1)


class ClassifyBatchRequest(BaseModel):
    items: list[ClassifyItem] = Field(..., min_length=1)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "translation": "disabled",
        "classify_loaded": _classify_model is not None,
        "classify_labels": list(_CATEGORY_ORDER),
        "classify_multi_label": True,
    }


@app.post("/translate")
def translate(
    body: TranslateSingleRequest,
    _: None = Depends(_optional_api_key),
):
    raise HTTPException(
        status_code=501,
        detail="Translation is not enabled on this worker (classification-only).",
    )


@app.post("/translate/batch")
def translate_batch(
    body: TranslateBatchRequest,
    _: None = Depends(_optional_api_key),
):
    raise HTTPException(
        status_code=501,
        detail="Translation is not enabled on this worker (classification-only).",
    )


@app.post("/classify")
def classify(
    body: ClassifyRequest,
    _: None = Depends(_optional_api_key),
):
    try:
        cats = _classify_one(body.text)
    except Exception as e:
        raise _classify_http_exception(e) from e
    return {
        "text": body.text,
        "categories": cats,
        "category": ",".join(cats),
    }


@app.post("/classify/batch")
def classify_batch(
    body: ClassifyBatchRequest,
    _: None = Depends(_optional_api_key),
):
    results = []
    for idx, it in enumerate(body.items):
        try:
            cats = _classify_one(it.text)
        except Exception as e:
            raise _classify_http_exception(e, item_index=idx) from e
        results.append(
            {
                "text": it.text,
                "categories": cats,
                "category": ",".join(cats),
            }
        )
    return {"results": results}
