"""Load sources.yaml and optional .env for auth (HF_TOKEN, GITHUB_TOKEN, etc.)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES_PATH = REPO_ROOT / "sources.yaml"


def load_sources(path: Path | None = None) -> dict[str, Any]:
    """Parse sources.yaml and, if auth.dotenv_path is set, load that file into os.environ."""
    filepath = path or SOURCES_PATH
    with filepath.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    auth = data.get("auth") or {}
    dotenv_rel = auth.get("dotenv_path")
    if dotenv_rel:
        raw = Path(dotenv_rel)
        env_file = raw if raw.is_absolute() else (REPO_ROOT / raw).resolve()
        load_dotenv(env_file)

    return data


def token_from_sources(data: dict[str, Any], key: str) -> str | None:
    """Read token using env var name from data['auth'][key] (e.g. hf_token_env)."""
    env_name = (data.get("auth") or {}).get(key)
    if not env_name:
        return None
    return os.environ.get(env_name)
