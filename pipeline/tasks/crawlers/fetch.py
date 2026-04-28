"""Fetch datasets by `type` in sources.yaml (git clone or file download; extensible)."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from prefect import task

from pipeline.sources_config import REPO_ROOT

logger = logging.getLogger(__name__)


def slugify(label: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-]+", "_", label, flags=re.UNICODE).strip("_")
    return s[:max_len] if s else "dataset"


def resolve_output_root(data: dict[str, Any]) -> Path:
    raw = (data.get("defaults") or {}).get("output_root", "./data/raw")
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _dataset_dest_dir(output_root: Path, ds: dict[str, Any]) -> Path:
    label = ds.get("name") or ds.get("id") or "dataset"
    return output_root / slugify(str(label))


def _is_hf_host(netloc: str) -> bool:
    n = netloc.lower().split("@")[-1]  # strip userinfo if present
    return n.endswith("huggingface.co") or n == "hf.co"


def _hf_clone_url(ds: dict[str, Any]) -> str:
    url = (ds.get("url") or "").strip()
    if url and _is_hf_host(urlparse(url).netloc):
        return url.rstrip("/")
    ds_id = ds.get("id")
    if not ds_id:
        raise ValueError("HF dataset needs `id` or huggingface `url`")
    return f"https://huggingface.co/datasets/{str(ds_id).strip().rstrip('/')}"


def _hf_git_clone_url(clone_base: str, token: str | None) -> str:
    """Embed token for private HF repos (oauth2 pattern)."""
    if not token:
        return clone_base
    p = urlparse(clone_base)
    if p.scheme not in ("http", "https"):
        return clone_base
    host = p.netloc.split("@")[-1]
    path = p.path or "/"
    safe = quote(token, safe="")
    return f"{p.scheme}://oauth2:{safe}@{host}{path}"


def _is_github_repo_clone_url(url: str) -> bool:
    p = urlparse(url.strip())
    if p.scheme not in ("http", "https"):
        return False
    host = p.netloc.lower()
    # raw.githubusercontent.com contains "github.com" but is not a git remote root
    if host in ("github.com", "www.github.com"):
        parts = [x for x in p.path.strip("/").split("/") if x]
        return len(parts) >= 2
    return False


def _github_clone_url(url: str, token: str | None) -> str:
    u = url.strip().rstrip("/")
    if not token:
        return u if u.endswith(".git") else f"{u}.git"
    p = urlparse(u)
    if p.scheme not in ("http", "https") or "github.com" not in p.netloc.lower():
        return u if u.endswith(".git") else f"{u}.git"
    host = p.netloc.split("@")[-1]
    path = p.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    safe = quote(token, safe="")
    return f"{p.scheme}://x-access-token:{safe}@{host}{path}.git"


def _git_lfs_pull(dest: Path, *, enabled: bool = True) -> None:
    """Fetch Git LFS blobs for ``dest`` (HF/GitHub datasets often use LFS for parquet).

    No-op if ``enabled`` is false, ``dest`` is not a git repo, or ``git-lfs`` is missing.
    """
    if not enabled or not (dest / ".git").is_dir():
        return
    logger.info("git lfs install + pull -> %s", dest)
    try:
        r1 = subprocess.run(
            ["git", "-C", str(dest), "lfs", "install", "--local"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        logger.warning("git not on PATH; skip LFS for %s", dest)
        return
    err1 = ((r1.stderr or "") + (r1.stdout or "")).strip()
    if r1.returncode != 0:
        low = err1.lower()
        if "not a git command" in low or "git-lfs" in low and "not found" in low:
            logger.warning("git-lfs not installed; skip LFS for %s (%s)", dest, err1 or r1.returncode)
            return
        logger.warning("git lfs install --local non-zero for %s (continuing pull): %s", dest, err1 or r1.returncode)

    try:
        r2 = subprocess.run(
            ["git", "-C", str(dest), "lfs", "pull"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return
    err2 = ((r2.stderr or "") + (r2.stdout or "")).strip()
    if r2.returncode != 0:
        logger.warning("git lfs pull failed for %s: %s", dest, err2 or r2.returncode)
    else:
        logger.info("git lfs pull finished for %s", dest)


def _git_clone(url: str, dest: Path, *, shallow: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [url, str(dest)]
    logger.info("git clone -> %s", dest)
    try:
        subprocess.run(cmd, check=True, cwd=str(dest.parent), capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        detail = ((e.stderr or "") + (e.stdout or "")).strip()
        if detail:
            logger.error("git clone failed: %s", detail)
        raise


def _should_skip_clone(dest: Path, resume: bool) -> bool:
    """If resuming, skip when a previous git clone left a repo (.git present)."""
    if not resume or not dest.exists():
        return False
    return (dest / ".git").is_dir()


def _download_file(url: str, dest_file: Path, github_token: str | None) -> None:
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    headers: dict[str, str] = {"User-Agent": "lacuna-data-pipeline/1.0"}
    if github_token and "github.com" in url.lower():
        headers["Authorization"] = f"token {github_token}"
    req = Request(url, headers=headers)
    logger.info("download -> %s", dest_file)
    with urlopen(req, timeout=600) as resp:
        dest_file.write_bytes(resp.read())


def _should_skip_file(dest_file: Path, resume: bool) -> bool:
    return resume and dest_file.is_file() and dest_file.stat().st_size > 0


def fetch_hf(
    ds: dict[str, Any],
    dest: Path,
    hf_token: str | None,
    *,
    resume: bool,
    git_lfs: bool = True,
) -> Path:
    base = _hf_clone_url(ds)
    url = _hf_git_clone_url(base, hf_token)
    if _should_skip_clone(dest, resume):
        logger.info("skip (exists): %s", dest)
        _git_lfs_pull(dest, enabled=git_lfs)
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    _git_clone(url, dest)
    _git_lfs_pull(dest, enabled=git_lfs)
    return dest


def fetch_github_repo(
    ds: dict[str, Any],
    dest: Path,
    github_token: str | None,
    *,
    resume: bool,
    git_lfs: bool = True,
) -> Path:
    url = _github_clone_url(str(ds["url"]), github_token)
    if _should_skip_clone(dest, resume):
        logger.info("skip (exists): %s", dest)
        _git_lfs_pull(dest, enabled=git_lfs)
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    _git_clone(url, dest)
    _git_lfs_pull(dest, enabled=git_lfs)
    return dest


def fetch_github_file(ds: dict[str, Any], dest_dir: Path, github_token: str | None, *, resume: bool) -> Path:
    url = str(ds["url"])
    name = Path(urlparse(url).path).name or "download"
    dest_file = dest_dir / name
    if _should_skip_file(dest_file, resume):
        logger.info("skip (exists): %s", dest_file)
        return dest_file
    dest_dir.mkdir(parents=True, exist_ok=True)
    _download_file(url, dest_file, github_token)
    return dest_file


@task(name="fetch-dataset", retries=0, log_prints=True)
def fetch_dataset_task(
    ds: dict[str, Any],
    output_root: Path,
    *,
    hf_token: str | None,
    github_token: str | None,
    resume: bool,
    on_error: str,
    git_lfs: bool = True,
) -> Path | None:
    """Clone or download one dataset entry; returns path or None if skipped on error."""
    dtype = (ds.get("type") or "").strip().lower()
    dest_dir = _dataset_dest_dir(output_root, ds)
    try:
        if dtype == "hf":
            return fetch_hf(ds, dest_dir, hf_token, resume=resume, git_lfs=git_lfs)
        if dtype == "github":
            url = str(ds.get("url") or "")
            if _is_github_repo_clone_url(url):
                return fetch_github_repo(ds, dest_dir, github_token, resume=resume, git_lfs=git_lfs)
            return fetch_github_file(ds, dest_dir, github_token, resume=resume)
        logger.warning("unknown dataset type %r for %s — skipped", dtype, ds.get("name"))
        return None
    except (subprocess.CalledProcessError, HTTPError, URLError, OSError, ValueError) as e:
        msg = f"{ds.get('name')!r} ({dtype}): {e}"
        if on_error == "continue":
            logger.exception("collect error (continue): %s", msg)
            return None
        raise RuntimeError(msg) from e


