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


def _git_lfs_unavailable(stderr_stdout: str) -> bool:
    """True when git failed because the LFS extension/binary is missing or broken."""
    low = (stderr_stdout or "").lower()
    return (
        "not a git command" in low
        or ("git-lfs" in low and "not found" in low)
        or "git-lfs is broken" in low
        or ("unable to execute" in low and "lfs" in low)
        or ("appears to be a git command" in low and "lfs" in low)
    )


_LFS_REMEDIATION = (
    "Install Git LFS (https://git-lfs.com), run `git lfs install`, then `git -C <repo> lfs pull`. "
    "Without it, Parquet/JSON in clones are LFS pointer stubs and refine fails "
    "(JSONDecodeError, Parquet magic bytes not found)."
)

# Large HF dataset repos can take a long time to download LFS objects.
_GIT_LFS_PULL_TIMEOUT_SEC = 14_400


def _any_clone_dataset_uses_git_lfs(data: dict[str, Any]) -> bool:
    """True if any hf or github *repo* clone entry will run Git LFS (per-dataset or default)."""
    defaults = data.get("defaults") or {}
    default_lfs = bool(defaults.get("git_lfs", True))
    for ds in data.get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        dtype = (ds.get("type") or "").strip().lower()
        use_lfs = bool(ds["git_lfs"]) if "git_lfs" in ds else default_lfs
        if not use_lfs:
            continue
        if dtype == "hf":
            return True
        if dtype == "github" and _is_github_repo_clone_url(str(ds.get("url") or "")):
            return True
    return False


def assert_git_lfs_for_collect(data: dict[str, Any]) -> None:
    """Fail fast before cloning when ``sources.yaml`` needs Git LFS but it is not usable."""
    if not _any_clone_dataset_uses_git_lfs(data):
        return
    try:
        r = subprocess.run(
            ["git", "lfs", "version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "`git` is not on PATH; cannot run Git LFS. " + _LFS_REMEDIATION
        ) from e
    out = ((r.stderr or "") + (r.stdout or "")).strip()
    if r.returncode != 0 or _git_lfs_unavailable(out):
        raise RuntimeError(
            "This collect needs Git LFS (HF/GitHub dataset clones) but `git lfs version` failed. "
            f"{_LFS_REMEDIATION} Raw output: {out or r.returncode}"
        )
    logger.info("Git LFS available: %s", out.splitlines()[0] if out else "ok")


def _git_lfs_pull(dest: Path, *, enabled: bool = True) -> None:
    """Fetch and checkout Git LFS blobs for ``dest``.

    Raises ``RuntimeError`` when ``enabled`` but LFS cannot run or ``git lfs pull`` fails,
    so collect does not finish with pointer-only files.

    No-op if ``enabled`` is false or ``dest`` is not a git repo.
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
    except FileNotFoundError as e:
        raise RuntimeError(f"git not on PATH; cannot run LFS for {dest}") from e
    err1 = ((r1.stderr or "") + (r1.stdout or "")).strip()
    if r1.returncode != 0:
        if _git_lfs_unavailable(err1):
            raise RuntimeError(
                f"git-lfs missing or broken for {dest}. {_LFS_REMEDIATION} Raw error: {err1 or r1.returncode}"
            )
        logger.warning("git lfs install --local non-zero for %s (continuing pull): %s", dest, err1 or r1.returncode)

    try:
        r2 = subprocess.run(
            ["git", "-C", str(dest), "lfs", "pull"],
            capture_output=True,
            text=True,
            timeout=_GIT_LFS_PULL_TIMEOUT_SEC,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"git not on PATH during lfs pull for {dest}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"git lfs pull timed out after {_GIT_LFS_PULL_TIMEOUT_SEC}s for {dest}. "
            "Retry collect or run `git lfs pull` manually in that directory."
        ) from e
    err2 = ((r2.stderr or "") + (r2.stdout or "")).strip()
    if r2.returncode != 0:
        if _git_lfs_unavailable(err2):
            raise RuntimeError(
                f"git lfs pull failed (LFS unavailable) for {dest}. {_LFS_REMEDIATION} Raw error: {err2 or r2.returncode}"
            )
        raise RuntimeError(
            f"git lfs pull failed for {dest} (exit {r2.returncode}). "
            f"Check network, credentials (HF_TOKEN / GITHUB_TOKEN), and disk space. Output:\n{err2}"
        )
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


