"""Git clone / pull for HF dataset staging dirs (before copying train/test)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_git_clone_or_pull(local_repo: Path, remote: str, *, shallow: bool = True) -> None:
    """If ``local_repo`` is already a git repo, ``git pull``; else ``git clone`` into it.

    ``remote`` should be a clone URL (e.g. ``https://huggingface.co/datasets/org/name.git``).
    When ``remote`` is empty, this function is a no-op (caller creates dirs only).
    """
    remote = remote.strip()
    if not remote:
        return

    parent = local_repo.parent
    parent.mkdir(parents=True, exist_ok=True)
    git_dir = local_repo / ".git"

    if git_dir.is_dir():
        logger.info("git pull --ff-only in %s", local_repo)
        proc = subprocess.run(
            ["git", "-C", str(local_repo), "pull", "--ff-only"],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            logger.warning("git pull failed (continuing): %s", (proc.stderr or proc.stdout or "").strip())
        return

    if local_repo.exists() and any(local_repo.iterdir()):
        raise FileExistsError(
            f"{local_repo} exists, is not a git repo, and is not empty — "
            "remove it or pick another path before clone."
        )

    cmd = ["git", "clone"]
    if shallow:
        cmd.append("--depth")
        cmd.append("1")
    cmd.extend([remote, str(local_repo)])
    logger.info("git clone -> %s", local_repo)
    subprocess.run(cmd, check=True, cwd=str(parent), capture_output=True, text=True)
