"""Collect stage: clone or download sources from sources.yaml."""

from prefect import flow

from pipeline.sources_config import load_sources, token_from_sources
from pipeline.tasks.crawlers.fetch import fetch_dataset_task, resolve_output_root


@flow(name="collect", log_prints=True)
def collect_flow() -> None:
    data = load_sources()
    hf_token = token_from_sources(data, "hf_token_env")
    github_token = token_from_sources(data, "github_token_env")

    defaults = data.get("defaults") or {}
    resume = bool(defaults.get("resume", True))
    on_error = str(defaults.get("on_error", "continue")).strip().lower()
    git_lfs_default = bool(defaults.get("git_lfs", True))
    output_root = resolve_output_root(data)

    for ds in data.get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        git_lfs = bool(ds["git_lfs"]) if "git_lfs" in ds else git_lfs_default
        fetch_dataset_task(
            ds,
            output_root,
            hf_token=hf_token,
            github_token=github_token,
            resume=resume,
            on_error=on_error,
            git_lfs=git_lfs,
        )


if __name__ == "__main__":
    collect_flow()
