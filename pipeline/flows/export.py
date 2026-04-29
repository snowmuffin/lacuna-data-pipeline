"""Export: Label Studio annotations → training JSONL under ``output_dir``."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from prefect import flow, task

from pipeline.sources_config import REPO_ROOT

logger = logging.getLogger(__name__)


@task(name="label-studio-export", log_prints=True)
def label_studio_export_task(input_path: str, output_dir: str) -> list[str]:
    """Write ``messages.jsonl`` (and pass-through ``.jsonl``) into ``output_dir``."""
    from pipeline.tasks.export.label_studio_export import export_path_to_dir

    paths = export_path_to_dir(Path(input_path), Path(output_dir))
    return [str(p) for p in paths]


@flow(name="export", log_prints=True)
def export_flow(
    label_studio_export_path: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Convert a Label Studio ``export.json`` (or ``.jsonl``) into JSONL under ``output_dir``."""
    load_dotenv(REPO_ROOT / ".env")
    inp = Path(label_studio_export_path).resolve()
    out = Path(output_dir).resolve()
    if not inp.exists():
        raise FileNotFoundError(str(inp))
    return label_studio_export_task(str(inp), str(out))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Label Studio export.json or a .jsonl file",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write messages.jsonl",
    )
    args = p.parse_args()
    export_flow(label_studio_export_path=args.input, output_dir=args.output_dir)
