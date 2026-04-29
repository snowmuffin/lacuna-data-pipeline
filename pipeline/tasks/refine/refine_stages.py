"""Orchestrate refine filter stages [1]–[9] (raw → splits; upload excluded)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pipeline.sources_config import REPO_ROOT
from pipeline.tasks.refine.refine_filter_steps import (
    collect_all_samples,
    duplication_filter,
    filter_empty_turns,
    split_dataset,
    sync_message_format,
    sync_multi_turn_files,
    sync_single_turn_files,
    sync_single_turn_to_multi_turn,
)


@dataclass(frozen=True)
class RefinePaths:
    raw_dir: Path
    jsonl_dir: Path
    multi_turn_dir: Path
    single_turn_dir: Path
    final_dir: Path


def default_refine_paths(
    raw_root: str | Path,
    *,
    refined_root: str | Path | None = None,
) -> RefinePaths:
    """Resolve paths for refine stages [1]–[9].

    ``raw_root`` is the directory of per-dataset raw clones (same as collect output).

    ``refined_root`` is the parent directory for ``jsonl/``, ``multi_turn/``,
    ``single_turn/``, and ``final/``. When omitted, defaults to
    ``<REPO_ROOT>/data/refined`` so outputs stay under the repo regardless of where
    raw data lives.
    """
    raw = Path(raw_root).resolve()
    if refined_root is None:
        refined = (REPO_ROOT / "data" / "refined").resolve()
    else:
        p = Path(refined_root)
        refined = p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()
    return RefinePaths(
        raw_dir=raw,
        jsonl_dir=(refined / "jsonl").resolve(),
        multi_turn_dir=(refined / "multi_turn").resolve(),
        single_turn_dir=(refined / "single_turn").resolve(),
        final_dir=(refined / "final").resolve(),
    )


def remove_endwith_suffixes(remove_list: list[str], dir_path: str | Path) -> None:
    """Delete files under ``dir_path`` whose names end with any suffix in ``remove_list``."""
    dp = str(dir_path)
    count = 0
    for root, _dirs, files in os.walk(dp):
        for file in files:
            if file.endswith(tuple(remove_list)):
                os.remove(os.path.join(root, file))
                count += 1
    print(f"total files removed: {count}")


def run_refine_filter_stages(paths: RefinePaths) -> tuple[str, str]:
    """Run filter stages [1]–[9]; returns ``(train_path, test_path)``."""
    paths.jsonl_dir.mkdir(parents=True, exist_ok=True)
    paths.multi_turn_dir.mkdir(parents=True, exist_ok=True)
    paths.single_turn_dir.mkdir(parents=True, exist_ok=True)
    paths.final_dir.mkdir(parents=True, exist_ok=True)

    print("[1] convert raw → jsonl …")
    from pipeline.tasks.refine.post_processing import filter_data

    filter_data(str(paths.raw_dir), str(paths.jsonl_dir))
    remove_endwith_suffixes(["-res.jsonl", "config.jsonl", "_infos.jsonl"], paths.jsonl_dir)

    print("[2] sync multi_turn files …")
    sync_multi_turn_files(
        str(paths.jsonl_dir),
        str(paths.multi_turn_dir),
        ["conversation", "conversations", "text"],
        ["id", "source"],
        "messages",
    )

    print("[3] normalize multi_turn message format …")
    sync_message_format(
        str(paths.multi_turn_dir),
        str(paths.multi_turn_dir),
        "messages",
        ["from"],
        [],
        "role",
    )
    sync_message_format(
        str(paths.multi_turn_dir),
        str(paths.multi_turn_dir),
        "messages",
        ["value"],
        [],
        "content",
    )

    print("[4] normalize single_turn files …")
    sync_single_turn_files(
        str(paths.jsonl_dir),
        str(paths.single_turn_dir),
        instruction_keys=["instruction", "question", "prompt"],
        output_keys=["output", "answer", "solution", "completion"],
    )

    print("[5] convert single_turn → multi_turn(messages) …")
    sync_single_turn_to_multi_turn(str(paths.single_turn_dir), str(paths.multi_turn_dir))

    print("[6] collect all samples → all.jsonl …")
    all_path = paths.final_dir / "all.jsonl"
    collect_all_samples(
        source_dirs=[str(paths.multi_turn_dir)],
        out_path=str(all_path),
    )

    print("[7] deduplicate …")
    dedup_path = paths.final_dir / "all_unique.jsonl"
    duplication_filter(str(all_path), str(dedup_path))

    print("[8] filter empty turns …")
    filtered_path = paths.final_dir / "all_filtered.jsonl"
    filter_empty_turns(str(dedup_path), str(filtered_path), min_turns=1)

    print("[9] split into train/test …")
    train_path, test_path = split_dataset(
        input_path=str(filtered_path),
        out_dir=str(paths.final_dir),
        train_ratio=0.98,
        seed=42,
    )
    return train_path, test_path
