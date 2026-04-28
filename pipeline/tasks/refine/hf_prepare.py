"""Prepare a local Hugging Face dataset folder (train/test + README); optional git sync first."""

from __future__ import annotations

import datetime
import os
import shutil
from pathlib import Path

from pipeline.tasks.refine.git_staging import ensure_git_clone_or_pull


def stage_hf_classification_artifacts_into_ds_dir(
    ds_dir: str,
    *,
    classify_state_path: str | None = None,
    classify_output_jsonl: str | None = None,
    classify_category_dir: str | None = None,
) -> list[str]:
    """Copy classification outputs under ``ds_dir/classification/``. Returns README bullet lines."""
    cls_lines: list[str] = []
    cls_root = os.path.join(ds_dir, "classification")
    for label, p in (
        ("classify_state_path", classify_state_path),
        ("classify_output_jsonl", classify_output_jsonl),
    ):
        if not p:
            continue
        if not os.path.isfile(p):
            print(f"WARNING: {label} missing on disk, skip HF staging: {p!r}")
            continue
        os.makedirs(cls_root, exist_ok=True)
        dst = os.path.join(cls_root, os.path.basename(p))
        shutil.copy2(p, dst)
        cls_lines.append(f"  - `classification/{os.path.basename(p)}`")
    if classify_category_dir:
        if not os.path.isdir(classify_category_dir):
            print(f"WARNING: classify_category_dir not a directory, skip: {classify_category_dir!r}")
        else:
            cat_dst = os.path.join(cls_root, "categories")
            os.makedirs(cat_dst, exist_ok=True)
            n = 0
            for fn in sorted(os.listdir(classify_category_dir)):
                if not fn.endswith(".jsonl"):
                    continue
                shutil.copy2(
                    os.path.join(classify_category_dir, fn),
                    os.path.join(cat_dst, fn),
                )
                n += 1
            if n:
                cls_lines.append("  - `classification/categories/*.jsonl` (per-category shards)")
            elif not cls_lines:
                print(f"WARNING: no .jsonl under classify_category_dir: {classify_category_dir!r}")
    if cls_lines:
        print(f"  Staged classification under {cls_root}/ ({len(cls_lines)} artifact group(s))")
    return cls_lines


def prepare_hf_dataset_repo(
    final_train_path: str,
    final_test_path: str,
    repo_name: str,
    hf_datasets_root: str | Path,
    namespace: str | None = None,
    task_tag: str = "text2text-generation",
    language: str = "ko",
    *,
    remote_git_url: str | None = None,
    classify_state_path: str | None = None,
    classify_output_jsonl: str | None = None,
    classify_category_dir: str | None = None,
) -> str:
    """Create ``hf_datasets_root/<repo_name>/`` with train/test (+ README). Optionally clone/pull first."""
    root = Path(hf_datasets_root)
    ds_dir = root / repo_name

    if remote_git_url:
        ensure_git_clone_or_pull(ds_dir, remote_git_url)
    ds_dir.mkdir(parents=True, exist_ok=True)

    for src, dst_name in [
        (final_train_path, "train.jsonl"),
        (final_test_path, "test.jsonl"),
    ]:
        if not os.path.exists(src):
            raise FileNotFoundError(f"missing source file: {src}")
        dst = ds_dir / dst_name
        shutil.copy2(src, dst)

    cls_lines = stage_hf_classification_artifacts_into_ds_dir(
        str(ds_dir),
        classify_state_path=classify_state_path,
        classify_output_jsonl=classify_output_jsonl,
        classify_category_dir=classify_category_dir,
    )

    cls_bullets = ""
    if cls_lines:
        cls_bullets = (
            "\n- Classification artifacts (optional; same commit as train/test):\n"
            + "\n".join(cls_lines)
            + "\n"
        )

    readme_path = ds_dir / "README.md"
    created = datetime.date.today().isoformat()
    card = f"""---
language:
  - {language}
tags:
  - {task_tag}
  - lacuna
pretty_name: Refined conversational dataset
license: cc-by-sa-4.0
---

# Dataset Card

Pipeline: lacuna-data-pipeline refine (filter stages + HF staging).

- Created: {created}
- Training split: `train.jsonl`
- Evaluation split: `test.jsonl`{cls_bullets}
Each sample follows the schema below.

```json
{{
  "messages": [
    {{"role": "system"|"user"|"assistant", "content": "..."}},
    ...
  ]
}}
```
"""

    readme_path.write_text(card, encoding="utf-8")

    print(f"HF dataset directory prepared at: {ds_dir}")
    if namespace:
        print("Suggested commands (run in shell):")
        print(f"  huggingface-cli repo create {namespace}/{repo_name} --type dataset")
        print(f"  cd {ds_dir} && huggingface-cli upload . {namespace}/{repo_name} --commit-message 'Add dataset'")
    else:
        print("Suggested commands (run in shell):")
        print(f"  huggingface-cli repo create {repo_name} --type dataset")
        print(f"  cd {ds_dir} && huggingface-cli upload . {repo_name} --commit-message 'Add dataset'")
    return str(ds_dir)
