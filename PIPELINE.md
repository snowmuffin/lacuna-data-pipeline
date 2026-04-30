# Lacuna data pipeline

This repo supports two **ingest** paths; pick one per project and stay consistent.

## Quick start

1. **Python env**  
   `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)  
   `pip install -r requirements.txt`  
   For **convmerge** ingest or refine paths that need it, also install convmerge (see [Dependencies](#dependencies)).

2. **Local config from templates** (paths are gitignored unless noted as “tracked template”):
   - Git collect: `cp sources.example.yaml sources.yaml` and edit (only if you use `--collect-via git` / `collect_flow`).
   - Convmerge collect: `cp config/convmerge_manifest.example.yaml config/convmerge_manifest.yaml` and edit (default manifest path for CLI and flows).
   - Secrets: `cp .env.example .env` and fill values (see [Environment](#environment)).

3. **Git LFS** if you use **git+LFS collect**: follow [Git LFS (required for many raw clones)](#git-lfs-required-for-many-raw-clones).

4. **After a history rewrite on `main`** (e.g. filter-repo), existing clones may diverge from `origin/main`. Refresh with `git fetch origin` and `git reset --hard origin/main`, or clone the repo again into a clean directory.

## Ingest (raw data)

| Path | When to use |
|------|-------------|
| **Prefect `collect_flow`** ([`pipeline/flows/collect.py`](pipeline/flows/collect.py)) + repo-root `sources.yaml` (tracked template: [`sources.example.yaml`](sources.example.yaml)) | Copy the example to `sources.yaml`, then `git clone` + Git LFS under Hugging Face dataset URLs. Fine when LFS is reliable; otherwise large Parquet rows stay as pointers. |
| **convmerge fetch** — [`scripts/fetch_datasets_convmerge.py`](scripts/fetch_datasets_convmerge.py), tracked template [`config/convmerge_manifest.example.yaml`](config/convmerge_manifest.example.yaml) (copy to **`config/convmerge_manifest.yaml`** for default CLI paths; that file is gitignored), or Prefect [`collect_convmerge_flow`](pipeline/flows/collect_convmerge.py) | HF hub / snapshot download (notebook-style `convmerge fetch`). Use for large gated Parquet datasets. Requires `pip install 'convmerge[preset,fetch-all,parquet]'`. |
| **End-to-end** | `python -m pipeline.flows.dataset_pipeline --collect-via convmerge --skip-hf-staging` runs convmerge collect then refine into `./data/raw` by default. |

Both ingest modes can target the same directory (e.g. `./data/raw`). For **large multi-million-row** HF splits, prefer **`--collect-via convmerge`** (or the script) over git+LFS collect.

## Refine → classify → HF

1. **Refine** ([`pipeline/flows/refine.py`](pipeline/flows/refine.py)): raw tree → `jsonl/` → `multi_turn/` / `single_turn/` → `final/train.jsonl` (+ test).  
   - `-i/--raw-root`: input clones.  
   - `--refined-root`: parent for `jsonl`, `multi_turn`, `single_turn`, `final` (default: `./data/refined` under repo root). Use this when raw lives outside the repo but you still want a predictable refined tree.  
   - Optional HF staging: `--hf-repo-name`, `--skip-hf-staging`, and `hf_classify_*` paths to copy RunPod classify artifacts.

2. **Classify** ([`pipeline/flows/classify.py`](pipeline/flows/classify.py)): RunPod worker HTTP batch. Env: `LACUNA_DATA_API_BASE`, `LACUNA_DATA_API_KEY`, etc.

3. **End-to-end** ([`pipeline/flows/dataset_pipeline.py`](pipeline/flows/dataset_pipeline.py)): `collect → refine → (optional) classify` in one flow. CLI: `python -m pipeline.flows.dataset_pipeline` with `--collect-via git|convmerge`, `--no-collect`, `--no-refine`, `--classify`, `--convmerge-manifest`, `--convmerge-only`, paths as flags.

## Optional Anthropic steps

[`pipeline/flows/refine_llm.py`](pipeline/flows/refine_llm.py) runs **identity batch eval** and/or **assistant rewrite+translate** on JSONL (Anthropic API, `ANTHROPIC_API_KEY` in `.env`). Not part of the default refine flow; invoke when needed.

## Export (Label Studio → JSONL)

[`pipeline/flows/export.py`](pipeline/flows/export.py) reads a Label Studio JSON export (array of tasks) or copies an existing `.jsonl`, and writes `messages.jsonl` under `-o/--output-dir`. See [`pipeline/tasks/export/label_studio_export.py`](pipeline/tasks/export/label_studio_export.py).

## Environment

- Copy [`.env.example`](.env.example) to repo-root `.env` and set values. Common keys: `HF_TOKEN`, `GITHUB_TOKEN`, `RUNPOD_API_KEY`, `LACUNA_DATA_*`, `HF_DATASET_REPO_NAME`, `ANTHROPIC_API_KEY`, etc. (see comments in `.env.example`).
- GPU worker on RunPod: [`scripts/data_worker_api.py`](scripts/data_worker_api.py), deploy [`scripts/runpod_deploy_worker.sh`](scripts/runpod_deploy_worker.sh).

### Git LFS (required for many raw clones)

Hugging Face and GitHub dataset repos often store large `.parquet` and `.json` files through **Git LFS**. **`collect` checks that `git lfs` works before cloning** (when any listed HF or GitHub *repo* clone uses LFS). After each clone or resume, it runs `git lfs install --local` and `git lfs pull` (with a long timeout); **if that fails, collect raises** instead of leaving **LFS pointer stubs** on disk. To opt out of LFS for every dataset, set `defaults.git_lfs: false` in your local `sources.yaml` (from [`sources.example.yaml`](sources.example.yaml); refine may then fail on LFS-backed files unless you use another ingest path).

1. Install the **`git-lfs` package** (the `git lfs` subcommand is provided by this binary; installing only `git` is not enough).
   - **WSL2 / Ubuntu or Debian:**  
     `sudo apt update && sudo apt install -y git-lfs`  
     then **once per user:**  
     `git lfs install`
   - **Fedora:** `sudo dnf install git-lfs && git lfs install`  
   - **macOS (Homebrew):** `brew install git-lfs && git lfs install`
2. **Check:** `git lfs version` must print a version line (not “not found” or “broken”). Open a **new terminal** if the shell was started before `apt install`.
3. For existing clones under `data/raw`, materialize blobs: run [`scripts/git_lfs_pull_raw.sh`](scripts/git_lfs_pull_raw.sh) from the repo root, or manually `git lfs pull` inside each dataset directory that contains a `.git` folder.
4. Re-run refine only if raw is already present:  
   `python -m pipeline.flows.dataset_pipeline --no-collect --skip-hf-staging`  
   Or run full `collect` again after LFS works so `git lfs pull` runs right after each clone.

Per-dataset LFS can be turned off with `git_lfs: false` on an entry (not recommended if that repo stores data in LFS). If **every** clone that would use LFS has it disabled, the preflight check is skipped.

## Dependencies

Prefect flows need `prefect`, `python-dotenv`, and (for collect/refine) `convmerge` where `filter_data` / normalize is used. The convmerge fetch script needs the convmerge package installed separately.
