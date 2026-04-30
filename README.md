# Lacuna data pipeline

Prefect-backed flows for collecting conversation datasets, refining them to JSONL, optional RunPod classification, and related export paths.

**Full documentation and stage-by-stage commands:** see [PIPELINE.md](PIPELINE.md).

**Quick start:** create a virtualenv, `pip install -r requirements.txt`, copy `sources.example.yaml` → `sources.yaml` and/or `config/convmerge_manifest.example.yaml` → `config/convmerge_manifest.yaml` as needed, copy `.env.example` → `.env`, then follow the ingest and Git LFS sections in [PIPELINE.md](PIPELINE.md).
