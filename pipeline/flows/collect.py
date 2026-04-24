"""Collect stage: raw ingestion (skeleton)."""

from prefect import flow


@flow(name="collect", log_prints=True)
def collect_flow() -> None:
    """Wire `pipeline.tasks.crawlers` tasks here."""
    print("collect: stub")


if __name__ == "__main__":
    collect_flow()
