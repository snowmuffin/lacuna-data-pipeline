"""Refine stage: cleaning / normalization (skeleton)."""

from prefect import flow


@flow(name="refine", log_prints=True)
def refine_flow() -> None:
    """Wire `pipeline.tasks.filters` tasks here."""
    print("refine: stub")


if __name__ == "__main__":
    refine_flow()
