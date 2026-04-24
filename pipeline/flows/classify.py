"""Classify stage: model inference (skeleton)."""

from prefect import flow


@flow(name="classify", log_prints=True)
def classify_flow() -> None:
    """Wire `pipeline.tasks.classifiers` tasks here."""
    print("classify: stub")


if __name__ == "__main__":
    classify_flow()
