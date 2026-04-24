"""Export: Label Studio annotations → training dataset (skeleton)."""

from prefect import flow


@flow(name="export", log_prints=True)
def export_flow() -> None:
    """Use `labeling` helpers and write under `data/export/`."""
    print("export: stub")


if __name__ == "__main__":
    export_flow()
