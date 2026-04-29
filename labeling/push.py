"""Send pipeline outputs to Label Studio (project-specific; not automated here)."""

from __future__ import annotations


def main() -> None:
    """Configure Label Studio URL / API key in ``.env`` and use the LS SDK or UI to import tasks.

    See ``config/label_studio.yaml`` if present. For JSONL → LS, use your project’s import format.
    """
    print(
        "labeling.push: no default automation. "
        "Import JSONL via Label Studio UI or SDK; see PIPELINE.md and config/label_studio.yaml."
    )


if __name__ == "__main__":
    main()
