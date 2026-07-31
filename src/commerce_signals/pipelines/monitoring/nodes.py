"""Node functions for the `monitoring` pipeline (M0 stub)."""

from __future__ import annotations


def generate_drift_report(
    baseline_features,  # noqa: ARG001
    current_features,  # noqa: ARG001
) -> None:
    """Generate an Evidently drift report comparing two feature batches.

    Args:
        baseline_features: Reference feature batch (e.g. the training
            set).
        current_features: Recent feature batch (e.g. last N days of
            served traffic).

    Returns:
        A drift-report HTML saved via the ``drift_report`` output.

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M8")


def publish_metrics(
    drift_report,  # noqa: ARG001
) -> None:
    """Publish drift and operational metrics to Cloud Monitoring.

    Args:
        drift_report: Output of ``generate_drift_report``.

    Returns:
        None. Side-effect only.

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M8")
