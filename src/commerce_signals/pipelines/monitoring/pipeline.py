"""Pipeline definition for the `monitoring` stage (M0 stub)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.monitoring.nodes import (
    generate_drift_report,
    publish_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the monitoring pipeline."""
    return Pipeline(
        [
            node(
                func=generate_drift_report,
                inputs=["customer_features", "predictions"],
                outputs="drift_report",
                name="generate_drift_report",
                tags=["m8-stub"],
            ),
            node(
                func=publish_metrics,
                inputs="drift_report",
                outputs=None,
                name="publish_metrics",
                tags=["m8-stub"],
            ),
        ],
    )
