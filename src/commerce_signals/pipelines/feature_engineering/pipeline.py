"""Pipeline definition for the `feature_engineering` stage (M3)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.feature_engineering.nodes import (
    build_customer_snapshots,
    verify_feature_table_row_count,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=build_customer_snapshots,
                inputs=[
                    "clean_transactions",
                    "params:feature_engineering.snapshot_start_date",
                    "params:churn.threshold_days",
                ],
                outputs=["customer_snapshots", "feature_engineering_report"],
                name="build_customer_snapshots",
            ),
            node(
                func=verify_feature_table_row_count,
                inputs=["customer_snapshots", "feature_engineering_report"],
                outputs=None,
                name="verify_feature_table_row_count",
            ),
        ],
    )
