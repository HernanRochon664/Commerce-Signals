"""Pipeline definition for the `feature_engineering` stage (M0 stub)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.feature_engineering.nodes import (
    build_customer_features,
    split_train_test,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=build_customer_features,
                inputs=["clean_transactions", "params:churn.threshold_days"],
                outputs="customer_features",
                name="build_customer_features",
                tags=["m3-stub"],
            ),
            node(
                func=split_train_test,
                inputs={
                    "customer_features": "customer_features",
                    "train_cutoff": "params:feature_engineering.train_cutoff",
                },
                outputs={
                    "X_train": "X_train",
                    "y_train": "y_train",
                    "X_test": "X_test",
                    "y_test": "y_test",
                },
                name="split_train_test",
                tags=["m3-stub"],
            ),
        ],
    )
