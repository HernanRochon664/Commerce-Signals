"""Pipeline definition for the `training` stage (M0 stub)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.training.nodes import (
    evaluate_model,
    train_lightgbm_model,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the training pipeline."""
    return Pipeline(
        [
            node(
                func=train_lightgbm_model,
                inputs={
                    "X_train": "X_train",
                    "y_train": "y_train",
                    "model_params": "params:training.model_params",
                },
                outputs="churn_model",
                name="train_lightgbm_model",
                tags=["m4-stub"],
            ),
            node(
                func=evaluate_model,
                inputs=["churn_model", "X_test", "y_test"],
                outputs="training_metrics",
                name="evaluate_model",
                tags=["m4-stub"],
            ),
        ],
    )
