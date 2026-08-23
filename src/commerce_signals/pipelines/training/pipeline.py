"""Pipeline definition for the `training` stage (M4)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.training.nodes import (
    evaluate_churn_model,
    log_training_run_and_build_report,
    split_snapshots_temporally,
    train_churn_model,
    verify_training_run_is_retrievable,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the training pipeline."""
    return Pipeline(
        [
            node(
                func=split_snapshots_temporally,
                inputs=[
                    "customer_snapshots",
                    "params:training.test_snapshots_count",
                    "params:training.val_snapshots_count",
                ],
                outputs=["train_snapshots", "val_snapshots", "test_snapshots", "split_info"],
                name="split_snapshots_temporally",
            ),
            node(
                func=train_churn_model,
                inputs=[
                    "train_snapshots",
                    "val_snapshots",
                    "params:training.model_params",
                    "params:training.eval_metric",
                    "params:training.early_stopping_rounds",
                ],
                outputs="trained_model",
                name="train_churn_model",
            ),
            node(
                func=evaluate_churn_model,
                inputs=[
                    "trained_model",
                    "val_snapshots",
                    "test_snapshots",
                    "params:training.precision_targets",
                ],
                outputs="evaluation_metrics",
                name="evaluate_churn_model",
            ),
            node(
                func=log_training_run_and_build_report,
                inputs=[
                    "trained_model",
                    "params:training.model_params",
                    "params:training.eval_metric",
                    "params:training.early_stopping_rounds",
                    "split_info",
                    "evaluation_metrics",
                    "params:training.mlflow_experiment_name",
                ],
                outputs="training_report",
                name="log_training_run_and_build_report",
            ),
            node(
                func=verify_training_run_is_retrievable,
                inputs="training_report",
                outputs=None,
                name="verify_training_run_is_retrievable",
            ),
        ],
    )
