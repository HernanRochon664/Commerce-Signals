"""Node functions for the `training` pipeline (M0 stub)."""

from __future__ import annotations

from typing import Any


def train_lightgbm_model(
    X_train: Any,  # noqa: ARG001
    y_train: Any,  # noqa: ARG001
    model_params: dict[str, Any],  # noqa: ARG001
) -> None:
    """Train a LightGBM churn classifier and log to MLflow.

    Args:
        X_train: Training features.
        y_train: Training labels (binary churn indicator).
        model_params: LightGBM hyperparameters (from
            ``params:training.model_params``).

    Returns:
        The trained model, saved via the ``churn_model`` output dataset
        (and tracked in MLflow).

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M4")


def evaluate_model(
    churn_model,  # noqa: ARG001
    X_test,  # noqa: ARG001
    y_test,  # noqa: ARG001
) -> None:
    """Evaluate the trained model and report PR-AUC and recall.

    Args:
        churn_model: Trained model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        A metrics dict saved via the ``training_metrics`` output.

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M4")
