"""Node functions for the `inference` pipeline (M0 stub)."""

from __future__ import annotations


def predict_for_customer(
    customer_id: str,  # noqa: ARG001
    churn_model,  # noqa: ARG001
    customer_features,  # noqa: ARG001
) -> None:
    """Predict churn probability for a single customer.

    Args:
        customer_id: The customer ID (from the FastAPI request body).
        churn_model: Trained model.
        customer_features: Feature store (Mongo collection) keyed by
            customer.

    Returns:
        A prediction object (probability + risk level + timestamp)
        saved via the ``predictions`` output dataset.

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M6")
