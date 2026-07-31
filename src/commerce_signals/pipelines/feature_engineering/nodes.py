"""Node functions for the `feature_engineering` pipeline (M0 stub)."""

from __future__ import annotations


def build_customer_features(
    clean_transactions,  # noqa: ARG001
    churn_threshold_days: int,  # noqa: ARG001
) -> None:
    """Compute per-customer features (RFM + behaviour) and churn label.

    Args:
        clean_transactions: DataFrame-style object from the
            ``clean_transactions`` MongoDB collection.
        churn_threshold_days: Days without a purchase after which a
            customer is labelled as churned (from
            ``params:churn.threshold_days``).

    Returns:
        A feature table (one row per customer) saved via the
        ``customer_features`` output dataset.

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M3")


def split_train_test(
    customer_features,  # noqa: ARG001
    train_cutoff,  # noqa: ARG001
) -> None:
    """Split features into train/test sets using a temporal cutoff.

    Args:
        customer_features: Per-customer features (point-in-time safe).
        train_cutoff: The temporal cutoff date (from
            ``params:feature_engineering.train_cutoff``).

    Returns:
        Two datasets, ``X_train`` and ``X_test`` (model input).

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M3")
