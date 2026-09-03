"""Node functions for the `inference` pipeline (M6).

Batch utility: predict for a single customer given an already-loaded
transactions DataFrame. The live HTTP path lives in app/main.py; this
module stays pure (no Mongo, no MLflow) so it is testable without
infrastructure.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd

from commerce_signals.features import compute_customer_features


def predict_customer_churn(
    customer_id: str,
    clean_transactions: pd.DataFrame,
    trained_model: lgb.LGBMClassifier,
    threshold: float,
) -> dict[str, Any]:
    """Predict churn for ``customer_id`` from ``clean_transactions``.

    Filters ``clean_transactions`` to the requested customer, computes
    features via ``compute_customer_features`` (single source of truth
    with training), scores with ``trained_model``, and thresholds.

    Args:
        customer_id: Customer to score.
        clean_transactions: Full clean transactions table (already in
            memory; filtering by customer is this function's job).
        trained_model: Fitted LightGBM classifier.
        threshold: Probability threshold from training_report.

    Returns:
        Dict with ``customer_id``, ``churn_probability``, ``is_at_risk``,
        ``as_of_date`` (isoformat), or ``error`` if the customer has no
        real purchase history. Never raises for missing customer so a
        batch loop over many ids does not crash on one bad id.
    """
    filtered = clean_transactions[clean_transactions["customer_id"] == str(customer_id)]

    if filtered.empty:
        return {"error": "customer not found or no purchase history"}

    as_of_date = pd.Timestamp.now()
    features = compute_customer_features(filtered, as_of_date=as_of_date)

    if features.empty:
        return {"error": "customer not found or no purchase history"}

    # Single-row frame for this customer; compute_customer_features returns
    # one row per customer in filtered population (here: at most 1).
    # Ensure we use that single row.
    # Features index is customer_id; retrieve row.
    if str(customer_id) not in features.index:
        return {"error": "customer not found or no purchase history"}

    X = features.loc[[str(customer_id)]]

    proba = float(trained_model.predict_proba(X)[:, 1][0])
    is_at_risk = bool(proba >= float(threshold))

    return {
        "customer_id": str(customer_id),
        "churn_probability": float(proba),
        "is_at_risk": bool(is_at_risk),
        "as_of_date": as_of_date.isoformat(),
    }
