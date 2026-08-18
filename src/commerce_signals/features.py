"""Shared, point-in-time-safe customer feature computation (M3).

This module is deliberately NOT part of any pipeline package because it
is shared between two very different runtimes:

* **Training (feature_engineering pipeline, M3/M4):** called once per
  snapshot date, for every customer at once, using the full
  ``clean_transactions`` frame batched through the pipeline.
* **Inference (FastAPI ``/predict`` endpoint, M6):** called with the
  complete history of a single ``customer_id`` and "now" as
  ``as_of_date``, right before scoring the model.

Keeping a single implementation guarantees that training and inference
compute features identically: if M6 ever computed features with its own
copy of this logic, any drift between the two implementations would
silently corrupt every prediction. Do not reimplement this in another
module.

The function is pure: a DataFrame and a Timestamp go in, a DataFrame
comes out. No logging, no Mongo, no Kedro, no side effects.
"""

from __future__ import annotations

import pandas as pd

# Feature columns, in the order they are returned. `customer_id` is the
# index, not a column.
FEATURE_COLUMNS: list[str] = [
    "recency_days",
    "frequency",
    "monetary_total",
    "monetary_avg_order",
    "customer_tenure_days",
    "n_distinct_products",
    "return_rate",
]


def compute_customer_features(
    clean_transactions: pd.DataFrame, as_of_date: pd.Timestamp
) -> pd.DataFrame:
    """Compute point-in-time-safe RFM-style features for every customer.

    Only transactions with ``invoice_date`` strictly before
    ``as_of_date`` are used, so features never leak purchases that
    happened after the moment being evaluated. Cancellations
    (``is_cancellation=True``) never count as a real purchase, either
    for the population or for the feature values.

    Population: customers with at least one real (non-cancellation)
    purchase before ``as_of_date``. A customer without one does not
    appear in the result (no zero-filled row, the row simply does not
    exist).

    Args:
        clean_transactions: ``clean_transactions`` DataFrame from the
            validation pipeline (snake_case columns + boolean flags).
        as_of_date: Snapshot moment. Only rows with
            ``invoice_date < as_of_date`` (strict) are considered.

    Returns:
        A DataFrame indexed by ``customer_id`` with one row per
        customer in the population and these columns:

        - ``recency_days``: days between ``as_of_date`` and the last
          real purchase before it.
        - ``frequency``: count of distinct real invoices (orders, not
          product lines; cancellations excluded) before ``as_of_date``.
        - ``monetary_total``: net sum of ``quantity * price`` over ALL
          lines before ``as_of_date``, cancellation lines included
          (they typically subtract, which is the point).
        - ``monetary_avg_order``: ``monetary_total / frequency``.
        - ``customer_tenure_days``: days between the first real
          purchase and ``as_of_date``.
        - ``n_distinct_products``: distinct ``stock_code`` values in
          real (non-cancelled) lines before ``as_of_date``.
        - ``return_rate``: share of lines with
          ``has_non_positive_quantity=True`` over all lines (real or
          not) before ``as_of_date``.

    This function is shared with inference (M6): it must stay pure and
    identical for batch training and single-customer scoring.
    """
    as_of = pd.Timestamp(as_of_date)
    before = clean_transactions[clean_transactions["invoice_date"] < as_of]
    real = before[~before["is_cancellation"]]

    real_by_customer = real.groupby("customer_id")
    population = real_by_customer.size()

    recency_days = (as_of - real_by_customer["invoice_date"].max()).dt.days
    customer_tenure_days = (as_of - real_by_customer["invoice_date"].min()).dt.days
    frequency = real_by_customer["invoice"].nunique()
    n_distinct_products = real_by_customer["stock_code"].nunique()

    line_value = before["quantity"] * before["price"]
    monetary_total = line_value.groupby(before["customer_id"]).sum()
    return_rate = before.groupby("customer_id")["has_non_positive_quantity"].mean()
    # `monetary_total`/`return_rate` are grouped over `before`, which
    # may include customers whose only lines are cancellations; they
    # are outside the population. Reindexing to the population index
    # before assembling the DataFrame keeps every column on exactly the
    # same index (no NaN upcasting), so dtypes are identical whether a
    # customer is processed in a batch or alone — the invariant the M6
    # inference path relies on.
    monetary_total = monetary_total.reindex(population.index)
    return_rate = return_rate.reindex(population.index)

    features = pd.DataFrame(
        {
            "recency_days": recency_days,
            "frequency": frequency,
            "monetary_total": monetary_total,
            "monetary_avg_order": monetary_total / frequency,
            "customer_tenure_days": customer_tenure_days,
            "n_distinct_products": n_distinct_products,
            "return_rate": return_rate,
        }
    )
    features.index.name = "customer_id"
    return features
