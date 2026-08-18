"""Node functions for the `feature_engineering` pipeline (M3).

Design: a long snapshot table instead of a single temporal cut. For
each usable snapshot date ``T`` (first day of a month), features are
computed for every customer from transactions strictly before ``T``
(shared, pure implementation in ``commerce_signals.features``) and a
churn label is computed from the window ``(T, T + threshold_days]``.
The result is one row per ``(customer_id, snapshot_date)`` pair.

Cancellations (``is_cancellation=True``) never count as a real
purchase, neither in features nor in the churn label.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from commerce_signals.features import FEATURE_COLUMNS, compute_customer_features

logger = logging.getLogger(__name__)

# Columns of the long snapshot table, in a stable, documented order.
SNAPSHOT_TABLE_COLUMNS: list[str] = [
    "customer_id",
    "snapshot_date",
    *FEATURE_COLUMNS,
    "is_churned",
]


def compute_churn_labels(
    clean_transactions: pd.DataFrame,
    as_of_date: pd.Timestamp,
    threshold_days: int,
) -> pd.Series:
    """Compute the churn label for every customer at ``as_of_date``.

    Label semantics:
        * ``True`` (churned): NO real (non-cancellation) purchase with
          ``invoice_date`` in the open-closed window
          ``(as_of_date, as_of_date + threshold_days]``.
        * ``False`` (active): at least one real purchase inside the
          window. A purchase exactly on the upper bound
          (``as_of_date + threshold_days``) counts; a purchase exactly
          on ``as_of_date`` does not (strict lower bound).

    The population is identical to ``compute_customer_features``:
    customers with at least one real purchase strictly before
    ``as_of_date``.

    IMPORTANT: this function is exclusive to training (M3/M4). It must
    NEVER be used in inference (M6): there the churn label is exactly
    what the model has to predict, so it cannot be computed from data
    that has not happened yet.

    Args:
        clean_transactions: ``clean_transactions`` DataFrame from the
            validation pipeline.
        as_of_date: Snapshot moment; window starts strictly after it.
        threshold_days: Days without a real purchase after which a
            customer is labelled churned (from ``params:churn.threshold_days``).

    Returns:
        A boolean ``pd.Series`` named ``is_churned``, indexed by
        ``customer_id``, ``True`` meaning churned.
    """
    as_of = pd.Timestamp(as_of_date)
    before = clean_transactions[clean_transactions["invoice_date"] < as_of]
    population = before[~before["is_cancellation"]].groupby("customer_id").size()

    window_end = as_of + pd.Timedelta(threshold_days, unit="D")
    in_window = clean_transactions[
        (clean_transactions["invoice_date"] > as_of)
        & (clean_transactions["invoice_date"] <= window_end)
        & (~clean_transactions["is_cancellation"])
    ]
    returned_customers = in_window["customer_id"].unique()

    return pd.Series(
        ~population.index.isin(returned_customers),
        index=population.index,
        name="is_churned",
        dtype=bool,
    )


def build_customer_snapshots(
    clean_transactions: pd.DataFrame,
    snapshot_start_date: str,
    threshold_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the long customer snapshot table (features + churn label).

    Candidate snapshot dates are the first day of each month from
    ``snapshot_start_date`` up to ``max_invoice_date``. Only candidates
    on or before ``last_labelable_cutoff`` (``max_invoice_date -
    threshold_days``, derived at run time from the real data, never
    hardcoded) are usable: a snapshot later than that would have an
    incomplete churn-label window.

    Each usable snapshot is processed in a simple loop (filter +
    groupby per snapshot, which is fast enough at this data volume); no
    incremental/cumulative aggregation is used. Features and labels
    must cover exactly the same population; a mismatch is a bug and is
    raised, not silently ignored.

    Args:
        clean_transactions: ``clean_transactions`` DataFrame from the
            validation pipeline.
        snapshot_start_date: First candidate snapshot (from
            ``params:feature_engineering.snapshot_start_date``).
        threshold_days: Churn window length in days (from
            ``params:churn.threshold_days``).

    Returns:
        A tuple ``(spine_df, report)``:

        * ``spine_df``: the long table, one row per
          ``(customer_id, snapshot_date)``, with columns
          ``customer_id``, ``snapshot_date`` and the seven feature
          columns from ``compute_customer_features`` plus ``is_churned``.
        * ``report``: a dict with the exact shape::

            {
                "threshold_days": int,
                "max_invoice_date_observed": str (isoformat),
                "last_labelable_cutoff": str (isoformat),
                "snapshot_start_date": str (isoformat),
                "candidate_snapshot_dates": int,
                "usable_snapshot_dates": int,
                "rows_total": int,
                "churn_rate_overall": float,
                "per_snapshot": [
                    {"snapshot_date": str, "n_customers": int,
                     "n_churned": int, "churn_rate": float},
                    ...
                ],
            }
    """
    max_invoice_date = pd.Timestamp(clean_transactions["invoice_date"].max())
    last_labelable_cutoff = max_invoice_date - pd.Timedelta(threshold_days, unit="D")

    candidates = pd.date_range(
        start=pd.Timestamp(snapshot_start_date), end=max_invoice_date, freq="MS"
    )
    usable_dates = candidates[candidates <= last_labelable_cutoff]
    logger.info(
        "Feature engineering: %d candidate snapshot dates generated "
        "(%s to %s); %d usable after filtering snapshots <= "
        "last_labelable_cutoff (%s).",
        len(candidates),
        candidates[0].isoformat() if len(candidates) else "n/a",
        candidates[-1].isoformat() if len(candidates) else "n/a",
        len(usable_dates),
        last_labelable_cutoff.isoformat(),
    )

    iterations: list[pd.DataFrame] = []
    per_snapshot: list[dict[str, Any]] = []
    for snapshot_date in usable_dates:
        features = compute_customer_features(clean_transactions, snapshot_date)
        labels = compute_churn_labels(
            clean_transactions, snapshot_date, threshold_days
        )
        if not features.index.equals(labels.index):
            raise AssertionError(
                "Feature/label population mismatch at snapshot "
                f"{snapshot_date.isoformat()}: {len(features)} customers "
                f"with features vs {len(labels)} with labels. Both are "
                "derived from the same population rule, so this is a bug."
            )
        snapshot_df = features.assign(is_churned=labels, snapshot_date=snapshot_date)
        snapshot_df = snapshot_df.reset_index()
        iterations.append(snapshot_df)

        n_customers = int(len(labels))
        n_churned = int(labels.sum())
        per_snapshot.append(
            {
                "snapshot_date": snapshot_date.isoformat(),
                "n_customers": n_customers,
                "n_churned": n_churned,
                "churn_rate": n_churned / n_customers if n_customers else 0.0,
            }
        )

    if iterations:
        spine_df = pd.concat(iterations, ignore_index=True)
    else:
        spine_df = pd.DataFrame(columns=SNAPSHOT_TABLE_COLUMNS)
    spine_df = spine_df[SNAPSHOT_TABLE_COLUMNS]

    rows_total = int(len(spine_df))
    total_churned = sum(entry["n_churned"] for entry in per_snapshot)
    report: dict[str, Any] = {
        "threshold_days": threshold_days,
        "max_invoice_date_observed": max_invoice_date.isoformat(),
        "last_labelable_cutoff": last_labelable_cutoff.isoformat(),
        "snapshot_start_date": pd.Timestamp(snapshot_start_date).isoformat(),
        "candidate_snapshot_dates": int(len(candidates)),
        "usable_snapshot_dates": int(len(usable_dates)),
        "rows_total": rows_total,
        "churn_rate_overall": (
            total_churned / rows_total if rows_total else 0.0
        ),
        "per_snapshot": per_snapshot,
    }

    per_snapshot_lines = "\n".join(
        f"  {entry['snapshot_date']}: n_customers={entry['n_customers']}, "
        f"n_churned={entry['n_churned']}, churn_rate={entry['churn_rate']:.4f}"
        for entry in report["per_snapshot"]
    )
    logger.info(
        "Feature engineering report:\n"
        "  threshold_days: %d\n"
        "  max_invoice_date_observed: %s\n"
        "  last_labelable_cutoff: %s\n"
        "  snapshot_start_date: %s\n"
        "  candidate_snapshot_dates: %d\n"
        "  usable_snapshot_dates: %d\n"
        "  rows_total: %d\n"
        "  churn_rate_overall: %.4f\n"
        "%s",
        report["threshold_days"],
        report["max_invoice_date_observed"],
        report["last_labelable_cutoff"],
        report["snapshot_start_date"],
        report["candidate_snapshot_dates"],
        report["usable_snapshot_dates"],
        report["rows_total"],
        report["churn_rate_overall"],
        per_snapshot_lines,
    )
    return spine_df, report


def verify_feature_table_row_count(
    customer_snapshots: pd.DataFrame,
    feature_engineering_report: dict[str, Any],
) -> None:
    """Compare the Mongo round-trip row count to the persisted report.

    ``customer_snapshots`` is re-loaded from Mongo and
    ``feature_engineering_report`` is re-read from disk by Kedro, so
    this node verifies the full save/load round-trip of both outputs,
    not just the in-memory objects that produced them.

    This node is observability, not a quality gate: it never raises
    and never fails the pipeline. If the counts match it logs INFO; if
    they differ it logs WARNING with both numbers so the mismatch is
    visible in any log scrape.

    Args:
        customer_snapshots: DataFrame re-loaded from Mongo by Kedro.
        feature_engineering_report: Dict re-loaded from the persisted
            JSON by Kedro; the ``"rows_total"`` key is compared against
            ``len(customer_snapshots)``.
    """
    loaded = int(len(customer_snapshots))
    expected = int(feature_engineering_report["rows_total"])
    if loaded == expected:
        logger.info(
            "Feature engineering verification OK: %d rows in customer_snapshots "
            "matches the persisted report (%d).",
            loaded,
            expected,
        )
    else:
        logger.warning(
            "Feature engineering verification MISMATCH: customer_snapshots has "
            "%d rows, the report says %d (delta=%d). Data and report were saved "
            "but the counts do not agree — investigate before downstream "
            "pipelines run.",
            loaded,
            expected,
            loaded - expected,
        )
