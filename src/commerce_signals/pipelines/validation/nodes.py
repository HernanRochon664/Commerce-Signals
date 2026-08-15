"""Node functions for the `validation` pipeline (M2).

Cleaning rules, applied in this exact order:
    1. Rename columns to snake_case.
    2. Drop exact duplicate rows (on all columns, ``keep="first"``).
    3. Exclude rows without ``customer_id`` (counted in the report,
       never persisted anywhere else).
    4. Compute five independent boolean flag columns on the surviving
       rows. No decision is taken here on what the flags mean for
       training; that is feature engineering (M3).
    5. Cast ``customer_id`` from float64 to a clean string without a
       trailing ``.0`` decimal point.

``raw_transactions`` is an immutable source: this pipeline only reads
it, never writes back to it.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Rename map from the raw Online Retail II schema (as loaded by the
# ingestion pipeline) to the snake_case contract every downstream
# pipeline works with.
COLUMN_RENAME_MAP: dict[str, str] = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def _resolve_reference_date(reference_date: str | pd.Timestamp | None) -> pd.Timestamp:
    """Coerce the ``params:validation.reference_date`` value.

    If the parameter is ``None`` (the default in parameters.yml), the
    run-time clock is used. The parameter exists so tests can pass a
    fixed date and historical reprocessing is not contaminated by
    today's date.

    Args:
        reference_date: From ``params:validation.reference_date``.

    Returns:
        A timezone-naive ``pd.Timestamp`` to compare against.
    """
    if reference_date is None:
        resolved = pd.Timestamp(datetime.now())
        logger.info(
            "validation.reference_date not set; using run-time now (%s).",
            resolved.isoformat(),
        )
    else:
        resolved = pd.Timestamp(reference_date)
    return resolved


def _compute_flags(df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Attach the five independent boolean flag columns to ``df``.

    Every flag is computed over the already-deduplicated,
    customer-id-non-null frame. Flags are mutually independent and
    none of them drops rows: what they mean for training is decided in
    M3 (feature engineering).

    Args:
        df: Frame that survived dedup and the ``customer_id`` exclusion.
        reference_date: Fixed comparison date for the future-invoice flag.

    Returns:
        A copy of ``df`` with the five ``is_*`` / ``has_*`` columns
        added, all with dtype ``bool``.
    """
    flagged = df.copy()

    # .str.startswith("C") propagates NaN for null invoices; the nullable
    # "boolean" dtype lets fillna(False) run without pandas' deprecated
    # object downcasting, and the final astype(bool) guarantees a plain
    # bool column, never float or object.
    flagged["is_cancellation"] = (
        flagged["invoice"]
        .str.upper()
        .str.strip()
        .str.startswith("C")
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    flagged["has_non_positive_quantity"] = flagged["quantity"] <= 0
    flagged["has_non_positive_price"] = flagged["price"] <= 0
    flagged["has_null_invoice_date"] = flagged["invoice_date"].isna()
    # NaT > reference_date evaluates to False, so null dates are never
    # also counted as future dates.
    flagged["has_future_invoice_date"] = flagged["invoice_date"] > reference_date
    return flagged


def validate_and_clean_transactions(
    raw_transactions: pd.DataFrame,
    reference_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate and clean raw transactions per the M2 rules.

    Applies the five cleaning rules in their fixed order and returns
    the cleaned frame together with a report counting how many rows
    each rule affected. The report's ``flags`` counts are computed on
    the cleaned frame (after dedup and customer-id exclusion), never
    on the raw input.

    Args:
        raw_transactions: Full ``raw_transactions`` DataFrame from the
            Mongo collection (via the catalog).
        reference_date: ``params:validation.reference_date``; ``None``
            falls back to run-time ``datetime.now()``.

    Returns:
        A tuple ``(clean_df, validation_report)`` where ``clean_df`` is
        the cleaned DataFrame (consumed by ``clean_transactions``) and
        ``validation_report`` has the exact shape::

            {
                "source_rows": int,
                "duplicate_rows_dropped": int,
                "rows_without_customer_id_excluded": int,
                "clean_transactions_rows": int,
                "reference_date_used": str,
                "flags": {
                    "is_cancellation": int,
                    "has_non_positive_quantity": int,
                    "has_non_positive_price": int,
                    "has_null_invoice_date": int,
                    "has_future_invoice_date": int,
                },
            }

        The report always satisfies the invariant
        ``source_rows - duplicate_rows_dropped - rows_without_customer_id_excluded
        == clean_transactions_rows``.
    """
    # Rule 1: rename first; everything below works with the new names.
    df = raw_transactions.rename(columns=COLUMN_RENAME_MAP)
    source_rows = int(len(df))

    # Rule 2: exact duplicates on all columns; the only unconditional
    # row removal in this pipeline.
    is_duplicate = df.duplicated(keep="first")
    duplicate_rows_dropped = int(is_duplicate.sum())
    df = df[~is_duplicate]

    # Rule 3: rows without a customer_id are unusable for per-customer
    # churn; they are counted and dropped, never persisted elsewhere.
    has_no_customer = df["customer_id"].isna()
    rows_without_customer_id_excluded = int(has_no_customer.sum())
    df = df[~has_no_customer]

    # Rule 4: flag columns over the surviving rows.
    resolved_reference_date = _resolve_reference_date(reference_date)
    df = _compute_flags(df, resolved_reference_date)

    # Rule 5: float64 -> Int64 -> str. A direct .astype(str) would
    # produce "13085.0" with a decimal point (silent bug, no error).
    df["customer_id"] = df["customer_id"].astype("Int64").astype(str)

    clean_transactions_rows = int(len(df))
    report: dict[str, Any] = {
        "source_rows": source_rows,
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "rows_without_customer_id_excluded": rows_without_customer_id_excluded,
        "clean_transactions_rows": clean_transactions_rows,
        "reference_date_used": resolved_reference_date.isoformat(),
        "flags": {
            "is_cancellation": int(df["is_cancellation"].sum()),
            "has_non_positive_quantity": int(df["has_non_positive_quantity"].sum()),
            "has_non_positive_price": int(df["has_non_positive_price"].sum()),
            "has_null_invoice_date": int(df["has_null_invoice_date"].sum()),
            "has_future_invoice_date": int(df["has_future_invoice_date"].sum()),
        },
    }
    logger.info(
        "Validation report raw_transactions -> clean_transactions:\n"
        "  source_rows: %d\n"
        "  duplicate_rows_dropped: %d\n"
        "  rows_without_customer_id_excluded: %d\n"
        "  clean_transactions_rows: %d\n"
        "  reference_date_used: %s\n"
        "  flags: is_cancellation=%d, has_non_positive_quantity=%d, "
        "has_non_positive_price=%d, has_null_invoice_date=%d, "
        "has_future_invoice_date=%d",
        report["source_rows"],
        report["duplicate_rows_dropped"],
        report["rows_without_customer_id_excluded"],
        report["clean_transactions_rows"],
        report["reference_date_used"],
        report["flags"]["is_cancellation"],
        report["flags"]["has_non_positive_quantity"],
        report["flags"]["has_non_positive_price"],
        report["flags"]["has_null_invoice_date"],
        report["flags"]["has_future_invoice_date"],
    )
    return df, report


def verify_validation_row_count(
    clean_transactions: pd.DataFrame,
    validation_report: dict[str, Any],
) -> None:
    """Compare the Mongo round-trip row count to the persisted report.

    ``clean_transactions`` is re-loaded from Mongo and
    ``validation_report`` is re-read from disk by Kedro, so this node
    verifies the full save/load round-trip of both outputs, not just
    the in-memory objects that produced them.

    This node is observability, not a quality gate: it never raises
    and never fails the pipeline. If the counts match it logs INFO; if
    they differ it logs WARNING with both numbers so the mismatch is
    visible in any log scrape.

    Args:
        clean_transactions: DataFrame re-loaded from Mongo by Kedro.
        validation_report: Dict re-loaded from the persisted JSON by
            Kedro; the ``"clean_transactions_rows"`` key is compared
            against ``len(clean_transactions)``.
    """
    loaded = int(len(clean_transactions))
    expected = int(validation_report["clean_transactions_rows"])
    if loaded == expected:
        logger.info(
            "Validation verification OK: %d rows in clean_transactions matches "
            "the persisted report (%d).",
            loaded,
            expected,
        )
    else:
        logger.warning(
            "Validation verification MISMATCH: clean_transactions has %d rows, "
            "the report says %d (delta=%d). Data and report were saved but the "
            "counts do not agree — investigate before downstream pipelines run.",
            loaded,
            expected,
            loaded - expected,
        )
