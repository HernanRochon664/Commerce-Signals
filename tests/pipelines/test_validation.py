"""Unit tests for the `validation` pipeline nodes (M2).

Both nodes are pure functions (DataFrame/dict in, DataFrame/dict out);
Mongo and the JSON file are resolved by Kedro at the catalog boundary,
so none of these tests needs the integration fixture in conftest.py.

The synthetic dataset is built by hand to deliberately exercise every
cleaning rule at least once:
    * one exact duplicate row (dropped),
    * one row without a customer_id (excluded, counted only),
    * a cancellation (invoice starting with ``C``, one in lowercase to
      prove the ``.upper()`` path),
    * a negative quantity, a zero price,
    * two null invoice dates,
    * a future invoice date relative to a fixed ``reference_date``.

Expected counts on the 12-row fixture (reference_date=2011-01-15):

    source_rows=12, duplicates dropped=1, no-customer excluded=1,
    clean rows=10,
    flags: is_cancellation=1, has_non_positive_quantity=1,
    has_non_positive_price=1, has_null_invoice_date=2,
    has_future_invoice_date=1.

The excluded no-customer row carries a ``C`` invoice and a future date
on purpose: because flags are computed after dedup + exclusion, it must
NOT be counted in any flag. That assertion double-checks the
"flags over clean_transactions" contract.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest

from commerce_signals.pipelines.validation.nodes import (
    validate_and_clean_transactions,
    verify_validation_row_count,
)

REFERENCE_DATE = "2011-01-15"

SNAKE_COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_date",
    "price",
    "customer_id",
    "country",
]

FLAG_COLUMNS = [
    "is_cancellation",
    "has_non_positive_quantity",
    "has_non_positive_price",
    "has_null_invoice_date",
    "has_future_invoice_date",
]


@pytest.fixture(scope="module")
def synthetic_raw_transactions() -> pd.DataFrame:
    """The 12-row fixture described in the module docstring.

    ``InvoiceDate`` is parsed with ``pd.to_datetime`` (``None`` becomes
    NaT) and ``Customer ID`` stays float64 with NaN for the missing
    customer, mirroring what the ingestion pipeline wrote to Mongo.
    """
    return pd.DataFrame(
        {
            "Invoice": [
                "536365",  # 0 clean
                "536365",  # 1 exact duplicate of row 0 -> dropped
                "536366",  # 2 clean
                "c536367",  # 3 cancellation (lowercase c proves .upper())
                "536368",  # 4 negative quantity
                "536369",  # 5 zero price
                "536370",  # 6 null invoice date
                "536371",  # 7 future invoice date
                "C536372",  # 8 no customer_id (also C-invoice + future date)
                None,  # 9 null invoice (fillna(False) path)
                "536374",  # 10 second null invoice date
                "536375",  # 11 near-duplicate of row 0, kept (different invoice)
            ],
            "StockCode": [
                "85123A",
                "85123A",
                "71053",
                "84406B",
                "84029G",
                "84029E",
                "84029G",
                "84029E",
                "85123A",
                "71053",
                "84406B",
                "85123A",
            ],
            "Description": [
                "WHITE HANGING HEART",
                "WHITE HANGING HEART",
                "WHITE METAL LANTERN",
                "CREAM CUPID HEARTS",
                "KNITTED UNION FLAG",
                "RED WOOLLY HOTTIE",
                "KNITTED UNION FLAG",
                "RED WOOLLY HOTTIE",
                "WHITE HANGING HEART",
                "WHITE METAL LANTERN",
                "CREAM CUPID HEARTS",
                "WHITE HANGING HEART",
            ],
            "Quantity": [6, 6, 6, 8, -1, 1, 1, 1, 0, 3, 5, 6],
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-01",  # 0
                    "2011-01-01",  # 1
                    "2011-01-02",  # 2
                    "2011-01-03",  # 3
                    "2011-01-04",  # 4
                    "2011-01-05",  # 5
                    None,  # 6
                    "2011-02-01",  # 7 future vs 2011-01-15
                    "2011-03-01",  # 8 future too, but excluded
                    "2011-01-07",  # 9
                    None,  # 10
                    "2011-01-01",  # 11
                ]
            ),
            "Price": [2.55, 2.55, 3.39, 2.75, 3.39, 0.0, 5.95, 5.95, 0.0, 1.50, 2.00, 2.55],
            "Customer ID": [
                17850.0,
                17850.0,
                13047.0,
                13047.0,
                13047.0,
                13047.0,
                13047.0,
                13047.0,
                None,  # 8 no customer
                13047.0,
                13047.0,
                17850.0,
            ],
            "Country": ["United Kingdom"] * 12,
        }
    )


@pytest.fixture(scope="module")
def expected_report() -> dict:
    """The exact report the fixture must produce with REFERENCE_DATE."""
    return {
        "source_rows": 12,
        "duplicate_rows_dropped": 1,
        "rows_without_customer_id_excluded": 1,
        "clean_transactions_rows": 10,
        "reference_date_used": pd.Timestamp(REFERENCE_DATE).isoformat(),
        "flags": {
            "is_cancellation": 1,
            "has_non_positive_quantity": 1,
            "has_non_positive_price": 1,
            "has_null_invoice_date": 2,
            "has_future_invoice_date": 1,
        },
    }


def test_each_rule_count_matches_exact_expected_numbers(
    synthetic_raw_transactions: pd.DataFrame, expected_report: dict
) -> None:
    """Every counter in the report matches the hand-computed numbers."""
    _, report = validate_and_clean_transactions(
        synthetic_raw_transactions, reference_date=REFERENCE_DATE
    )

    assert report == expected_report


def test_arithmetic_invariant_holds(
    synthetic_raw_transactions: pd.DataFrame,
) -> None:
    """source_rows - duplicates - no_customer == clean rows, always."""
    clean_df, report = validate_and_clean_transactions(
        synthetic_raw_transactions, reference_date=REFERENCE_DATE
    )

    assert report["source_rows"] - report["duplicate_rows_dropped"] - report[
        "rows_without_customer_id_excluded"
    ] == report["clean_transactions_rows"]
    assert len(clean_df) == report["clean_transactions_rows"]


def test_output_columns_are_snake_case(
    synthetic_raw_transactions: pd.DataFrame,
) -> None:
    """Renaming happens first; everything downstream uses snake_case."""
    clean_df, _ = validate_and_clean_transactions(
        synthetic_raw_transactions, reference_date=REFERENCE_DATE
    )

    assert list(clean_df.columns[:8]) == SNAKE_COLUMNS
    assert "Customer ID" not in clean_df.columns


def test_exact_duplicates_dropped_and_similar_rows_kept(
    synthetic_raw_transactions: pd.DataFrame,
) -> None:
    """Only the exact full-row duplicate is dropped, not near-duplicates."""
    clean_df, report = validate_and_clean_transactions(
        synthetic_raw_transactions, reference_date=REFERENCE_DATE
    )

    assert report["duplicate_rows_dropped"] == 1
    invoices = clean_df["invoice"].dropna().astype(str).tolist()
    # 10 clean rows; 9 with a non-null invoice (row 9's invoice is null).
    assert len(clean_df) == 10
    assert set(invoices) == {
        "536365", "536366", "c536367", "536368", "536369",
        "536370", "536371", "536374", "536375",
    }
    # The near-duplicate (same values as row 0, different invoice) survived.
    assert "536375" in invoices


def test_customer_id_is_clean_string_without_decimal_point(
    synthetic_raw_transactions: pd.DataFrame,
) -> None:
    """The float64 -> '13085.0' silent bug must never appear."""
    clean_df, _ = validate_and_clean_transactions(
        synthetic_raw_transactions, reference_date=REFERENCE_DATE
    )

    customer_ids = clean_df["customer_id"].tolist()
    assert set(customer_ids) == {"17850", "13047"}
    # The specific regression: no value may contain a decimal point.
    assert all("." not in value for value in customer_ids)
    assert clean_df["customer_id"].dtype == object


def test_all_five_flag_columns_exist_and_are_boolean(
    synthetic_raw_transactions: pd.DataFrame,
) -> None:
    """Flags are real booleans, not float or object columns."""
    clean_df, _ = validate_and_clean_transactions(
        synthetic_raw_transactions, reference_date=REFERENCE_DATE
    )

    assert list(clean_df.columns[8:]) == FLAG_COLUMNS
    for flag in FLAG_COLUMNS:
        assert clean_df[flag].dtype == bool
    # A null invoice must not leak NaN into the cancellation flag.
    assert not clean_df["is_cancellation"].isna().any()


def test_reference_date_none_falls_back_to_run_time_clock() -> None:
    """reference_date=None uses now, so a future-dated row is flagged."""
    df = pd.DataFrame(
        {
            "Invoice": ["536999"],
            "StockCode": ["85123A"],
            "Description": ["FUTURE ORDER"],
            "Quantity": [1],
            "InvoiceDate": pd.to_datetime([datetime.now() + timedelta(days=30)]),
            "Price": [1.0],
            "Customer ID": [13047.0],
            "Country": ["United Kingdom"],
        }
    )

    _, report = validate_and_clean_transactions(df, reference_date=None)

    assert report["flags"]["has_future_invoice_date"] == 1


def test_verify_logs_info_on_match(
    synthetic_raw_transactions: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Matching counts log at INFO and never raise."""
    clean_df, report = validate_and_clean_transactions(
        synthetic_raw_transactions, reference_date=REFERENCE_DATE
    )

    with caplog.at_level(
        logging.INFO, logger="commerce_signals.pipelines.validation.nodes"
    ):
        # Must NOT raise.
        verify_validation_row_count(clean_df, report)

    assert any(
        "Validation verification OK" in rec.message and rec.levelno == logging.INFO
        for rec in caplog.records
    )


def test_verify_logs_warning_on_mismatch(
    synthetic_raw_transactions: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mismatched counts log at WARNING; the pipeline does not fail."""
    clean_df, report = validate_and_clean_transactions(
        synthetic_raw_transactions, reference_date=REFERENCE_DATE
    )
    # Simulate Mongo/JSON losing one row between save and reload.
    truncated_df = clean_df.iloc[:-1].reset_index(drop=True)

    with caplog.at_level(
        logging.WARNING, logger="commerce_signals.pipelines.validation.nodes"
    ):
        # Must NOT raise.
        verify_validation_row_count(truncated_df, report)

    assert any(
        "MISMATCH" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    )
