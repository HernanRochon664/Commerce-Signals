"""Unit tests for the `feature_engineering` pipeline nodes (M3).

All nodes are pure functions over DataFrames; Mongo and the JSON file
are resolved by Kedro at the catalog boundary, so these tests need no
running Mongo and no integration fixtures.

Churn-label fixture (``as_of_date = 2010-03-01``, ``threshold_days = 30``,
window ``(2010-03-01, 2010-03-31]``):

    8001 — real purchase on 2010-02-01 plus a real purchase exactly at
           the upper bound (2010-03-31) -> NOT churned (inclusive).
    8002 — real purchase on 2010-02-05; the only purchase inside the
           window is a cancellation (2010-03-15) -> churned.
    8003 — real purchases on 2010-02-20 and 2010-03-10 -> NOT churned.
    8004 — real purchases on 2010-02-10 and exactly at the lower bound
           (2010-03-01 00:00) -> churned (exclusive lower bound).
    8005 — a single real purchase on 2010-02-20 -> churned.
    8006 — a single real purchase on 2010-03-20 (no purchase before
           ``as_of_date``) -> outside the population, no row at all.

Snapshot fixture (``snapshot_start_date = 2010-01-01``,
``threshold_days = 30``): the last transaction is 2010-06-15, so
``max_invoice_date = 2010-06-15`` and
``last_labelable_cutoff = 2010-05-16``. Candidates are the first day of
each month from 2010-01-01 to 2010-06-01 (6); the 2010-06-01 candidate
is excluded (>= cutoff), leaving 5 usable snapshots:

    7001: 2009-12-20 (real), 2010-01-10 (real), 2010-03-05 (real),
          2010-06-15 (real).
    7002: 2010-01-15 (real), 2010-04-20 (real, negative quantity),
          2010-05-25 (cancellation).
    7003: 2010-02-10 (real).

Hand-computed report (threshold 30 days):

    S1 2010-01-01: pop {7001}; 7001 buys 01-10 in (01-01, 01-31]  -> 1, 0, 0.0
    S2 2010-02-01: pop {7001, 7002}; next real buys 03-05 / 04-20,
                   both after (02-01, 03-03]                        -> 2, 2, 1.0
    S3 2010-03-01: pop {7001, 7002, 7003}; 7001 buys 03-05 in
                   (03-01, 03-31]; others churned                   -> 3, 2, 2/3
    S4 2010-04-01: pop all three; 7002 buys 04-20 in (04-01, 05-01];
                   others churned                                   -> 3, 2, 2/3
    S5 2010-05-01: pop all three; only in-window purchase for 7002
                   is the 05-25 cancellation (does not count)       -> 3, 3, 1.0

    rows_total = 12, n_churned over all snapshots = 9,
    churn_rate_overall = 9/12 = 0.75.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from commerce_signals.features import compute_customer_features
from commerce_signals.pipelines.feature_engineering.nodes import (
    SNAPSHOT_TABLE_COLUMNS,
    build_customer_snapshots,
    compute_churn_labels,
    verify_feature_table_row_count,
)

AS_OF_DATE = "2010-03-01"
THRESHOLD_DAYS = 30


def _row(
    invoice: str,
    stock_code: str,
    quantity: int,
    invoice_date: str,
    price: float,
    customer_id: str,
) -> dict:
    """Compact row builder mirroring the M2 flag logic (same as test_features)."""
    return {
        "invoice": invoice,
        "stock_code": stock_code,
        "description": f"item {stock_code}",
        "quantity": quantity,
        "invoice_date": invoice_date,
        "price": price,
        "customer_id": customer_id,
        "country": "United Kingdom",
        "is_cancellation": invoice.upper().startswith("C"),
        "has_non_positive_quantity": quantity <= 0,
        "has_non_positive_price": price <= 0,
        "has_null_invoice_date": invoice_date is None,
        "has_future_invoice_date": False,
    }


def _make_clean_transactions(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="mixed")
    return df


@pytest.fixture(scope="module")
def label_fixture() -> pd.DataFrame:
    """Transactions for the churn-label cases described above."""
    return _make_clean_transactions(
        [
            _row("801", "AAA", 1, "2010-02-01", 10.0, "8001"),
            _row("802", "AAA", 1, "2010-03-31", 10.0, "8001"),
            _row("811", "BBB", 1, "2010-02-05", 10.0, "8002"),
            _row("C812", "BBB", -1, "2010-03-15", 10.0, "8002"),
            _row("821", "CCC", 1, "2010-02-20", 10.0, "8003"),
            _row("822", "CCC", 1, "2010-03-10", 10.0, "8003"),
            _row("831", "DDD", 1, "2010-02-10", 10.0, "8004"),
            _row("832", "DDD", 1, "2010-03-01", 10.0, "8004"),
            _row("841", "EEE", 1, "2010-02-20", 10.0, "8005"),
            _row("851", "FFF", 1, "2010-03-20", 10.0, "8006"),
        ]
    )


@pytest.fixture(scope="module")
def snapshot_fixture() -> pd.DataFrame:
    """Transactions for the 5-snapshot case described above."""
    return _make_clean_transactions(
        [
            _row("701", "P1", 2, "2009-12-20", 10.0, "7001"),
            _row("702", "P2", 1, "2010-01-10", 5.0, "7001"),
            _row("703", "P1", 3, "2010-03-05", 4.0, "7001"),
            _row("704", "P2", 1, "2010-06-15", 50.0, "7001"),
            _row("711", "Q1", 5, "2010-01-15", 2.0, "7002"),
            _row("712", "Q1", -2, "2010-04-20", 2.0, "7002"),
            _row("C713", "Q2", -5, "2010-05-25", 2.0, "7002"),
            _row("721", "R1", 1, "2010-02-10", 100.0, "7003"),
        ]
    )


# --- compute_churn_labels -----------------------------------------------------

def test_churn_labels_boundaries_and_cancellation(
    label_fixture: pd.DataFrame,
) -> None:
    """Every label matches the hand-computed expectations."""
    labels = compute_churn_labels(
        label_fixture, pd.Timestamp(AS_OF_DATE), THRESHOLD_DAYS
    )

    assert labels.dtype == bool
    assert list(labels.index) == ["8001", "8002", "8003", "8004", "8005"]
    assert labels.to_dict() == {
        "8001": False,  # upper bound inclusive: purchase exactly on 03-31
        "8002": True,  # only a cancellation inside the window
        "8003": False,  # real purchase inside the window
        "8004": True,  # lower bound exclusive: purchase exactly on 03-01
        "8005": True,  # no purchase inside the window
    }
    # 8006 has no purchase before as_of_date: outside the population.
    assert "8006" not in labels.index


def test_churn_labels_population_matches_features_population(
    label_fixture: pd.DataFrame,
) -> None:
    """Labels and features must cover exactly the same customers."""
    labels = compute_churn_labels(
        label_fixture, pd.Timestamp(AS_OF_DATE), THRESHOLD_DAYS
    )
    features = compute_customer_features(label_fixture, pd.Timestamp(AS_OF_DATE))

    assert labels.index.equals(features.index)


# --- build_customer_snapshots -------------------------------------------------

def test_snapshot_dates_filtering_and_report(
    snapshot_fixture: pd.DataFrame,
) -> None:
    """Candidate/usable counts, cutoff derivation and the exact report."""
    spine_df, report = build_customer_snapshots(
        snapshot_fixture, "2010-01-01", THRESHOLD_DAYS
    )

    assert report["threshold_days"] == 30
    assert report["max_invoice_date_observed"] == "2010-06-15T00:00:00"
    # Derived at run time from the data, never hardcoded upstream.
    assert report["last_labelable_cutoff"] == "2010-05-16T00:00:00"
    assert report["snapshot_start_date"] == "2010-01-01T00:00:00"
    assert report["candidate_snapshot_dates"] == 6
    assert report["usable_snapshot_dates"] == 5
    assert report["rows_total"] == 12
    assert report["churn_rate_overall"] == pytest.approx(9 / 12)
    assert set(report.keys()) == {
        "threshold_days",
        "max_invoice_date_observed",
        "last_labelable_cutoff",
        "snapshot_start_date",
        "candidate_snapshot_dates",
        "usable_snapshot_dates",
        "rows_total",
        "churn_rate_overall",
        "per_snapshot",
    }

    assert report["per_snapshot"] == [
        {
            "snapshot_date": "2010-01-01T00:00:00",
            "n_customers": 1,
            "n_churned": 0,
            "churn_rate": 0.0,
        },
        {
            "snapshot_date": "2010-02-01T00:00:00",
            "n_customers": 2,
            "n_churned": 2,
            "churn_rate": 1.0,
        },
        {
            "snapshot_date": "2010-03-01T00:00:00",
            "n_customers": 3,
            "n_churned": 2,
            "churn_rate": pytest.approx(2 / 3),
        },
        {
            "snapshot_date": "2010-04-01T00:00:00",
            "n_customers": 3,
            "n_churned": 2,
            "churn_rate": pytest.approx(2 / 3),
        },
        {
            "snapshot_date": "2010-05-01T00:00:00",
            "n_customers": 3,
            "n_churned": 3,
            "churn_rate": 1.0,
        },
    ]


def test_snapshot_table_structure(
    snapshot_fixture: pd.DataFrame,
) -> None:
    """The long table has one row per (customer_id, snapshot_date)."""
    spine_df, _ = build_customer_snapshots(
        snapshot_fixture, "2010-01-01", THRESHOLD_DAYS
    )

    assert list(spine_df.columns) == SNAPSHOT_TABLE_COLUMNS
    assert len(spine_df) == 12
    assert spine_df["is_churned"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(spine_df["snapshot_date"])
    # One row per (customer_id, snapshot_date) pair, no duplicates.
    assert (
        spine_df.groupby(["customer_id", "snapshot_date"]).size().max() == 1
    )
    assert set(spine_df["customer_id"]) == {"7001", "7002", "7003"}
    assert set(spine_df["snapshot_date"]) == set(
        pd.to_datetime(
            [
                "2010-01-01",
                "2010-02-01",
                "2010-03-01",
                "2010-04-01",
                "2010-05-01",
            ]
        )
    )
    churned_7002 = spine_df[
        (spine_df["customer_id"] == "7002")
        & (spine_df["snapshot_date"] == pd.Timestamp("2010-05-01"))
    ]
    assert churned_7002["is_churned"].item() is True


def test_usability_filter_is_logged(
    snapshot_fixture: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Candidate and usable counts are logged at INFO."""
    with caplog.at_level(
        logging.INFO, logger="commerce_signals.pipelines.feature_engineering.nodes"
    ):
        build_customer_snapshots(snapshot_fixture, "2010-01-01", THRESHOLD_DAYS)

    assert any(
        "6 candidate snapshot dates" in rec.message
        and "5 usable" in rec.message
        and rec.levelno == logging.INFO
        for rec in caplog.records
    )


# --- verify_feature_table_row_count --------------------------------------------

def test_verify_logs_info_on_match(
    snapshot_fixture: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Matching counts log at INFO and never raise."""
    spine_df, report = build_customer_snapshots(
        snapshot_fixture, "2010-01-01", THRESHOLD_DAYS
    )

    with caplog.at_level(
        logging.INFO, logger="commerce_signals.pipelines.feature_engineering.nodes"
    ):
        # Must NOT raise.
        verify_feature_table_row_count(spine_df, report)

    assert any(
        "Feature engineering verification OK" in rec.message
        and rec.levelno == logging.INFO
        for rec in caplog.records
    )


def test_verify_logs_warning_on_mismatch(
    snapshot_fixture: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mismatched counts log at WARNING; the pipeline does not fail."""
    spine_df, report = build_customer_snapshots(
        snapshot_fixture, "2010-01-01", THRESHOLD_DAYS
    )
    # Simulate Mongo losing one row between save and reload.
    truncated_df = spine_df.iloc[:-1].reset_index(drop=True)

    with caplog.at_level(
        logging.WARNING, logger="commerce_signals.pipelines.feature_engineering.nodes"
    ):
        # Must NOT raise.
        verify_feature_table_row_count(truncated_df, report)

    assert any(
        "MISMATCH" in rec.message and rec.levelno == logging.WARNING
        for rec in caplog.records
    )
