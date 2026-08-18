"""Unit tests for the shared feature module (M3).

``compute_customer_features`` lives in ``src/commerce_signals/features.py``,
outside any pipeline package, because it is shared: the feature
engineering pipeline calls it once per snapshot for all customers, and
the M6 inference endpoint will call it with a single customer's history
and "now" as ``as_of_date``. These tests must stay green with or
without Mongo: the function is pure and the fixture is built by hand
(small and purpose-specific, not the integration fixtures).

The synthetic dataset (``as_of_date = 2010-03-01``):

    17850 — several real purchases + one cancellation + one
            negative-quantity real line:
        * 2009-12-15, invoice 1001, qty 5, price 10.0      ->  50.0
        * 2010-01-20, invoice 1002, qty 1, price 20.0      ->  20.0
        * 2010-01-25, invoice C1002, qty -5, price 10.0    -> -50.0 (cancellation)
        * 2010-02-10, invoice 1003, qty 2, price 30.0      ->  60.0
        * 2010-02-20, invoice 1004, qty 3, price 5.0       ->  15.0
        * 2010-02-28, invoice 1005, qty -2, price 30.0     -> -60.0
          (return line, not a cancellation)

    13047 — a single real purchase:
        * 2010-02-14, invoice 2001, qty 10, price 2.0      ->  20.0

    30000 — a real purchase exactly ON as_of_date (must be excluded):
        * 2010-03-01 10:00, invoice 3001 (strict ``<`` boundary).

    40000 — only a cancellation before as_of_date (must not appear).

    50000 — a real purchase after as_of_date (must not appear).

Expected values at ``as_of_date = 2010-03-01``:

    17850: recency_days=1, frequency=5, monetary_total=35.0,
           monetary_avg_order=7.0, customer_tenure_days=76,
           n_distinct_products=3 (AAA, BBB, CCC),
           return_rate=2/6 (lines with non-positive quantity: C1002 and
           1005, over all 6 lines).
    13047: recency_days=15, frequency=1, monetary_total=20.0,
           monetary_avg_order=20.0, customer_tenure_days=15,
           n_distinct_products=1, return_rate=0.0.
"""

from __future__ import annotations

import pandas as pd
import pytest

from commerce_signals.features import compute_customer_features

AS_OF_DATE = "2010-03-01"

# Compact row builder mirroring the M2 flag logic on the clean schema.
def _row(
    invoice: str,
    stock_code: str,
    quantity: int,
    invoice_date: str,
    price: float,
    customer_id: str,
) -> dict:
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


@pytest.fixture(scope="module")
def clean_transactions() -> pd.DataFrame:
    """Small hand-built dataset covering all the cases above."""
    rows = [
        _row("1001", "AAA", 5, "2009-12-15", 10.0, "17850"),
        _row("1002", "BBB", 1, "2010-01-20", 20.0, "17850"),
        _row("C1002", "AAA", -5, "2010-01-25", 10.0, "17850"),
        _row("1003", "AAA", 2, "2010-02-10", 30.0, "17850"),
        _row("1004", "CCC", 3, "2010-02-20", 5.0, "17850"),
        _row("1005", "AAA", -2, "2010-02-28", 30.0, "17850"),
        _row("2001", "ZZZ", 10, "2010-02-14", 2.0, "13047"),
        _row("3001", "PPP", 1, "2010-03-01 10:00:00", 9.0, "30000"),
        _row("C4001", "QQQ", -3, "2010-02-10", 5.0, "40000"),
        _row("5001", "RRR", 1, "2010-03-10", 7.0, "50000"),
    ]
    df = pd.DataFrame(rows)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="mixed")
    return df


@pytest.fixture(scope="module")
def features(clean_transactions: pd.DataFrame) -> pd.DataFrame:
    """The feature table the fixture must produce at AS_OF_DATE."""
    return compute_customer_features(clean_transactions, pd.Timestamp(AS_OF_DATE))


def test_exact_feature_values_for_two_distinct_customers(
    features: pd.DataFrame,
) -> None:
    """Every column is verified with its exact expected value."""
    assert list(features.index) == ["13047", "17850"]

    customer_13047 = features.loc["13047"]
    assert customer_13047["recency_days"] == 15
    assert customer_13047["frequency"] == 1
    assert customer_13047["monetary_total"] == 20.0
    assert customer_13047["monetary_avg_order"] == 20.0
    assert customer_13047["customer_tenure_days"] == 15
    assert customer_13047["n_distinct_products"] == 1
    assert customer_13047["return_rate"] == 0.0

    customer_17850 = features.loc["17850"]
    assert customer_17850["recency_days"] == 1
    assert customer_17850["frequency"] == 5
    assert customer_17850["monetary_total"] == pytest.approx(35.0)
    assert customer_17850["monetary_avg_order"] == pytest.approx(7.0)
    assert customer_17850["customer_tenure_days"] == 76
    assert customer_17850["n_distinct_products"] == 3
    # C1002 (cancellation) and 1005 are the non-positive-quantity lines,
    # over 6 total lines: cancellations count in the denominator, and a
    # negative-quantity line that is NOT a cancellation is a return.
    assert customer_17850["return_rate"] == pytest.approx(2 / 6)


def test_customer_with_purchase_exactly_at_as_of_date_is_excluded(
    features: pd.DataFrame,
) -> None:
    """Strict ``<`` boundary: a purchase on the cut-off itself is not
    usable for features, so the customer is not in the population."""
    assert "30000" not in features.index


def test_customer_without_real_purchase_before_cutoff_is_absent(
    features: pd.DataFrame,
) -> None:
    """No zero-filled rows: only-cancellation and future-only customers
    simply do not exist in the result."""
    assert "40000" not in features.index
    assert "50000" not in features.index
    assert set(features.index) == {"13047", "17850"}


def test_single_customer_inference_equals_batch_result(
    clean_transactions: pd.DataFrame,
) -> None:
    """M6 inference will call this function with one customer's full
    history; the result must be identical to the batch call."""
    batch = compute_customer_features(clean_transactions, pd.Timestamp(AS_OF_DATE))
    single_customer_history = clean_transactions[
        clean_transactions["customer_id"] == "17850"
    ]
    single = compute_customer_features(
        single_customer_history, pd.Timestamp(AS_OF_DATE)
    )

    pd.testing.assert_frame_equal(batch.loc[["17850"]], single)


def test_index_dtype_and_name(features: pd.DataFrame) -> None:
    """The result is indexed by customer_id with the documented name."""
    assert features.index.name == "customer_id"
    assert features.index.dtype == object
