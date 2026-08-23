"""Tests for the `training` pipeline (M4).

Everything here runs without Mongo: the functions are pure or train a
LightGBM model directly in memory (no Kedro/catalog involved). MLflow
uses a temporary local SQLite tracking store per test (``tmp_path``),
never the project's real ``mlflow.db`` — an invalid tracking URI would
mix test runs with the real training history.

The ``log_training_run_and_build_report`` node hardcodes
``sqlite:///mlflow.db`` by design; tests stub ``mlflow.set_tracking_uri``
to a no-op and set the temp URI *before* calling the node, so nothing
touches the project's real database.
"""

from __future__ import annotations

import logging

import mlflow
import pandas as pd
import pytest

from commerce_signals.pipelines.training.nodes import (
    _best_f1_threshold,
    _recall_at_precision,
    evaluate_churn_model,
    log_training_run_and_build_report,
    split_snapshots_temporally,
    train_churn_model,
    verify_training_run_is_retrievable,
)

SNAPSHOT_DATES = [
    "2010-01-01",
    "2010-02-01",
    "2010-03-01",
    "2010-04-01",
    "2010-05-01",
    "2010-06-01",
    "2010-07-01",
]

# (customer_id, recency_days, is_churned) — churn is a deterministic
# function of recency so the synthetic model can actually learn.
_CUSTOMERS = [
    ("C001", 5, 0),
    ("C002", 10, 0),
    ("C003", 18, 0),
    ("C004", 35, 1),
    ("C005", 60, 1),
    ("C006", 90, 1),
]

_N_ROWS_PER_SNAPSHOT = len(_CUSTOMERS)


def _customer_row(
    customer_id: str, recency_days: int, churn: int, snapshot_date: str
) -> dict:
    i = int(customer_id[1:])
    monetary_total = 1000.0 + 100.0 * i
    frequency = 10 - i
    return {
        "customer_id": customer_id,
        "snapshot_date": snapshot_date,
        "recency_days": recency_days,
        "frequency": frequency,
        "monetary_total": monetary_total,
        "monetary_avg_order": monetary_total / frequency,
        "customer_tenure_days": 300 + 25 * i,
        "n_distinct_products": 5 + i,
        "return_rate": 0.02 + 0.01 * i,
        "is_churned": churn,
    }


@pytest.fixture
def customer_snapshots() -> pd.DataFrame:
    """Synthetic snapshots: 7 monthly dates x 6 customers = 42 rows.

    ``snapshot_date`` is stored as plain strings on purpose, so the
    defensive ``pd.to_datetime`` coercion in the split node is actually
    exercised.
    """
    rows = [
        _customer_row(customer_id, recency, churn, snapshot_date)
        for snapshot_date in SNAPSHOT_DATES
        for customer_id, recency, churn in _CUSTOMERS
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def mini_training_run(tmp_path, monkeypatch, customer_snapshots):
    """Train, evaluate, and log a tiny run to MLflow in tmp_path.

    Returns the model, split info, evaluation metrics, and the final
    training report.
    """
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    train_df, val_df, test_df, split_info = split_snapshots_temporally(
        customer_snapshots, test_snapshots_count=2, val_snapshots_count=2
    )
    model_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 4,
        "learning_rate": 0.05,
        "n_estimators": 30,
        "random_state": 42,
        # Default min_data_in_leaf=20 exceeds the 18-row train split,
        # which would make LightGBM stop before any split.
        "min_data_in_leaf": 3,
    }
    model = train_churn_model(
        train_df,
        val_df,
        model_params,
        eval_metric="binary_logloss",
        early_stopping_rounds=5,
    )
    evaluation_metrics = evaluate_churn_model(
        model, val_df, test_df, precision_targets=[0.5, 0.7]
    )
    report = log_training_run_and_build_report(
        model,
        model_params,
        "binary_logloss",
        5,
        split_info,
        evaluation_metrics,
        "commerce-signals-churn-test",
    )
    return model, split_info, evaluation_metrics, report


# --- split_snapshots_temporally ------------------------------------------------

def test_split_snapshots_temporally_splits_by_date(customer_snapshots) -> None:
    train_df, val_df, test_df, split_info = split_snapshots_temporally(
        customer_snapshots, test_snapshots_count=2, val_snapshots_count=2
    )

    assert split_info["train_snapshot_dates"] == [
        "2010-01-01T00:00:00",
        "2010-02-01T00:00:00",
        "2010-03-01T00:00:00",
    ]
    assert split_info["val_snapshot_dates"] == [
        "2010-04-01T00:00:00",
        "2010-05-01T00:00:00",
    ]
    assert split_info["test_snapshot_dates"] == [
        "2010-06-01T00:00:00",
        "2010-07-01T00:00:00",
    ]

    # Row counts match a manual count over the fixture: 6 rows per date.
    assert split_info["train_rows"] == 3 * _N_ROWS_PER_SNAPSHOT
    assert split_info["val_rows"] == 2 * _N_ROWS_PER_SNAPSHOT
    assert split_info["test_rows"] == 2 * _N_ROWS_PER_SNAPSHOT
    assert len(train_df) == 3 * _N_ROWS_PER_SNAPSHOT
    assert len(val_df) == 2 * _N_ROWS_PER_SNAPSHOT
    assert len(test_df) == 2 * _N_ROWS_PER_SNAPSHOT

    # Strict temporal boundaries and no row overlap.
    train_dates = pd.to_datetime(train_df["snapshot_date"])
    val_dates = pd.to_datetime(val_df["snapshot_date"])
    test_dates = pd.to_datetime(test_df["snapshot_date"])
    assert train_dates.max() < val_dates.min()
    assert val_dates.max() < test_dates.min()
    assert val_dates.max() < test_dates.min()
    assert set(train_df.index).isdisjoint(val_df.index)


def test_split_snapshots_temporally_raises_with_too_few_snapshots(
    customer_snapshots,
) -> None:
    with pytest.raises(ValueError, match="Not enough snapshots"):
        split_snapshots_temporally(
            customer_snapshots, test_snapshots_count=4, val_snapshots_count=3
        )


# --- _recall_at_precision / _best_f1_threshold --------------------------------

def test_recall_at_precision_reachable_target() -> None:
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.7, 0.5, 0.3]
    # PR pairs per threshold (ascending): (0.5, 1.0), (2/3, 1.0),
    # (0.5, 0.5), (1.0, 0.5). Max recall at precision >= 0.6 is 1.0
    # (threshold 0.5); at precision >= 0.9 only threshold 0.9 qualifies
    # with recall 0.5.
    assert _recall_at_precision(y_true, y_score, 0.6) == pytest.approx(1.0)
    assert _recall_at_precision(y_true, y_score, 0.9) == pytest.approx(0.5)


def test_recall_at_precision_unreachable_target_returns_zero() -> None:
    y_true = [0, 1, 1, 0]
    y_score = [0.9, 0.7, 0.5, 0.3]
    # First sample is negative, so the curve's max precision is 2/3:
    # a target of 0.75 is unreachable and must yield 0.0.
    assert _recall_at_precision(y_true, y_score, 0.75) == 0.0
    assert _recall_at_precision(y_true, y_score, 0.65) == pytest.approx(1.0)


def test_best_f1_threshold() -> None:
    y_true = [1, 0, 1, 0]
    y_score = [0.9, 0.7, 0.5, 0.3]
    # F1 per threshold: 0.3 -> 0.67, 0.5 -> 0.8, 0.7 -> 0.5, 0.9 -> 0.67.
    assert _best_f1_threshold(y_true, y_score) == pytest.approx(0.5)


# --- End-to-end: train -> evaluate -> log -> report ---------------------------

def test_end_to_end_training_run(mini_training_run) -> None:
    _, split_info, evaluation_metrics, report = mini_training_run

    assert set(report) == {
        "run_id",
        "mlflow_experiment_name",
        "model_params",
        "eval_metric",
        "early_stopping_rounds",
        "best_iteration",
        "split_info",
        "evaluation_metrics",
    }
    assert report["run_id"]
    assert report["mlflow_experiment_name"] == "commerce-signals-churn-test"
    assert report["eval_metric"] == "binary_logloss"
    assert report["early_stopping_rounds"] == 5
    assert report["best_iteration"] >= 1
    assert report["split_info"] == split_info
    assert set(report["evaluation_metrics"]) == {"val", "test", "test_per_snapshot"}

    for split in ("val", "test"):
        metrics = report["evaluation_metrics"][split]
        for key, value in metrics.items():
            if key == "confusion_matrix":
                continue
            assert 0.0 <= value <= 1.0, f"{split}.{key}={value} out of [0, 1]"
        assert "recall_at_precision_0.5" in metrics
        assert "recall_at_precision_0.7" in metrics

    cm = report["evaluation_metrics"]["test"]["confusion_matrix"]
    assert set(cm) == {"tn", "fp", "fn", "tp"}
    assert all(isinstance(v, int) for v in cm.values())
    assert sum(cm.values()) == report["split_info"]["test_rows"]

    per_snapshot = report["evaluation_metrics"]["test_per_snapshot"]
    assert len(per_snapshot) == 2
    for entry in per_snapshot:
        assert set(entry) == {
            "snapshot_date",
            "n_rows",
            "churn_rate",
            "pr_auc",
            "roc_auc",
        }
        assert entry["n_rows"] == _N_ROWS_PER_SNAPSHOT
        assert entry["churn_rate"] == pytest.approx(0.5)
        assert 0.0 <= entry["pr_auc"] <= 1.0
        assert 0.0 <= entry["roc_auc"] <= 1.0


# --- verify_training_run_is_retrievable --------------------------------------

def test_verify_training_run_is_retrievable_with_real_run(
    mini_training_run, caplog
) -> None:
    _, _, _, report = mini_training_run
    with caplog.at_level(logging.INFO, logger="commerce_signals.pipelines.training.nodes"):
        verify_training_run_is_retrievable(report)
    assert any(
        "n_features_in_" in record.message and "matches" in record.message
        for record in caplog.records
    )


def test_verify_training_run_is_retrievable_invalid_run_id_never_raises(
    tmp_path, monkeypatch, caplog
) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    report = {"run_id": "definitely-not-a-real-run-id"}
    verify_training_run_is_retrievable(report)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and "definitely-not-a-real-run-id" in warnings[0].message
