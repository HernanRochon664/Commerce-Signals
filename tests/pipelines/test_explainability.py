"""Tests for the `explainability` pipeline nodes (M5).

No Mongo, MLflow uses a temporary SQLite store per test (tmp_path)
like test_training.py.
"""

from __future__ import annotations

import logging

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import pytest

from commerce_signals.features import FEATURE_COLUMNS
from commerce_signals.pipelines.explainability.nodes import (
    compute_churn_explanations,
    load_model_for_explanation,
    prepare_explanation_inputs,
    verify_shap_additivity,
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

_CUSTOMERS = [
    ("C001", 5, 0),
    ("C002", 10, 0),
    ("C003", 18, 0),
    ("C004", 35, 1),
    ("C005", 60, 1),
    ("C006", 90, 1),
]


def _customer_row(customer_id: str, recency_days: int, churn: int, snapshot_date: str) -> dict:
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
    rows = [
        _customer_row(customer_id, recency, churn, snapshot_date)
        for snapshot_date in SNAPSHOT_DATES
        for customer_id, recency, churn in _CUSTOMERS
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_training_report() -> dict:
    return {
        "run_id": "synthetic-run-id",
        "split_info": {
            "train_snapshot_dates": [
                "2010-01-01T00:00:00",
                "2010-02-01T00:00:00",
                "2010-03-01T00:00:00",
            ],
            "val_snapshot_dates": [
                "2010-04-01T00:00:00",
                "2010-05-01T00:00:00",
            ],
            "test_snapshot_dates": [
                "2010-06-01T00:00:00",
                "2010-07-01T00:00:00",
            ],
            "train_rows": 18,
            "val_rows": 12,
            "test_rows": 12,
        },
    }


# --- prepare_explanation_inputs -----------------------------------------------


def test_prepare_explanation_inputs_filters_by_exact_dates(
    customer_snapshots, synthetic_training_report, caplog
) -> None:
    with caplog.at_level(logging.INFO, logger="commerce_signals.pipelines.explainability.nodes"):
        background_df, explanation_df, sample_info = prepare_explanation_inputs(
            customer_snapshots,
            synthetic_training_report,
            background_sample_size=5,
            explanation_sample_size=4,
            random_state=42,
        )
    # Must filter to exact dates from training_report, not recalc from params
    train_dates = pd.to_datetime(synthetic_training_report["split_info"]["train_snapshot_dates"])
    test_dates = pd.to_datetime(synthetic_training_report["split_info"]["test_snapshot_dates"])

    assert set(pd.to_datetime(background_df["snapshot_date"]).unique()).issubset(set(train_dates))
    assert set(pd.to_datetime(explanation_df["snapshot_date"]).unique()).issubset(set(test_dates))

    assert len(background_df) == 5
    assert len(explanation_df) == 4

    # Ensure customer_ids come from correct pools
    assert len(background_df) <= 3 * len(_CUSTOMERS)
    assert len(explanation_df) <= 2 * len(_CUSTOMERS)

    # Info log should mention counts
    assert any("Prepared explainability inputs" in r.message for r in caplog.records)


def test_prepare_explanation_inputs_small_pool_warning(
    customer_snapshots, synthetic_training_report, caplog
) -> None:
    # Request more than available: train pool has 18 rows (3 dates x6), test 12
    with caplog.at_level(logging.WARNING, logger="commerce_signals.pipelines.explainability.nodes"):
        background_df, explanation_df, sample_info = prepare_explanation_inputs(
            customer_snapshots,
            synthetic_training_report,
            background_sample_size=100,
            explanation_sample_size=100,
            random_state=42,
        )
    # Should use full pools
    assert len(background_df) == 18
    assert len(explanation_df) == 12
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Background pool" in w.message for w in warnings)
    assert any("Explanation pool" in w.message for w in warnings)

    # Also check sample_info is correctly returned
    assert sample_info["background_sample_size_requested"] == 100
    assert sample_info["background_sample_size_used"] == 18
    assert sample_info["explanation_sample_size_requested"] == 100
    assert sample_info["explanation_sample_size_used"] == 12
    assert sample_info["run_id"] == "synthetic-run-id"


def test_prepare_explanation_inputs_coercion_string_dates(
    synthetic_training_report,
) -> None:
    # Ensure defensive pd.to_datetime coercion is exercised: store snapshot_date as strings
    rows = [
        _customer_row("C001", 5, 0, "2010-01-01"),
        _customer_row("C002", 60, 1, "2010-07-01"),
    ]
    df = pd.DataFrame(rows)
    # snapshot_date is object string dtype
    assert df["snapshot_date"].dtype == object
    bg, exp, sample_info = prepare_explanation_inputs(
        df,
        synthetic_training_report,
        background_sample_size=1,
        explanation_sample_size=1,
        random_state=0,
    )
    assert len(bg) == 1
    assert len(exp) == 1
    assert sample_info["background_sample_size_used"] == 1
    assert sample_info["explanation_sample_size_used"] == 1


# --- load_model_for_explanation ----------------------------------------------


def test_load_model_for_explanation(tmp_path, monkeypatch) -> None:
    # Train tiny model and log to tmp_path mlflow
    df = pd.DataFrame(
        [
            _customer_row(cid, rec, churn, "2010-01-01")
            for cid, rec, churn in _CUSTOMERS
        ]
        * 3
    )
    X = df[FEATURE_COLUMNS]
    y = df["is_churned"]
    model_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "num_leaves": 4,
        "n_estimators": 10,
        "random_state": 42,
        "min_data_in_leaf": 2,
        "verbose": -1,
    }
    model = lgb.LGBMClassifier(**model_params)
    model.fit(X, y)

    # Log to tmp_path mlflow
    tmp_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tmp_uri)
    # The node hardcodes sqlite:///mlflow.db, so stub set_tracking_uri to no-op
    # and pre-set the tmp URI; but we need to ensure the node will still
    # load from the tmp DB. Trick: monkeypatch the node's mlflow.set_tracking_uri
    # to keep our tmp_uri, or patch to set to tmp_uri when called.
    original_set_uri = mlflow.set_tracking_uri

    def fake_set_uri(uri: str) -> None:
        # Redirect the hardcoded sqlite:///mlflow.db to our tmp DB
        if uri == "sqlite:///mlflow.db":
            original_set_uri(tmp_uri)
        else:
            original_set_uri(uri)

    monkeypatch.setattr(mlflow, "set_tracking_uri", fake_set_uri)
    # Also need to ensure experiment is set correctly for logging
    mlflow.set_experiment("test-explain-load")
    with mlflow.start_run() as run:
        mlflow.lightgbm.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    training_report = {"run_id": run_id}
    # Now call the node: it will call fake_set_uri("sqlite:///mlflow.db") -> tmp_uri
    loaded = load_model_for_explanation(training_report)
    assert isinstance(loaded, lgb.LGBMClassifier)
    assert loaded.n_features_in_ == len(FEATURE_COLUMNS)
    # Predict sanity
    proba = loaded.predict_proba(X)[:, 1]
    assert proba.shape[0] == len(X)


# --- compute_churn_explanations ----------------------------------------------


def _train_tiny_model_for_explain() -> tuple[lgb.LGBMClassifier, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 100
    data: dict[str, np.ndarray] = {}
    for col in FEATURE_COLUMNS:
        if col == "recency_days":
            data[col] = rng.integers(0, 120, size=n).astype(float)
        elif col == "frequency":
            data[col] = rng.integers(1, 15, size=n).astype(float)
        elif col == "return_rate":
            data[col] = rng.random(n) * 0.3
        else:
            data[col] = rng.random(n) * 100 + 10
    df = pd.DataFrame(data)
    df["is_churned"] = (df["recency_days"] > 50).astype(int)
    # Add customer_id / snapshot_date for report plumbing
    df["customer_id"] = [f"C{i:04d}" for i in range(n)]
    df["snapshot_date"] = pd.Timestamp("2010-07-01")
    X = df[FEATURE_COLUMNS]
    y = df["is_churned"]
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=4,
        n_estimators=15,
        random_state=42,
        min_data_in_leaf=3,
        verbose=-1,
    )
    model.fit(X, y)
    return model, df


def test_compute_churn_explanations_report_shape() -> None:
    model, df = _train_tiny_model_for_explain()
    # Split into background (train-like) and explanation (test-like)
    background = df.sample(20, random_state=42)
    explanation_pool = df.sample(30, random_state=1)
    sample_info = {
        "run_id": "test-run-123",
        "background_sample_size_requested": 20,
        "background_sample_size_used": 20,
        "explanation_sample_size_requested": 30,
        "explanation_sample_size_used": 30,
    }

    report = compute_churn_explanations(
        model,
        background,
        explanation_pool,
        sample_info,
        top_n_examples=3,
    )

    # Exact shape
    assert set(report.keys()) == {
        "run_id",
        "background_sample_size_requested",
        "background_sample_size_used",
        "explanation_sample_size_requested",
        "explanation_sample_size_used",
        "global_importance",
        "example_explanations",
    }
    assert report["run_id"] == "test-run-123"
    assert report["background_sample_size_requested"] == 20
    assert report["background_sample_size_used"] == 20
    assert report["explanation_sample_size_requested"] == 30
    assert report["explanation_sample_size_used"] == 30

    # global_importance sorted descending
    gi = report["global_importance"]
    assert len(gi) == len(FEATURE_COLUMNS)
    mean_abs = [e["mean_abs_shap"] for e in gi]
    assert mean_abs == sorted(mean_abs, reverse=True)
    for entry in gi:
        assert "feature" in entry and "mean_abs_shap" in entry
        assert isinstance(entry["feature"], str)
        assert isinstance(entry["mean_abs_shap"], float)

    # example_explanations length and sorting
    examples = report["example_explanations"]
    assert len(examples) == 3
    probs = [e["predicted_probability"] for e in examples]
    assert probs == sorted(probs, reverse=True)
    for ex in examples:
        assert set(ex.keys()) == {
            "customer_id",
            "snapshot_date",
            "predicted_probability",
            "base_value",
            "feature_contributions",
        }
        assert isinstance(ex["customer_id"], str)
        assert isinstance(ex["snapshot_date"], str)
        assert isinstance(ex["predicted_probability"], float)
        assert isinstance(ex["base_value"], float)
        assert isinstance(ex["feature_contributions"], dict)
        assert set(ex["feature_contributions"].keys()) == set(FEATURE_COLUMNS)
        for v in ex["feature_contributions"].values():
            assert isinstance(v, float)

    # No numpy types anywhere in report (check recursively)
    def _assert_no_numpy(obj) -> None:
        assert not isinstance(obj, (np.generic, np.ndarray)), f"Found numpy type {type(obj)}"
        if isinstance(obj, dict):
            for v in obj.values():
                _assert_no_numpy(v)
        elif isinstance(obj, list):
            for v in obj:
                _assert_no_numpy(v)

    _assert_no_numpy(report)


# --- verify_shap_additivity ---------------------------------------------------


def test_verify_shap_additivity_passes(caplog) -> None:
    model, df = _train_tiny_model_for_explain()
    background = df.sample(15, random_state=42)
    explanation_pool = df.sample(8, random_state=99)
    sample_info = {
        "run_id": "run-pass",
        "background_sample_size_requested": 15,
        "background_sample_size_used": 15,
        "explanation_sample_size_requested": 8,
        "explanation_sample_size_used": 8,
    }

    report = compute_churn_explanations(
        model,
        background,
        explanation_pool,
        sample_info,
        top_n_examples=3,
    )

    with caplog.at_level(logging.INFO, logger="commerce_signals.pipelines.explainability.nodes"):
        verify_shap_additivity(report, tolerance=0.0001)

    # Should log INFO for each passed example and a summary, no WARNING
    assert any("additivity passed" in r.message for r in caplog.records)
    assert any("additivity summary" in r.message for r in caplog.records)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    # Might have zero warnings for a correct report
    assert not any("FAILED" in w.message for w in warnings)


def test_verify_shap_additivity_corrupted_logs_warning(caplog) -> None:
    model, df = _train_tiny_model_for_explain()
    background = df.sample(15, random_state=42)
    explanation_pool = df.sample(8, random_state=5)
    sample_info = {
        "run_id": "run-corrupt",
        "background_sample_size_requested": 15,
        "background_sample_size_used": 15,
        "explanation_sample_size_requested": 8,
        "explanation_sample_size_used": 8,
    }

    report = compute_churn_explanations(
        model,
        background,
        explanation_pool,
        sample_info,
        top_n_examples=2,
    )

    # Corrupt one feature contribution to break additivity
    corrupted = report.copy()
    # Deep copy example_explanations to avoid mutating original
    import copy

    corrupted = copy.deepcopy(report)
    corrupted["example_explanations"][0]["feature_contributions"][FEATURE_COLUMNS[0]] += 5.0

    with caplog.at_level(logging.INFO, logger="commerce_signals.pipelines.explainability.nodes"):
        # Must NOT raise
        verify_shap_additivity(corrupted, tolerance=0.0001)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("FAILED" in w.message for w in warnings)
    # Delta should be mentioned
    assert any("delta" in w.message.lower() for w in warnings)
    # Summary still logged (INFO)
    assert any("additivity summary" in r.message for r in caplog.records)

    # Original correct report should still pass without exception
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="commerce_signals.pipelines.explainability.nodes"):
        verify_shap_additivity(report, tolerance=0.0001)
    assert any("passed" in r.message.lower() for r in caplog.records)
