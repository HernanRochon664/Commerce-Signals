"""Tests for FastAPI /predict (M6).

Uses TestClient + dependency overrides to inject a tiny synthetic model,
so no real mlflow.db / training_report.json is needed.
Mongo I/O (load_filtered + predictions save) goes against the real test
Mongo (TEST_MONGODB_URI), same pattern as test_mongo_dataset.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

import lightgbm as lgb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from commerce_signals.datasets import MongoCollectionDataset
from commerce_signals.explainability import build_explainer
from commerce_signals.features import FEATURE_COLUMNS
from tests.conftest import TEST_DB_NAME

pytestmark = pytest.mark.integration


def _credentials(uri: str) -> dict[str, str]:
    return {"uri": uri, "db": TEST_DB_NAME}


def _train_synthetic_model() -> tuple[lgb.LGBMClassifier, float, object]:
    """Train tiny LGBM on synthetic FEATURE_COLUMNS, return model, threshold, explainer."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 200
    data: dict[str, np.ndarray] = {}
    for col in FEATURE_COLUMNS:
        if col == "recency_days":
            data[col] = rng.integers(0, 200, size=n).astype(float)
        elif col == "frequency":
            data[col] = rng.integers(1, 20, size=n).astype(float)
        elif col == "return_rate":
            data[col] = rng.random(n) * 0.3
        else:
            data[col] = rng.random(n) * 100 + 10
    df = pd.DataFrame(data)
    df["is_churned"] = (df["recency_days"] > 90).astype(int)
    X = df[FEATURE_COLUMNS]
    y = df["is_churned"]
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=4,
        n_estimators=20,
        random_state=42,
        min_data_in_leaf=2,
        verbose=-1,
    )
    model.fit(X, y)
    threshold = 0.5
    # Build explainer from a small background
    background = df[FEATURE_COLUMNS].sample(20, random_state=42)
    explainer = build_explainer(model, background)
    return model, threshold, explainer


def _make_tx(
    customer_id: str,
    invoice: str,
    invoice_date: str,
    quantity: int,
    price: float,
    stock_code: str,
    is_cancellation: bool,
) -> dict:
    return {
        "customer_id": customer_id,
        "invoice": invoice,
        "invoice_date": pd.Timestamp(invoice_date),
        "quantity": quantity,
        "price": price,
        "stock_code": stock_code,
        "is_cancellation": bool(is_cancellation),
        "has_non_positive_quantity": quantity <= 0,
    }


@asynccontextmanager
async def _noop_lifespan(app):  # type: ignore[no-untyped-def]
    yield


@pytest.fixture
def synthetic_model():
    model, thr, explainer = _train_synthetic_model()
    return model, thr, explainer


def _make_client_with_overrides(
    mongo_uri: str,
    collection_name: str,
    model,
    threshold: float,
    explainer,
    predictions_collection: str | None = None,
):
    """Create TestClient with lifespan disabled and dependencies overridden."""
    from app.main import (
        app,
        get_clean_transactions_dataset,
        get_explainer,
        get_predictions_dataset,
        get_threshold,
        get_trained_model,
    )

    # Patch lifespan to noop so we don't need real mlflow/report/Mongo snapshot
    app.router.lifespan_context = _noop_lifespan  # type: ignore[attr-defined]

    clean_ds = MongoCollectionDataset(
        collection=collection_name, credentials=_credentials(mongo_uri)
    )
    pred_coll = predictions_collection or f"pred_{uuid.uuid4().hex[:8]}"
    pred_ds = MongoCollectionDataset(
        collection=pred_coll, credentials=_credentials(mongo_uri), mode="append"
    )

    # Preserve original overrides to restore after
    orig = dict(app.dependency_overrides)
    app.dependency_overrides[get_trained_model] = lambda: model  # type: ignore[assignment]
    # also alias get_model
    from app.main import get_model

    app.dependency_overrides[get_model] = lambda: model  # type: ignore[assignment]
    app.dependency_overrides[get_threshold] = lambda: threshold  # type: ignore[assignment]
    app.dependency_overrides[get_explainer] = lambda: explainer  # type: ignore[assignment]
    app.dependency_overrides[get_clean_transactions_dataset] = lambda: clean_ds  # type: ignore[assignment]
    app.dependency_overrides[get_predictions_dataset] = lambda: pred_ds  # type: ignore[assignment]

    client = TestClient(app, raise_server_exceptions=False)
    # Also set app.state for any code that still reads it directly
    client.app.state.trained_model = model  # type: ignore[attr-defined]
    client.app.state.threshold = threshold  # type: ignore[attr-defined]
    client.app.state.explainer = explainer  # type: ignore[attr-defined]
    client.app.state.clean_transactions_ds = clean_ds  # type: ignore[attr-defined]
    client.app.state.predictions_ds = pred_ds  # type: ignore[attr-defined]

    return client, app, orig, pred_ds, clean_ds


def test_health(synthetic_model, mongo_uri, mongo_test_db):
    model, thr, explainer = synthetic_model
    coll = f"health_{uuid.uuid4().hex[:8]}"
    client, app, orig, _, _ = _make_client_with_overrides(mongo_uri, coll, model, thr, explainer)
    try:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
    finally:
        app.dependency_overrides = orig  # type: ignore[assignment]


def test_predict_success(synthetic_model, mongo_uri, mongo_test_db):
    model, thr, explainer = synthetic_model
    coll = f"clean_{uuid.uuid4().hex[:8]}"
    client, app, orig, pred_ds, clean_ds = _make_client_with_overrides(
        mongo_uri, coll, model, thr, explainer
    )
    try:
        # Insert synthetic history for customer C123
        txs = pd.DataFrame(
            [
                _make_tx("C123", "10001", "2011-10-01", 5, 10.0, "A001", False),
                _make_tx("C123", "10002", "2011-11-01", 2, 20.0, "A002", False),
                _make_tx("C123", "10003", "2011-11-15", 1, 30.0, "A003", False),
                # Add another customer's noise
                _make_tx("OTHER", "99999", "2011-10-02", 3, 15.0, "B001", False),
            ]
        )
        clean_ds._save(txs)

        resp = client.post("/predict", json={"customer_id": "C123"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["customer_id"] == "C123"
        assert 0 <= body["churn_probability"] <= 1
        assert isinstance(body["is_at_risk"], bool)
        assert "as_of_date" in body
        assert len(body["top_risk_factors"]) == 3
        for fac in body["top_risk_factors"]:
            assert "feature" in fac and "shap_value" in fac
            assert isinstance(fac["feature"], str)
            assert isinstance(fac["shap_value"], float)

        # Verify audit write to predictions collection
        pred_records = pred_ds._load()
        assert not pred_records.empty
        assert "C123" in pred_records["customer_id"].values
    finally:
        app.dependency_overrides = orig  # type: ignore[assignment]
        # cleanup collections
        from pymongo import MongoClient

        with MongoClient(mongo_uri) as cli:
            cli[TEST_DB_NAME][coll].drop()
            # pred collection name is inside pred_ds
            cli[TEST_DB_NAME][pred_ds.collection].drop()


def test_predict_not_found(synthetic_model, mongo_uri, mongo_test_db):
    model, thr, explainer = synthetic_model
    coll = f"clean_{uuid.uuid4().hex[:8]}"
    client, app, orig, pred_ds, clean_ds = _make_client_with_overrides(
        mongo_uri, coll, model, thr, explainer
    )
    try:
        # Insert history for OTHER only
        txs = pd.DataFrame(
            [_make_tx("OTHER", "99999", "2011-10-02", 3, 15.0, "B001", False)]
        )
        clean_ds._save(txs)

        resp = client.post("/predict", json={"customer_id": "NONEXISTENT"})
        assert resp.status_code == 404
    finally:
        app.dependency_overrides = orig  # type: ignore[assignment]
        from pymongo import MongoClient

        with MongoClient(mongo_uri) as cli:
            cli[TEST_DB_NAME][coll].drop()
            cli[TEST_DB_NAME][pred_ds.collection].drop()


def test_predict_cancellations_only_404(synthetic_model, mongo_uri, mongo_test_db):
    model, thr, explainer = synthetic_model
    coll = f"clean_{uuid.uuid4().hex[:8]}"
    client, app, orig, pred_ds, clean_ds = _make_client_with_overrides(
        mongo_uri, coll, model, thr, explainer
    )
    try:
        # Customer whose only rows are cancellations
        txs = pd.DataFrame(
            [
                _make_tx("CANCEL_ONLY", "C10001", "2011-10-01", -5, 10.0, "A001", True),
                _make_tx("CANCEL_ONLY", "C10002", "2011-11-01", -2, 20.0, "A002", True),
            ]
        )
        clean_ds._save(txs)

        resp = client.post("/predict", json={"customer_id": "CANCEL_ONLY"})
        assert resp.status_code == 404
        assert "no real purchase" in resp.text.lower() or "customer not found" in resp.text.lower()
    finally:
        app.dependency_overrides = orig  # type: ignore[assignment]
        from pymongo import MongoClient

        with MongoClient(mongo_uri) as cli:
            cli[TEST_DB_NAME][coll].drop()
            cli[TEST_DB_NAME][pred_ds.collection].drop()


def test_predict_predictions_save_failure_logs_error(  # noqa: E501
    synthetic_model, mongo_uri, mongo_test_db, caplog
):
    model, thr, explainer = synthetic_model
    coll = f"clean_{uuid.uuid4().hex[:8]}"
    client, app, orig, pred_ds, clean_ds = _make_client_with_overrides(
        mongo_uri, coll, model, thr, explainer
    )
    try:
        txs = pd.DataFrame(
            [_make_tx("C123", "10001", "2011-10-01", 5, 10.0, "A001", False)]
        )
        clean_ds._save(txs)

        # Make predictions save fail: replace _save with raising mock
        original_save = pred_ds._save

        def failing_save(data):  # type: ignore[no-untyped-def]
            raise RuntimeError("injected save failure")

        pred_ds._save = failing_save  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR, logger="app.main"):
            resp = client.post("/predict", json={"customer_id": "C123"})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["customer_id"] == "C123"
        assert 0 <= body["churn_probability"] <= 1
        # Error was logged at ERROR level
        assert any(r.levelno == logging.ERROR for r in caplog.records)
        assert any("Failed to save prediction" in r.message for r in caplog.records)

        # Restore so cleanup can drop
        pred_ds._save = original_save  # type: ignore[method-assign]
    finally:
        app.dependency_overrides = orig  # type: ignore[assignment]
        from pymongo import MongoClient

        with MongoClient(mongo_uri) as cli:
            cli[TEST_DB_NAME][coll].drop()
            cli[TEST_DB_NAME][pred_ds.collection].drop()
