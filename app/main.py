"""FastAPI application for Commerce Signals (M6).

Provides:
    GET /health  — liveness probe
    POST /predict — churn prediction per customer

Startup (lifespan) loads once:
    - MLflow tracking URI
    - training_report.json (direct file read, no Kedro)
    - trained LightGBM model via mlflow.lightgbm
    - threshold from training report
    - Mongo datasets for clean_transactions / customer_snapshots / predictions
    - SHAP background + explainer
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from commerce_signals.datasets import MongoCollectionDataset
from commerce_signals.explainability import (
    build_explainer,
    build_shap_background,
    compute_shap_explanations,
)
from commerce_signals.features import FEATURE_COLUMNS, compute_customer_features

load_dotenv()

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# --- Pydantic models ---------------------------------------------------------


class PredictRequest(BaseModel):
    customer_id: str


class RiskFactor(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float
    is_at_risk: bool
    as_of_date: str
    top_risk_factors: list[RiskFactor]


# --- Helpers for lifespan ----------------------------------------------------


def _get_mongo_credentials() -> dict[str, str]:
    """Credentials from env vars (NOT from Kedro conf)."""
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    db = os.environ.get("MONGODB_DB_NAME", "commerce_signals")
    return {"uri": uri, "db": db}


def _load_training_report(path: str = "data/08_reporting/training_report.json") -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _extract_threshold(training_report: dict[str, Any]) -> float:
    # Spec says training_report["evaluation_metrics"]["threshold_used"].
    # The training pipeline's actual report (M4) may nest threshold inside
    # evaluation_metrics or at top-level; be defensive and support both.
    em = training_report.get("evaluation_metrics", {})
    if isinstance(em, dict) and "threshold_used" in em:
        return float(em["threshold_used"])
    # Fallback: _best_f1_threshold result stored elsewhere
    if "threshold_used" in training_report:
        return float(training_report["threshold_used"])
    # Last resort: evaluation_metrics may have been flattened differently
    raise KeyError("threshold_used not found in training_report")


# --- Lifespan ----------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    # Load once at startup; let any exception propagate (fail hard, not half-ready).
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    training_report = _load_training_report("data/08_reporting/training_report.json")
    run_id = training_report["run_id"]
    trained_model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    threshold = _extract_threshold(training_report)

    creds = _get_mongo_credentials()
    clean_ds = MongoCollectionDataset(collection="clean_transactions", credentials=creds)
    predictions_ds = MongoCollectionDataset(
        collection="predictions", credentials=creds, mode="append"
    )
    # Customer snapshots needed for SHAP background (train dates filter)
    snapshots_ds = MongoCollectionDataset(collection="customer_snapshots", credentials=creds)
    customer_snapshots = snapshots_ds._load()

    # Background params hard-coded to match M5 parameters.yml
    # (explainability.background_sample_size=100, random_state=42).
    # Reading parameters.yml at runtime would couple the API image to Kedro
    # conf/ layout; hard-coding with a comment keeps the values in sync
    # without that coupling and matches the spec's instruction.
    background_sample_size = 100  # == parameters.yml explainability.background_sample_size
    background_random_state = 42  # == parameters.yml explainability.random_state
    train_snapshot_dates: list[str] = training_report["split_info"]["train_snapshot_dates"]
    background_df = build_shap_background(
        customer_snapshots, train_snapshot_dates, background_sample_size, background_random_state
    )
    # build_explainer expects FEATURE_COLUMNS subset
    background_for_explainer = background_df[FEATURE_COLUMNS]
    explainer = build_explainer(trained_model, background_for_explainer)

    app.state.training_report = training_report
    app.state.trained_model = trained_model
    app.state.threshold = threshold
    app.state.clean_transactions_ds = clean_ds
    app.state.predictions_ds = predictions_ds
    app.state.explainer = explainer
    app.state.background_df = background_df

    logger.info("API startup complete: model run %s threshold %.4f", run_id, threshold)

    yield

    # Shutdown: pymongo opens a client per request via MongoCollectionDataset
    # (no persistent pool held in lifespan), so no explicit close needed.
    logger.info("API shutdown")


app = FastAPI(
    title="Commerce Signals",
    description="Churn prediction API for e-commerce customers.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Dependency injectors (overridable in tests) -----------------------------


def get_trained_model(request: Request):  # type: ignore[no-untyped-def]
    return request.app.state.trained_model


def get_threshold(request: Request):  # type: ignore[no-untyped-def]
    return request.app.state.threshold


def get_explainer(request: Request):  # type: ignore[no-untyped-def]
    return request.app.state.explainer


def get_clean_transactions_dataset(request: Request):  # type: ignore[no-untyped-def]
    return request.app.state.clean_transactions_ds


def get_predictions_dataset(request: Request):  # type: ignore[no-untyped-def]
    return request.app.state.predictions_ds


# Compat aliases so tests can override via either name style
# (e.g. app.dependency_overrides[get_model] )
get_model = get_trained_model  # noqa: N816


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    request: Request,
    trained_model=Depends(get_trained_model),  # noqa: B008
    threshold: float = Depends(get_threshold),  # noqa: B008
    explainer=Depends(get_explainer),  # noqa: B008
    clean_ds=Depends(get_clean_transactions_dataset),  # noqa: B008
    predictions_ds=Depends(get_predictions_dataset),  # noqa: B008
) -> PredictResponse:
    customer_id = str(payload.customer_id)

    # 1. Lookup history via Mongo filtered query
    # clean_ds is a MongoCollectionDataset; use load_filtered for targeted lookup.
    # Dependency injection in tests replaces this with a dataset pointing at TEST_MONGODB.
    try:
        # Use load_filtered if available (M6), fallback to _load filtered manually
        if hasattr(clean_ds, "load_filtered"):
            history = clean_ds.load_filtered({"customer_id": customer_id})
        else:
            # Should not happen in prod, but keep for test mocks
            history = clean_ds._load()  # type: ignore[attr-defined]
            history = (
                history[history["customer_id"] == customer_id]
                if not history.empty
                else history
            )
    except Exception as exc:  # pragma: no cover - unexpected Mongo failure
        logger.error("Failed to load history for %s: %s", customer_id, exc)
        raise HTTPException(status_code=500, detail="Failed to load customer history") from exc

    if history.empty:
        raise HTTPException(status_code=404, detail="customer not found or no purchase history")

    # 2. Compute features at now
    as_of = pd.Timestamp.now()
    # Ensure invoice_date is datetime for compute_customer_features
    if "invoice_date" in history.columns:
        history = history.copy()
        history["invoice_date"] = pd.to_datetime(history["invoice_date"])
    features = compute_customer_features(history, as_of_date=as_of)

    if features.empty:
        raise HTTPException(
            status_code=404, detail="customer has no real purchase history before as_of_date"
        )

    # Single row for this customer
    # compute_customer_features returns one row per customer; take our customer
    if customer_id not in features.index:
        raise HTTPException(
            status_code=404, detail="customer has no real purchase history before as_of_date"
        )
    X = features.loc[[customer_id]]

    # 3. Score
    churn_probability = float(trained_model.predict_proba(X)[:, 1][0])
    is_at_risk = bool(churn_probability >= float(threshold))

    # 4. SHAP per-customer risk factors (top 3 by abs shap)
    explanation = compute_shap_explanations(explainer, X)
    values = explanation.values
    # values is 2-D (1, n_features) after defensive slicing in compute_shap_explanations
    if values.ndim == 3:  # fallback guard
        values = values[:, :, 1] if values.shape[2] == 2 else values[:, :, 0]
    row_shap = values[0]
    # Pair feature -> shap, sort by abs value desc
    pairs = sorted(
        zip(FEATURE_COLUMNS, row_shap, strict=True),
        key=lambda x: abs(float(x[1])),
        reverse=True,
    )
    top_factors = [
        RiskFactor(feature=str(feat), shap_value=float(val)) for feat, val in pairs[:3]
    ]

    response = PredictResponse(
        customer_id=customer_id,
        churn_probability=float(churn_probability),
        is_at_risk=bool(is_at_risk),
        as_of_date=as_of.isoformat(),
        top_risk_factors=top_factors,
    )

    # 5. Audit write to predictions (append-only). Failure must not fail the request.
    try:
        record = pd.DataFrame(
            [
                {
                    "customer_id": customer_id,
                    "churn_probability": float(churn_probability),
                    "is_at_risk": bool(is_at_risk),
                    "timestamp": as_of.isoformat(),
                }
            ]
        )
        predictions_ds._save(record)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - tested via mock
        logger.error("Failed to save prediction for %s: %s", customer_id, exc)

    return response
