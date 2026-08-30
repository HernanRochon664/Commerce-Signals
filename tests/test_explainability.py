"""Tests for the shared explainability helpers (M5).

Pure, no Mongo, no MLflow: trains a tiny LGBMClassifier on synthetic
data and exercises build_explainer + compute_shap_explanations
directly, including the central SHAP additivity contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from commerce_signals.explainability import build_explainer, compute_shap_explanations
from commerce_signals.features import FEATURE_COLUMNS


def _synthetic_frame(n_rows: int = 60, seed: int = 42) -> pd.DataFrame:
    """Tiny synthetic frame with FEATURE_COLUMNS + binary label.

    Churn is a deterministic function of recency_days so the model
    has signal; other features are random but reproducible.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for col in FEATURE_COLUMNS:
        if col == "recency_days":
            data[col] = rng.integers(0, 120, size=n_rows).astype(float)
        elif col == "frequency":
            data[col] = rng.integers(1, 15, size=n_rows).astype(float)
        elif col == "return_rate":
            data[col] = rng.random(n_rows) * 0.3
        else:
            data[col] = rng.random(n_rows) * 100 + 10
    df = pd.DataFrame(data)
    # Deterministic label: churn if recency > 50
    df["is_churned"] = (df["recency_days"] > 50).astype(int)
    return df


@pytest.fixture
def trained_model():
    """Tiny LGBMClassifier trained on synthetic data."""
    import lightgbm as lgb

    df = _synthetic_frame(n_rows=80, seed=0)
    X = df[FEATURE_COLUMNS]
    y = df["is_churned"]
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=4,
        n_estimators=20,
        learning_rate=0.05,
        random_state=42,
        min_data_in_leaf=3,
        verbose=-1,
    )
    model.fit(X, y)
    return model, df


def test_build_explainer_and_compute_runs(trained_model) -> None:
    model, df = trained_model
    X = df[FEATURE_COLUMNS]
    background = X.sample(20, random_state=42)
    explainer = build_explainer(model, background)
    # explainer should be a shap.TreeExplainer
    assert explainer is not None
    X_explain = X.sample(10, random_state=1)
    explanation = compute_shap_explanations(explainer, X_explain)
    assert explanation is not None
    assert hasattr(explanation, "values")
    assert hasattr(explanation, "base_values")


def test_explanation_values_shape(trained_model) -> None:
    model, df = trained_model
    X = df[FEATURE_COLUMNS]
    background = X.sample(20, random_state=42)
    explainer = build_explainer(model, background)
    n_rows = 15
    X_explain = X.sample(n_rows, random_state=7)
    explanation = compute_shap_explanations(explainer, X_explain)
    # Shape must be (n_rows, n_features) in probability space for
    # binary LGBMClassifier. Observed with shap==0.52.0: 2-D.
    assert explanation.values.shape == (n_rows, len(FEATURE_COLUMNS))
    # base_values should be 1-D with length n_rows (or scalar)
    bv = explanation.base_values
    if hasattr(bv, "__len__") and not isinstance(bv, float):
        # Could be ndarray with shape (n_rows,) or scalar broadcast
        if isinstance(bv, np.ndarray):
            assert bv.shape == (n_rows,) or bv.ndim == 0
        else:
            assert len(bv) == n_rows


def test_shap_additivity_direct(trained_model) -> None:
    """Central mathematical contract: base + sum(SHAP) == predict_proba.

    For each row, base_value + sum(shap_values[row]) must approximate
    model.predict_proba(row)[:,1] within a small tolerance (probability
    space, interventional perturbation).
    """
    model, df = trained_model
    X = df[FEATURE_COLUMNS]
    background = X.sample(25, random_state=42)
    explainer = build_explainer(model, background)
    X_explain = X.sample(12, random_state=99)
    explanation = compute_shap_explanations(explainer, X_explain)

    proba = model.predict_proba(X_explain)[:, 1]
    values = explanation.values
    base_vals = explanation.base_values

    # Normalize base_vals to 1-D
    if np.ndim(base_vals) == 0:
        base_vals = np.full(len(X_explain), float(base_vals))
    elif isinstance(base_vals, np.ndarray) and base_vals.ndim == 2:
        if base_vals.shape[1] == 2:
            base_vals = base_vals[:, 1]
        else:
            base_vals = base_vals[:, 0]

    for i in range(len(X_explain)):
        reconstructed = float(base_vals[i]) + float(values[i].sum())
        assert reconstructed == pytest.approx(float(proba[i]), abs=1e-4), (
            f"Row {i}: base {base_vals[i]} + sum SHAP {values[i].sum()} = "
            f"{reconstructed} vs proba {proba[i]}"
        )
