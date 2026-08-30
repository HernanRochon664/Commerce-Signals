"""Node functions for the `explainability` pipeline (M5)."""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from commerce_signals.explainability import build_explainer, compute_shap_explanations
from commerce_signals.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def prepare_explanation_inputs(
    customer_snapshots: pd.DataFrame,
    training_report: dict[str, Any],
    background_sample_size: int,
    explanation_sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare background and explanation DataFrames for SHAP.

    The split is never recomputed from parameters: exact snapshot dates
    stored in ``training_report["split_info"]`` are used, guaranteeing
    the same test set that was evaluated in M4 is explained.

    Args:
        customer_snapshots: Full ``customer_snapshots`` table (all
            snapshot dates, from Mongo).
        training_report: Report from the training pipeline (contains
            ``split_info`` with ``train_snapshot_dates`` and
            ``test_snapshot_dates`` as ISO strings).
        background_sample_size: Requested number of train rows for the
            SHAP background distribution (100-1000 per SHAP docs).
        explanation_sample_size: Requested number of test rows to
            explain.
        random_state: Seed for deterministic sampling.

    Returns:
        A tuple ``(background_df, explanation_df, sample_info)`` filtered
        to the exact train / test snapshot dates and sampled to the
        requested sizes (or the full pool if the pool is smaller, with a
        WARNING). ``sample_info`` carries requested/used sizes and
        ``run_id`` for downstream nodes.
    """
    snapshots = customer_snapshots.copy()
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])

    train_dates = pd.to_datetime(training_report["split_info"]["train_snapshot_dates"])
    test_dates = pd.to_datetime(training_report["split_info"]["test_snapshot_dates"])

    train_pool = snapshots[snapshots["snapshot_date"].isin(train_dates)]
    test_pool = snapshots[snapshots["snapshot_date"].isin(test_dates)]

    if len(train_pool) < background_sample_size:
        logger.warning(
            "Background pool has %d rows, less than requested background_sample_size=%d; "
            "using the full pool (%d rows)",
            len(train_pool),
            background_sample_size,
            len(train_pool),
        )
        background_df = train_pool
    else:
        background_df = train_pool.sample(
            n=background_sample_size, random_state=random_state
        )

    if len(test_pool) < explanation_sample_size:
        logger.warning(
            "Explanation pool has %d rows, less than requested explanation_sample_size=%d; "
            "using the full pool (%d rows)",
            len(test_pool),
            explanation_sample_size,
            len(test_pool),
        )
        explanation_df = test_pool
    else:
        explanation_df = test_pool.sample(
            n=explanation_sample_size, random_state=random_state
        )

    sample_info = {
        "run_id": str(training_report.get("run_id", "")),
        "background_sample_size_requested": int(background_sample_size),
        "background_sample_size_used": int(len(background_df)),
        "explanation_sample_size_requested": int(explanation_sample_size),
        "explanation_sample_size_used": int(len(explanation_df)),
    }

    logger.info(
        "Prepared explainability inputs: background %d/%d rows (requested %d), "
        "explanation %d/%d rows (requested %d)",
        len(background_df),
        len(train_pool),
        background_sample_size,
        len(explanation_df),
        len(test_pool),
        explanation_sample_size,
    )

    return background_df, explanation_df, sample_info


def load_model_for_explanation(training_report: dict[str, Any]) -> lgb.LGBMClassifier:
    """Load the trained LightGBM model from MLflow for explanation.

    The tracking URI is set explicitly because this pipeline runs in a
    separate Kedro invocation from training (M4), so the URI is not
    inherited in memory.

    Args:
        training_report: Report dict containing ``run_id``.

    Returns:
        The fitted ``lightgbm.LGBMClassifier`` reloaded from
        ``runs:/<run_id>/model``.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    run_id = training_report["run_id"]
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    logger.info("Loaded model for explanation from run %s", run_id)
    return model


def compute_churn_explanations(
    trained_model: lgb.LGBMClassifier,
    background_snapshots: pd.DataFrame,
    explanation_snapshots: pd.DataFrame,
    sample_info: dict[str, Any],
    top_n_examples: int,
) -> dict[str, Any]:
    """Compute SHAP explanations in probability space.

    SHAP values are expressed in probability points (model_output
    ``probability`` + interventional perturbation), so each contribution
    reads as a change in churn risk.

    Args:
        trained_model: LightGBM classifier loaded from MLflow.
        background_snapshots: Background pool (sample of train).
        explanation_snapshots: Test rows to explain.
        sample_info: Dict with ``run_id``, ``background_sample_size_requested``,
            ``background_sample_size_used``, ``explanation_sample_size_requested``,
            ``explanation_sample_size_used`` (produced by
            ``prepare_explanation_inputs``).
        top_n_examples: Number of highest-risk examples to include in
            ``example_explanations`` (sorted by predicted probability).

    Returns:
        A dict with the exact shape::

            {
                "run_id": str,
                "background_sample_size_requested": int,
                "background_sample_size_used": int,
                "explanation_sample_size_requested": int,
                "explanation_sample_size_used": int,
                "global_importance": [
                    {"feature": str, "mean_abs_shap": float}, ...
                ],
                "example_explanations": [
                    {
                        "customer_id": str,
                        "snapshot_date": str,
                        "predicted_probability": float,
                        "base_value": float,
                        "feature_contributions": {feature: float, ...}
                    }, ...
                ]
            }
    """
    X_background = background_snapshots[FEATURE_COLUMNS]
    X_explain = explanation_snapshots[FEATURE_COLUMNS]

    run_id = str(sample_info.get("run_id", ""))
    bg_requested = int(sample_info["background_sample_size_requested"])
    bg_used = int(sample_info["background_sample_size_used"])
    exp_requested = int(sample_info["explanation_sample_size_requested"])
    exp_used = int(sample_info["explanation_sample_size_used"])

    explainer = build_explainer(trained_model, X_background)
    explanation = compute_shap_explanations(explainer, X_explain)

    values = explanation.values
    # Extra safety: if wrapper still returned 3-D, handle here as well
    # (compute_shap_explanations already slices, but keep guard).
    if values.ndim == 3:
        if values.shape[2] == 2:
            values = values[:, :, 1]
        elif values.shape[2] == 1:
            values = values[:, :, 0]
        else:
            values = values.squeeze(axis=2)

    # Global importance: mean |SHAP| per feature, sorted descending
    mean_abs = np.abs(values).mean(axis=0)
    importance_pairs = sorted(
        zip(FEATURE_COLUMNS, mean_abs, strict=True),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    global_importance = [
        {"feature": str(feat), "mean_abs_shap": float(val)}
        for feat, val in importance_pairs
    ]

    # Predicted probabilities for ranking
    proba = trained_model.predict_proba(X_explain)[:, 1]

    # Select top_n_examples by predicted prob
    n_examples = min(int(top_n_examples), len(explanation_snapshots))
    top_indices = np.argsort(proba)[::-1][:n_examples]

    # base_values: per-row; shap may return scalar, 1-D, or 2-D
    base_vals = explanation.base_values
    # Normalize to 1-D array of length n_rows
    if np.ndim(base_vals) == 0:
        base_vals = np.full(len(X_explain), float(base_vals))
    elif isinstance(base_vals, np.ndarray) and base_vals.ndim == 2:
        # Shape (n_rows, n_classes) e.g. (n,2) -> take positive class
        if base_vals.shape[1] == 2:
            base_vals = base_vals[:, 1]
        elif base_vals.shape[1] == 1:
            base_vals = base_vals[:, 0]
        else:
            base_vals = base_vals.squeeze()

    example_explanations: list[dict[str, Any]] = []
    for idx in top_indices:
        row = explanation_snapshots.iloc[int(idx)]
        # customer_id and snapshot_date are in the original snapshots frame
        cid = str(row["customer_id"])
        sdate = pd.to_datetime(row["snapshot_date"]).isoformat()
        pred_prob = float(proba[int(idx)])
        bv = float(base_vals[int(idx)]) if hasattr(base_vals, "__len__") else float(base_vals)
        contributions = {
            str(feat): float(values[int(idx), j])
            for j, feat in enumerate(FEATURE_COLUMNS)
        }
        example_explanations.append(
            {
                "customer_id": cid,
                "snapshot_date": sdate,
                "predicted_probability": pred_prob,
                "base_value": bv,
                "feature_contributions": contributions,
            }
        )

    # Already sorted descending by predicted_probability due to argsort
    if global_importance:
        top3 = ", ".join(
            f"{e['feature']}={e['mean_abs_shap']:.4f}" for e in global_importance[:3]
        )
        top_prob = example_explanations[0]["predicted_probability"] if example_explanations else 0.0
        logger.info(
            "SHAP global importance top 3: %s | top risk example prob=%.4f",
            top3,
            top_prob,
        )

    return {
        "run_id": str(run_id),
        "background_sample_size_requested": int(bg_requested),
        "background_sample_size_used": int(bg_used),
        "explanation_sample_size_requested": int(exp_requested),
        "explanation_sample_size_used": int(exp_used),
        "global_importance": global_importance,
        "example_explanations": example_explanations,
    }


def verify_shap_additivity(
    explainability_report: dict[str, Any],
    tolerance: float,
) -> None:
    """Verify SHAP additivity: base_value + sum(SHAP) ≈ predicted_probability.

    This is a mathematical property of the SHAP decomposition, not an
    integrity check: a violation indicates the SHAP calculation itself
    is wrong.

    Args:
        explainability_report: Report dict as persisted to JSON (round-
            trip via Kedro). Must contain ``example_explanations``.
        tolerance: Absolute tolerance for the additivity check.
    """
    examples = explainability_report.get("example_explanations", [])
    n = len(examples)
    if n == 0:
        logger.info("SHAP additivity check: no examples to verify (0/0 passed)")
        return

    passed = 0
    for ex in examples:
        base = float(ex["base_value"])
        contribs = ex.get("feature_contributions", {})
        s = float(sum(float(v) for v in contribs.values()))
        reconstructed = base + s
        predicted = float(ex["predicted_probability"])
        delta = abs(reconstructed - predicted)
        cid = ex.get("customer_id", "?")
        sdate = ex.get("snapshot_date", "?")
        if delta <= tolerance:
            passed += 1
            logger.info(
                "SHAP additivity passed for customer %s @ %s: "
                "reconstructed=%.6f predicted=%.6f delta=%.2e <= tolerance=%.2e",
                cid,
                sdate,
                reconstructed,
                predicted,
                delta,
                tolerance,
            )
        else:
            logger.warning(
                "SHAP additivity FAILED for customer %s @ %s: "
                "base=%.6f + sum(SHAP)=%.6f => reconstructed=%.6f vs "
                "predicted=%.6f delta=%.6f exceeds tolerance=%.6f",
                cid,
                sdate,
                base,
                s,
                reconstructed,
                predicted,
                delta,
                tolerance,
            )

    logger.info(
        "SHAP additivity summary: %d/%d examples passed (tolerance=%.6f)",
        passed,
        n,
        tolerance,
    )
