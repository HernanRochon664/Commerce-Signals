"""Shared SHAP explainability helpers (M5).

Pure functions shared between the explainability pipeline and potential
future consumers. No logging, no Kedro, no Mongo -- same contract as
``commerce_signals.features``.
"""

from __future__ import annotations

import pandas as pd
import shap

from commerce_signals.features import FEATURE_COLUMNS


def build_explainer(
    model, background_data: pd.DataFrame  # noqa: ANN001
) -> shap.TreeExplainer:
    """Build a TreeExplainer in probability space.

    Args:
        model: Fitted ``lightgbm.LGBMClassifier``.
        background_data: DataFrame with ``FEATURE_COLUMNS`` as columns,
            used as the interventional background distribution (100-1000
            rows per SHAP docs; sampling is handled by the caller).

    Returns:
        A ``shap.TreeExplainer`` configured with
        ``model_output="probability"`` and
        ``feature_perturbation="interventional"`` so each SHAP value
        is expressed in percentage-points of churn risk.
    """
    return shap.TreeExplainer(
        model,
        data=background_data,
        model_output="probability",
        feature_perturbation="interventional",
    )


def compute_shap_explanations(
    explainer: shap.TreeExplainer, X: pd.DataFrame  # noqa: ANN001
) -> shap.Explanation:
    """Compute SHAP explanations for *X*.

    Defensive handling of the ``values`` shape: for a binary
    ``LGBMClassifier`` with ``model_output="probability"`` the SHAP docs
    and ecosystem history are ambiguous about whether
    ``explainer(X).values`` is 2-D ``(n_rows, n_features)`` or carries
    an extra class dimension e.g. ``(n_rows, n_features, 2)`` or
    ``(n_rows, n_features, 1)``.  In practice (verified 2026-08-24 with
    ``shap==0.52.0`` + ``lightgbm>=4.7``) the observed shape is strictly
    2-D ``(n_rows, n_features)`` for binary classification in probability
    mode -- see the manual probe in the dev log / final summary.

    This function verifies the shape explicitly and, if an extra trailing
    class dimension is present, slices the positive-class channel so the
    caller always receives a 2-D ``(n_rows, n_features)`` view aligned
    with ``FEATURE_COLUMNS``.

    Args:
        explainer: TreeExplainer built by ``build_explainer``.
        X: DataFrame with ``FEATURE_COLUMNS`` columns to explain.

    Returns:
        A ``shap.Explanation`` whose ``.values`` is 2-D
        ``(len(X), len(FEATURE_COLUMNS))`` and ``.base_values`` is 1-D
        ``(len(X),)`` (or scalar broadcast).
    """
    explanation = explainer(X)

    values = explanation.values
    n_rows = len(X)
    n_features = len(FEATURE_COLUMNS)
    expected_2d = (n_rows, n_features)

    # Defensive: handle historical / version-dependent extra class axis.
    # Observed in practice with shap==0.52.0: shape is exactly 2-D, so
    # this branch is not taken, but it guards against a future
    # ``(n_rows, n_features, n_classes)`` layout (e.g. ``(n, 7, 2)``)
    # that some LightGBM + SHAP combinations have reportedly emitted.
    if values.shape != expected_2d:
        # Common variant: (n_rows, n_features, n_classes) or (n_rows, n_features, 1)
        if values.ndim == 3 and values.shape[0] == n_rows and values.shape[1] == n_features:
            # Binary classifier: keep the positive-class channel (last axis index 1
            # if 2 classes, index 0 if only one class dimension was emitted).
            # We slice explicitly and document which index was taken.
            if values.shape[2] == 2:
                # Shape (n, n_features, 2): [:,:,1] is the positive class.
                explanation.values = values[:, :, 1]
            elif values.shape[2] == 1:
                # Shape (n, n_features, 1): squeeze the singleton class axis.
                explanation.values = values[:, :, 0]
            else:
                # Unexpected 3-D shape: fall back to squeezing last axis if singleton,
                # otherwise raise so the mismatch is visible rather than silent.
                if values.shape[2] == 1:
                    explanation.values = values.squeeze(axis=2)
                else:
                    raise ValueError(  # noqa: TRY003
                        f"Unexpected SHAP values shape {values.shape}: expected {expected_2d} "
                        f"or (n_rows, n_features, n_classes); got ndim={values.ndim}."
                    )
            # base_values may also carry a class axis in this regime.
            base = explanation.base_values
            if hasattr(base, "ndim") and getattr(base, "ndim", 0) == 2:
                # Shape (n_rows, n_classes) -> keep positive class.
                if base.shape[1] == 2:
                    explanation.base_values = base[:, 1]
                elif base.shape[1] == 1:
                    explanation.base_values = base[:, 0]
            elif hasattr(base, "ndim") and getattr(base, "ndim", 0) == 1 and base.shape[0] == 2:
                # Rare: base_values shape (2,) for binary -> take index 1.
                explanation.base_values = base[1]  # type: ignore[index]
        elif values.ndim == 3 and values.shape[0] == n_rows and values.shape[2] == n_features:
            # Transposed variant (n_rows, n_classes, n_features) seen in some wrappers.
            # Reorder to (n_rows, n_features) by taking positive class.
            explanation.values = values[:, 1, :] if values.shape[1] == 2 else values[:, 0, :]
            base = explanation.base_values
            if hasattr(base, "ndim") and getattr(base, "ndim", 0) == 2 and base.shape[1] == 2:
                explanation.base_values = base[:, 1]
        else:
            raise ValueError(  # noqa: TRY003
                f"Unexpected SHAP values shape {values.shape}: expected {expected_2d}. "
                f"Observed ndim={values.ndim}. If SHAP emitted an extra class dimension, "
                f"update the defensive handler above to slice it explicitly."
            )

    return explanation
