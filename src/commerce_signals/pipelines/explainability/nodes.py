"""Node functions for the `explainability` pipeline (M0 stub)."""

from __future__ import annotations


def compute_shap_values(
    churn_model,  # noqa: ARG001
    X_test,  # noqa: ARG001
) -> None:
    """Compute SHAP values for test-set predictions.

    Args:
        churn_model: Trained LightGBM model.
        X_test: Test features used to compute SHAP.

    Returns:
        A SHAP explanation object saved via the ``shap_explanations``
        output (used by global reporting and per-customer explanations).

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M5")


def build_global_importance_report(
    shap_explanations,  # noqa: ARG001
) -> None:
    """Build the global feature-importance report from SHAP values.

    Args:
        shap_explanations: Output of ``compute_shap_values``.

    Returns:
        A report artefact saved via the ``global_importance_report``
        output (HTML or JSON).

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M5")
