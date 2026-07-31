"""Pipeline definition for the `explainability` stage (M0 stub)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.explainability.nodes import (
    build_global_importance_report,
    compute_shap_values,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the explainability pipeline."""
    return Pipeline(
        [
            node(
                func=compute_shap_values,
                inputs=["churn_model", "X_test"],
                outputs="shap_explanations",
                name="compute_shap_values",
                tags=["m5-stub"],
            ),
            node(
                func=build_global_importance_report,
                inputs="shap_explanations",
                outputs="global_importance_report",
                name="build_global_importance_report",
                tags=["m5-stub"],
            ),
        ],
    )
