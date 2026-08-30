"""Pipeline definition for the `explainability` stage (M5)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.explainability.nodes import (
    compute_churn_explanations,
    load_model_for_explanation,
    prepare_explanation_inputs,
    verify_shap_additivity,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the explainability pipeline."""
    return Pipeline(
        [
            node(
                func=prepare_explanation_inputs,
                inputs=[
                    "customer_snapshots",
                    "training_report",
                    "params:explainability.background_sample_size",
                    "params:explainability.explanation_sample_size",
                    "params:explainability.random_state",
                ],
                outputs=["background_snapshots", "explanation_snapshots", "sample_info"],
                name="prepare_explanation_inputs",
            ),
            node(
                func=load_model_for_explanation,
                inputs="training_report",
                outputs="trained_model_for_explanation",
                name="load_model_for_explanation",
            ),
            node(
                func=compute_churn_explanations,
                inputs=[
                    "trained_model_for_explanation",
                    "background_snapshots",
                    "explanation_snapshots",
                    "sample_info",
                    "params:explainability.top_n_examples",
                ],
                outputs="explainability_report",
                name="compute_churn_explanations",
            ),
            node(
                func=verify_shap_additivity,
                inputs=[
                    "explainability_report",
                    "params:explainability.additivity_tolerance",
                ],
                outputs=None,
                name="verify_shap_additivity",
            ),
        ],
    )
