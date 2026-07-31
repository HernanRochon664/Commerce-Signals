"""Pipeline definition for the `inference` stage (M0 stub).

In M6 this pipeline is invoked by the FastAPI app via the
``KedroSession`` ``run()`` API with ``runtime_params`` containing the
``customer_id`` from the request. For M0 the pipeline exists with a
single stub node so it can be discovered and validated.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.inference.nodes import predict_for_customer


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the inference pipeline."""
    return Pipeline(
        [
            node(
                func=predict_for_customer,
                inputs={
                    "customer_id": "params:customer_id",
                    "churn_model": "churn_model",
                    "customer_features": "customer_features",
                },
                outputs="predictions",
                name="predict_for_customer",
                tags=["m6-stub"],
            ),
        ],
    )
