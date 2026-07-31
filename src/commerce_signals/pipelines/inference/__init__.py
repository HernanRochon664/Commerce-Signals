"""Inference pipeline (M0 stub).

Exposes the trained model behind a FastAPI endpoint that:
- looks up customer features in MongoDB,
- predicts churn probability,
- persists the prediction (with timestamp) to the `predictions` collection,
- returns the prediction (and optional SHAP summary) to the caller.

Real implementation lands in M6.
"""

from commerce_signals.pipelines.inference.pipeline import create_pipeline  # noqa: F401
