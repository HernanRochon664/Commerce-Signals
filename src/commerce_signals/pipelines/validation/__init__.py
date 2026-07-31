"""Validation pipeline (M0 stub).

Validates and cleans `raw_transactions`, producing a `clean_transactions`
MongoDB collection plus a validation report describing what was dropped
and why. Real implementation lands in M2.
"""

from commerce_signals.pipelines.validation.pipeline import create_pipeline  # noqa: F401
