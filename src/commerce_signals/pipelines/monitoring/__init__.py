"""Monitoring pipeline (M0 stub).

Generates drift reports with Evidently (local HTML artefacts) and
exposes latency/error metrics to Cloud Monitoring. Real implementation
lands in M8.
"""

from commerce_signals.pipelines.monitoring.pipeline import create_pipeline  # noqa: F401
