"""Composition root for Kedro pipelines.

`register_pipelines` is invoked by Kedro to discover all pipelines in
this project. Each pipeline module exposes a `create_pipeline()`
function in its `pipeline.py`. New pipelines are added as discovered.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline

from commerce_signals.pipelines import (
    explainability,
    feature_engineering,
    inference,
    ingestion,
    monitoring,
    training,
    validation,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Map pipeline names to their constructed pipelines.

    Returns:
        A dict mapping pipeline names (e.g. ``"ingestion"``) to
        ``Pipeline`` instances. Run a specific pipeline via
        ``kedro run --pipeline <name>``.
    """
    pipelines: dict[str, Pipeline] = {
        "ingestion": ingestion.create_pipeline(),
        "validation": validation.create_pipeline(),
        "feature_engineering": feature_engineering.create_pipeline(),
        "training": training.create_pipeline(),
        "explainability": explainability.create_pipeline(),
        "inference": inference.create_pipeline(),
        "monitoring": monitoring.create_pipeline(),
    }

    return pipelines
