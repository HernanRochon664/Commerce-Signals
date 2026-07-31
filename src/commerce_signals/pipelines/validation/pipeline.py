"""Pipeline definition for the `validation` stage (M0 stub)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.validation.nodes import (
    build_validation_report,
    validate_and_clean,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the validation pipeline."""
    return Pipeline(
        [
            node(
                func=validate_and_clean,
                inputs="raw_transactions",
                outputs="clean_transactions",
                name="validate_and_clean",
                tags=["m2-stub"],
            ),
            node(
                func=build_validation_report,
                inputs=["raw_transactions", "clean_transactions"],
                outputs="validation_report",
                name="build_validation_report",
                tags=["m2-stub"],
            ),
        ],
    )
