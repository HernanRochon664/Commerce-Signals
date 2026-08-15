"""Pipeline definition for the `validation` stage (M2)."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.validation.nodes import (
    validate_and_clean_transactions,
    verify_validation_row_count,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the validation pipeline.

    ``validate_and_clean_transactions`` cleans ``raw_transactions``
    into ``clean_transactions`` and a ``validation_report`` JSON;
    ``verify_validation_row_count`` then re-loads both persisted
    outputs through the catalog to confirm the round-trip.
    """
    return Pipeline(
        [
            node(
                func=validate_and_clean_transactions,
                inputs=["raw_transactions", "params:validation.reference_date"],
                outputs=["clean_transactions", "validation_report"],
                name="validate_and_clean_transactions",
            ),
            node(
                func=verify_validation_row_count,
                inputs=["clean_transactions", "validation_report"],
                outputs=None,
                name="verify_validation_row_count",
            ),
        ],
    )
