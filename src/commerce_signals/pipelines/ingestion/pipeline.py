"""Pipeline definition for the `ingestion` stage.

M1:
    * Read the Online Retail II source workbook untouched.
    * Persist it to the ``raw_transactions`` MongoDB collection
      (which is configured with ``mode: replace`` in the catalog, so
      the collection is fully refreshed on every run).
    * Reload from Mongo and log a row-count comparison against the
      source file. The comparison is observability only; mismatches
      do not fail the pipeline.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.ingestion.nodes import (
    load_source_to_raw_transactions,
    verify_ingestion_row_count,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the ingestion pipeline.

    Returns:
        A ``Pipeline`` with two nodes:

        1. ``load_source_to_raw_transactions`` reads the workbook and
           writes the concatenated DataFrame to ``raw_transactions``
           (Mongo). It also returns ``ingestion_row_counts`` (an
           in-memory dict, no catalog entry) with per-sheet totals.
        2. ``verify_ingestion_row_count`` reads ``raw_transactions``
           back from Mongo and compares its length to the source
           count, logging INFO on match and WARNING on mismatch.
    """
    return Pipeline(
        [
            node(
                func=load_source_to_raw_transactions,
                inputs="params:ingestion.source_path",
                outputs=["raw_transactions", "ingestion_row_counts"],
                name="load_source_to_raw_transactions",
            ),
            node(
                func=verify_ingestion_row_count,
                inputs=["raw_transactions", "ingestion_row_counts"],
                outputs=None,
                name="verify_ingestion_row_count",
            ),
        ],
    )
