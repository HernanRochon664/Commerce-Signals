"""Pipeline definition for the `ingestion` stage (M0 stub).

This pipeline will, in M1, download or read the Online Retail II dataset
and write it as-is into the `raw_transactions` MongoDB collection. For
M0 the pipeline is empty (no nodes), so Kedro can still import the
project and report it in `kedro registry list`.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from commerce_signals.pipelines.ingestion.nodes import load_source_to_raw_transactions


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the ingestion pipeline.

    Returns:
        A ``Pipeline`` with the stub node wired to
        ``params:ingestion.source_path`` as input and
        ``raw_transactions`` as output.
    """
    return Pipeline(
        [
            node(
                func=load_source_to_raw_transactions,
                inputs="params:ingestion.source_path",
                outputs="raw_transactions",
                name="load_source_to_raw_transactions",
                tags=["m1-stub"],
            ),
        ],
    )
