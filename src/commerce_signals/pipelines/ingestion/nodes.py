"""Node functions for the `ingestion` pipeline.

Each node is a stub for M0 — it documents the contract and raises
`NotImplementedError` to make explicit that real behaviour lands in M1.
"""

from __future__ import annotations


def load_source_to_raw_transactions(
    source_path: str,  # noqa: ARG001
) -> None:
    """Load the Online Retail II source file into the raw Mongo collection.

    Args:
        source_path: Path to the Online Retail II source file
            (Excel/CSV), taken from ``params:ingestion.source_path``.

    Returns:
        None. Data is persisted via the `raw_transactions` output
        dataset (MongoDB collection).

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M1")
