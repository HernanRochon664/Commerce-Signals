"""Ingestion pipeline (M0 stub).

Reads the Online Retail II source and loads it, untouched, into the
`raw_transactions` MongoDB collection. Real implementation lands in M1.
"""

from commerce_signals.pipelines.ingestion.pipeline import create_pipeline  # noqa: F401
