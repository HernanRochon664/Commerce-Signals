"""Commerce Signals source package.

Re-exports the custom MongoDB dataset so Kedro's catalog can reference
it via ``type: commerce_signals.datasets.MongoCollectionDataset``.
"""

from __future__ import annotations

from commerce_signals.datasets import MongoCollectionDataset

__version__ = "0.1.0"

__all__ = ["MongoCollectionDataset", "__version__"]
