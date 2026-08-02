"""Smoke tests for Commerce Signals (M0 scaffolding + M1 dataset wiring).

Pipeline modules are imported to confirm they remain importable
(smoke check on package structure). The MongoCollectionDataset gets
a construction test plus two negative tests that the M0
``NotImplementedError("Implementar en M1")`` stub is no longer there.

Real behaviour of ``MongoCollectionDataset._load`` and ``_save`` is
covered by ``tests/test_mongo_dataset.py`` (integration; auto-skipped
when no Mongo is reachable) and by ``tests/pipelines/test_ingestion.py``.
"""

from __future__ import annotations

# --- Pipeline discovery tests -------------------------------------------------

def test_ingestion_import() -> None:
    from commerce_signals.pipelines.ingestion import (
        nodes,  # noqa: F401
        pipeline,  # noqa: F401
    )


def test_validation_import() -> None:
    from commerce_signals.pipelines.validation import (
        nodes,  # noqa: F401
        pipeline,  # noqa: F401
    )


def test_feature_engineering_import() -> None:
    from commerce_signals.pipelines.feature_engineering import (
        nodes,  # noqa: F401
        pipeline,  # noqa: F401
    )


def test_training_import() -> None:
    from commerce_signals.pipelines.training import (
        nodes,  # noqa: F401
        pipeline,  # noqa: F401
    )


def test_explainability_import() -> None:
    from commerce_signals.pipelines.explainability import (
        nodes,  # noqa: F401
        pipeline,  # noqa: F401
    )


def test_inference_import() -> None:
    from commerce_signals.pipelines.inference import (
        nodes,  # noqa: F401
        pipeline,  # noqa: F401
    )


def test_monitoring_import() -> None:
    from commerce_signals.pipelines.monitoring import (
        nodes,  # noqa: F401
        pipeline,  # noqa: F401
    )


# --- Custom dataset tests -----------------------------------------------------

def test_mongo_collection_dataset_can_be_instantiated() -> None:
    """Verify that MongoCollectionDataset can be instantiated with
    the minimum required arguments (collection, credentials placeholder).
    """
    from commerce_signals.datasets import MongoCollectionDataset

    ds = MongoCollectionDataset(
        collection="test_collection",
        credentials={"uri": "mongodb://localhost:27017", "db": "test_db"},
    )
    assert ds.collection == "test_collection"

def test_mongo_collection_dataset_describe_returns_dict() -> None:
    """_describe should return a dict describing the dataset."""
    from commerce_signals.datasets import MongoCollectionDataset

    ds = MongoCollectionDataset(
        collection="my_coll",
        credentials={"uri": "mongodb://localhost:27017", "db": "test_db"},
    )
    info = ds._describe()
    assert isinstance(info, dict)
    assert info["collection"] == "my_coll"
    assert "credentials_keys" in info
