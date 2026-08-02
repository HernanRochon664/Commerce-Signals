"""Integration tests for `MongoCollectionDataset`.

These tests talk to a real MongoDB instance (see ``conftest.py`` for
the connection fixture). They cover the four M1-mandated behaviours:

* Round-trip: save then load returns the same data.
* ``mode="replace"`` wipes the collection before inserting.
* ``mode="append"`` does not touch existing documents.
* ``_save`` correctly batches large inputs (>chunk size).
"""

from __future__ import annotations

import pandas as pd
import pytest

from commerce_signals.datasets import MongoCollectionDataset

pytestmark = pytest.mark.integration


def _make_df(n_rows: int) -> pd.DataFrame:
    """Build a small DataFrame with mixed types for round-trip checks."""
    return pd.DataFrame(
        {
            "customer_id": list(range(1, n_rows + 1)),
            "amount": [round(0.99 + i * 0.01, 2) for i in range(n_rows)],
            "label": ["a" if i % 2 == 0 else "b" for i in range(n_rows)],
        }
    )


def _credentials(mongo_uri: str) -> dict[str, str]:
    return {"uri": mongo_uri, "db": "commerce_signals_test"}


def test_roundtrip_save_then_load_returns_same_data(
    mongo_test_collection, mongo_uri, mongo_test_db
) -> None:
    """Saving and reloading a DataFrame yields equivalent data."""
    _, client, name = mongo_test_collection
    df = _make_df(7)

    ds = MongoCollectionDataset(
        collection=name, credentials=_credentials(mongo_uri)
    )
    ds._save(df)

    loaded = ds._load()
    assert len(loaded) == 7
    assert sorted(loaded.columns.tolist()) == ["amount", "customer_id", "label"]
    # Row-level equality, order-insensitive (Mongo does not guarantee
    # insertion order on a round-trip without an explicit sort key).
    pd.testing.assert_frame_equal(
        loaded.sort_values("customer_id").reset_index(drop=True),
        df.sort_values("customer_id").reset_index(drop=True),
        check_dtype=False,
    )
    # Sanity: the dataset was actually written to the live collection.
    assert client["commerce_signals_test"][name].count_documents({}) == 7


def test_load_on_empty_collection_returns_empty_dataframe(
    mongo_test_collection, mongo_uri, mongo_test_db
) -> None:
    """``_load`` on an empty collection returns an empty DataFrame, not an error."""
    _, _, name = mongo_test_collection
    ds = MongoCollectionDataset(
        collection=name, credentials=_credentials(mongo_uri)
    )
    loaded = ds._load()
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.empty


def test_replace_mode_drops_previous_documents(
    mongo_test_collection, mongo_uri, mongo_test_db
) -> None:
    """A second ``_save`` with ``mode="replace"`` removes the first batch."""
    _, client, name = mongo_test_collection

    ds_replace = MongoCollectionDataset(
        collection=name, credentials=_credentials(mongo_uri), mode="replace"
    )
    ds_replace._save(_make_df(5))
    assert client["commerce_signals_test"][name].count_documents({}) == 5

    ds_replace._save(_make_df(3))
    assert client["commerce_signals_test"][name].count_documents({}) == 3

    loaded = ds_replace._load()
    assert len(loaded) == 3


def test_append_mode_preserves_previous_documents(
    mongo_test_collection, mongo_uri, mongo_test_db
) -> None:
    """A second ``_save`` with ``mode="append"`` does not drop the first batch."""
    _, client, name = mongo_test_collection

    ds = MongoCollectionDataset(
        collection=name, credentials=_credentials(mongo_uri), mode="append"
    )
    ds._save(_make_df(4))
    assert client["commerce_signals_test"][name].count_documents({}) == 4

    ds._save(_make_df(6))
    assert client["commerce_signals_test"][name].count_documents({}) == 10

    loaded = ds._load()
    assert len(loaded) == 10


def test_save_batches_large_inputs(mongo_test_collection, mongo_uri, mongo_test_db) -> None:
    """Saving more rows than the chunk size still inserts all of them.

    Uses 12 000 rows against the 5 000-row chunk size, so the save
    must complete in at least three chunks. We assert the resulting
    collection count rather than the chunk count (the latter is a
    private implementation detail).
    """
    _, client, name = mongo_test_collection

    ds = MongoCollectionDataset(
        collection=name, credentials=_credentials(mongo_uri)
    )
    ds._save(_make_df(12_000))

    assert client["commerce_signals_test"][name].count_documents({}) == 12_000
    loaded = ds._load()
    assert len(loaded) == 12_000
    # Spot-check that all values made it across (no silent drop in a
    # later chunk).
    assert loaded["customer_id"].min() == 1
    assert loaded["customer_id"].max() == 12_000


def test_invalid_mode_raises_at_construction() -> None:
    """``mode`` is validated at construction; a typo fails fast."""
    with pytest.raises(ValueError, match="mode must be"):
        MongoCollectionDataset(
            collection="any",
            credentials={"uri": "mongodb://localhost:27017", "db": "x"},
            mode="overwrite",  # type: ignore[arg-type]
        )


def test_save_rejects_non_dataframe(
    mongo_test_collection, mongo_uri, mongo_test_db
) -> None:
    """``_save`` refuses non-DataFrame inputs to avoid silent data loss."""
    _, _, name = mongo_test_collection
    ds = MongoCollectionDataset(
        collection=name, credentials=_credentials(mongo_uri)
    )
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        ds._save({"not": "a dataframe"})


def test_missing_credentials_keys_raise() -> None:
    """A credentials dict without ``uri``/``db`` produces a clear error."""
    ds = MongoCollectionDataset(collection="x", credentials={})
    with pytest.raises(KeyError, match="uri"):
        ds._load()
