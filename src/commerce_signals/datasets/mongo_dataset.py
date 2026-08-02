"""Custom Kedro dataset for reading and writing MongoDB collections.

`MongoCollectionDataset` lets Kedro pipelines load and save arbitrary
Python objects (DataFrames, dicts, list-of-dicts, etc.) to/from a
specific MongoDB collection. Pipelines can consume and produce plain
Python objects without knowing that the backing store is Mongo, which
keeps them testable without a running Mongo instance.

Configuration in `catalog.yml` typically looks like::

    raw_transactions:
      type: commerce_signals.datasets.MongoCollectionDataset
      collection: raw_transactions
      credentials: mongo_creds
      mode: replace   # or 'append' (default: replace)

The `mode` flag only affects ``_save``:

* ``"replace"`` (default) drops the target collection before inserting.
  Suited for batch pipeline outputs that should be fully refreshed on
  every run (e.g. ``raw_transactions``, ``clean_transactions``).
* ``"append"`` inserts without dropping. Required for collections that
  are written incrementally one row at a time and where history must
  survive (e.g. ``predictions`` written by the FastAPI ``/predict``
  endpoint, M6).
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pandas as pd
from kedro.io import AbstractDataset
from pymongo import MongoClient

# Number of documents per `insert_many` batch. Tuned for the Online
# Retail II volume (~1M rows). pymongo's default request size limit is
# 16 MiB, so this is well under it even with wide rows.
_INSERT_CHUNK_SIZE: int = 5000


class MongoCollectionDataset(AbstractDataset[Any, Any]):
    """``MongoCollectionDataset`` loads/saves data from/to a MongoDB collection.

    The dataset is registered in the Data Catalog via
    ``type: commerce_signals.datasets.MongoCollectionDataset``. Credentials
    (MongoDB URI and database name) are passed via the catalog's
    ``credentials: <name>`` reference, which Kedro resolves from
    `conf/local/credentials.yml`.
    """

    # Public alias so catalog YAML can reference the class by short name.
    # Kedro loads the class via `importlib` so the full dotted path also works.
    DEFAULT_CLASS_NAME: ClassVar[str] = "MongoCollectionDataset"

    def __init__(
        self,
        collection: str,
        credentials: dict[str, Any] | None = None,
        mode: Literal["replace", "append"] = "replace",
    ) -> None:
        """Initialise the dataset.

        Args:
            collection: Name of the MongoDB collection.
            credentials: Dict with at least ``uri`` and ``db`` keys.
                Other keys are accepted and ignored.
            mode: ``"replace"`` drops the collection before each
                ``_save`` (idempotent batch rewrites); ``"append"``
                inserts without dropping (per-row append-only writes).
        """
        if mode not in ("replace", "append"):
            raise ValueError(
                f"mode must be 'replace' or 'append', got {mode!r}"
            )
        self._collection: str = collection
        self._credentials: dict[str, Any] = credentials or {}
        self._mode: Literal["replace", "append"] = mode

    @property
    def collection(self) -> str:
        """Return the MongoDB collection name this dataset targets."""
        return self._collection

    @property
    def mode(self) -> Literal["replace", "append"]:
        """Return the write mode of this dataset (``replace`` or ``append``)."""
        return self._mode

    def _load(self) -> pd.DataFrame:
        """Load all documents from the MongoDB collection as a DataFrame.

        The full collection is read with ``find()`` (no filter, no
        projection) and materialised into a ``pandas.DataFrame``. No
        column-name assumptions are made: the Mongo ``_id`` field is
        dropped, but every other field is kept as-is. M6 may add
        filter/projection support when it needs to look up a single
        ``customer_id``.

        Returns:
            A ``pandas.DataFrame`` with one row per document. If the
            collection is empty, an empty DataFrame is returned (no
            exception).

        Raises:
            KeyError: If ``credentials`` does not contain both
                ``"uri"`` and ``"db"`` keys.
            pymongo.errors.PyMongoError: On any Mongo-side failure.
        """
        uri, db_name = self._resolve_credentials()
        with MongoClient(uri) as client:
            collection = client[db_name][self._collection]
            cursor = collection.find({})
            records = list(cursor)

        if not records:
            return pd.DataFrame()

        # Drop Mongo's synthetic primary key; everything else is user data.
        for record in records:
            record.pop("_id", None)
        return pd.DataFrame.from_records(records)

    def _save(self, data: Any) -> None:
        """Persist ``data`` to the MongoDB collection.

        ``data`` is expected to be a ``pandas.DataFrame`` and is
        serialised via ``df.to_dict("records")`` before being inserted
        in fixed-size chunks. With ``mode="replace"`` the collection is
        dropped first, so a second run of the same pipeline produces
        the same collection contents (idempotent).

        Note on NaT/NaN: pandas ``NaT``/``NaN`` values are converted
        to ``None`` by ``to_dict("records")`` for object-dtype columns,
        which serialises cleanly to BSON. If a future column type fails
        BSON encoding (e.g. nullable datetime64[ns]), the fix is to
        coerce those columns to native Python types or ``None`` before
        the ``insert_many`` call. Not needed in M1; flagged here as a
        known footgun for later milestones.

        Args:
            data: Object to persist. Expected to be a
                ``pandas.DataFrame``; anything else is rejected loudly
                (better fail fast than silently drop rows).

        Raises:
            TypeError: If ``data`` is not a ``pandas.DataFrame``.
            KeyError: If ``credentials`` is missing ``"uri"`` or ``"db"``.
            pymongo.errors.PyMongoError: On any Mongo-side failure
                (errors propagate; partial inserts are not masked).
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"MongoCollectionDataset._save expects a pandas.DataFrame, "
                f"got {type(data).__name__}"
            )

        if self._mode == "replace":
            self._replace_collection(data)
        else:
            self._append_collection(data)

    def _replace_collection(self, df: pd.DataFrame) -> None:
        """Drop the collection, then insert all rows from ``df`` in chunks."""
        uri, db_name = self._resolve_credentials()
        records = df.to_dict("records")
        with MongoClient(uri) as client:
            collection = client[db_name][self._collection]
            collection.drop()
            if not records:
                return
            self._insert_in_chunks(collection, records)

    def _append_collection(self, df: pd.DataFrame) -> None:
        """Insert rows from ``df`` into the collection without dropping."""
        if df.empty:
            return
        uri, db_name = self._resolve_credentials()
        records = df.to_dict("records")
        with MongoClient(uri) as client:
            collection = client[db_name][self._collection]
            self._insert_in_chunks(collection, records)

    @staticmethod
    def _insert_in_chunks(collection: Any, records: list[dict[str, Any]]) -> None:
        """Insert ``records`` into ``collection`` in ``_INSERT_CHUNK_SIZE`` chunks.

        ``ordered=True`` (the pymongo default) is left as-is: on a
        batch failure, the pipeline should fail loudly, not swallow
        partial inserts. M2+ may revisit this if validation needs
        partial-success semantics.
        """
        for start in range(0, len(records), _INSERT_CHUNK_SIZE):
            chunk = records[start : start + _INSERT_CHUNK_SIZE]
            collection.insert_many(chunk)

    def _resolve_credentials(self) -> tuple[str, str]:
        """Return ``(uri, db)`` from the credentials dict.

        Raises:
            KeyError: If either key is missing. We surface a clear
                message rather than letting ``MongoClient(None)`` fail
                with a confusing low-level error.
        """
        try:
            return self._credentials["uri"], self._credentials["db"]
        except KeyError as exc:
            raise KeyError(
                "MongoCollectionDataset requires credentials with 'uri' and 'db' keys; "
                f"got keys={sorted(self._credentials.keys())}"
            ) from exc

    def _describe(self) -> dict[str, Any]:
        """Return a description of the dataset for logging.

        Returns:
            A dictionary describing the dataset instance. Useful for
            ``kedro catalog describe-datasets`` and runtime logs.
        """
        # We deliberately do not include the credentials URI in the
        # description: secrets should never end up in logs.
        return {
            "collection": self._collection,
            "mode": self._mode,
            "credentials_keys": sorted(self._credentials.keys()),
        }
