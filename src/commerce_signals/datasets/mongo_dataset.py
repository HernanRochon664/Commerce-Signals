"""Custom Kedro dataset for reading and writing MongoDB collections.

`MongoCollectionDataset` lets Kedro pipelines load and save arbitrary
Python objects (DataFrames, dicts, list-of-dicts, etc.) to/from a
specific MongoDB collection. Pipelines can consume and produce plain
Python objects without knowing that the backing store is Mongo, which
keeps them testable without a running Mongo instance.

Implementation note (M0):
    `_load`, `_save`, and `_describe` are deliberately stubs that raise
    `NotImplementedError`. The real implementation (pymongo connection
    handling, batched inserts, query/load semantics) lands in M1.

Configuration in `catalog.yml` typically looks like::

    raw_transactions:
      type: commerce_signals.datasets.MongoCollectionDataset
      collection: raw_transactions
      credentials: mongo_creds
"""

from __future__ import annotations

from typing import Any, ClassVar

from kedro.io import AbstractDataset


class MongoCollectionDataset(AbstractDataset[Any, Any]):
    """``MongoCollectionDataset`` loads/saves data from/to a MongoDB collection.

    The dataset is registered in the Data Catalog via
    ``type: commerce_signals.datasets.MongoCollectionDataset``.
    Credentials (MongoDB URI and database name) are passed via the
    catalog's ``credentials: <name>`` reference, which Kedro resolves
    from `conf/local/credentials.yml`.
    """

    # Public alias so catalog YAML can reference the class by short name.
    # Kedro loads the class via `importlib` so the full dotted path also works.
    DEFAULT_CLASS_NAME: ClassVar[str] = "MongoCollectionDataset"

    def __init__(
        self,
        collection: str,
        credentials: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            collection: Name of the MongoDB collection.
            credentials: Dict with at least ``uri`` and ``db`` keys
                (other keys are allowed and ignored by M0).
        """
        self._collection: str = collection
        self._credentials: dict[str, Any] = credentials or {}

    @property
    def collection(self) -> str:
        """Return the MongoDB collection name this dataset targets."""
        return self._collection

    def _load(self) -> Any:
        """Load data from the MongoDB collection.

        Stub for M0. Real implementation lands in M1.

        Returns:
            Whatever the dataset returns at the ``__call__`` boundary
            (a ``pandas.DataFrame`` is the expected default, but the
            concrete type is decided in M1).

        Raises:
            NotImplementedError: Always (M0).
        """
        raise NotImplementedError("Implementar en M1")

    def _save(self, data: Any) -> None:
        """Save ``data`` to the MongoDB collection.

        Stub for M0. Real implementation lands in M1.

        Args:
            data: Object to persist. Expected to be a ``pandas.DataFrame``
                by default, but the contract is decided in M1.

        Raises:
            NotImplementedError: Always (M0).
        """
        raise NotImplementedError("Implementar en M1")

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
            "credentials_keys": sorted(self._credentials.keys()),
        }
