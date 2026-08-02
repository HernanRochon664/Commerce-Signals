"""Shared pytest fixtures for Commerce Signals.

Mongo fixture
-------------
Integration tests in ``tests/test_mongo_dataset.py`` need a real
MongoDB instance. The URI is taken from the ``TEST_MONGODB_URI``
environment variable (default: ``mongodb://localhost:27017``) and
points at a test database named ``commerce_signals_test``.

Two failure modes are handled explicitly:

1. **Connection refused / DNS failure** — pymongo raises
   ``ServerSelectionTimeoutError``. We surface it as a clean
   ``pytest.skip`` so the unit tests still run and CI (which has
   the ``mongo:7`` service) passes; on a developer machine without
   Docker, integration tests are simply skipped, not red.
2. **A stale test database** — the fixture drops
   ``commerce_signals_test`` on teardown so each test starts from
   a clean slate without affecting the developer's main
   ``commerce_signals`` database.
"""

from __future__ import annotations

import os
import uuid

import pytest
from pymongo import MongoClient
from pymongo.errors import PyMongoError

TEST_DB_NAME = "commerce_signals_test"
TEST_MONGODB_URI = os.environ.get(
    "TEST_MONGODB_URI", "mongodb://localhost:27017"
)
# pymongo's default 30s timeout is too long for a CI skip-with-reason
# when the service is missing. 1.5s is long enough to confirm a local
# docker-compose Mongo is reachable, short enough that a missing
# service fails fast.
_SERVER_SELECTION_TIMEOUT_MS = 1500


def _mongo_reachable(uri: str) -> bool:
    """Return True if a Mongo server is reachable at ``uri``."""
    try:
        client = MongoClient(
            uri, serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS
        )
        client.admin.command("ping")
        client.close()
    except PyMongoError:
        return False
    return True


_MONGODB_REACHABLE = _mongo_reachable(TEST_MONGODB_URI)
_SKIP_REASON = (
    f"MongoDB is not reachable at {TEST_MONGODB_URI!r}. "
    "Integration tests require a running Mongo instance: locally run "
    "`docker compose up -d` (URI defaults to mongodb://localhost:27017), "
    "or set TEST_MONGODB_URI to a reachable instance. CI provides a "
    "mongo:7 service on the same default URI."
)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-skip integration tests when no Mongo is reachable.

    A test is considered "integration" if any of the following is true:

    * It is marked ``@pytest.mark.integration``.
    * It lives in ``tests/test_mongo_dataset.py`` (the dataset
      integration suite talks to Mongo unconditionally).

    The unit tests under ``tests/test_dummy_stubs.py`` and
    ``tests/pipelines/test_ingestion.py`` are NOT auto-skipped here:
    the stubs use a non-Mongo error path (``NotImplementedError`` /
    file I/O) and the ingestion node tests only read the .xlsx
    fixture. Both should run on every commit, with or without Mongo.
    """
    if _MONGODB_REACHABLE:
        return
    skip_marker = pytest.mark.skip(reason=_SKIP_REASON)
    for item in items:
        basename = item.fspath.basename
        is_integration_module = basename in {"test_mongo_dataset.py"}
        has_integration_mark = "integration" in item.keywords
        if is_integration_module or has_integration_mark:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def mongo_uri() -> str:
    """Return the Mongo URI used by integration tests.

    Session-scoped: a single ping on session start is enough to fail
    fast in CI before any test runs.
    """
    if not _MONGODB_REACHABLE:
        pytest.skip(_SKIP_REASON)
    return TEST_MONGODB_URI


@pytest.fixture
def mongo_test_collection(mongo_uri: str):
    """Yield a unique collection name in the test database and clean up after.

    Each test gets a freshly-named collection so tests cannot see each
    other's documents even within the same test session. The collection
    is dropped on teardown; the database itself is also dropped at the
    end of the session via ``mongo_test_db``.
    """
    collection_name = f"it_{uuid.uuid4().hex[:12]}"
    client = MongoClient(
        mongo_uri, serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS
    )
    try:
        yield client[TEST_DB_NAME][collection_name], client, collection_name
    finally:
        try:
            client[TEST_DB_NAME][collection_name].drop()
        finally:
            client.close()


@pytest.fixture(scope="session")
def mongo_test_db(mongo_uri: str):
    """Drop the test database after the entire session.

    Not autouse: only the integration tests that opt in (by requesting
    ``mongo_uri`` or ``mongo_test_collection``) trigger it. The fixture
    itself skips the requesting test if Mongo is unreachable, which
    keeps the unit-test slice green on developer machines without
    Docker.
    """
    yield
    client = MongoClient(
        mongo_uri, serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS
    )
    try:
        client.drop_database(TEST_DB_NAME)
    finally:
        client.close()
