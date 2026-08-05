# AGENTS.md - Commerce Signals

E-commerce churn prediction. Kedro 1.5 + MongoDB + LightGBM + MLflow (local) + FastAPI + Docker. Currently M0 (scaffolding only): every pipeline node and MongoCollectionDataset._load / _save are stubs that raise NotImplementedError("Implementar en M1"). See docs/plan-milestones.md for the full M0-M8 roadmap and the rationale behind the stack choices.

## Project layout (the only file boundaries an agent must respect)

- src/commerce_signals/ - Kedro package (pipelines, datasets, pipeline_registry.py, settings.py).
- app/main.py - FastAPI app (separate from the Kedro runtime, containerized via Dockerfile).
- conf/base/catalog.yml, conf/base/parameters.yml - Kedro data catalog and runtime params.
- conf/local/credentials.yml - git-ignored, Mongo URI + db. Template at conf/local/credentials.yml.example.
- conf/logging.yml - Kedro logging config.
- tests/ - pytest (testpaths set in pyproject.toml).
- docker-compose.yml - local MongoDB only (mongo:7 on 27017, volume commerce_signals_mongo_data).
- .env.example - MONGODB_URI, MONGODB_DB_NAME, MLFLOW_TRACKING_URI template; .env itself is git-ignored.
- .telemetry - kedro-telemetry opt out, committed to the repo. See "Telemetry" below.

Each Kedro pipeline lives at src/commerce_signals/pipelines/<name>/ with pipeline.py (create_pipeline()) and nodes.py. Pipelines are wired in src/commerce_signals/pipeline_registry.py, inside register_pipelines(). New pipelines must be added in three places: the package __init__.py (src/commerce_signals/pipelines/__init__.py), the registry (pipeline_registry.py), and the module itself.

The Mongo dataset is referenced from catalog.yml as type: commerce_signals.datasets.MongoCollectionDataset and re-exported from src/commerce_signals/__init__.py. Catalog credentials resolve from the mongo_creds: block in conf/local/credentials.yml.

## Local setup (run in this order)

```bash
uv sync
docker compose up -d
cp conf/local/credentials.yml.example conf/local/credentials.yml
uv run kedro info
uv run uvicorn app.main:app --reload
curl http://localhost:8000/health
```

Python is pinned to 3.12 (requires-python = ">=3.12,<3.13" in pyproject.toml). uv run is the canonical command runner, do not cd and run raw commands.

## Verify commands

```bash
uv run ruff check .
uv run pytest
uv run pytest -v --cov=src/commerce_signals --no-cov-on-fail
uv run kedro run --pipeline <name>
```

CI order in .github/workflows/ci.yml: uv sync --all-extras, then ruff check ., then pytest -v --cov=..., then a separate docker-build job. Triggered on push to main/develop and PRs to main.

Ruff config in pyproject.toml: line-length 100, target py312, rules E,F,I,B,UP,W; B is ignored under tests/**.

## Telemetry

kedro-telemetry is disabled, deliberately and redundantly. A .telemetry file with consent: false lives at the project root and is committed to the repo (removed from .gitignore on purpose), so the opt out persists for anyone who clones it, including CI, without depending on an environment variable being set in every shell. ci.yml also sets KEDRO_DISABLE_TELEMETRY: "true" as a backup mechanism. Do not delete .telemetry, do not re-add it to .gitignore, and do not remove the CI env var without discussing it first: the business's whole positioning is "nothing runs silently without you knowing about it," so silently re-enabling telemetry would contradict that on principle, not just in this codebase.

## M0 gotchas agents will trip on

- uv run kedro run and any --pipeline <name> will fail at the first node with NotImplementedError("Implementar en M1"). That is expected, the M0 goal is to confirm wiring, not data flow.
- MongoCollectionDataset._load() and _save() are stubs that always raise. Tests in tests/test_dummy_stubs.py assert on the literal substring "M1" in the message, keep that exact string when implementing.
- Only MongoCollectionDataset.__init__ has actually run so far. _load and _save have never executed against a real Mongo instance (the M0 ingestion node raises before ever calling _save). When M1 implements the real logic, decide up front whether the dataset gets unit tests with a mocked pymongo client, integration tests against a real Mongo (would need a mongo service added to ci.yml, not just docker-compose locally), or both.
- FastAPI only exposes GET /health (app/main.py). There is no /predict yet (planned for M6).
- Dockerfile copies only pyproject.toml, src/, and app/; if you add top-level assets the API image needs, copy them in. The image is run as root, M6 plan calls for hardening (non-root user, healthcheck).
- mlruns/, data/0[1-8]_*, drift_reports/, conf/local/credentials.yml, .env, .pytest_cache/, .ruff_cache/ are all git-ignored. Do not commit them. .telemetry is the one deliberate exception to "when in doubt, git-ignore it": it must stay tracked.

## Conventions to keep

- Every module uses from __future__ import annotations and PEP 604 union syntax (X | Y).
- Docstrings are used liberally to encode the contract; nodes and dataset methods document the expected return type even when the body is a stub.
- _describe() on datasets intentionally omits the credentials URI, never log secrets.
- Pipeline tags (e.g., tags=["m1-stub"] in pipelines/ingestion/pipeline.py) mark M0 placeholders; remove the tag when replacing the stub with real logic.

## Entry points summary

- Kedro CLI: uv run kedro ... (or uv run python -m commerce_signals, which delegates to kedro.framework.cli.main in src/commerce_signals/__main__.py).
- FastAPI: uv run uvicorn app.main:app, defined in app/main.py.
- Mongo locally: docker compose up -d (data in named volume commerce_signals_mongo_data; docker compose down -v wipes it).
