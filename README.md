# Commerce Signals

Churn prediction for e-commerce customers.

A dual-purpose project: ML engineering portfolio + proof-of-concept for a
real product that identifies customers at risk of leaving an online store.

**Stack:** Python 3.12 · Kedro 1.5 · MongoDB (via custom `AbstractDataset`) · LightGBM · SHAP · MLflow · FastAPI · Docker · Evidently · pytest · ruff · GitHub Actions · uv

---

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker & Docker Compose

### Local development

```bash
# 1. Create and activate the virtual environment
uv sync

# 2. Start MongoDB (local dev)
docker compose up -d

# 3. Copy credential template and fill in your local Mongo URI
cp conf/local/credentials.yml.example conf/local/credentials.yml

# 4. Verify the project loads
uv run kedro info

# 5. Run the (empty) pipeline to confirm environment wiring
uv run kedro run

# 6. Start the FastAPI server
uv run uvicorn app.main:app --reload

# 7. Check health
curl http://localhost:8000/health
```

### Tests & linting

```bash
uv run ruff check .
uv run pytest
```

---

## Project structure

```
commerce-signals/
├── .github/workflows/ci.yml   # CI: ruff + pytest + Docker build
├── app/
│   └── main.py                 # FastAPI /health stub (M6 → /predict)
├── conf/
│   ├── base/
│   │   ├── catalog.yml         # Dataset registry (MongoCollectionDataset)
│   │   └── parameters.yml      # Runtime parameters
│   └── local/
│       ├── credentials.yml      # Git-ignored; local Mongo creds
│       └── credentials.yml.example
├── docker-compose.yml           # MongoDB service
├── Dockerfile                   # FastAPI container (hardened in M6)
├── src/
│   └── commerce_signals/
│       ├── datasets/
│       │   └── mongo_dataset.py # MongoCollectionDataset (AbstractDataset)
│       └── pipelines/
│           ├── ingestion/       # M1 — load source → raw_transactions
│           ├── validation/      # M2 — clean & validate
│           ├── feature_engineering/  # M3 — RFM + churn label
│           ├── training/        # M4 — LightGBM + MLflow
│           ├── explainability/  # M5 — SHAP
│           ├── inference/       # M6 — FastAPI /predict
│           └── monitoring/      # M8 — Evidently drift
├── tests/
│   └── test_dummy_stubs.py
├── .env.example
├── .gitignore
├── .kedro.yml
└── pyproject.toml
```

---

## Milestones

| #  | Name                | What it delivers                                      |
|----|---------------------|-------------------------------------------------------|
| M0 | Scaffolding         | **You are here** — structure, stubs, config, CI       |
| M1 | Ingestion           | Load Online Retail II into MongoDB                    |
| M2 | Validation          | Clean rules + validation report                       |
| M3 | Feature engineering | RFM features, churn label, temporal split             |
| M4 | Training            | LightGBM + MLflow tracking                            |
| M5 | Explainability      | SHAP values + global importance report                |
| M6 | Serving             | FastAPI /predict endpoint (Dockerized)                |
| M7 | Deployment          | Cloud Run + CI/CD                                     |
| M8 | Observability       | Evidently drift + Cloud Monitoring                    |

---

## Decisions

| Topic                     | Choice                                     |
|---------------------------|--------------------------------------------|
| Python                    | 3.12 (stable, Kedro 1.5 supported)         |
| Package manager           | uv                                          |
| MongoDB driver            | pymongo (via custom dataset, not in nodes)  |
| ML tracking               | MLflow (local, `mlruns/`)                  |
| Drift                     | Evidently local (no Evidently Cloud)        |
| Linting & formatting      | ruff (single tool)                          |
| CI                        | GitHub Actions                              |
| Dataset source            | Online Retail II (Kaggle / UCI)             |
