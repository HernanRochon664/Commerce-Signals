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

## Dataset

This project uses the **Online Retail II** dataset (UCI Machine Learning
Repository, [dataset page](https://archive.ics.uci.edu/dataset/502/online+retail+ii)).
It is a transactional dataset from a UK-based online retailer covering
2009-12-01 to 2011-12-09, with roughly 1 million rows across two sheets
("Year 2009-2010" and "Year 2010-2011").

**Download is manual, on purpose.** The pipeline does not scrape or
auto-fetch the file: dataset licensing, version drift, and reproducibility
all argue for pinning a specific copy under version control of the
analyst's choice. To add the dataset:

1. Download the `.xlsx` from the UCI page above.
2. Place it in `data/01_raw/` (the directory is git-ignored; the file
   is never committed).
3. Open `conf/base/parameters.yml` and set
   `ingestion.source_path` to the filename, e.g.
   `data/01_raw/online_retail_ii.xlsx`.

The ingestion pipeline reads this path, loads both sheets untouched
(no column renaming, no row filtering — that is M2's job), and writes
the concatenated rows into the `raw_transactions` MongoDB collection.

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
