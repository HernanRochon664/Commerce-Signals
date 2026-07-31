"""FastAPI application skeleton for Commerce Signals (M0 stub).

Endpoints:
    GET /health  — returns {"status": "ok"} as a liveness check.

The full prediction endpoint (POST /predict) lands in M6.
"""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Commerce Signals",
    description="Churn prediction API for e-commerce customers.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe.

    Returns:
        JSON with ``status: "ok"`` when the service is running.
    """
    return {"status": "ok"}
