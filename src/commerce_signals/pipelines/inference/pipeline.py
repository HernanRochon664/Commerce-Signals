"""Pipeline definition for the `inference` stage (M6).

Vacío a propósito (opción A).

``predict_customer_churn`` en ``nodes.py`` es una función pura, reutilizable
y testeada, usada de verdad por ``app/main.py`` vía FastAPI, no a través de
Kedro. Wirearla como pipeline batch de Kedro necesitaría un dataset custom que
resuelva un modelo desde MLflow por ``run_id`` (no existe hoy) y una
conversión de ``dict`` a ``DataFrame`` antes de guardar en ``predictions``.
Se difiere hasta que haya un consumidor real (ej. re-scoring batch de toda la
base de clientes), no se construye a medias sin necesidad concreta.
"""

from __future__ import annotations

from kedro.pipeline import Pipeline


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    """Create the inference pipeline (intencionalmente vacío, ver docstring)."""
    return Pipeline([])
