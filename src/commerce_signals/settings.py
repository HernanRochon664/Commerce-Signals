"""Kedro settings for Commerce Signals.

Anything Kedro conventionally reads from ``settings.py`` goes here.
The pipeline registry lives in ``pipeline_registry.py``.
"""

from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog

# Config loader — Omega supports YAML and credential resolution.
CONFIG_LOADER_CLASS = OmegaConfigLoader

# DataCatalog — supports custom AbstractDataset subclasses.
DATA_CATALOG_CLASS = DataCatalog
