"""Node functions for the `ingestion` pipeline.

M1 scope:
    Load the Online Retail II source file into MongoDB untouched
    (no column renames, no type coercion, no filtering — those
    decisions belong to M2). Verify round-trip row counts.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Sheet names the published Online Retail II workbook is expected to
# carry. We check these first and fall back to positional reads if the
# publisher changes them in a future release.
EXPECTED_SHEETS: tuple[str, ...] = ("Year 2009-2010", "Year 2010-2011")


def load_source_to_raw_transactions(source_path: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read both sheets of the Online Retail II workbook, untouched.

    The workbook is expected to have two sheets named
    ``"Year 2009-2010"`` and ``"Year 2010-2011"``. We verify those
    exact names first; if they are missing, a warning is logged and
    the first two sheets (by position) are read instead. The fallback
    is loud, on purpose: a silent fallback would let a renamed
    workbook publish partial data without anyone noticing.

    No column renaming, no type coercion, no row filtering. Any
    cleaning rule is the job of the validation pipeline (M2). The
    output DataFrame schema is whatever the source gives us.

    Args:
        source_path: Absolute or Kedro-resolved path to the
            ``.xlsx`` file, from ``params:ingestion.source_path``.

    Returns:
        A tuple ``(df, row_counts)`` where:
            * ``df`` is the concatenated, untransformed DataFrame
              (consumed downstream by the ``raw_transactions``
              MongoCollectionDataset).
            * ``row_counts`` is a dict like
              ``{"sheet_2009_2010": N, "sheet_2010_2011": M, "total": N + M}``
              used by ``verify_ingestion_row_count`` to confirm the
              round-trip through Mongo.
    """
    sheet_names = pd.ExcelFile(source_path).sheet_names

    if all(name in sheet_names for name in EXPECTED_SHEETS):
        sheet_2009 = pd.read_excel(source_path, sheet_name=EXPECTED_SHEETS[0])
        sheet_2010 = pd.read_excel(source_path, sheet_name=EXPECTED_SHEETS[1])
        used_fallback = False
    else:
        logger.warning(
            "Expected sheet names %s not all found in %s; found %s. "
            "Falling back to the first two sheets by position. "
            "Verify the source workbook matches the Online Retail II layout.",
            list(EXPECTED_SHEETS),
            source_path,
            list(sheet_names),
        )
        if len(sheet_names) < 2:
            raise ValueError(
                f"Source workbook {source_path!r} has fewer than two sheets "
                f"(found: {sheet_names!r}); cannot load Online Retail II."
            )
        sheet_2009 = pd.read_excel(source_path, sheet_name=sheet_names[0])
        sheet_2010 = pd.read_excel(source_path, sheet_name=sheet_names[1])
        used_fallback = True

    df = pd.concat([sheet_2009, sheet_2010], ignore_index=True)
    row_counts: dict[str, int] = {
        "sheet_2009_2010": int(len(sheet_2009)),
        "sheet_2010_2011": int(len(sheet_2010)),
        "total": int(len(df)),
    }
    if used_fallback:
        logger.warning(
            "Ingestion read %d rows total via positional fallback; "
            "expected-sheet match failed.",
            row_counts["total"],
        )
    else:
        logger.info(
            "Ingestion read %d rows total from %s and %s.",
            row_counts["total"],
            EXPECTED_SHEETS[0],
            EXPECTED_SHEETS[1],
        )
    return df, row_counts


def verify_ingestion_row_count(
    raw_transactions: pd.DataFrame,
    ingestion_row_counts: dict[str, Any],
) -> None:
    """Compare Mongo-roundtrip row count to the source-file row count.

    This node is observability, not a quality gate: it never raises
    and never fails the pipeline. Quality rules live in the
    validation pipeline (M2). If the counts match we log INFO; if
    they differ we log WARNING with both numbers so the mismatch is
    visible in any log scrape, not just on the console.

    Args:
        raw_transactions: DataFrame re-loaded from Mongo by Kedro
            (forces a real ``_load()`` round-trip, not just an
            in-memory count).
        ingestion_row_counts: Dict produced by
            ``load_source_to_raw_transactions`` with the
            ``"total"`` key the comparison is made against.
    """
    loaded = int(len(raw_transactions))
    expected = int(ingestion_row_counts["total"])
    if loaded == expected:
        logger.info(
            "Ingestion verification OK: %d rows in raw_transactions matches "
            "source row count (%d).",
            loaded,
            expected,
        )
    else:
        logger.warning(
            "Ingestion verification MISMATCH: raw_transactions has %d rows, "
            "source had %d (delta=%d). Data was saved to Mongo but the counts "
            "do not match — investigate before downstream pipelines run.",
            loaded,
            expected,
            loaded - expected,
        )
