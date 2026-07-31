"""Node functions for the `validation` pipeline (M0 stub)."""

from __future__ import annotations


def validate_and_clean(
    raw_transactions,  # noqa: ARG001
) -> None:
    """Validate and clean raw transactions, producing cleaned rows.

    Args:
        raw_transactions: DataFrame-style object loaded from the
            ``raw_transactions`` MongoDB collection.

    Returns:
        The cleaned transactions (output via ``clean_transactions``).
        A separate validation report artifact is also produced.

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M2")


def build_validation_report(
    raw_transactions,  # noqa: ARG001
    clean_transactions,  # noqa: ARG001
) -> None:
    """Summarise the validation rules and the rows dropped by each.

    Args:
        raw_transactions: Original raw rows (before cleaning).
        clean_transactions: Rows that survived cleaning.

    Returns:
        A validation report (output via ``validation_report``).

    Raises:
        NotImplementedError: Always (M0 stub).
    """
    raise NotImplementedError("Implementar en M2")
