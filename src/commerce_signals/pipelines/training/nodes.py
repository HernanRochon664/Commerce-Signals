"""Node functions for the `training` pipeline (M4).

The pipeline trains a LightGBM churn classifier on the customer
snapshots built by the feature engineering pipeline, using a strictly
temporal split by snapshot date (never a random row split): the last
``test_snapshots_count`` snapshots are the test set, the
``val_snapshots_count`` before those are the validation set, and
everything earlier is training. A single ``customer_id`` may appear in
multiple splits in different months: that is intentional, the model
will score the same customers repeatedly in production.

The trained model lives ONLY in the MLflow artifact store (local
SQLite tracking, ``sqlite:///mlflow.db``). MongoDB stays for business
data, and the Kedro catalog only persists ``training_report``; all
DataFrames between nodes are MemoryDatasets.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from commerce_signals.features import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


def split_snapshots_temporally(
    customer_snapshots: pd.DataFrame,
    test_snapshots_count: int,
    val_snapshots_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split customer snapshots into train/val/test by snapshot date.

    Test = the last ``test_snapshots_count`` snapshot dates. Validation
    = the ``val_snapshots_count`` dates immediately before those.
    Training = every remaining (earlier) date.

    Args:
        customer_snapshots: ``customer_snapshots`` DataFrame from the
            feature engineering pipeline (customer_id, snapshot_date,
            the FEATURE_COLUMNS, is_churned).
        test_snapshots_count: Number of most recent snapshots reserved
            for testing.
        val_snapshots_count: Number of snapshots just before the test
            ones reserved for validation/early stopping.

    Returns:
        A tuple of (train_df, val_df, test_df, split_info).
        ``split_info`` has the exact shape::

            {
                "train_snapshot_dates": [str, ...] (ISO format, sorted),
                "val_snapshot_dates": [str, ...],
                "test_snapshot_dates": [str, ...],
                "train_rows": int,
                "val_rows": int,
                "test_rows": int,
            }

    Raises:
        ValueError: If there are not more snapshots than
            test_snapshots_count + val_snapshots_count (the training
            set would be empty or nearly empty).
    """
    snapshots = customer_snapshots.copy()
    # Defensive: not yet confirmed that snapshot_date survives the
    # Mongo round-trip with a datetime dtype. Cheap no-op if it does.
    snapshots["snapshot_date"] = pd.to_datetime(snapshots["snapshot_date"])

    sorted_dates = sorted(snapshots["snapshot_date"].unique())
    if len(sorted_dates) <= test_snapshots_count + val_snapshots_count:
        raise ValueError(
            "Not enough snapshots for a temporal split: found "
            f"{len(sorted_dates)} unique snapshot date(s) but "
            f"test_snapshots_count + val_snapshots_count = "
            f"{test_snapshots_count} + {val_snapshots_count} = "
            f"{test_snapshots_count + val_snapshots_count}. Increase the "
            "snapshot history or lower the test/val counts."
        )

    test_dates = sorted_dates[-test_snapshots_count:]
    val_dates = (
        sorted_dates[-test_snapshots_count - val_snapshots_count : -test_snapshots_count]
        if val_snapshots_count > 0
        else []
    )
    train_dates = sorted_dates[: -test_snapshots_count - val_snapshots_count]

    train_df = snapshots[snapshots["snapshot_date"].isin(train_dates)]
    val_df = snapshots[snapshots["snapshot_date"].isin(val_dates)]
    test_df = snapshots[snapshots["snapshot_date"].isin(test_dates)]

    split_info = {
        "train_snapshot_dates": [d.isoformat() for d in train_dates],
        "val_snapshot_dates": [d.isoformat() for d in val_dates],
        "test_snapshot_dates": [d.isoformat() for d in test_dates],
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }

    logger.info(
        "Temporal split: %d train snapshot(s) (%d rows), %d val snapshot(s) "
        "(%d rows), %d test snapshot(s) (%d rows)",
        len(train_dates),
        len(train_df),
        len(val_dates),
        len(val_df),
        len(test_dates),
        len(test_df),
    )
    return train_df, val_df, test_df, split_info


def train_churn_model(
    train_snapshots: pd.DataFrame,
    val_snapshots: pd.DataFrame,
    model_params: dict[str, Any],
    eval_metric: str,
    early_stopping_rounds: int,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier with early stopping on the val split.

    Only the FEATURE_COLUMNS are used as features (imported from
    ``commerce_signals.features``); ``customer_id`` and
    ``snapshot_date`` never enter the model. ``is_churned`` is the
    label. No resampling or class reweighting: the real churn rate by
    snapshot (~28-60%) is not severe enough to warrant it. Validation
    data is passed via ``eval_X=(X_val,)`` and ``eval_y=(y_val,)``
    (single-element tuples, one entry per validation set).

    Args:
        train_snapshots: Training split (train_df from
            ``split_snapshots_temporally``).
        val_snapshots: Validation split, used for early stopping.
        model_params: LightGBM hyperparameters (from
            ``params:training.model_params``).
        eval_metric: LightGBM eval metric name (e.g.
            ``"binary_logloss"``).
        early_stopping_rounds: Stop training if the eval metric does
            not improve for this many rounds.

    Returns:
        The fitted ``lightgbm.LGBMClassifier``.
    """
    X_train = train_snapshots[FEATURE_COLUMNS]
    y_train = train_snapshots["is_churned"]
    X_val = val_snapshots[FEATURE_COLUMNS]
    y_val = val_snapshots["is_churned"]

    model = lgb.LGBMClassifier(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_X=(X_val,),
        eval_y=(y_val,),
        eval_metric=eval_metric,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        ],
    )

    logger.info(
        "Early stopping at iteration %s with best validation score %s",
        model.best_iteration_,
        model.best_score_,
    )
    return model


def _recall_at_precision(y_true, y_score, target_precision: float) -> float:
    """Maximum recall achievable at a minimum precision.

    Builds the precision-recall curve and returns the highest recall
    among all operating points whose precision is at least
    ``target_precision``.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities (positive class).
        target_precision: Minimum precision the operating point must
            reach.

    Returns:
        The maximum recall over operating points with
        ``precision >= target_precision``, or ``0.0`` if no threshold
        achieves that precision (defined behavior, not an error).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    achievable = precision >= target_precision
    if not achievable.any():
        return 0.0
    return float(recall[achievable].max())


def _best_f1_threshold(y_true, y_score) -> float:
    """Threshold maximizing F1 on the precision-recall curve.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities (positive class).

    Returns:
        The score threshold with the highest F1. The last
        precision/recall pair (which has no associated threshold) is
        excluded.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    best_f1 = -1.0
    best_threshold = thresholds[0]
    for i, threshold in enumerate(thresholds):
        p, r = precision[i], recall[i]
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold)


def evaluate_churn_model(
    trained_model: lgb.LGBMClassifier,
    val_snapshots: pd.DataFrame,
    test_snapshots: pd.DataFrame,
    precision_targets: list[float],
) -> dict[str, Any]:
    """Evaluate the trained model on val and test splits.

    The decision threshold is picked on VALIDATION predictions only
    (never on test) via ``_best_f1_threshold``. PR-AUC / ROC-AUC /
    recall-at-precision are reported for both splits; the confusion
    matrix at the validation threshold is reported for test only, plus
    a per-snapshot breakdown over test snapshots.

    Args:
        trained_model: Fitted ``lightgbm.LGBMClassifier``.
        val_snapshots: Validation split.
        test_snapshots: Test split.
        precision_targets: Precision levels for the
            ``recall_at_precision_<p>`` metrics (e.g. ``[0.5, 0.7]``).

    Returns:
        A dict with the exact shape::

            {
                "threshold_used": float,
                "val": {"pr_auc": float, "roc_auc": float,
                        "recall_at_precision_<p>": float, ...},
                "test": {"pr_auc": float, "roc_auc": float,
                         "recall_at_precision_<p>": float, ...,
                         "confusion_matrix": {"tn": int, "fp": int,
                                              "fn": int, "tp": int}},
                "test_per_snapshot": [
                    {"snapshot_date": str, "n_rows": int,
                     "churn_rate": float, "pr_auc": float,
                     "roc_auc": float}, ...
                ],
            }
    """
    X_val = val_snapshots[FEATURE_COLUMNS]
    y_val = val_snapshots["is_churned"]
    X_test = test_snapshots[FEATURE_COLUMNS]
    y_test = test_snapshots["is_churned"]

    val_score = trained_model.predict_proba(X_val)[:, 1]
    test_score = trained_model.predict_proba(X_test)[:, 1]
    threshold = _best_f1_threshold(y_val, val_score)

    def _split_metrics(y_true, y_score) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "roc_auc": float(roc_auc_score(y_true, y_score)),
        }
        for p in precision_targets:
            metrics[f"recall_at_precision_{p}"] = _recall_at_precision(
                y_true, y_score, p
            )
        return metrics

    val_metrics = _split_metrics(y_val, val_score)
    test_metrics = _split_metrics(y_test, test_score)

    tn, fp, fn, tp = confusion_matrix(
        y_test, (test_score >= threshold).astype(int)
    ).ravel()
    test_metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    # Per-snapshot breakdown, test only. Skips snapshots missing either
    # class: PR-AUC / ROC-AUC are not defined then.
    labeled = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(test_snapshots["snapshot_date"].to_numpy()),
            "churn_label": y_test.to_numpy(),
            "churn_score": test_score,
        }
    )
    test_per_snapshot: list[dict[str, Any]] = []
    for snapshot_date, group in labeled.groupby("snapshot_date", sort=True):
        n_positive = int(group["churn_label"].sum())
        n_negative = len(group) - n_positive
        if n_positive == 0 or n_negative == 0:
            logger.warning(
                "Skipping snapshot %s in the per-snapshot breakdown of the test "
                "split: PR-AUC and ROC-AUC need at least one row of each class, "
                "found %d positive / %d negative",
                snapshot_date.isoformat(),
                n_positive,
                n_negative,
            )
            continue
        test_per_snapshot.append(
            {
                "snapshot_date": snapshot_date.isoformat(),
                "n_rows": int(len(group)),
                "churn_rate": float(n_positive / len(group)),
                "pr_auc": float(
                    average_precision_score(group["churn_label"], group["churn_score"])
                ),
                "roc_auc": float(
                    roc_auc_score(group["churn_label"], group["churn_score"])
                ),
            }
        )

    logger.info(
        "Evaluation: threshold=%.4f | val pr_auc=%.4f roc_auc=%.4f | "
        "test pr_auc=%.4f roc_auc=%.4f",
        threshold,
        val_metrics["pr_auc"],
        val_metrics["roc_auc"],
        test_metrics["pr_auc"],
        test_metrics["roc_auc"],
    )
    return {
        "threshold_used": float(threshold),
        "val": val_metrics,
        "test": test_metrics,
        "test_per_snapshot": test_per_snapshot,
    }


def log_training_run_and_build_report(
    trained_model: lgb.LGBMClassifier,
    model_params: dict[str, Any],
    eval_metric: str,
    early_stopping_rounds: int,
    split_info: dict[str, Any],
    evaluation_metrics: dict[str, Any],
    mlflow_experiment_name: str,
) -> dict[str, Any]:
    """Log the training run to MLflow and build the persisted report.

    Manual logging only (no autolog: it is marked experimental for
    LightGBM in the MLflow docs and misses the custom metrics we
    produce). The model artifact goes to the MLflow artifact store,
    nothing is persisted to Mongo or as its own Kedro dataset.

    Args:
        trained_model: Fitted ``lightgbm.LGBMClassifier``.
        model_params: LightGBM hyperparameters used for the run.
        eval_metric: Eval metric name used for early stopping.
        early_stopping_rounds: Early stopping patience used.
        split_info: From ``split_snapshots_temporally``.
        evaluation_metrics: From ``evaluate_churn_model``.
        mlflow_experiment_name: MLflow experiment to log into.

    Returns:
        The complete ``training_report`` dict: run id, experiment name,
        model params, eval metric, early stopping rounds, best
        iteration, split info, and all evaluation metrics (val, test,
        test_per_snapshot).

    Note: ``artifact_path`` raises a DeprecationWarning in MLflow 3.x but remains
    compatible with ``runs:/<run_id>/<path>``; migrating to ``name=`` requires
    adopting the new Logged Models scheme, which has been reported as unstable in
    certain flows — revisit in M6 when building the real model loading path for
    production inference.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(mlflow_experiment_name)

    with mlflow.start_run():
        mlflow.log_params(model_params)
        mlflow.log_param("eval_metric", eval_metric)
        mlflow.log_param("early_stopping_rounds", early_stopping_rounds)

        mlflow.set_tags(
            {
                "train_snapshot_range": (
                    f"{split_info['train_snapshot_dates'][0]}.."
                    f"{split_info['train_snapshot_dates'][-1]}"
                ),
                "val_snapshot_range": (
                    f"{split_info['val_snapshot_dates'][0]}.."
                    f"{split_info['val_snapshot_dates'][-1]}"
                ),
                "test_snapshot_range": (
                    f"{split_info['test_snapshot_dates'][0]}.."
                    f"{split_info['test_snapshot_dates'][-1]}"
                ),
            }
        )

        flattened_metrics: dict[str, float] = {}
        for prefix, block in (
            ("val", evaluation_metrics["val"]),
            ("test", evaluation_metrics["test"]),
        ):
            for key, value in block.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flattened_metrics[f"{prefix}_{key}_{sub_key}"] = sub_value
                else:
                    flattened_metrics[f"{prefix}_{key}"] = value
        mlflow.log_metrics(flattened_metrics)

        mlflow.lightgbm.log_model(trained_model, artifact_path="model")

        # active_run() returns None once the run is finished, so capture
        # the run id inside the with block.
        run_id = mlflow.active_run().info.run_id

    training_report = {
        "run_id": run_id,
        "mlflow_experiment_name": mlflow_experiment_name,
        "model_params": model_params,
        "eval_metric": eval_metric,
        "early_stopping_rounds": early_stopping_rounds,
        "best_iteration": int(trained_model.best_iteration_),
        "split_info": split_info,
        "evaluation_metrics": {
            "val": evaluation_metrics["val"],
            "test": evaluation_metrics["test"],
            "test_per_snapshot": evaluation_metrics["test_per_snapshot"],
        },
    }

    logger.info(
        "Training run %s logged to experiment %r (test pr_auc=%.4f "
        "roc_auc=%.4f)",
        run_id,
        mlflow_experiment_name,
        evaluation_metrics["test"]["pr_auc"],
        evaluation_metrics["test"]["roc_auc"],
    )
    return training_report


def verify_training_run_is_retrievable(training_report: dict[str, Any]) -> None:
    """Verify the logged model can be reloaded from the MLflow artifact store.

    ``training_report`` arrives re-read from disk (a real JSON
    round-trip through the Kedro catalog). This node is a
    verifier/observability step: any failure is logged as a WARNING,
    never raised, so a corrupt artifact cannot take the pipeline down.

    Args:
        training_report: The report dict built by
            ``log_training_run_and_build_report`` and persisted by Kedro.
    """
    run_id = training_report["run_id"]
    try:
        model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    except Exception as exc:
        logger.warning(
            "Could not reload the model for run %s from MLflow: %s", run_id, exc
        )
        return
    if model.n_features_in_ == len(FEATURE_COLUMNS):
        logger.info(
            "Model for run %s reloaded from MLflow: n_features_in_=%d matches "
            "FEATURE_COLUMNS (%d)",
            run_id,
            model.n_features_in_,
            len(FEATURE_COLUMNS),
        )
    else:
        logger.warning(
            "Model for run %s reloaded from MLflow but n_features_in_=%d differs "
            "from len(FEATURE_COLUMNS)=%d",
            run_id,
            model.n_features_in_,
            len(FEATURE_COLUMNS),
        )
