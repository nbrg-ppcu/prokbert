
"""Utilities for segment-level and sequence-level classification evaluation.

This module is designed for `ProkBertForSequenceClassification` and for the
current Hugging Face Trainer evaluation flow.

Public API
----------
Table-first / external models
    - compute_classification_metrics_from_predictions
    - evaluate_classification_prediction_table

Segment level
    - build_segment_classification_prediction_table
    - evaluate_segment_classification_predictions

Sequence level
    - aggregate_sequence_classification_prediction_table
    - build_sequence_classification_prediction_table
    - evaluate_sequence_classification_predictions
    - build_sequence_classification_compute_metrics

Trainer hooks
    - compute_metrics
    - preprocess_logits_for_metrics

Design notes
------------
- Single-label classification only.
- Binary and multiclass classification are supported.
- Table-first evaluation supports externally generated predictions, including
  non-Transformer models that already provide `predicted_label_id`.
- Sequence aggregation requires `sequence_id` in the original segment dataset.
- The public `evaluate_*` functions return `(prediction_table, metrics)`.
- The public `compute_metrics` functions return metrics only, as expected by
  `transformers.Trainer`.
- Sequence aggregation is implemented with NumPy on factorized sequence ids,
  which is significantly faster than building a large pandas groupby pipeline.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = [
    "aggregate_sequence_classification_prediction_table",
    "build_segment_classification_prediction_table",
    "build_sequence_classification_compute_metrics",
    "build_sequence_classification_prediction_table",
    "compute_classification_metrics",
    "compute_classification_metrics_from_predictions",
    "compute_classification_metrics_from_probabilities",
    "compute_metrics",
    "evaluate_classification_prediction_table",
    "evaluate_segment_classification_predictions",
    "evaluate_sequence_classification_predictions",
    "preprocess_logits_for_metrics",
]

_ALLOWED_SEQUENCE_AGGREGATION_METHODS = {
    "mean_logits",
    "mean_probabilities",
    "mean_log_probabilities",
    "weighted_mean_log_probabilities",
    "topk_mean_logits",
    "topk_mean_probabilities",
    "topk_mean_log_probabilities",
}


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors, arrays, and sequences to a NumPy array."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _as_2d_logits_array(value: Any) -> np.ndarray:
    """Return a validated 2D logits array."""
    logits = _to_numpy(value)
    if logits.ndim != 2:
        raise ValueError(
            "Expected logits with shape (n_samples, n_classes). "
            f"Received {getattr(logits, 'shape', None)!r}."
        )
    if not np.all(np.isfinite(logits)):
        raise ValueError("Logits contain non-finite values.")
    return logits


def _as_1d_label_array(value: Any) -> np.ndarray:
    """Return a validated 1D integer labels array."""
    label_ids = _to_numpy(value)

    if label_ids.ndim == 2 and label_ids.shape[1] == 1:
        label_ids = label_ids[:, 0]

    if label_ids.ndim != 1:
        raise ValueError(
            "Expected labels with shape (n_samples,). "
            f"Received {getattr(label_ids, 'shape', None)!r}."
        )

    if not np.all(np.isfinite(label_ids)):
        raise ValueError("Labels contain non-finite values.")

    rounded = np.rint(label_ids)
    if not np.all(label_ids == rounded):
        raise ValueError("Single-label classification requires integer class ids.")

    return rounded.astype(np.int64, copy=False)


def _extract_logits_and_label_ids(
    predictions: Any,
    label_ids: Optional[Any] = None,
    *,
    allow_missing_labels: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract logits and optional labels from a Trainer-style prediction object.

    Accepted forms for `predictions`
    --------------------------------
    - `EvalPrediction`-like objects with `.predictions` and optional `.label_ids`
    - `(logits, label_ids)` tuple
    - raw logits array
    - model output tuples where logits are the first element
    """
    raw_predictions = predictions
    raw_label_ids = label_ids

    if hasattr(raw_predictions, "predictions"):
        raw_predictions = raw_predictions.predictions
        if raw_label_ids is None and hasattr(predictions, "label_ids"):
            raw_label_ids = predictions.label_ids
    elif isinstance(raw_predictions, (tuple, list)) and raw_label_ids is None and len(raw_predictions) == 2:
        raw_predictions, raw_label_ids = raw_predictions

    while isinstance(raw_predictions, (tuple, list)):
        if len(raw_predictions) == 0:
            raise ValueError("Received an empty predictions tuple/list.")
        raw_predictions = raw_predictions[0]

    logits = _as_2d_logits_array(raw_predictions)

    if raw_label_ids is None:
        if allow_missing_labels:
            return logits, None
        raise ValueError("Labels are required for evaluation.")

    while isinstance(raw_label_ids, (tuple, list)):
        if len(raw_label_ids) != 1:
            raise ValueError("Expected label_ids to contain a single array.")
        raw_label_ids = raw_label_ids[0]

    labels = _as_1d_label_array(raw_label_ids)

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "The number of predictions and labels does not match: "
            f"{logits.shape[0]} != {labels.shape[0]}."
        )

    return logits, labels


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable log-softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _normalize_probabilities(probabilities: Any) -> np.ndarray:
    """Return a validated probability matrix with rows summing to one."""
    probabilities = _to_numpy(probabilities)
    if probabilities.ndim != 2:
        raise ValueError(
            "Expected probabilities with shape (n_samples, n_classes). "
            f"Received {getattr(probabilities, 'shape', None)!r}."
        )
    probabilities = probabilities.astype(np.float64, copy=False)
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each probability row must have a positive sum.")
    return probabilities / row_sums


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return `numerator / denominator`, or 0.0 when the denominator is zero."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _validate_label_ids(label_ids: np.ndarray, num_classes: int) -> None:
    """Validate integer class ids for single-label classification."""
    if num_classes < 2:
        raise ValueError(
            "Classification evaluation requires at least 2 classes. "
            f"Received num_classes={num_classes}."
        )
    if np.any(label_ids < 0) or np.any(label_ids >= num_classes):
        raise ValueError(
            f"Labels must be in the range [0, {num_classes - 1}]. "
            f"Received min={label_ids.min()} and max={label_ids.max()}."
        )


def _binary_confusion_metrics(
    label_ids: np.ndarray,
    predicted_label_ids: np.ndarray,
) -> Dict[str, float]:
    """Compute binary-only confusion-matrix-derived metrics."""
    tn, fp, fn, tp = confusion_matrix(label_ids, predicted_label_ids, labels=[0, 1]).ravel()

    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    negative_predicted_value = _safe_divide(tn, tn + fn)

    return {
        "recall": recall,
        "sensitivity": recall,
        "precision": precision,
        "neg_pred_val": negative_predicted_value,
        "specificity": specificity,
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "Np": float(tp + fn),
        "Nn": float(tn + fp),
    }


def _per_class_auc(label_ids: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    """Compute one-vs-rest AUROC for each class."""
    num_classes = probabilities.shape[1]
    auc_results: Dict[str, float] = {}

    for class_idx in range(num_classes):
        class_targets = (label_ids == class_idx).astype(np.int64)

        if class_targets.min() == class_targets.max():
            auc_results[f"auc_class{class_idx}"] = float("nan")
            continue

        auc_results[f"auc_class{class_idx}"] = float(
            roc_auc_score(class_targets, probabilities[:, class_idx])
        )

    return auc_results


def _validate_predicted_label_ids(predicted_label_ids: np.ndarray, num_classes: int) -> None:
    """Validate predicted integer class ids for single-label classification."""
    _validate_label_ids(predicted_label_ids, num_classes)


def _infer_num_classes(
    label_ids: np.ndarray,
    predicted_label_ids: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    *,
    num_classes: Optional[int] = None,
) -> int:
    """Infer the number of classes from explicit input or observed arrays."""
    if probabilities is not None:
        inferred_num_classes = int(probabilities.shape[1])
    elif num_classes is not None:
        inferred_num_classes = int(num_classes)
    else:
        inferred_num_classes = int(
            max(
                label_ids.max(initial=0),
                predicted_label_ids.max(initial=0),
            )
        ) + 1

    if num_classes is not None and int(num_classes) != inferred_num_classes:
        raise ValueError(
            "num_classes does not match the provided probability matrix: "
            f"{int(num_classes)} != {inferred_num_classes}."
        )

    if inferred_num_classes < 2:
        raise ValueError(
            "Classification evaluation requires at least 2 classes. "
            f"Received num_classes={inferred_num_classes}."
        )

    return inferred_num_classes


def compute_classification_metrics_from_predictions(
    predicted_label_ids: Any,
    label_ids: Any,
    probabilities: Optional[Any] = None,
    *,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Compute single-label classification metrics from class predictions.

    Parameters
    ----------
    predicted_label_ids
        Predicted single-label class ids. These are used for all discrete
        metrics such as accuracy, balanced accuracy, F1, MCC, recall, and
        precision.
    label_ids
        Ground-truth single-label class ids.
    probabilities
        Optional class probabilities or non-negative class scores with shape
        `(n_samples, n_classes)`. When provided, they are used for probability-
        dependent metrics (`CE_loss` and per-class AUROC). The predicted labels
        are *not* recomputed from these probabilities; this keeps externally
        supplied `predicted_label_id` values intact.
    num_classes
        Optional explicit class count. This is only needed when probabilities
        are not provided and the observed labels do not span all classes.

    Notes
    -----
    - When `probabilities` is omitted, `CE_loss` and per-class AUROC values are
      returned as `NaN`.
    - This function is intended for table-first evaluation and external models
      that already provide `predicted_label_id`.
    """
    label_ids_np = _as_1d_label_array(label_ids)
    predicted_label_ids_np = _as_1d_label_array(predicted_label_ids)

    if label_ids_np.shape[0] == 0:
        raise ValueError("At least one row is required for evaluation.")

    if label_ids_np.shape[0] != predicted_label_ids_np.shape[0]:
        raise ValueError(
            "The number of label_ids and predicted_label_ids does not match: "
            f"{label_ids_np.shape[0]} != {predicted_label_ids_np.shape[0]}."
        )

    probabilities_np = None
    if probabilities is not None:
        probabilities_np = _normalize_probabilities(probabilities)
        if probabilities_np.shape[0] != label_ids_np.shape[0]:
            raise ValueError(
                "The number of probability rows and label_ids does not match: "
                f"{probabilities_np.shape[0]} != {label_ids_np.shape[0]}."
            )

    num_classes_np = _infer_num_classes(
        label_ids_np,
        predicted_label_ids_np,
        probabilities_np,
        num_classes=num_classes,
    )
    _validate_label_ids(label_ids_np, num_classes_np)
    _validate_predicted_label_ids(predicted_label_ids_np, num_classes_np)

    is_binary = num_classes_np == 2
    average = "binary" if is_binary else "macro"

    recall = float(
        recall_score(
            label_ids_np,
            predicted_label_ids_np,
            average=average,
            zero_division=0,
        )
    )
    precision = float(
        precision_score(
            label_ids_np,
            predicted_label_ids_np,
            average=average,
            zero_division=0,
        )
    )

    if probabilities_np is None:
        cross_entropy_loss = float("nan")
    else:
        epsilon = np.finfo(np.float64).eps
        true_class_probabilities = probabilities_np[np.arange(label_ids_np.shape[0]), label_ids_np]
        cross_entropy_loss = float(
            -np.mean(np.log(np.clip(true_class_probabilities, epsilon, 1.0)))
        )

    metrics: Dict[str, float] = {
        "CE_loss": cross_entropy_loss,
        "acc": float(accuracy_score(label_ids_np, predicted_label_ids_np)),
        "bal_acc": float(balanced_accuracy_score(label_ids_np, predicted_label_ids_np)),
        "f1": float(
            f1_score(
                label_ids_np,
                predicted_label_ids_np,
                average=average,
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(label_ids_np, predicted_label_ids_np)),
        "recall": recall,
        "sensitivity": recall,
        "precision": precision,
    }

    if probabilities_np is not None:
        metrics.update(_per_class_auc(label_ids_np, probabilities_np))
    else:
        for class_idx in range(num_classes_np):
            metrics[f"auc_class{class_idx}"] = float("nan")

    if is_binary:
        metrics.update(_binary_confusion_metrics(label_ids_np, predicted_label_ids_np))

    return metrics


def compute_classification_metrics_from_probabilities(
    probabilities: Any,
    label_ids: Any,
) -> Dict[str, float]:
    """Compute single-label classification metrics from probabilities."""
    probabilities_np = _normalize_probabilities(probabilities)
    label_ids_np = _as_1d_label_array(label_ids)
    predicted_label_ids = np.argmax(probabilities_np, axis=1).astype(np.int64, copy=False)

    return compute_classification_metrics_from_predictions(
        predicted_label_ids=predicted_label_ids,
        label_ids=label_ids_np,
        probabilities=probabilities_np,
        num_classes=probabilities_np.shape[1],
    )


def compute_classification_metrics(
    logits: Any,
    label_ids: Any,
) -> Dict[str, float]:
    """Compute single-label classification metrics from logits."""
    logits_np = _as_2d_logits_array(logits)
    label_ids_np = _as_1d_label_array(label_ids)

    if logits_np.shape[0] != label_ids_np.shape[0]:
        raise ValueError(
            "The number of prediction rows and label_ids does not match: "
            f"{logits_np.shape[0]} != {label_ids_np.shape[0]}."
        )

    _validate_label_ids(label_ids_np, logits_np.shape[1])
    probabilities_np = _softmax(logits_np)
    return compute_classification_metrics_from_probabilities(probabilities_np, label_ids_np)


def _get_dataset_columns(dataset: Any) -> list[str]:
    """Return the column names of a pandas or Hugging Face dataset."""
    if isinstance(dataset, pd.DataFrame):
        return list(dataset.columns)
    return list(dataset.column_names)


def _select_dataset_columns(dataset: Any, columns: Sequence[str]) -> pd.DataFrame:
    """Select columns from a pandas or Hugging Face dataset."""
    if isinstance(dataset, pd.DataFrame):
        return dataset.loc[:, list(columns)].reset_index(drop=True).copy()
    return dataset.select_columns(list(columns)).to_pandas().reset_index(drop=True)


def _maybe_get_dataset_labels(
    segment_dataset: Optional[Any],
    label_col: str,
    expected_length: int,
) -> Optional[np.ndarray]:
    """Extract labels from the dataset when available."""
    if segment_dataset is None:
        return None

    dataset_columns = set(_get_dataset_columns(segment_dataset))
    if label_col not in dataset_columns:
        return None

    label_df = _select_dataset_columns(segment_dataset, [label_col])
    if len(label_df) != expected_length:
        raise ValueError(
            "Prediction row count and segment_dataset row count must match: "
            f"{expected_length} != {len(label_df)}."
        )

    return _as_1d_label_array(label_df[label_col].to_numpy())


def build_segment_classification_prediction_table(
    predictions: Any,
    segment_dataset: Optional[Any] = None,
    label_ids: Optional[Any] = None,
    *,
    segment_id_col: str = "segment_id",
    sequence_id_col: str = "sequence_id",
    label_col: str = "labels",
    extra_metadata_cols: Optional[Sequence[str]] = None,
    include_probabilities: bool = True,
) -> pd.DataFrame:
    """Build a segment-level classification prediction table.

    Notes
    -----
    - This function is row-order based. When `predictions` comes from
      `trainer.predict(eval_dataset)`, pass the same `eval_dataset` as
      `segment_dataset`.
    - `segment_id_col` and `sequence_id_col` are optional. They are included only
      when present in `segment_dataset`.
    """
    logits_np, label_ids_np = _extract_logits_and_label_ids(
        predictions,
        label_ids,
        allow_missing_labels=True,
    )
    num_segments, num_classes = logits_np.shape

    if label_ids_np is None:
        label_ids_np = _maybe_get_dataset_labels(segment_dataset, label_col, num_segments)

    if label_ids_np is not None:
        _validate_label_ids(label_ids_np, num_classes)

    predicted_label_ids = np.argmax(logits_np, axis=1)

    columns: Dict[str, Any] = {}

    if segment_dataset is not None:
        dataset_columns = set(_get_dataset_columns(segment_dataset))
        metadata_columns: list[str] = []

        for column_name in (segment_id_col, sequence_id_col):
            if column_name in dataset_columns and column_name not in metadata_columns:
                metadata_columns.append(column_name)

        if extra_metadata_cols is not None:
            for column_name in extra_metadata_cols:
                if column_name in dataset_columns and column_name not in metadata_columns:
                    metadata_columns.append(column_name)

        if metadata_columns:
            metadata_df = _select_dataset_columns(segment_dataset, metadata_columns)
            if len(metadata_df) != num_segments:
                raise ValueError(
                    "Prediction row count and segment_dataset row count must match: "
                    f"{num_segments} != {len(metadata_df)}."
                )
            for column_name in metadata_columns:
                columns[column_name] = metadata_df[column_name].to_numpy()

    if label_ids_np is not None:
        columns["label_id"] = label_ids_np

    columns["predicted_label_id"] = predicted_label_ids.astype(np.int64)

    for class_idx in range(num_classes):
        columns[f"logit_class_{class_idx}"] = logits_np[:, class_idx]

    if include_probabilities:
        probabilities_np = _softmax(logits_np)
        for class_idx in range(num_classes):
            columns[f"prob_class_{class_idx}"] = probabilities_np[:, class_idx]

    return pd.DataFrame(columns)


def evaluate_classification_prediction_table(
    prediction_table: pd.DataFrame,
    *,
    label_col: str = "label_id",
    predicted_label_col: str = "predicted_label_id",
    probability_col_prefix: str = "prob_class_",
    use_probability_metrics: bool = True,
    num_classes: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate a precomputed classification prediction table.

    This is the table-first entry point for externally generated predictions and
    non-Transformer models. The table may already be segment-level or
    sequence-level.

    Required columns
    ----------------
    - `label_col`
    - either `predicted_label_col`, probability columns named
      `f"{probability_col_prefix}{i}"`, or both

    Notes
    -----
    - When `predicted_label_col` is present, it is used as-is for all discrete
      metrics; predictions are not recomputed from the probability columns.
    - When `use_probability_metrics=False`, probability columns are ignored even
      if they are present. In that case `CE_loss` and AUROC metrics are
      returned as `NaN`.
    """
    if not isinstance(prediction_table, pd.DataFrame):
        raise TypeError("prediction_table must be a pandas DataFrame.")

    if label_col not in prediction_table.columns:
        raise ValueError(f"{label_col!r} is required in prediction_table.")

    evaluation_table = prediction_table.reset_index(drop=True).copy()
    label_ids_np = _as_1d_label_array(evaluation_table[label_col].to_numpy())

    predicted_label_ids_np: Optional[np.ndarray]
    if predicted_label_col in evaluation_table.columns:
        predicted_label_ids_np = _as_1d_label_array(
            evaluation_table[predicted_label_col].to_numpy()
        )
    else:
        predicted_label_ids_np = None

    num_classes_hint = num_classes
    if predicted_label_ids_np is not None and num_classes_hint is None:
        num_classes_hint = int(
            max(
                label_ids_np.max(initial=0),
                predicted_label_ids_np.max(initial=0),
            )
        ) + 1

    probabilities_np = None
    if use_probability_metrics:
        probabilities_np = _extract_probability_matrix_from_prediction_table(
            evaluation_table,
            probability_col_prefix=probability_col_prefix,
            num_classes=num_classes_hint,
        )
        if probabilities_np is not None and num_classes_hint is None:
            num_classes_hint = int(probabilities_np.shape[1])

    if predicted_label_ids_np is None:
        if probabilities_np is None:
            raise ValueError(
                f"{predicted_label_col!r} is required when probability columns are absent "
                "or when use_probability_metrics=False."
            )
        predicted_label_ids_np = np.argmax(probabilities_np, axis=1).astype(np.int64, copy=False)
        evaluation_table[predicted_label_col] = predicted_label_ids_np

    metrics = compute_classification_metrics_from_predictions(
        predicted_label_ids=predicted_label_ids_np,
        label_ids=label_ids_np,
        probabilities=probabilities_np,
        num_classes=num_classes_hint,
    )
    return evaluation_table, metrics


def evaluate_segment_classification_predictions(
    predictions: Any,
    segment_dataset: Optional[Any] = None,
    label_ids: Optional[Any] = None,
    *,
    segment_id_col: str = "segment_id",
    sequence_id_col: str = "sequence_id",
    label_col: str = "labels",
    extra_metadata_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate segment-level single-label classification predictions."""
    segment_prediction_table = build_segment_classification_prediction_table(
        predictions=predictions,
        segment_dataset=segment_dataset,
        label_ids=label_ids,
        segment_id_col=segment_id_col,
        sequence_id_col=sequence_id_col,
        label_col=label_col,
        extra_metadata_cols=extra_metadata_cols,
        include_probabilities=True,
    )

    if "label_id" not in segment_prediction_table.columns:
        raise ValueError(
            "Labels are required for segment-level evaluation. Provide them via "
            "`predictions.label_ids`, the `label_ids` argument, or `segment_dataset`."
        )

    return evaluate_classification_prediction_table(segment_prediction_table)


def _validate_aggregation_method(aggregation_method: str) -> str:
    """Validate the sequence aggregation method."""
    if aggregation_method not in _ALLOWED_SEQUENCE_AGGREGATION_METHODS:
        allowed = ", ".join(sorted(_ALLOWED_SEQUENCE_AGGREGATION_METHODS))
        raise ValueError(
            f"Unsupported aggregation_method={aggregation_method!r}. "
            f"Supported values: {allowed}."
        )
    return aggregation_method


def _get_indexed_class_columns(frame: pd.DataFrame, prefix: str) -> list[tuple[int, str]]:
    """Return sorted `(class_index, column_name)` pairs for class score columns."""
    indexed_columns: list[tuple[int, str]] = []

    for column_name in frame.columns:
        if not column_name.startswith(prefix):
            continue
        suffix = column_name[len(prefix) :]
        if suffix.isdigit():
            indexed_columns.append((int(suffix), column_name))

    indexed_columns.sort(key=lambda item: item[0])
    return indexed_columns


def _get_class_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    """Return sorted class columns with names like `{prefix}{i}`."""
    return [column_name for _, column_name in _get_indexed_class_columns(frame, prefix)]


def _extract_probability_matrix_from_prediction_table(
    prediction_table: pd.DataFrame,
    *,
    probability_col_prefix: str,
    num_classes: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Extract probability columns from a prediction table.

    Supported layouts
    -----------------
    - `prob_class_0`, `prob_class_1`, ..., `prob_class_{n-1}`
    - binary-only shorthand with a single `prob_class_1` or `prob_class_0`
      column, which is expanded to two columns by complementing to one
    """
    indexed_prob_columns = _get_indexed_class_columns(prediction_table, probability_col_prefix)
    if not indexed_prob_columns:
        return None

    class_indices = [class_idx for class_idx, _ in indexed_prob_columns]

    if class_indices == [1] and (num_classes is None or num_classes == 2):
        positive_probabilities = prediction_table[indexed_prob_columns[0][1]].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        if not np.all(np.isfinite(positive_probabilities)):
            raise ValueError("Probability columns contain non-finite values.")
        if np.any((positive_probabilities < 0.0) | (positive_probabilities > 1.0)):
            raise ValueError(
                "A single `prob_class_1` column must contain values in the range [0, 1]."
            )
        return np.column_stack([1.0 - positive_probabilities, positive_probabilities])

    if class_indices == [0] and (num_classes is None or num_classes == 2):
        negative_probabilities = prediction_table[indexed_prob_columns[0][1]].to_numpy(
            dtype=np.float64,
            copy=False,
        )
        if not np.all(np.isfinite(negative_probabilities)):
            raise ValueError("Probability columns contain non-finite values.")
        if np.any((negative_probabilities < 0.0) | (negative_probabilities > 1.0)):
            raise ValueError(
                "A single `prob_class_0` column must contain values in the range [0, 1]."
            )
        return np.column_stack([negative_probabilities, 1.0 - negative_probabilities])

    expected_indices = list(range(class_indices[-1] + 1))
    if class_indices != expected_indices:
        raise ValueError(
            "Probability columns must be consecutive from class 0. "
            f"Received class indices {class_indices!r}."
        )

    return prediction_table[
        [column_name for _, column_name in indexed_prob_columns]
    ].to_numpy(dtype=np.float64, copy=False)


def _factorize_sequence_ids(sequence_ids: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Factorize sequence ids while preserving first-seen order."""
    codes, unique_values = pd.factorize(sequence_ids, sort=False)

    if np.any(codes < 0):
        raise ValueError("sequence_id contains missing values, which are not supported.")

    return codes.astype(np.int64, copy=False), np.asarray(unique_values)


def _group_counts(codes: np.ndarray, num_groups: int) -> np.ndarray:
    """Count the number of rows in each group."""
    return np.bincount(codes, minlength=num_groups).astype(np.int64, copy=False)


def _group_mean(
    values: np.ndarray,
    codes: np.ndarray,
    num_groups: int,
    *,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a grouped mean for a 2D array.

    This implementation is optimized for a relatively small number of classes and
    a large number of rows. It loops over the class axis and uses `np.bincount`
    for the group reduction.
    """
    num_classes = values.shape[1]
    out = np.empty((num_groups, num_classes), dtype=np.float64)

    if weights is None:
        denom = np.bincount(codes, minlength=num_groups).astype(np.float64)
        for class_idx in range(num_classes):
            out[:, class_idx] = (
                np.bincount(codes, weights=values[:, class_idx], minlength=num_groups) / denom
            )
        return out

    denom = np.bincount(codes, weights=weights, minlength=num_groups).astype(np.float64)
    if np.any(denom <= 0.0):
        raise ValueError("All sequence weights must sum to a positive value.")

    for class_idx in range(num_classes):
        out[:, class_idx] = (
            np.bincount(
                codes,
                weights=values[:, class_idx] * weights,
                minlength=num_groups,
            )
            / denom
        )

    return out


def _group_first_and_validate_constant(
    values: np.ndarray,
    codes: np.ndarray,
    num_groups: int,
    *,
    value_name: str,
) -> np.ndarray:
    """Return the first value per group and validate that all values in a group match."""
    if values.shape[0] != codes.shape[0]:
        raise ValueError(f"{value_name} length does not match the number of rows.")

    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    values_sorted = values[order]

    starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
    mins = np.minimum.reduceat(values_sorted, starts)
    maxs = np.maximum.reduceat(values_sorted, starts)

    inconsistent_mask = mins != maxs
    if np.any(inconsistent_mask):
        bad_positions = np.flatnonzero(inconsistent_mask)[:10]
        raise ValueError(
            f"Each sequence must have exactly one {value_name} value. "
            f"Found conflicting values for {bad_positions.size} sequence(s); "
            f"first failing group index examples: {bad_positions.tolist()}"
        )

    if starts.shape[0] != num_groups:
        raise ValueError("Internal grouping error: number of groups does not match the starts array.")

    return values_sorted[starts]


def _group_topk_mean(
    values: np.ndarray,
    codes: np.ndarray,
    num_groups: int,
    *,
    top_k: int,
) -> np.ndarray:
    """Compute the mean of the top-k values within each group for each class."""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, received top_k={top_k}.")

    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    values_sorted = values[order]

    starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
    ends = np.r_[starts[1:], len(codes_sorted)]

    if starts.shape[0] != num_groups:
        raise ValueError("Internal grouping error: number of groups does not match the starts array.")

    out = np.empty((num_groups, values.shape[1]), dtype=np.float64)

    for group_idx, (start, end) in enumerate(zip(starts, ends)):
        block = values_sorted[start:end]
        k = min(top_k, end - start)

        if k == block.shape[0]:
            out[group_idx] = block.mean(axis=0)
            continue

        partitioned = np.partition(block, kth=block.shape[0] - k, axis=0)
        out[group_idx] = partitioned[-k:].mean(axis=0)

    return out


def _aggregate_sequence_probabilities_from_arrays(
    sequence_ids: Any,
    *,
    logits: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    label_ids: Optional[np.ndarray] = None,
    aggregation_method: str,
    weights: Optional[np.ndarray] = None,
    top_k: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Aggregate segment predictions into sequence-level probabilities.

    This is the fast internal path used by the public sequence-level helpers.
    """
    aggregation_method = _validate_aggregation_method(aggregation_method)

    if logits is None and probabilities is None:
        raise ValueError("At least one of `logits` or `probabilities` is required.")

    if logits is not None:
        logits = _as_2d_logits_array(logits)
        num_rows = logits.shape[0]
        num_classes = logits.shape[1]
    else:
        probabilities = _normalize_probabilities(probabilities)
        num_rows = probabilities.shape[0]
        num_classes = probabilities.shape[1]

    if probabilities is not None and probabilities.shape != (num_rows, num_classes):
        raise ValueError("Probabilities shape does not match logits shape.")

    sequence_ids_np = _to_numpy(sequence_ids)
    if sequence_ids_np.shape[0] != num_rows:
        raise ValueError(
            "The number of sequence ids and prediction rows does not match: "
            f"{sequence_ids_np.shape[0]} != {num_rows}."
        )

    codes, unique_sequence_ids = _factorize_sequence_ids(sequence_ids_np)
    num_groups = unique_sequence_ids.shape[0]
    segment_counts = _group_counts(codes, num_groups)

    if label_ids is not None:
        label_ids = _as_1d_label_array(label_ids)
        if label_ids.shape[0] != num_rows:
            raise ValueError(
                "The number of label_ids and prediction rows does not match: "
                f"{label_ids.shape[0]} != {num_rows}."
            )
        _validate_label_ids(label_ids, num_classes)
        sequence_label_ids = _group_first_and_validate_constant(
            label_ids,
            codes,
            num_groups,
            value_name="label_id",
        ).astype(np.int64, copy=False)
    else:
        sequence_label_ids = None

    if weights is not None:
        weights = _to_numpy(weights).reshape(-1)
        if weights.shape[0] != num_rows:
            raise ValueError(
                "The number of weights and prediction rows does not match: "
                f"{weights.shape[0]} != {num_rows}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("Weights contain non-finite values.")
        if np.any(weights < 0.0):
            raise ValueError("Weights must be non-negative.")

    # Prepare the score matrix required by the selected method.
    if aggregation_method == "mean_logits":
        if logits is None:
            raise ValueError("mean_logits aggregation requires logits.")
        sequence_probabilities = _softmax(_group_mean(logits, codes, num_groups))

    elif aggregation_method == "mean_probabilities":
        if probabilities is None:
            if logits is None:
                raise ValueError("mean_probabilities aggregation requires logits or probabilities.")
            probabilities = _softmax(logits)
        sequence_probabilities = _group_mean(probabilities, codes, num_groups)
        sequence_probabilities = _normalize_probabilities(sequence_probabilities)

    elif aggregation_method == "mean_log_probabilities":
        if logits is None:
            raise ValueError("mean_log_probabilities aggregation requires logits.")
        log_probabilities = _log_softmax(logits)
        sequence_probabilities = _softmax(_group_mean(log_probabilities, codes, num_groups))

    elif aggregation_method == "weighted_mean_log_probabilities":
        if logits is None:
            raise ValueError("weighted_mean_log_probabilities aggregation requires logits.")
        if weights is None:
            raise ValueError("weighted_mean_log_probabilities aggregation requires `weights`.")
        log_probabilities = _log_softmax(logits)
        sequence_probabilities = _softmax(
            _group_mean(log_probabilities, codes, num_groups, weights=weights)
        )

    elif aggregation_method == "topk_mean_logits":
        if logits is None:
            raise ValueError("topk_mean_logits aggregation requires logits.")
        if top_k is None:
            raise ValueError("topk_mean_logits aggregation requires `top_k`.")
        sequence_probabilities = _softmax(
            _group_topk_mean(logits, codes, num_groups, top_k=top_k)
        )

    elif aggregation_method == "topk_mean_probabilities":
        if top_k is None:
            raise ValueError("topk_mean_probabilities aggregation requires `top_k`.")
        if probabilities is None:
            if logits is None:
                raise ValueError("topk_mean_probabilities aggregation requires logits or probabilities.")
            probabilities = _softmax(logits)
        sequence_probabilities = _group_topk_mean(probabilities, codes, num_groups, top_k=top_k)
        sequence_probabilities = _normalize_probabilities(sequence_probabilities)

    elif aggregation_method == "topk_mean_log_probabilities":
        if logits is None:
            raise ValueError("topk_mean_log_probabilities aggregation requires logits.")
        if top_k is None:
            raise ValueError("topk_mean_log_probabilities aggregation requires `top_k`.")
        log_probabilities = _log_softmax(logits)
        sequence_probabilities = _softmax(
            _group_topk_mean(log_probabilities, codes, num_groups, top_k=top_k)
        )

    else:
        raise RuntimeError(f"Unhandled aggregation_method={aggregation_method!r}.")

    predicted_label_ids = np.argmax(sequence_probabilities, axis=1).astype(np.int64, copy=False)

    result: Dict[str, np.ndarray] = {
        "sequence_id": unique_sequence_ids,
        "segment_count": segment_counts,
        "predicted_label_id": predicted_label_ids,
        "sequence_probabilities": sequence_probabilities,
    }

    if sequence_label_ids is not None:
        result["label_id"] = sequence_label_ids

    return result


def aggregate_sequence_classification_prediction_table(
    segment_prediction_table: pd.DataFrame,
    *,
    sequence_id_col: str = "sequence_id",
    aggregation_method: str = "mean_logits",
    weight_col: Optional[str] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate a segment-level prediction table to sequence level.

    Expected segment-level columns
    ------------------------------
    Required
        - `sequence_id`
        - either `logit_class_{i}` columns or `prob_class_{i}` columns,
          depending on `aggregation_method`

    Optional
        - `label_id`
        - `weight_col`, for `weighted_mean_log_probabilities`

    Notes
    -----
    This function is meant for post-hoc aggregation of externally constructed
    segment prediction tables. For predictions coming directly from
    `trainer.predict(...)`, use `build_sequence_classification_prediction_table`,
    which avoids building a large intermediate segment table.
    """
    aggregation_method = _validate_aggregation_method(aggregation_method)

    if not isinstance(segment_prediction_table, pd.DataFrame):
        raise TypeError("segment_prediction_table must be a pandas DataFrame.")

    if sequence_id_col not in segment_prediction_table.columns:
        raise ValueError(
            f"{sequence_id_col!r} is required for sequence aggregation, but it is missing."
        )

    logit_columns = _get_class_columns(segment_prediction_table, "logit_class_")
    prob_columns = _get_class_columns(segment_prediction_table, "prob_class_")

    logits_np = None
    probabilities_np = None

    if logit_columns:
        logits_np = segment_prediction_table[logit_columns].to_numpy(dtype=np.float64, copy=False)

    if prob_columns:
        probabilities_np = segment_prediction_table[prob_columns].to_numpy(dtype=np.float64, copy=False)

    if aggregation_method in {
        "mean_logits",
        "mean_log_probabilities",
        "weighted_mean_log_probabilities",
        "topk_mean_logits",
        "topk_mean_log_probabilities",
    } and logits_np is None:
        raise ValueError(
            f"{aggregation_method} aggregation requires columns named 'logit_class_{{i}}'."
        )

    if aggregation_method in {
        "mean_probabilities",
        "topk_mean_probabilities",
    } and logits_np is None and probabilities_np is None:
        raise ValueError(
            f"{aggregation_method} aggregation requires either 'prob_class_{{i}}' "
            "or 'logit_class_{i}' columns."
        )

    label_ids_np = None
    if "label_id" in segment_prediction_table.columns:
        label_ids_np = segment_prediction_table["label_id"].to_numpy(dtype=np.int64, copy=False)

    weights_np = None
    if aggregation_method == "weighted_mean_log_probabilities":
        if weight_col is None:
            raise ValueError("weighted_mean_log_probabilities aggregation requires `weight_col`.")
        if weight_col not in segment_prediction_table.columns:
            raise ValueError(f"{weight_col!r} is missing from segment_prediction_table.")
        weights_np = _to_numpy(segment_prediction_table[weight_col].to_numpy()).reshape(-1)

    aggregated = _aggregate_sequence_probabilities_from_arrays(
        segment_prediction_table[sequence_id_col].to_numpy(),
        logits=logits_np,
        probabilities=probabilities_np,
        label_ids=label_ids_np,
        aggregation_method=aggregation_method,
        weights=weights_np,
        top_k=top_k,
    )

    sequence_prediction_table = {
        sequence_id_col: aggregated["sequence_id"],
        "segment_count": aggregated["segment_count"],
    }

    if "label_id" in aggregated:
        sequence_prediction_table["label_id"] = aggregated["label_id"]

    sequence_prediction_table["predicted_label_id"] = aggregated["predicted_label_id"]

    for class_idx in range(aggregated["sequence_probabilities"].shape[1]):
        sequence_prediction_table[f"prob_class_{class_idx}"] = aggregated["sequence_probabilities"][
            :, class_idx
        ]

    return pd.DataFrame(sequence_prediction_table)


def build_sequence_classification_prediction_table(
    predictions: Any,
    segment_dataset: Any,
    label_ids: Optional[Any] = None,
    *,
    sequence_id_col: str = "sequence_id",
    label_col: str = "labels",
    aggregation_method: str = "mean_logits",
    weight_col: Optional[str] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Build a sequence-level prediction table from Trainer-style predictions.

    This is the recommended fast path for `trainer.predict(eval_dataset)` outputs.
    It aggregates directly from arrays and only builds a pandas DataFrame at the end.
    """
    aggregation_method = _validate_aggregation_method(aggregation_method)

    dataset_columns = set(_get_dataset_columns(segment_dataset))
    if sequence_id_col not in dataset_columns:
        raise ValueError(
            f"{sequence_id_col!r} is required in segment_dataset for sequence aggregation."
        )

    logits_np, label_ids_np = _extract_logits_and_label_ids(
        predictions,
        label_ids,
        allow_missing_labels=True,
    )

    num_segments = logits_np.shape[0]

    if label_ids_np is None:
        label_ids_np = _maybe_get_dataset_labels(segment_dataset, label_col, num_segments)

    sequence_df = _select_dataset_columns(segment_dataset, [sequence_id_col])
    if len(sequence_df) != num_segments:
        raise ValueError(
            "Prediction row count and segment_dataset row count must match: "
            f"{num_segments} != {len(sequence_df)}."
        )

    weights_np = None
    if aggregation_method == "weighted_mean_log_probabilities":
        if weight_col is None:
            raise ValueError("weighted_mean_log_probabilities aggregation requires `weight_col`.")
        if weight_col not in dataset_columns:
            raise ValueError(
                f"{weight_col!r} is required in segment_dataset for weighted aggregation."
            )
        weight_df = _select_dataset_columns(segment_dataset, [weight_col])
        if len(weight_df) != num_segments:
            raise ValueError(
                "Prediction row count and segment_dataset row count must match: "
                f"{num_segments} != {len(weight_df)}."
            )
        weights_np = _to_numpy(weight_df[weight_col].to_numpy()).reshape(-1)

    aggregated = _aggregate_sequence_probabilities_from_arrays(
        sequence_df[sequence_id_col].to_numpy(),
        logits=logits_np,
        label_ids=label_ids_np,
        aggregation_method=aggregation_method,
        weights=weights_np,
        top_k=top_k,
    )

    sequence_prediction_table = {
        sequence_id_col: aggregated["sequence_id"],
        "segment_count": aggregated["segment_count"],
    }

    if "label_id" in aggregated:
        sequence_prediction_table["label_id"] = aggregated["label_id"]

    sequence_prediction_table["predicted_label_id"] = aggregated["predicted_label_id"]

    for class_idx in range(aggregated["sequence_probabilities"].shape[1]):
        sequence_prediction_table[f"prob_class_{class_idx}"] = aggregated["sequence_probabilities"][
            :, class_idx
        ]

    return pd.DataFrame(sequence_prediction_table)


def evaluate_sequence_classification_predictions(
    predictions: Any,
    segment_dataset: Any,
    label_ids: Optional[Any] = None,
    *,
    sequence_id_col: str = "sequence_id",
    label_col: str = "labels",
    aggregation_method: str = "mean_logits",
    weight_col: Optional[str] = None,
    top_k: Optional[int] = 5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Evaluate sequence-level single-label classification predictions.

    Returns
    -------
    tuple[pandas.DataFrame, dict[str, float]]
        `sequence_prediction_table, metrics`
    """
    sequence_prediction_table = build_sequence_classification_prediction_table(
        predictions=predictions,
        segment_dataset=segment_dataset,
        label_ids=label_ids,
        sequence_id_col=sequence_id_col,
        label_col=label_col,
        aggregation_method=aggregation_method,
        weight_col=weight_col,
        top_k=top_k,
    )

    if "label_id" not in sequence_prediction_table.columns:
        raise ValueError(
            "Labels are required for sequence-level evaluation. Provide them via "
            "`predictions.label_ids`, the `label_ids` argument, or `segment_dataset`."
        )

    return evaluate_classification_prediction_table(sequence_prediction_table)


def build_sequence_classification_compute_metrics(
    segment_dataset: Any,
    *,
    sequence_id_col: str = "sequence_id",
    label_col: str = "labels",
    aggregation_method: str = "mean_logits",
    weight_col: Optional[str] = None,
    top_k: Optional[int] = None,
) -> Callable[[Any], Dict[str, float]]:
    """Build a Trainer-compatible `compute_metrics` for sequence-level evaluation.

    Notes
    -----
    - Pass the same dataset to this builder that you use in `trainer.evaluate(...)`
      or `trainer.predict(...)`.
    - `sequence_id` must be present in that dataset.
    """
    _validate_aggregation_method(aggregation_method)

    def compute_metrics_fn(
        eval_prediction: Any,
        compute_result: bool = True,
    ) -> Dict[str, float]:
        if not compute_result:
            return {}

        _, metrics = evaluate_sequence_classification_predictions(
            predictions=eval_prediction,
            segment_dataset=segment_dataset,
            sequence_id_col=sequence_id_col,
            label_col=label_col,
            aggregation_method=aggregation_method,
            weight_col=weight_col,
            top_k=top_k,
        )
        return metrics

    return compute_metrics_fn


def compute_metrics(
    eval_prediction: Any,
    compute_result: bool = True,
) -> Dict[str, float]:
    """Trainer-compatible metrics function for segment-level classification."""
    if not compute_result:
        return {}

    _, metrics = evaluate_segment_classification_predictions(eval_prediction)
    return metrics


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
    """Keep only the logits before Hugging Face caches predictions for metrics."""
    del labels

    while isinstance(logits, (tuple, list)):
        if len(logits) == 0:
            raise ValueError("Received an empty logits tuple/list.")
        logits = logits[0]

    return logits
