"""Evaluation utilities for trained ECG200 models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Main classification metrics for a model."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray


def evaluate_classifier(
    y_true_labels: np.ndarray,
    y_pred_probabilities: np.ndarray,
) -> ClassificationMetrics:
    """Compute classification metrics from predicted probabilities."""

    y_pred_labels = np.argmax(y_pred_probabilities, axis=1)

    return ClassificationMetrics(
        accuracy=accuracy_score(y_true_labels, y_pred_labels),
        precision=precision_score(
            y_true_labels,
            y_pred_labels,
            average="macro",
            zero_division=0,
        ),
        recall=recall_score(
            y_true_labels,
            y_pred_labels,
            average="macro",
            zero_division=0,
        ),
        f1=f1_score(
            y_true_labels,
            y_pred_labels,
            average="macro",
            zero_division=0,
        ),
        confusion_matrix=confusion_matrix(y_true_labels, y_pred_labels),
    )
