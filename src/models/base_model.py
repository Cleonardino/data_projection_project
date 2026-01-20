"""
base_model.py - Abstract Base Class for ML Models

Defines the common interface that all model wrappers must implement.
Provides consistent API for training, prediction, and evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef,
)


@dataclass
class TrainingHistory:
    """
    Container for training history and metrics.

    Stores per-epoch metrics for neural network models and
    final metrics for traditional ML models.

    Attributes:
        epochs: List of epoch numbers.
        train_loss: Training loss per epoch.
        val_loss: Validation loss per epoch.
        train_accuracy: Training accuracy per epoch.
        val_accuracy: Validation accuracy per epoch.
        best_epoch: Epoch with best validation metric.
        training_time_seconds: Total training time.
    """
    epochs: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    best_epoch: int = 0
    training_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert history to dictionary for JSON serialization."""
        return {
            "epochs": self.epochs,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "best_epoch": self.best_epoch,
            "training_time_seconds": self.training_time_seconds,
        }

    def save(self, path: Path) -> None:
        """Save history to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class EvaluationMetrics:
    """
    Container for model evaluation metrics.

    Attributes:
        accuracy: Overall accuracy.
        balanced_accuracy: Balanced accuracy (for imbalanced classes).
        precision_macro: Macro-averaged precision.
        recall_macro: Macro-averaged recall.
        f1_macro: Macro-averaged F1 score.
        precision_weighted: Weighted precision.
        recall_weighted: Weighted recall.
        f1_weighted: Weighted F1 score.
        mcc: Matthews Correlation Coefficient.
        confusion_matrix: Confusion matrix as 2D array.
        per_class_precision: Precision per class.
        per_class_recall: Recall per class.
        per_class_f1: F1 score per class.
    """
    accuracy: float
    balanced_accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    mcc: float
    confusion_matrix: NDArray[np.int64]
    per_class_precision: NDArray[np.float64]
    per_class_recall: NDArray[np.float64]
    per_class_f1: NDArray[np.float64]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "precision_weighted": self.precision_weighted,
            "recall_weighted": self.recall_weighted,
            "f1_weighted": self.f1_weighted,
            "mcc": self.mcc,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_precision": self.per_class_precision.tolist(),
            "per_class_recall": self.per_class_recall.tolist(),
            "per_class_f1": self.per_class_f1.tolist(),
        }

    def save(self, path: Path) -> None:
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BaseModel(ABC):
    """
    Abstract base class for all ML model wrappers.

    Defines the common interface for training, prediction, evaluation,
    and model persistence. All model implementations must inherit from
    this class and implement the abstract methods.

    Attributes:
        name: Human-readable model name.
        model: The underlying model instance.
        is_fitted: Whether the model has been trained.
        hyperparameters: Dictionary of model hyperparameters.
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        """
        Initialize the base model.

        Args:
            hyperparameters: Dictionary of model-specific hyperparameters.
        """
        self.name: str = self.__class__.__name__
        self.model: Any = None
        self.is_fitted: bool = False
        self.hyperparameters: dict[str, Any] = hyperparameters or {}

    @abstractmethod
    def build(self, n_features: int, n_classes: int) -> None:
        """
        Build/initialize the model architecture.

        Args:
            n_features: Number of input features.
            n_classes: Number of output classes.
        """
        pass

    @abstractmethod
    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
        class_weights: dict[int, float] | None = None,
    ) -> TrainingHistory:
        """
        Train the model on the provided data.

        Args:
            X_train: Training features of shape (n_samples, n_features).
            y_train: Training labels of shape (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            TrainingHistory object with training metrics.
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Make predictions on input data.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Predicted class labels of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict class probabilities.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Class probabilities of shape (n_samples, n_classes).
        """
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model file.
        """
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model file.
        """
        pass

    def evaluate(
        self,
        X: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> EvaluationMetrics:
        """
        Evaluate the model on test data.

        Computes comprehensive metrics including accuracy, precision,
        recall, F1, balanced accuracy, and MCC.

        Args:
            X: Test features of shape (n_samples, n_features).
            y_true: True labels of shape (n_samples,).

        Returns:
            EvaluationMetrics object with all computed metrics.
        """
        y_pred: NDArray[np.int64] = self.predict(X)

        # Compute metrics
        accuracy: float = accuracy_score(y_true, y_pred)
        balanced_acc: float = balanced_accuracy_score(y_true, y_pred)

        precision_macro: float = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro: float = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro: float = f1_score(y_true, y_pred, average="macro", zero_division=0)

        precision_weighted: float = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall_weighted: float = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_weighted: float = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        mcc: float = matthews_corrcoef(y_true, y_pred)
        conf_matrix: NDArray[np.int64] = confusion_matrix(y_true, y_pred)

        # Per-class metrics
        per_class_precision: NDArray[np.float64] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        per_class_recall: NDArray[np.float64] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        per_class_f1: NDArray[np.float64] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )

        return EvaluationMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            precision_macro=precision_macro,
            recall_macro=recall_macro,
            f1_macro=f1_macro,
            precision_weighted=precision_weighted,
            recall_weighted=recall_weighted,
            f1_weighted=f1_weighted,
            mcc=mcc,
            confusion_matrix=conf_matrix,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
        )

    def get_sample_errors(
        self,
        X: NDArray[np.float64],
        y_true: NDArray[np.int64],
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
        """
        Get sample-by-sample prediction results.

        Args:
            X: Features of shape (n_samples, n_features).
            y_true: True labels of shape (n_samples,).

        Returns:
            Tuple of (y_true, y_pred, is_correct) arrays.
        """
        y_pred: NDArray[np.int64] = self.predict(X)
        is_correct: NDArray[np.bool_] = y_true == y_pred

        return y_true, y_pred, is_correct
