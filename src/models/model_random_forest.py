"""
model_random_forest.py - Random Forest Classifier Wrapper

Wraps scikit-learn Random Forest with consistent interface.
"""

from __future__ import annotations

import time
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from .base_model import BaseModel, TrainingHistory


class RandomForestModel(BaseModel):
    """
    Random Forest classifier wrapper.

    Hyperparameters:
        - n_estimators: Number of trees (default: 100).
        - max_depth: Maximum tree depth (default: None = unlimited).
        - min_samples_split: Minimum samples to split node (default: 2).
        - min_samples_leaf: Minimum samples in leaf (default: 1).
        - max_features: Features to consider for split ('sqrt', 'log2', int, float).
        - bootstrap: Whether to use bootstrap sampling (default: True).
        - class_weight: Class weights ('balanced', 'balanced_subsample', dict).
        - criterion: Split criterion ('gini' or 'entropy').
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        """
        Initialize Random Forest model.

        Args:
            hyperparameters: RF-specific hyperparameters.
        """
        super().__init__(hyperparameters)
        self.name = "RandomForest"
        self.n_classes: int = 0

    def build(self, n_features: int, n_classes: int) -> None:
        """
        Build Random Forest classifier.

        Args:
            n_features: Number of input features.
            n_classes: Number of output classes.
        """
        self.n_classes = n_classes

        # Default hyperparameters
        defaults: dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True,
            "class_weight": None,
            "criterion": "gini",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": 0,
        }

        # Override with user hyperparameters
        params: dict[str, Any] = {**defaults, **self.hyperparameters}

        self.model = RandomForestClassifier(**params)

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
        class_weights: dict[int, float] | None = None,
    ) -> TrainingHistory:
        """
        Train Random Forest model.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            class_weights: Optional dictionary of class weights.

        Returns:
            TrainingHistory with metrics.
        """
        history = TrainingHistory()
        start_time: float = time.time()

        # Update class weights if provided
        if class_weights is not None:
            self.model.set_params(class_weight=class_weights)

        # Fit model
        self.model.fit(X_train, y_train)

        # Record training time
        history.training_time_seconds = time.time() - start_time

        # Compute accuracies
        train_pred: NDArray[np.int64] = self._predict_batched(X_train)
        train_acc: float = float(np.mean(train_pred == y_train))
        history.train_accuracy = [train_acc]

        if X_val is not None and y_val is not None:
            val_pred: NDArray[np.int64] = self._predict_batched(X_val)
            val_acc: float = float(np.mean(val_pred == y_val))
            history.val_accuracy = [val_acc]

        history.epochs = [1]
        history.best_epoch = 1
        self.is_fitted = True

        return history

    def _predict_batched(self, X: NDArray[np.float64], batch_size: int = 10000) -> NDArray[np.int64]:
        """
        Helper to predict in batches to manage memory and show progress.
        """
        n_samples = X.shape[0]
        predictions = []

        # Only use progress bar if enough samples
        disable_tqdm = n_samples < batch_size

        with tqdm(total=n_samples, desc="  Predicting", leave=False, disable=disable_tqdm) as pbar:
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_pred = self.model.predict(X[i:batch_end])
                predictions.append(batch_pred)
                pbar.update(batch_end - i)

        return np.concatenate(predictions).astype(np.int64)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels."""
        return self._predict_batched(X)

    def predict_proba(self, X: NDArray[np.float64], batch_size: int = 10000) -> NDArray[np.float64]:
        """Predict class probabilities."""
        n_samples = X.shape[0]
        probabilities = []

        # Only use progress bar if enough samples
        disable_tqdm = n_samples < batch_size

        with tqdm(total=n_samples, desc="  Predicting Proba", leave=False, disable=disable_tqdm) as pbar:
            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                batch_proba = self.model.predict_proba(X[i:batch_end])
                probabilities.append(batch_proba)
                pbar.update(batch_end - i)

        return np.concatenate(probabilities).astype(np.float64)

    def save(self, path: Path) -> None:
        """Save model to file."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "hyperparameters": self.hyperparameters,
                "n_classes": self.n_classes,
            }, f)

    def load(self, path: Path) -> None:
        """Load model from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.hyperparameters = data["hyperparameters"]
        self.n_classes = data["n_classes"]
        self.is_fitted = True

    def get_feature_importance(self) -> NDArray[np.float64]:
        """Get feature importance scores."""
        return self.model.feature_importances_.astype(np.float64)
