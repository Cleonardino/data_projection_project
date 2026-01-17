"""
model_knn.py - K-Nearest Neighbors Classifier Wrapper

Wraps scikit-learn KNN classifier with consistent interface.
"""

from __future__ import annotations

import time
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import KNeighborsClassifier

from .base_model import BaseModel, TrainingHistory


class KNNModel(BaseModel):
    """
    K-Nearest Neighbors classifier wrapper.

    Hyperparameters:
        - n_neighbors: Number of neighbors (default: 5).
        - weights: Weight function ('uniform' or 'distance', default: 'uniform').
        - algorithm: Algorithm ('auto', 'ball_tree', 'kd_tree', 'brute').
        - leaf_size: Leaf size for tree algorithms (default: 30).
        - p: Power parameter for Minkowski metric (default: 2 = Euclidean).
        - metric: Distance metric (default: 'minkowski').
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        """
        Initialize KNN model.

        Args:
            hyperparameters: KNN-specific hyperparameters.
        """
        super().__init__(hyperparameters)
        self.name = "KNN"
        self.n_classes: int = 0

    def build(self, n_features: int, n_classes: int) -> None:
        """
        Build KNN classifier.

        Args:
            n_features: Number of input features.
            n_classes: Number of output classes.
        """
        self.n_classes = n_classes

        # Default hyperparameters
        defaults: dict[str, Any] = {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
            "n_jobs": -1,
        }

        # Override with user hyperparameters
        params: dict[str, Any] = {**defaults, **self.hyperparameters}

        self.model = KNeighborsClassifier(**params)

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> TrainingHistory:
        """
        Train KNN model (stores training data).

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (for accuracy computation).
            y_val: Validation labels.

        Returns:
            TrainingHistory with metrics.
        """
        history = TrainingHistory()
        start_time: float = time.time()

        # Fit model
        self.model.fit(X_train, y_train)

        # Record training time
        history.training_time_seconds = time.time() - start_time

        # Compute accuracies
        # Optimization: Subsample for large datasets to avoid O(N^2) complexity/timeout
        eval_indices = np.arange(len(X_train))
        if len(X_train) > 50000:
            print(f"  Note: Subsampling training set for KNN accuracy ({len(X_train)} -> 50000)")
            rng = np.random.RandomState(42)
            eval_indices = rng.choice(eval_indices, 50000, replace=False)
            
        train_pred: NDArray[np.int64] = self.model.predict(X_train[eval_indices])
        train_acc: float = float(np.mean(train_pred == y_train[eval_indices]))
        history.train_accuracy = [train_acc]

        if X_val is not None and y_val is not None:
            val_pred: NDArray[np.int64] = self.model.predict(X_val)
            val_acc: float = float(np.mean(val_pred == y_val))
            history.val_accuracy = [val_acc]

        history.epochs = [1]
        history.best_epoch = 1
        self.is_fitted = True

        return history

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels."""
        return self.model.predict(X).astype(np.int64)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        return self.model.predict_proba(X).astype(np.float64)

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
