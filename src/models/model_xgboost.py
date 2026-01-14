"""
model_xgboost.py - XGBoost Classifier Model Wrapper

Wraps XGBoost gradient boosting classifier with consistent interface.
Supports all XGBoost hyperparameters via configuration.
"""

from __future__ import annotations

import time
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import xgboost as xgb

from .base_model import BaseModel, TrainingHistory


class XGBoostModel(BaseModel):
    """
    XGBoost gradient boosting classifier wrapper.
    
    Hyperparameters (common options):
        - n_estimators: Number of boosting rounds (default: 100).
        - max_depth: Maximum tree depth (default: 6).
        - learning_rate: Step size shrinkage (default: 0.1).
        - subsample: Subsample ratio of training instances (default: 1.0).
        - colsample_bytree: Subsample ratio of columns (default: 1.0).
        - min_child_weight: Minimum sum of instance weight in child (default: 1).
        - gamma: Minimum loss reduction for split (default: 0).
        - reg_alpha: L1 regularization (default: 0).
        - reg_lambda: L2 regularization (default: 1).
        - scale_pos_weight: Balance of positive/negative weights (default: 1).
        - use_gpu: Whether to use GPU acceleration (default: False).
    """
    
    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        """
        Initialize XGBoost model.
        
        Args:
            hyperparameters: XGBoost-specific hyperparameters.
        """
        super().__init__(hyperparameters)
        self.name = "XGBoost"
        self.n_classes: int = 0
        
    def build(self, n_features: int, n_classes: int) -> None:
        """
        Build XGBoost classifier with configured hyperparameters.
        
        Args:
            n_features: Number of input features (unused, auto-detected).
            n_classes: Number of output classes.
        """
        self.n_classes = n_classes
        
        # Default hyperparameters
        defaults: dict[str, Any] = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1,
            "gamma": 0,
            "reg_alpha": 0,
            "reg_lambda": 1,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        
        # Override with user hyperparameters
        params: dict[str, Any] = {**defaults, **self.hyperparameters}
        
        # Handle GPU setting
        use_gpu: bool = params.pop("use_gpu", False)
        if use_gpu:
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"
        
        # Set objective based on number of classes
        if n_classes == 2:
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
        else:
            params["objective"] = "multi:softprob"
            params["eval_metric"] = "mlogloss"
            params["num_class"] = n_classes
        
        self.model = xgb.XGBClassifier(**params)
        
    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> TrainingHistory:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (used for early stopping if provided).
            y_val: Validation labels.
        
        Returns:
            TrainingHistory with training metrics.
        """
        history = TrainingHistory()
        start_time: float = time.time()
        
        # Setup evaluation set for early stopping
        eval_set: list[tuple[NDArray[np.float64], NDArray[np.int64]]] | None = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # Fit model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        
        # Record training time
        history.training_time_seconds = time.time() - start_time
        
        # Get final accuracies
        train_pred: NDArray[np.int64] = self.model.predict(X_train)
        train_acc: float = float(np.mean(train_pred == y_train))
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
        """
        Predict class labels.
        
        Args:
            X: Input features.
        
        Returns:
            Predicted class labels.
        """
        return self.model.predict(X).astype(np.int64)
    
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict class probabilities.
        
        Args:
            X: Input features.
        
        Returns:
            Class probabilities.
        """
        return self.model.predict_proba(X).astype(np.float64)
    
    def save(self, path: Path) -> None:
        """
        Save model to file.
        
        Args:
            path: Path for model file.
        """
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "hyperparameters": self.hyperparameters,
                "n_classes": self.n_classes,
            }, f)
    
    def load(self, path: Path) -> None:
        """
        Load model from file.
        
        Args:
            path: Path to saved model.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.model = data["model"]
        self.hyperparameters = data["hyperparameters"]
        self.n_classes = data["n_classes"]
        self.is_fitted = True
    
    def get_feature_importance(self) -> NDArray[np.float64]:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importance values.
        """
        return self.model.feature_importances_.astype(np.float64)
