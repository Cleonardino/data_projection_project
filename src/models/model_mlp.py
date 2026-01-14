"""
model_mlp.py - Multi-Layer Perceptron Classifier Wrapper

Wraps PyTorch MLP with consistent interface and training loop.
Supports configurable architecture, optimizers, and learning schedules.
"""

from __future__ import annotations

import time
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel, TrainingHistory


class MLPNetwork(nn.Module):
    """
    PyTorch MLP neural network architecture.
    
    Args:
        n_features: Input feature dimension.
        n_classes: Number of output classes.
        hidden_layers: List of hidden layer sizes.
        dropout: Dropout rate between layers.
        activation: Activation function ('relu', 'gelu', 'silu').
    """
    
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_layers: list[int] = [256, 128, 64],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        
        # Build layers
        layers: list[nn.Module] = []
        prev_size: int = n_features
        
        # Activation mapping
        activation_fn: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "tanh": nn.Tanh,
        }
        act_class = activation_fn.get(activation, nn.ReLU)
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_class())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, n_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        return self.network(x)


class MLPModel(BaseModel):
    """
    PyTorch MLP classifier wrapper.
    
    Hyperparameters:
        - hidden_layers: List of hidden layer sizes (default: [256, 128, 64]).
        - dropout: Dropout rate (default: 0.3).
        - activation: Activation function ('relu', 'gelu', 'silu').
        - epochs: Number of training epochs (default: 100).
        - batch_size: Training batch size (default: 32).
        - learning_rate: Initial learning rate (default: 0.001).
        - optimizer: Optimizer type ('adam', 'adamw', 'sgd').
        - weight_decay: L2 regularization (default: 1e-4).
        - early_stopping: Enable early stopping (default: True).
        - patience: Early stopping patience (default: 10).
        - scheduler: LR scheduler ('cosine', 'step', 'none').
        - use_gpu: Use GPU if available (default: True).
    """
    
    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        """
        Initialize MLP model.
        
        Args:
            hyperparameters: MLP-specific hyperparameters.
        """
        super().__init__(hyperparameters)
        self.name = "MLP"
        self.n_classes: int = 0
        self.n_features: int = 0
        self.device: torch.device = torch.device("cpu")
        
    def build(self, n_features: int, n_classes: int) -> None:
        """
        Build MLP network.
        
        Args:
            n_features: Number of input features.
            n_classes: Number of output classes.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        
        # Defaults
        defaults: dict[str, Any] = {
            "hidden_layers": [256, 128, 64],
            "dropout": 0.3,
            "activation": "relu",
            "use_gpu": True,
        }
        params: dict[str, Any] = {**defaults, **self.hyperparameters}
        
        # Set device
        use_gpu: bool = params.get("use_gpu", True)
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Build network
        self.model = MLPNetwork(
            n_features=n_features,
            n_classes=n_classes,
            hidden_layers=params["hidden_layers"],
            dropout=params["dropout"],
            activation=params["activation"],
        ).to(self.device)
        
    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> TrainingHistory:
        """
        Train MLP model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
        
        Returns:
            TrainingHistory with per-epoch metrics.
        """
        history = TrainingHistory()
        start_time: float = time.time()
        
        # Get hyperparameters
        defaults: dict[str, Any] = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "early_stopping": True,
            "patience": 10,
            "scheduler": "cosine",
        }
        params: dict[str, Any] = {**defaults, **self.hyperparameters}
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=params["batch_size"],
            shuffle=True,
        )
        
        val_loader: DataLoader | None = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        optimizer_map: dict[str, type] = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            "sgd": optim.SGD,
        }
        opt_class = optimizer_map.get(params["optimizer"], optim.Adam)
        optimizer = opt_class(
            self.model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        
        # Learning rate scheduler
        scheduler: Any = None
        if params["scheduler"] == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params["epochs"]
            )
        elif params["scheduler"] == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        
        # Training loop
        best_val_acc: float = 0.0
        best_state: dict[str, Any] = {}
        patience_counter: int = 0
        
        for epoch in range(params["epochs"]):
            # Training phase
            self.model.train()
            train_loss: float = 0.0
            train_correct: int = 0
            train_total: int = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_y)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(batch_y).sum().item()
                train_total += len(batch_y)
            
            train_loss /= train_total
            train_acc: float = train_correct / train_total
            
            history.epochs.append(epoch + 1)
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)
            
            # Validation phase
            val_acc: float = 0.0
            val_loss: float = 0.0
            if val_loader is not None:
                self.model.eval()
                val_correct: int = 0
                val_total: int = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item() * len(batch_y)
                        _, predicted = outputs.max(1)
                        val_correct += predicted.eq(batch_y).sum().item()
                        val_total += len(batch_y)
                
                val_loss /= val_total
                val_acc = val_correct / val_total
                
                history.val_loss.append(val_loss)
                history.val_accuracy.append(val_acc)
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = self.model.state_dict().copy()
                    history.best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if params["early_stopping"] and patience_counter >= params["patience"]:
                    break
            
            if scheduler is not None:
                scheduler.step()
        
        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        history.training_time_seconds = time.time() - start_time
        self.is_fitted = True
        
        return history
    
    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict class labels."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = outputs.max(1)
            return predicted.cpu().numpy().astype(np.int64)
    
    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy().astype(np.float64)
    
    def save(self, path: Path) -> None:
        """Save model to file."""
        torch.save({
            "state_dict": self.model.state_dict(),
            "hyperparameters": self.hyperparameters,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
        }, path)
    
    def load(self, path: Path) -> None:
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.hyperparameters = checkpoint["hyperparameters"]
        self.n_features = checkpoint["n_features"]
        self.n_classes = checkpoint["n_classes"]
        
        self.build(self.n_features, self.n_classes)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = True
