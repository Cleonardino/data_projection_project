"""
model_tab_transformer.py - TabTransformer Model for Tabular Data

Implements TabTransformer architecture that applies self-attention
to categorical feature embeddings while keeping numerical features
as-is, then concatenates for final classification.

Reference: Huang et al., "TabTransformer: Tabular Data Modeling Using
Contextual Embeddings" (2020)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base_model import BaseModel, TrainingHistory

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class TransformerBlock(nn.Module):
    """
    Single Transformer encoder block with multi-head attention.

    Args:
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
        d_ff: Feedforward dimension (default: 4 * d_model).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        d_ff: int | None = None,
    ):
        super().__init__()

        d_ff = d_ff or (4 * d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class TabTransformerNetwork(nn.Module):
    """
    TabTransformer architecture for tabular data.

    Treats each feature as a token, applies self-attention,
    then pools for classification.

    Args:
        n_features: Number of input features.
        n_classes: Number of output classes.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_model = d_model

        # Feature embedding: project each feature to d_model
        self.feature_embedding = nn.Linear(1, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * n_features),
            nn.Linear(d_model * n_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, n_features).

        Returns:
            Class logits of shape (batch, n_classes).
        """
        batch_size = x.size(0)

        # Reshape: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)

        # Embed each feature: (batch, n_features, d_model)
        x = self.feature_embedding(x)

        # Add positional encoding
        x = x + self.pos_encoding

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Flatten: (batch, n_features * d_model)
        x = x.view(batch_size, -1)

        # Classify
        return self.classifier(x)


class TabTransformerModel(BaseModel):
    """
    TabTransformer classifier wrapper.

    Hyperparameters:
        - d_model: Hidden dimension (default: 64).
        - n_heads: Number of attention heads (default: 4).
        - n_layers: Number of transformer layers (default: 2).
        - dropout: Dropout rate (default: 0.1).
        - epochs: Training epochs (default: 100).
        - batch_size: Batch size (default: 32).
        - learning_rate: Learning rate (default: 0.001).
        - optimizer: Optimizer type (default: 'adamw').
        - weight_decay: L2 regularization (default: 1e-4).
        - early_stopping: Enable early stopping (default: True).
        - patience: Early stopping patience (default: 10).
        - use_gpu: Use GPU if available (default: True).
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        super().__init__(hyperparameters)
        self.name = "TabTransformer"
        self.n_classes: int = 0
        self.n_features: int = 0
        self.device: torch.device = torch.device("cpu")

    def build(self, n_features: int, n_classes: int) -> None:
        """Build TabTransformer network."""
        self.n_features = n_features
        self.n_classes = n_classes

        defaults: dict[str, Any] = {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.1,
            "use_gpu": True,
        }
        params: dict[str, Any] = {**defaults, **self.hyperparameters}

        # Set device
        if params["use_gpu"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = TabTransformerNetwork(
            n_features=n_features,
            n_classes=n_classes,
            d_model=params["d_model"],
            n_heads=params["n_heads"],
            n_layers=params["n_layers"],
            dropout=params["dropout"],
        ).to(self.device)

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> TrainingHistory:
        """Train TabTransformer model."""
        history = TrainingHistory()
        start_time: float = time.time()

        defaults: dict[str, Any] = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "early_stopping": True,
            "patience": 10,
        }
        params: dict[str, Any] = {**defaults, **self.hyperparameters}

        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        val_loader: DataLoader | None = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

        # Training loop
        best_val_acc: float = 0.0
        best_state: dict[str, Any] = {}
        patience_counter: int = 0

        for epoch in tqdm(range(params["epochs"]), desc=f"Training {self.name}", unit="epoch"):
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

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss: float = 0.0
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
                val_acc: float = val_correct / val_total

                history.val_loss.append(val_loss)
                history.val_accuracy.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = self.model.state_dict().copy()
                    history.best_epoch = epoch + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if params["early_stopping"] and patience_counter >= params["patience"]:
                    break

            scheduler.step()

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
