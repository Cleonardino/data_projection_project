"""
model_attention_mlp.py - Attention-based MLP for Tabular Data

Implements a lightweight attention mechanism on top of MLP layers.
Uses self-attention to learn feature interactions before final classification.
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


class AttentionBlock(nn.Module):
    """
    Single attention block for feature interaction.

    Computes attention weights across features and produces
    weighted feature representations.

    Args:
        n_features: Number of input features.
        hidden_dim: Hidden dimension for attention.
        n_heads: Number of attention heads.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Project features to query, key, value
        self.q_proj = nn.Linear(1, hidden_dim)
        self.k_proj = nn.Linear(1, hidden_dim)
        self.v_proj = nn.Linear(1, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention across features.

        Args:
            x: Input of shape (batch, n_features).

        Returns:
            Output of shape (batch, n_features, hidden_dim).
        """
        batch_size, n_features = x.shape

        # Reshape: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)

        # Project to Q, K, V: (batch, n_features, hidden_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch, n_features, n_heads, head_dim) -> (batch, n_heads, n_features, head_dim)
        q = q.view(batch_size, n_features, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_features, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_features, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention: (batch, n_heads, n_features, n_features)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values: (batch, n_heads, n_features, head_dim)
        out = torch.matmul(attn, v)

        # Reshape back: (batch, n_features, hidden_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_features, -1)
        out = self.out_proj(out)

        return out


class AttentionMLPNetwork(nn.Module):
    """
    Attention-based MLP network.

    Applies attention to learn feature interactions, then uses
    MLP layers for classification.

    Args:
        n_features: Number of input features.
        n_classes: Number of output classes.
        hidden_dim: Hidden dimension (default: 64).
        n_heads: Number of attention heads (default: 4).
        mlp_layers: MLP hidden layer sizes.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        mlp_layers: list[int] = [128, 64],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features

        # Attention block
        self.attention = AttentionBlock(n_features, hidden_dim, n_heads, dropout)

        # Layer norm after attention
        self.norm = nn.LayerNorm(hidden_dim)

        # Flatten attention output and build MLP
        mlp_input_dim = n_features * hidden_dim

        layers: list[nn.Module] = []
        prev_dim = mlp_input_dim

        for hidden_size in mlp_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_size

        layers.append(nn.Linear(prev_dim, n_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features of shape (batch, n_features).

        Returns:
            Class logits of shape (batch, n_classes).
        """
        batch_size = x.size(0)

        # Apply attention: (batch, n_features, hidden_dim)
        attn_out = self.attention(x)
        attn_out = self.norm(attn_out)

        # Flatten: (batch, n_features * hidden_dim)
        flat = attn_out.view(batch_size, -1)

        # MLP classification
        return self.mlp(flat)


class AttentionMLPModel(BaseModel):
    """
    Attention-based MLP classifier wrapper.

    Hyperparameters:
        - hidden_dim: Attention hidden dimension (default: 64).
        - n_heads: Number of attention heads (default: 4).
        - mlp_layers: MLP layer sizes (default: [128, 64]).
        - dropout: Dropout rate (default: 0.2).
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
        self.name = "AttentionMLP"
        self.n_classes: int = 0
        self.n_features: int = 0
        self.device: torch.device = torch.device("cpu")

    def build(self, n_features: int, n_classes: int) -> None:
        """Build Attention MLP network."""
        self.n_features = n_features
        self.n_classes = n_classes

        defaults: dict[str, Any] = {
            "hidden_dim": 64,
            "n_heads": 4,
            "mlp_layers": [128, 64],
            "dropout": 0.2,
            "use_gpu": True,
        }
        params: dict[str, Any] = {**defaults, **self.hyperparameters}

        if params["use_gpu"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = AttentionMLPNetwork(
            n_features=n_features,
            n_classes=n_classes,
            hidden_dim=params["hidden_dim"],
            n_heads=params["n_heads"],
            mlp_layers=params["mlp_layers"],
            dropout=params["dropout"],
        ).to(self.device)

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
    ) -> TrainingHistory:
        """Train Attention MLP model."""
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

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        val_loader: DataLoader | None = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

        best_val_acc: float = 0.0
        best_state: dict[str, Any] = {}
        patience_counter: int = 0

        for epoch in range(params["epochs"]):
            self.model.train()
            train_loss: float = 0.0
            train_correct: int = 0
            train_total: int = 0

            # Batch loop with progress bar
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['epochs']}", unit="batch") as tepoch:
                for batch_X, batch_y in tepoch:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    # Update metrics
                    current_loss = loss.item()
                    train_loss += current_loss * len(batch_y)
                    _, predicted = outputs.max(1)
                    train_correct += predicted.eq(batch_y).sum().item()
                    train_total += len(batch_y)

                    # Update progress bar
                    tepoch.set_postfix(loss=f"{current_loss:.4f}")

            train_loss /= train_total
            train_acc: float = train_correct / train_total

            history.epochs.append(epoch + 1)
            history.train_loss.append(train_loss)
            history.train_accuracy.append(train_acc)

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
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            # Print epoch summary
            val_str = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}" if val_loader else ""
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}{val_str}")

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
