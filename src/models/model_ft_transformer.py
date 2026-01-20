"""
model_ft_transformer.py - FT-Transformer (Feature Tokenizer + Transformer)

Implements FT-Transformer architecture for tabular data that tokenizes
all features (numerical and categorical) using learned embeddings,
then applies a Transformer encoder for classification.

Reference: Gorishniy et al., "Revisiting Deep Learning Models for
Tabular Data" (2021)
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


class FeatureTokenizer(nn.Module):
    """
    Feature tokenizer that converts each numerical feature
    to a learned embedding vector.

    Args:
        n_features: Number of input features.
        d_token: Dimension of each token embedding.
    """

    def __init__(self, n_features: int, d_token: int):
        super().__init__()

        # Each feature gets its own embedding layer
        self.weights = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.biases = nn.Parameter(torch.zeros(n_features, d_token))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize features.

        Args:
            x: Input of shape (batch, n_features).

        Returns:
            Tokens of shape (batch, n_features, d_token).
        """
        # x: (batch, n_features) -> (batch, n_features, 1)
        x = x.unsqueeze(-1)

        # Linear per feature: (batch, n_features, d_token)
        tokens = x * self.weights + self.biases

        return tokens


class FTTransformerNetwork(nn.Module):
    """
    FT-Transformer architecture.

    Tokenizes features, prepends a [CLS] token, applies Transformer,
    and uses [CLS] representation for classification.

    Args:
        n_features: Number of input features.
        n_classes: Number of output classes.
        d_token: Token embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate.
        d_ff_mult: Feedforward dimension multiplier.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_token: int = 96,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        d_ff_mult: int = 4,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_token = d_token

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(n_features, d_token)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * d_ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final layer norm
        self.norm = nn.LayerNorm(d_token)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes),
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

        # Tokenize features: (batch, n_features, d_token)
        tokens = self.tokenizer(x)

        # Prepend [CLS] token: (batch, 1 + n_features, d_token)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Apply transformer
        tokens = self.transformer(tokens)

        # Extract [CLS] representation
        cls_output = tokens[:, 0]
        cls_output = self.norm(cls_output)

        # Classify
        return self.classifier(cls_output)


class FTTransformerModel(BaseModel):
    """
    FT-Transformer classifier wrapper.

    Hyperparameters:
        - d_token: Token dimension (default: 96).
        - n_heads: Number of attention heads (default: 8).
        - n_layers: Number of transformer layers (default: 3).
        - dropout: Dropout rate (default: 0.1).
        - d_ff_mult: FF dimension multiplier (default: 4).
        - epochs: Training epochs (default: 100).
        - batch_size: Batch size (default: 32).
        - learning_rate: Learning rate (default: 1e-4).
        - optimizer: Optimizer type (default: 'adamw').
        - weight_decay: L2 regularization (default: 1e-5).
        - early_stopping: Enable early stopping (default: True).
        - patience: Early stopping patience (default: 15).
        - use_gpu: Use GPU if available (default: True).
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        super().__init__(hyperparameters)
        self.name = "FT-Transformer"
        self.n_classes: int = 0
        self.n_features: int = 0
        self.device: torch.device = torch.device("cpu")

    def build(self, n_features: int, n_classes: int) -> None:
        """Build FT-Transformer network."""
        self.n_features = n_features
        self.n_classes = n_classes

        defaults: dict[str, Any] = {
            "d_token": 96,
            "n_heads": 8,
            "n_layers": 3,
            "dropout": 0.1,
            "d_ff_mult": 4,
            "use_gpu": True,
        }
        params: dict[str, Any] = {**defaults, **self.hyperparameters}

        if params["use_gpu"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = FTTransformerNetwork(
            n_features=n_features,
            n_classes=n_classes,
            d_token=params["d_token"],
            n_heads=params["n_heads"],
            n_layers=params["n_layers"],
            dropout=params["dropout"],
            d_ff_mult=params["d_ff_mult"],
        ).to(self.device)

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.int64],
        X_val: NDArray[np.float64] | None = None,
        y_val: NDArray[np.int64] | None = None,
        class_weights: dict[int, float] | None = None,
    ) -> TrainingHistory:
        """Train FT-Transformer model."""
        history = TrainingHistory()
        start_time: float = time.time()

        defaults: dict[str, Any] = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "optimizer": "adamw",
            "weight_decay": 1e-5,
            "early_stopping": True,
            "patience": 15,
        }
        params: dict[str, Any] = {**defaults, **self.hyperparameters}

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        val_loader: DataLoader | None = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

        # Loss with optional class weights
        weight_tensor: torch.Tensor | None = None
        if class_weights is not None:
            sorted_weights = [class_weights[i] for i in range(len(class_weights))]
            weight_tensor = torch.FloatTensor(sorted_weights).to(self.device)
            print(f"  Using class weights: {class_weights}")

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
        
        # Use batching for inference to avoid OOM
        batch_size = self.hyperparameters.get("batch_size", 32) * 4
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self.device)
                outputs = self.model(X_batch)
                _, predicted = outputs.max(1)
                predictions.append(predicted.cpu().numpy())
                
        return np.concatenate(predictions).astype(np.int64)

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict class probabilities."""
        self.model.eval()
        
        # Use batching for inference to avoid OOM
        batch_size = self.hyperparameters.get("batch_size", 32) * 4
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        probabilities = []
        
        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(self.device)
                outputs = self.model(X_batch)
                probs = torch.softmax(outputs, dim=1)
                probabilities.append(probs.cpu().numpy())
                
        return np.concatenate(probabilities, axis=0).astype(np.float64)

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
