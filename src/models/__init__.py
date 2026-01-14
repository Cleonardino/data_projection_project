"""
models package initialization.

Provides unified access to all model implementations.
"""

from .base_model import BaseModel, TrainingHistory, EvaluationMetrics
from .model_xgboost import XGBoostModel
from .model_knn import KNNModel
from .model_random_forest import RandomForestModel
from .model_mlp import MLPModel
from .model_tab_transformer import TabTransformerModel
from .model_ft_transformer import FTTransformerModel
from .model_attention_mlp import AttentionMLPModel


# Registry of available models
MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "xgboost": XGBoostModel,
    "knn": KNNModel,
    "random_forest": RandomForestModel,
    "mlp": MLPModel,
    "tab_transformer": TabTransformerModel,
    "ft_transformer": FTTransformerModel,
    "attention_mlp": AttentionMLPModel,
}


def get_model(name: str, hyperparameters: dict | None = None) -> BaseModel:
    """
    Factory function to create model instances by name.

    Args:
        name: Model name (e.g., 'xgboost', 'mlp', 'tab_transformer').
        hyperparameters: Optional model hyperparameters.

    Returns:
        Instantiated model object.

    Raises:
        ValueError: If model name is not in registry.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {name}. Available: {available}")

    return MODEL_REGISTRY[name](hyperparameters)


__all__ = [
    "BaseModel",
    "TrainingHistory",
    "EvaluationMetrics",
    "XGBoostModel",
    "KNNModel",
    "RandomForestModel",
    "MLPModel",
    "TabTransformerModel",
    "FTTransformerModel",
    "AttentionMLPModel",
    "MODEL_REGISTRY",
    "get_model",
]
