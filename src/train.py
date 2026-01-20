#!/usr/bin/env python3
"""
train.py - Model Training Script

Trains ML models using YAML configuration files.
Creates experiment folders with timestamp, saves training history,
model weights, and sample-by-sample error analysis.

Usage:
    python src/train.py --config configs/physical_small.yaml
    python src/train.py --config configs/physical_medium.yaml --model mlp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lib_data import load_ml_ready_data, MLDataset
from models import get_model, BaseModel, TrainingHistory, EvaluationMetrics


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)
    return config


def create_experiment_folder(
    base_dir: Path,
    config: dict[str, Any],
    model_name: str,
) -> Path:
    """
    Create experiment folder with timestamp.

    Folder name format: YYYY-MM-DD_HH-MM-SS_{experiment_name}_{model_name}

    Args:
        base_dir: Base experiments directory.
        config: Configuration dictionary.
        model_name: Name of the model being trained.

    Returns:
        Path to created experiment folder.
    """
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name: str = config.get("experiment", {}).get("name", "experiment")

    folder_name: str = f"{timestamp}_{exp_name}_{model_name}"
    exp_path: Path = base_dir / folder_name

    exp_path.mkdir(parents=True, exist_ok=True)

    return exp_path


def save_config_copy(exp_path: Path, config: dict[str, Any]) -> None:
    """Save a copy of the configuration to the experiment folder."""
    with open(exp_path / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def save_sample_errors(
    exp_path: Path,
    model: BaseModel,
    dataset: MLDataset,
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Save sample-by-sample error analysis.

    Args:
        exp_path: Experiment folder path.
        model: Trained model.
        dataset: MLDataset for label decoding.
        split_name: Name of the split ('train', 'val', 'test').
        X: Features.
        y: True labels.
    """
    y_true, y_pred, is_correct = model.get_sample_errors(X, y)

    # Decode labels
    true_labels: list[str] = dataset.label_encoder.inverse_transform(y_true).tolist()
    pred_labels: list[str] = dataset.label_encoder.inverse_transform(y_pred).tolist()

    # Create DataFrame
    df = pd.DataFrame({
        "sample_idx": range(len(y_true)),
        "true_label_encoded": y_true,
        "pred_label_encoded": y_pred,
        "true_label": true_labels,
        "pred_label": pred_labels,
        "is_correct": is_correct,
    })

    df.to_csv(exp_path / f"{split_name}_errors.csv", index=False)


def save_metrics(
    exp_path: Path,
    metrics_dict: dict[str, EvaluationMetrics],
    history: TrainingHistory,
) -> None:
    """
    Save final metrics summary to JSON.

    Args:
        exp_path: Experiment folder path.
        metrics_dict: Dictionary of split_name -> metrics.
        history: Training history.
    """
    summary: dict[str, Any] = {
        "training": {
            "training_time_seconds": history.training_time_seconds,
            "best_epoch": history.best_epoch,
            "total_epochs": len(history.epochs),
        },
        "metrics": {},
    }

    for split_name, metrics in metrics_dict.items():
        summary["metrics"][split_name] = metrics.to_dict()

    with open(exp_path / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


def print_metrics(split_name: str, metrics: EvaluationMetrics) -> None:
    """Print metrics summary to console."""
    print(f"\n{split_name.upper()} Metrics:")
    print(f"  Accuracy:          {metrics.accuracy:.4f}")
    print(f"  Balanced Accuracy: {metrics.balanced_accuracy:.4f}")
    print(f"  F1 (macro):        {metrics.f1_macro:.4f}")
    print(f"  F1 (weighted):     {metrics.f1_weighted:.4f}")
    print(f"  MCC:               {metrics.mcc:.4f}")


def train(
    config_path: Path,
    model_override: str | None = None,
    experiments_dir: Path | None = None,
    epochs_override: int | None = None,
    balancing_override: str | None = None,
) -> Path:
    """
    Main training function.

    Args:
        config_path: Path to YAML configuration file.
        model_override: Override model name from command line.
        experiments_dir: Custom experiments directory.
        epochs_override: Override training epochs.
        balancing_override: Override balancing strategy.

    Returns:
        Path to experiment folder.
    """
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config: dict[str, Any] = load_config(config_path)

    # Extract settings
    data_config: dict[str, Any] = config.get("data", {})
    model_config: dict[str, Any] = config.get("model", {})
    training_config: dict[str, Any] = config.get("training", {})
    exp_config: dict[str, Any] = config.get("experiment", {})

    # Apply overrides
    if epochs_override is not None:
        print(f"Overriding epochs: {epochs_override}")
        training_config["epochs"] = epochs_override
    
    if balancing_override is not None:
        print(f"Overriding balancing strategy: {balancing_override}")
        data_config["balancing"] = balancing_override

    # Model selection (command line override takes precedence)
    model_name: str = model_override or model_config.get("name", "xgboost")

    # Create experiment folder
    if experiments_dir is None:
        experiments_dir = Path(__file__).parent.parent / "experiments"

    exp_path: Path = create_experiment_folder(experiments_dir, config, model_name)
    print(f"Experiment folder: {exp_path}")

    # Save config copy
    save_config_copy(exp_path, config)

    # Load data
    print("\nLoading data...")
    dataset: MLDataset = load_ml_ready_data(
        dataset_type=data_config.get("dataset_type", "physical"),
        train_ratio=data_config.get("train_ratio", 0.7),
        val_ratio=data_config.get("val_ratio", 0.15),
        test_ratio=data_config.get("test_ratio", 0.15),
        balancing=data_config.get("balancing", "none"),
        normalize=data_config.get("normalize", True),
        n_samples=data_config.get("n_samples"),
        random_state=data_config.get("seed", 42),
        noise_std=data_config.get("augmentation_noise_std", 0.1),
        datasets=data_config.get("datasets"),
        nrows=data_config.get("nrows_per_file"),
    )

    print(f"  Train samples: {len(dataset.X_train)}")
    print(f"  Val samples:   {len(dataset.X_val)}")
    print(f"  Test samples:  {len(dataset.X_test)}")
    print(f"  Features:      {dataset.X_train.shape[1]}")
    print(f"  Classes:       {dataset.n_classes} ({dataset.class_names})")

    # Merge training config into hyperparameters for neural nets
    hyperparameters: dict[str, Any] = model_config.get("hyperparameters", {}).copy()

    # For neural network models, add training params
    if model_name in ["mlp", "tab_transformer", "ft_transformer", "attention_mlp"]:
        hyperparameters.setdefault("epochs", training_config.get("epochs", 100))
        hyperparameters.setdefault("batch_size", training_config.get("batch_size", 32))
        hyperparameters.setdefault("learning_rate", training_config.get("learning_rate", 0.001))
        hyperparameters.setdefault("optimizer", training_config.get("optimizer", "adamw"))
        hyperparameters.setdefault("weight_decay", training_config.get("weight_decay", 0.0001))
        hyperparameters.setdefault("early_stopping", training_config.get("early_stopping", True))
        hyperparameters.setdefault("patience", training_config.get("patience", 10))
        hyperparameters.setdefault("use_gpu", training_config.get("use_gpu", True))

    # Create and build model
    print(f"\nBuilding model: {model_name}")
    model: BaseModel = get_model(model_name, hyperparameters)
    model.build(n_features=dataset.X_train.shape[1], n_classes=dataset.n_classes)

    # Calculate class weights if requested
    class_weights: dict[int, float] | None = None
    balancing_strategy = data_config.get("balancing", "none")
    
    if balancing_strategy == "class_weights":
        from lib_data import BalancingStrategy
        print("\nComputing class weights for imbalanced data...")
        class_weights = BalancingStrategy.get_class_weights(dataset.y_train)
        print(f"  Weights: {class_weights}")

    # Train model
    print("\nTraining...")
    history: TrainingHistory = model.fit(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        class_weights=class_weights,
    )

    print(f"Training completed in {history.training_time_seconds:.2f} seconds")
    print(f"Best epoch: {history.best_epoch}")

    # Save training history
    if exp_config.get("save_history", True):
        history.save(exp_path / "training_history.json")

        # Also save as CSV for easy plotting
        history_df = pd.DataFrame({
            "epoch": history.epochs,
            "train_loss": history.train_loss if history.train_loss else [None] * len(history.epochs),
            "val_loss": history.val_loss if history.val_loss else [None] * len(history.epochs),
            "train_accuracy": history.train_accuracy,
            "val_accuracy": history.val_accuracy if history.val_accuracy else [None] * len(history.epochs),
        })
        history_df.to_csv(exp_path / "training_history.csv", index=False)

    # Save model
    if exp_config.get("save_model", True):
        model_ext: str = ".pt" if model_name in ["mlp", "tab_transformer", "ft_transformer", "attention_mlp"] else ".pkl"
        model.save(exp_path / f"best_model{model_ext}")

    # Evaluate on all splits
    print("\nEvaluating...")
    metrics_dict: dict[str, EvaluationMetrics] = {}

    # Optimization: Subsample training evaluation for large datasets
    if len(dataset.X_train) > 50000:
        print(f"  Subsampling training evaluation ({len(dataset.X_train)} -> 50000) to save time/memory")
        rng = np.random.RandomState(42)
        indices = rng.choice(len(dataset.X_train), 50000, replace=False)
        X_train_eval = dataset.X_train[indices]
        y_train_eval = dataset.y_train[indices]
    else:
        X_train_eval = dataset.X_train
        y_train_eval = dataset.y_train

    train_metrics: EvaluationMetrics = model.evaluate(X_train_eval, y_train_eval)
    metrics_dict["train"] = train_metrics
    print_metrics("train", train_metrics)

    val_metrics: EvaluationMetrics = model.evaluate(dataset.X_val, dataset.y_val)
    metrics_dict["val"] = val_metrics
    print_metrics("val", val_metrics)

    test_metrics: EvaluationMetrics = model.evaluate(dataset.X_test, dataset.y_test)
    metrics_dict["test"] = test_metrics
    print_metrics("test", test_metrics)

    # Save sample-by-sample errors
    if exp_config.get("save_predictions", True):
        print("\nSaving sample-by-sample predictions...")
        
        # Skip training set prediction saving if too large
        if len(dataset.X_train) <= 50000:
            save_sample_errors(exp_path, model, dataset, "train", dataset.X_train, dataset.y_train)
        else:
            print(f"  Skipping training set predictions (size {len(dataset.X_train)} > 50000)")
            
        save_sample_errors(exp_path, model, dataset, "val", dataset.X_val, dataset.y_val)
        save_sample_errors(exp_path, model, dataset, "test", dataset.X_test, dataset.y_test)

    # Save metrics summary
    save_metrics(exp_path, metrics_dict, history)

    print(f"\n✅ Experiment saved to: {exp_path}")

    return exp_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ML models with YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/train.py --config configs/physical_small.yaml
    python src/train.py --config configs/physical_medium.yaml --model mlp
    python src/train.py --config configs/physical_large.yaml --model ft_transformer
        """,
    )

    parser.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        choices=[
            "xgboost", "knn", "random_forest", "mlp",
            "tab_transformer", "ft_transformer", "attention_mlp",
        ],
        help="Override model name from config",
    )

    parser.add_argument(
        "--experiments-dir", "-e",
        type=Path,
        default=None,
        help="Custom experiments output directory",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )

    parser.add_argument(
        "--balancing",
        type=str,
        default=None,
        help="Override balancing strategy (e.g., 'class_weights')",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    train(
        config_path=args.config,
        model_override=args.model,
        experiments_dir=args.experiments_dir,
        epochs_override=args.epochs,
        balancing_override=args.balancing,
    )


if __name__ == "__main__":
    main()
