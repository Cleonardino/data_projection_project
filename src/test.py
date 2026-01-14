#!/usr/bin/env python3
"""
test.py - Model Testing Script

Tests trained models on data and generates comprehensive metrics reports.

Usage:
    python src/test.py --experiment experiments/2024-01-14_19-30-00_physical_small_xgboost
    python src/test.py --experiment experiments/latest --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lib_data import load_ml_ready_data, MLDataset
from models import get_model, BaseModel, EvaluationMetrics


def find_model_file(exp_path: Path) -> tuple[Path, str]:
    """
    Find the model file in experiment folder and determine model type.

    Args:
        exp_path: Experiment folder path.

    Returns:
        Tuple of (model_path, model_name).
    """
    # Check for PyTorch model
    pt_files: list[Path] = list(exp_path.glob("*.pt"))
    if pt_files:
        model_path = pt_files[0]
        # Infer model name from experiment folder name
        folder_name = exp_path.name
        for name in ["ft_transformer", "tab_transformer", "attention_mlp", "mlp"]:
            if name in folder_name:
                return model_path, name
        return model_path, "mlp"  # Default to MLP

    # Check for pickle model
    pkl_files: list[Path] = list(exp_path.glob("*.pkl"))
    if pkl_files:
        model_path = pkl_files[0]
        folder_name = exp_path.name
        for name in ["xgboost", "random_forest", "knn"]:
            if name in folder_name:
                return model_path, name
        return model_path, "xgboost"  # Default

    raise FileNotFoundError(f"No model file found in {exp_path}")


def load_experiment_config(exp_path: Path) -> dict[str, Any]:
    """Load configuration from experiment folder."""
    config_path: Path = exp_path / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    return config


def print_detailed_metrics(metrics: EvaluationMetrics, class_names: list[str]) -> None:
    """
    Print detailed metrics report.

    Args:
        metrics: Evaluation metrics.
        class_names: List of class names.
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Overall Metrics ---")
    print(f"Accuracy:              {metrics.accuracy:.4f}")
    print(f"Balanced Accuracy:     {metrics.balanced_accuracy:.4f}")
    print(f"Matthews Corr. Coef.:  {metrics.mcc:.4f}")

    print("\n--- Macro Averaged ---")
    print(f"Precision:             {metrics.precision_macro:.4f}")
    print(f"Recall:                {metrics.recall_macro:.4f}")
    print(f"F1 Score:              {metrics.f1_macro:.4f}")

    print("\n--- Weighted Averaged ---")
    print(f"Precision:             {metrics.precision_weighted:.4f}")
    print(f"Recall:                {metrics.recall_weighted:.4f}")
    print(f"F1 Score:              {metrics.f1_weighted:.4f}")

    # Use actual number of classes from metrics (may differ if some classes absent)
    n_actual_classes: int = len(metrics.per_class_precision)
    actual_class_names: list[str] = class_names[:n_actual_classes] if n_actual_classes <= len(class_names) else class_names

    print("\n--- Per-Class Metrics ---")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 56)

    for i in range(n_actual_classes):
        name = actual_class_names[i] if i < len(actual_class_names) else f"Class_{i}"
        print(f"{name:<20} {metrics.per_class_precision[i]:.4f}       "
              f"{metrics.per_class_recall[i]:.4f}       {metrics.per_class_f1[i]:.4f}")

    print("\n--- Confusion Matrix ---")
    print("(Rows = True, Columns = Predicted)")

    # Print header
    print(f"{'':>15}", end="")
    for i in range(len(metrics.confusion_matrix)):
        name = actual_class_names[i][:8] if i < len(actual_class_names) else f"C{i}"
        print(f"{name:>10}", end="")
    print()

    # Print matrix
    for i, row in enumerate(metrics.confusion_matrix):
        name = actual_class_names[i][:15] if i < len(actual_class_names) else f"Class_{i}"
        print(f"{name:>15}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()


def test(
    exp_path: Path,
    split: str = "test",
    output_file: Path | None = None,
) -> EvaluationMetrics:
    """
    Test a trained model.

    Args:
        exp_path: Path to experiment folder.
        split: Data split to evaluate ('train', 'val', 'test').
        output_file: Optional path to save results.

    Returns:
        Evaluation metrics.
    """
    print(f"Loading experiment from: {exp_path}")

    # Load config
    config: dict[str, Any] = load_experiment_config(exp_path)
    data_config: dict[str, Any] = config.get("data", {})

    # Find and load model
    model_path, model_name = find_model_file(exp_path)
    print(f"Loading model: {model_name} from {model_path.name}")

    model: BaseModel = get_model(model_name)
    model.load(model_path)

    # Load data
    print("\nLoading data...")
    dataset: MLDataset = load_ml_ready_data(
        dataset_type=data_config.get("dataset_type", "physical"),
        train_ratio=data_config.get("train_ratio", 0.7),
        val_ratio=data_config.get("val_ratio", 0.15),
        test_ratio=data_config.get("test_ratio", 0.15),
        balancing="none",  # No balancing for testing
        normalize=data_config.get("normalize", True),
        n_samples=data_config.get("n_samples"),
        random_state=data_config.get("seed", 42),
        datasets=data_config.get("datasets"),
        nrows=data_config.get("nrows_per_file"),
    )

    # Select split
    if split == "train":
        X, y = dataset.X_train, dataset.y_train
    elif split == "val":
        X, y = dataset.X_val, dataset.y_val
    else:
        X, y = dataset.X_test, dataset.y_test

    print(f"Evaluating on {split} split ({len(y)} samples)")

    # Evaluate
    metrics: EvaluationMetrics = model.evaluate(X, y)

    # Print results
    print_detailed_metrics(metrics, dataset.class_names)

    # Save results if requested
    if output_file is not None:
        results: dict[str, Any] = {
            "experiment": str(exp_path),
            "model": model_name,
            "split": split,
            "n_samples": len(y),
            "metrics": metrics.to_dict(),
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test trained ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/test.py --experiment experiments/2024-01-14_19-30-00_physical_small_xgboost
    python src/test.py --experiment experiments/latest --split val
    python src/test.py --experiment experiments/best_model --output results.json
        """,
    )

    parser.add_argument(
        "--experiment", "-e",
        type=Path,
        required=True,
        help="Path to experiment folder",
    )

    parser.add_argument(
        "--split", "-s",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate (default: test)",
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to save results JSON",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.experiment.exists():
        print(f"Error: Experiment folder not found: {args.experiment}")
        sys.exit(1)

    test(
        exp_path=args.experiment,
        split=args.split,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
