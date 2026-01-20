#!/usr/bin/env python3
"""
generate_configs.py - Generate All Configuration Files

Creates organized config files for all model × dataset × size combinations.

Structure:
    configs/
    ├── xgboost/
    │   ├── physical_small.yaml
    │   ├── physical_medium.yaml
    │   ├── physical_large.yaml
    │   ├── network_small.yaml
    │   ├── network_medium.yaml
    │   └── network_large.yaml
    ├── knn/
    │   └── ...
    └── ...

Usage:
    python src/generate_configs.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# =============================================================================
# Configuration Templates
# =============================================================================

# Model-specific hyperparameters for each size
MODEL_HYPERPARAMS: dict[str, dict[str, dict[str, Any]]] = {
    "xgboost": {
        "small": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.2},
        "medium": {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.1, "subsample": 0.8},
        "large": {"n_estimators": 500, "max_depth": 12, "learning_rate": 0.05, "subsample": 0.9, "reg_lambda": 1.0},
    },
    "knn": {
        "small": {"n_neighbors": 3, "weights": "uniform"},
        "medium": {"n_neighbors": 5, "weights": "distance"},
        "large": {"n_neighbors": 7, "weights": "distance", "algorithm": "ball_tree"},
    },
    "random_forest": {
        "small": {"n_estimators": 50, "max_depth": 6},
        "medium": {"n_estimators": 200, "max_depth": 12, "min_samples_split": 5},
        "large": {"n_estimators": 500, "max_depth": 20, "min_samples_split": 2},
    },
    "mlp": {
        "small": {"hidden_layers": [64, 32], "dropout": 0.2},
        "medium": {"hidden_layers": [256, 128, 64], "dropout": 0.3},
        "large": {"hidden_layers": [512, 256, 128, 64], "dropout": 0.4, "activation": "gelu"},
    },
    "tab_transformer": {
        "small": {"d_model": 32, "n_heads": 2, "n_layers": 1, "dropout": 0.1},
        "medium": {"d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.15},
        "large": {"d_model": 128, "n_heads": 8, "n_layers": 4, "dropout": 0.2},
    },
    "ft_transformer": {
        "small": {"d_token": 48, "n_heads": 4, "n_layers": 1, "dropout": 0.1},
        "medium": {"d_token": 96, "n_heads": 8, "n_layers": 3, "dropout": 0.15},
        "large": {"d_token": 192, "n_heads": 8, "n_layers": 6, "dropout": 0.2, "d_ff_mult": 4},
    },
    "attention_mlp": {
        "small": {"hidden_dim": 32, "n_heads": 2, "mlp_layers": [64, 32], "dropout": 0.2},
        "medium": {"hidden_dim": 64, "n_heads": 4, "mlp_layers": [128, 64], "dropout": 0.25},
        "large": {"hidden_dim": 128, "n_heads": 8, "mlp_layers": [256, 128, 64], "dropout": 0.3},
    },
}

# Training settings per size
TRAINING_SETTINGS: dict[str, dict[str, Any]] = {
    "small": {
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.002,
        "patience": 5,
    },
    "medium": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "patience": 5,
    },
    "large": {
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 0.0005,
        "patience": 3,
    },
}

# Data settings per dataset type and size
DATA_SETTINGS: dict[str, dict[str, dict[str, Any]]] = {
    "physical": {
        "small": {"n_samples": None, "balancing": "oversampling_copy"},
        "medium": {"n_samples": None, "balancing": "oversampling_copy"},
        "large": {"n_samples": None, "balancing": "class_weights"},
    },
    "network": {
        # Network files have time-ordered data with attacks appearing after ~50k rows
        # Must load enough rows per file to capture attack labels ('anomaly')
        "small": {"nrows_per_file": 300000, "balancing": "oversampling_copy"},  # ~500k rows total
        "medium": {"nrows_per_file": 500000, "balancing": "oversampling_copy"},  # ~2.5M rows
        "large": {"nrows_per_file": 800000, "balancing": "class_weights"},  # Full dataset
    },
}


def create_config(
    model_name: str,
    dataset_type: str,
    size: str,
) -> dict[str, Any]:
    """
    Create a complete configuration dictionary.

    Args:
        model_name: Model identifier (e.g., 'xgboost', 'mlp').
        dataset_type: Either 'physical' or 'network'.
        size: Size profile ('small', 'medium', 'large').

    Returns:
        Complete configuration dictionary.
    """
    data_settings = DATA_SETTINGS[dataset_type][size]
    training_settings = TRAINING_SETTINGS[size]
    model_hyperparams = MODEL_HYPERPARAMS[model_name][size]

    config: dict[str, Any] = {
        "data": {
            "dataset_type": dataset_type,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "balancing": data_settings["balancing"],
            "normalize": True,
            "seed": 42,
        },
        "model": {
            "name": model_name,
            "hyperparameters": model_hyperparams,
        },
        "training": {
            "epochs": training_settings["epochs"],
            "batch_size": training_settings["batch_size"],
            "learning_rate": training_settings["learning_rate"],
            "optimizer": "adamw",
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "early_stopping": True,
            "patience": training_settings["patience"],
            "use_gpu": True,
        },
        "experiment": {
            "name": f"{dataset_type}_{size}",
            "save_predictions": True,
            "save_history": True,
            "save_model": True,
        },
    }

    # Add dataset-specific settings
    if dataset_type == "physical":
        if data_settings["n_samples"] is not None:
            config["data"]["n_samples"] = data_settings["n_samples"]
    else:  # network
        if data_settings["nrows_per_file"] is not None:
            config["data"]["nrows_per_file"] = data_settings["nrows_per_file"]

    return config


def generate_all_configs(output_dir: Path) -> list[Path]:
    """
    Generate all configuration files.

    Args:
        output_dir: Base configs directory.

    Returns:
        List of created config file paths.
    """
    created_files: list[Path] = []

    models = list(MODEL_HYPERPARAMS.keys())
    dataset_types = ["physical", "network"]
    sizes = ["small", "medium", "large"]

    for model_name in models:
        # Create model subfolder
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        for dataset_type in dataset_types:
            for size in sizes:
                # Create config
                config = create_config(model_name, dataset_type, size)

                # Write file
                filename = f"{dataset_type}_{size}.yaml"
                filepath = model_dir / filename

                with open(filepath, "w") as f:
                    # Add header comment
                    f.write(f"# {model_name.upper()} - {dataset_type.capitalize()} Data - {size.capitalize()} Config\n")
                    f.write(f"# Auto-generated by generate_configs.py\n\n")
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                created_files.append(filepath)
                print(f"Created: {filepath}")

    return created_files


def main() -> None:
    """Main entry point."""
    # Get configs directory
    script_dir = Path(__file__).parent
    configs_dir = script_dir.parent / "configs"

    print("=" * 60)
    print("GENERATING CONFIGURATION FILES")
    print("=" * 60)
    print(f"Output directory: {configs_dir}\n")

    created = generate_all_configs(configs_dir)

    print("\n" + "=" * 60)
    print(f"✅ Generated {len(created)} configuration files")
    print("=" * 60)

    # Print summary
    print("\nFolder structure:")
    for model in MODEL_HYPERPARAMS.keys():
        print(f"  configs/{model}/")
        print(f"    ├── physical_small.yaml")
        print(f"    ├── physical_medium.yaml")
        print(f"    ├── physical_large.yaml")
        print(f"    ├── network_small.yaml")
        print(f"    ├── network_medium.yaml")
        print(f"    └── network_large.yaml")


if __name__ == "__main__":
    main()
