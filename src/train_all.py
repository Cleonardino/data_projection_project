#!/usr/bin/env python3
"""
train_all.py - Batch Training Script

Trains all models using all available configuration files.
Supports parallel training, filtering by model/dataset/size, and resuming.

Usage:
    python src/train_all.py                          # Train all configs
    python src/train_all.py --model xgboost mlp      # Only specific models
    python src/train_all.py --dataset physical       # Only physical data
    python src/train_all.py --size small             # Only small configs
    python src/train_all.py --dry-run                # Show what would run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def find_all_configs(configs_dir: Path) -> list[Path]:
    """
    Find all YAML config files recursively.

    Args:
        configs_dir: Base configs directory.

    Returns:
        List of config file paths sorted by name.
    """
    configs: list[Path] = []

    for yaml_file in configs_dir.rglob("*.yaml"):
        # Skip schema file
        if "schema" in yaml_file.name:
            continue
        configs.append(yaml_file)

    return sorted(configs)


def parse_config_path(config_path: Path) -> dict[str, str]:
    """
    Extract model, dataset, and size from config path.

    Args:
        config_path: Path to config file.

    Returns:
        Dict with 'model', 'dataset', 'size' keys.
    """
    # Path structure: configs/<model>/<dataset>_<size>.yaml
    parts = config_path.stem.split("_")

    return {
        "model": config_path.parent.name,
        "dataset": parts[0] if len(parts) >= 2 else "unknown",
        "size": parts[1] if len(parts) >= 2 else "unknown",
    }


def filter_configs(
    configs: list[Path],
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    sizes: list[str] | None = None,
) -> list[Path]:
    """
    Filter configs by model, dataset, and/or size.

    Args:
        configs: List of all config paths.
        models: Filter by model names (if provided).
        datasets: Filter by dataset types (if provided).
        sizes: Filter by size profiles (if provided).

    Returns:
        Filtered list of config paths.
    """
    filtered: list[Path] = []

    for config in configs:
        info = parse_config_path(config)

        if models and info["model"] not in models:
            continue
        if datasets and info["dataset"] not in datasets:
            continue
        if sizes and info["size"] not in sizes:
            continue

        filtered.append(config)

    return filtered


def run_training(
    config_path: Path,
    train_script: Path,
    experiments_dir: Path,
    timeout: int = 3600,
) -> tuple[bool, float, str]:
    """
    Run training for a single config.

    Args:
        config_path: Path to config file.
        train_script: Path to train.py script.
        experiments_dir: Output directory for experiments.
        timeout: Maximum training time in seconds.

    Returns:
        Tuple of (success, duration_seconds, error_message).
    """
    start_time = time.time()

    cmd = [
        sys.executable,
        str(train_script),
        "--config", str(config_path),
        "--experiments-dir", str(experiments_dir),
    ]

    try:
        # Start subprocess, allowing it to write directly to stdout/stderr
        # This enables progress bars to be visible
        result = subprocess.run(
            cmd,
            cwd=train_script.parent.parent,
            check=False,  # We check returncode manually
            timeout=timeout,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            return True, duration, ""
        else:
            return False, duration, "Check console output for error details"

    except subprocess.TimeoutExpired:
        return False, timeout, f"Timeout after {timeout}s"
    except Exception as e:
        return False, time.time() - start_time, str(e)


def train_all(
    configs_dir: Path,
    experiments_dir: Path,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    sizes: list[str] | None = None,
    dry_run: bool = False,
    timeout: int = 3600,
) -> dict[str, Any]:
    """
    Train all filtered configurations.

    Args:
        configs_dir: Directory containing configs.
        experiments_dir: Output directory for experiments.
        models: Filter by model names.
        datasets: Filter by dataset types.
        sizes: Filter by size profiles.
        dry_run: If True, only show what would run.
        timeout: Max training time per config.

    Returns:
        Summary dict with success/failure counts and details.
    """
    # Find and filter configs
    all_configs = find_all_configs(configs_dir)
    configs = filter_configs(all_configs, models, datasets, sizes)

    if not configs:
        print("No configs found matching the filters!")
        return {"total": 0, "success": 0, "failed": 0, "details": []}

    print("=" * 70)
    print("BATCH TRAINING")
    print("=" * 70)
    print(f"Found {len(configs)} configurations to train")
    print(f"Output directory: {experiments_dir}")
    print()

    # Show configs
    for i, config in enumerate(configs, 1):
        info = parse_config_path(config)
        print(f"  {i:2}. {info['model']:20} | {info['dataset']:10} | {info['size']}")
    print()

    if dry_run:
        print("DRY RUN - No training will be performed")
        return {"total": len(configs), "success": 0, "failed": 0, "details": []}

    # Train each config
    train_script = Path(__file__).parent / "train.py"
    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0

    start_time = datetime.now()

    for i, config in enumerate(configs, 1):
        info = parse_config_path(config)
        config_name = f"{info['model']}/{info['dataset']}_{info['size']}"

        print("-" * 70)
        print(f"[{i}/{len(configs)}] Training: {config_name}")
        print("-" * 70)

        success, duration, error = run_training(
            config, train_script, experiments_dir, timeout
        )

        if success:
            print(f"  ✅ Completed in {duration:.1f}s")
            successful += 1
        else:
            print(f"  ❌ Failed after {duration:.1f}s: {error[:100]}")
            failed += 1

        results.append({
            "config": str(config),
            "model": info["model"],
            "dataset": info["dataset"],
            "size": info["size"],
            "success": success,
            "duration": duration,
            "error": error,
        })

    total_time = (datetime.now() - start_time).total_seconds()

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Total configs: {len(configs)}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")
    print(f"Total time:    {total_time/60:.1f} minutes")

    if failed > 0:
        print("\nFailed configs:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['model']}/{r['dataset']}_{r['size']}: {r['error'][:50]}")

    return {
        "total": len(configs),
        "success": successful,
        "failed": failed,
        "total_time_seconds": total_time,
        "details": results,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train all models with all configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/train_all.py                          # Train all
    python src/train_all.py --model xgboost mlp      # Only XGBoost and MLP
    python src/train_all.py --dataset physical       # Only physical data
    python src/train_all.py --size small medium      # Only small and medium
    python src/train_all.py --dry-run                # Show what would run
        """,
    )

    parser.add_argument(
        "--model", "-m",
        nargs="+",
        type=str,
        default=None,
        help="Filter by model names",
    )

    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        type=str,
        choices=["physical", "network"],
        default=None,
        help="Filter by dataset type",
    )

    parser.add_argument(
        "--size", "-s",
        nargs="+",
        type=str,
        choices=["small", "medium", "large"],
        default=None,
        help="Filter by size profile",
    )

    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=None,
        help="Configs directory (default: configs/)",
    )

    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=None,
        help="Experiments output directory (default: experiments/)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per config in seconds (default: 3600)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be trained without actually training",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Set default paths
    script_dir = Path(__file__).parent
    configs_dir = args.configs_dir or script_dir.parent / "configs"
    experiments_dir = args.experiments_dir or script_dir.parent / "experiments"

    if not configs_dir.exists():
        print(f"Error: Configs directory not found: {configs_dir}")
        print("Run 'python src/generate_configs.py' first to create configs.")
        sys.exit(1)

    experiments_dir.mkdir(parents=True, exist_ok=True)

    train_all(
        configs_dir=configs_dir,
        experiments_dir=experiments_dir,
        models=args.model,
        datasets=args.dataset,
        sizes=args.size,
        dry_run=args.dry_run,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
