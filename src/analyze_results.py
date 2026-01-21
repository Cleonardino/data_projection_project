#!/usr/bin/env python3
"""
analyze_results.py - Comprehensive Results Analysis

Analyzes experiment results and generates separate markdown reports for each dataset:
- Model leaderboard (ranked by F1 score)
- Training curves
- Error analysis (hardest classes/samples)
- Model correlation matrix

Output:
    results_analysis/
    ├── network/            # Analysis for network dataset
    │   ├── report.md
    │   └── ...
    ├── physical/           # Analysis for physical dataset
    │   ├── report.md
    │   └── ...
    └── summary.md          # Global summary (optional)

Usage:
    python src/analyze_results.py
    python src/analyze_results.py --experiments-dir experiments/
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")


def load_experiment_results(experiments_dir: Path) -> list[dict[str, Any]]:
    """
    Load results from all experiment folders.

    Args:
        experiments_dir: Directory containing experiment folders.

    Returns:
        List of experiment result dictionaries.
    """
    results: list[dict[str, Any]] = []

    if not experiments_dir.exists():
        return results

    for exp_folder in sorted(experiments_dir.iterdir()):
        if not exp_folder.is_dir():
            continue

        metrics_file = exp_folder / "metrics.json"
        config_file = exp_folder / "config.yaml"

        if not metrics_file.exists():
            continue

        # Load metrics
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {metrics_file}")
            continue

        # Load config if available
        config: dict[str, Any] = {}
        if config_file.exists():
            import yaml
            try:
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
            except Exception:
                pass

        # Parse folder name for metadata
        folder_name = exp_folder.name
        parts = folder_name.split("_")

        # Extract model name from folder (last part)
        model_name = parts[-1] if parts else "unknown"

        # Get dataset and size from config
        dataset_type = config.get("data", {}).get("dataset_type", "unknown")
        # If dataset not in config, infer from folder name if possible (heuristic)
        if dataset_type == "unknown":
            for part in parts:
                if part in ["network", "physical"]:
                    dataset_type = part
                    break

        exp_name = config.get("experiment", {}).get("name", "unknown")

        results.append({
            "experiment_folder": str(exp_folder),
            "folder_name": folder_name,
            "model": model_name,
            "dataset": dataset_type,
            "experiment_name": exp_name,
            "config": config,
            "metrics": metrics,
        })

    return results


def create_leaderboard(results: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Create a leaderboard DataFrame from experiment results.

    Args:
        results: List of experiment result dicts.

    Returns:
        DataFrame with model rankings.
    """
    rows: list[dict[str, Any]] = []

    for exp in results:
        metrics = exp["metrics"]
        test_metrics = metrics.get("metrics", {}).get("test", {})
        training = metrics.get("training", {})

        rows.append({
            "Experiment": exp["folder_name"],
            "Model": exp["model"],
            "Dataset": exp["dataset"],
            "Config": exp["experiment_name"],
            "Accuracy": test_metrics.get("accuracy", 0),
            "F1 (macro)": test_metrics.get("f1_macro", 0),
            "F1 (weighted)": test_metrics.get("f1_weighted", 0),
            "Balanced Acc": test_metrics.get("balanced_accuracy", 0),
            "MCC": test_metrics.get("mcc", 0),
            "Training Time (s)": training.get("training_time_seconds", 0),
            "Best Epoch": training.get("best_epoch", 0),
        })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        # Sort by F1 macro descending
        df = df.sort_values("F1 (macro)", ascending=False).reset_index(drop=True)
        df.index = df.index + 1  # 1-indexed ranking

    return df


def load_error_files(results: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    """
    Load test error CSVs for the specified experiments.

    Args:
        results: List of experiment result dicts to load errors for.

    Returns:
        Dict mapping experiment name to error DataFrame.
    """
    errors: dict[str, pd.DataFrame] = {}

    for exp in results:
        exp_folder = Path(exp["experiment_folder"])
        error_file = exp_folder / "test_errors.csv"

        if error_file.exists():
            try:
                df = pd.read_csv(error_file)
                errors[exp["folder_name"]] = df
            except Exception as e:
                print(f"Warning: Could not read errors for {exp['folder_name']}: {e}")

    return errors


def analyze_hard_samples(errors: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze which samples are hardest across all models.

    Args:
        errors: Dict of experiment name -> error DataFrame.

    Returns:
        Tuple of (hard_samples_df, hard_classes_df).
    """
    if not errors:
        return pd.DataFrame(), pd.DataFrame()

    # Combine all predictions
    sample_errors: dict[int, dict[str, Any]] = defaultdict(lambda: {"total_wrong": 0, "models_wrong": []})
    class_errors: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "wrong": 0})

    for exp_name, df in errors.items():
        for _, row in df.iterrows():
            sample_idx = row["sample_idx"]
            true_label = row["true_label"]
            is_correct = row["is_correct"]

            class_errors[true_label]["total"] += 1

            if not is_correct:
                sample_errors[sample_idx]["total_wrong"] += 1
                sample_errors[sample_idx]["models_wrong"].append(exp_name)
                # Store sample metadata (assuming consistent across models for same dataset)
                sample_errors[sample_idx]["true_label"] = true_label
                sample_errors[sample_idx]["pred_labels"] = sample_errors[sample_idx].get("pred_labels", [])
                sample_errors[sample_idx]["pred_labels"].append(row["pred_label"])
                class_errors[true_label]["wrong"] += 1

    # Create hard samples DataFrame
    hard_samples_rows: list[dict[str, Any]] = []
    for sample_idx, info in sample_errors.items():
        if info["total_wrong"] > 0:
            hard_samples_rows.append({
                "sample_idx": sample_idx,
                "true_label": info.get("true_label", ""),
                "times_wrong": info["total_wrong"],
                "wrong_by_models": len(info["models_wrong"]),
                "common_wrong_pred": max(set(info.get("pred_labels", [])), key=info.get("pred_labels", []).count) if info.get("pred_labels") else "",
            })

    hard_samples_df = pd.DataFrame(hard_samples_rows)
    if len(hard_samples_df) > 0:
        hard_samples_df = hard_samples_df.sort_values("times_wrong", ascending=False).head(50)

    # Create hard classes DataFrame
    hard_classes_rows: list[dict[str, Any]] = []
    for class_name, counts in class_errors.items():
        if counts["total"] > 0:
            error_rate = counts["wrong"] / counts["total"]
            hard_classes_rows.append({
                "class": class_name,
                "total_predictions": counts["total"],
                "total_errors": counts["wrong"],
                "error_rate": error_rate,
            })

    hard_classes_df = pd.DataFrame(hard_classes_rows)
    if len(hard_classes_df) > 0:
        hard_classes_df = hard_classes_df.sort_values("error_rate", ascending=False)

    return hard_samples_df, hard_classes_df


def compute_model_correlation(errors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute correlation between models based on error patterns.

    Two models are similar if they make mistakes on the same samples.

    Args:
        errors: Dict of experiment name -> error DataFrame.

    Returns:
        Correlation matrix DataFrame.
    """
    if len(errors) < 2:
        return pd.DataFrame()

    # Build error vectors for each model
    exp_names = list(errors.keys())

    # Get all sample indices
    all_samples: set[int] = set()
    for df in errors.values():
        all_samples.update(df["sample_idx"].tolist())

    sample_list = sorted(all_samples)
    sampling_needed = False

    # Optimization: If we have > 50k samples, subsample for correlation to save memory/time
    if len(sample_list) > 50000:
        sampling_needed = True
        rng = np.random.RandomState(42)
        sample_list = sorted(rng.choice(sample_list, 50000, replace=False))

    n_samples = len(sample_list)
    sample_to_idx = {s: i for i, s in enumerate(sample_list)}

    # Create binary error matrix (1 = wrong, 0 = correct)
    error_matrix = np.zeros((len(exp_names), n_samples))

    for i, exp_name in enumerate(exp_names):
        df = errors[exp_name]
        # Filter df to only samples we care about
        if sampling_needed:
             df = df[df['sample_idx'].isin(sample_to_idx.keys())]

        for _, row in df.iterrows():
            sample_idx = sample_to_idx.get(row["sample_idx"])
            if sample_idx is not None and not row["is_correct"]:
                error_matrix[i, sample_idx] = 1

    # Filter out identical duplicates: for same model name AND identical error vector, keep only most recent
    # keep_indices = []

    # 1. Parse metadata for all experiments
    exp_metadata = []
    for idx, name in enumerate(exp_names):
        parts = name.split('_')
        timestamp_obj = datetime.min
        clean_name = name

        # Parse timestamp if present (YYYY-MM-DD_HH-MM-SS)
        if len(parts) >= 2 and parts[0][0].isdigit():
            try:
                ts_str = f"{parts[0]}_{parts[1]}" # YYYY-MM-DD_HH-MM-SS
                timestamp_obj = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")
                # Clean name starts after timestamp
                clean_name = "_".join(parts[2:])
            except ValueError:
                pass

        exp_metadata.append({
            "index": idx,
            "timestamp": timestamp_obj,
            "clean_name": clean_name
        })

    # 2. Group by clean_name
    groups = defaultdict(list)
    for meta in exp_metadata:
        groups[meta["clean_name"]].append(meta)

    # 3. Identify indices to drop
    indices_to_drop = set()

    for model_name, group in groups.items():
        if len(group) < 2:
            continue

        # Sort by timestamp descending (newest first)
        group.sort(key=lambda x: x["timestamp"], reverse=True)

        # Keep the newest one always. Check others against kept ones.
        kept_in_group = [group[0]]

        for i in range(1, len(group)):
            candidate = group[i]
            candidate_vec = error_matrix[candidate["index"]]

            is_identical = False
            for kept in kept_in_group:
                kept_vec = error_matrix[kept["index"]]
                if np.array_equal(candidate_vec, kept_vec):
                    is_identical = True
                    break

            if is_identical:
                # It's identical to a newer run -> drop it
                indices_to_drop.add(candidate["index"])
            else:
                # It's different -> keep it
                kept_in_group.append(candidate)

    # 4. Filter lists
    if indices_to_drop:
        valid_indices = [i for i in range(len(exp_names)) if i not in indices_to_drop]
        exp_names = [exp_names[i] for i in valid_indices]
        error_matrix = error_matrix[valid_indices]
        print(f"  Dropped {len(indices_to_drop)} identical duplicate experiments from correlation analysis.")

    # Compute correlation
    # Using Jaccard similarity: intersection / union of errors
    n_models = len(exp_names)
    correlation = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            errors_i = error_matrix[i]
            errors_j = error_matrix[j]

            intersection = np.sum(errors_i * errors_j)
            union = np.sum(np.maximum(errors_i, errors_j))

            if union > 0:
                correlation[i, j] = intersection / union
            else:
                correlation[i, j] = 1.0 if i == j else 0.0

    # Create cleaner labels for display
    # 1. Remove timestamp (YYYY-MM-DD_HH-MM-SS)
    # 2. Find common prefix to strip (like "physical_" or "network_")
    # 3. Handle duplicates by prepending time
    # 4. Sort by reversed name

    parsed_names = []
    for name in exp_names:
        parts = name.split('_')
        timestamp = ""
        clean_name = name

        # Check if starts with timestamp pattern (date_time_...)
        if len(parts) >= 3 and parts[0][0].isdigit():
            # Capture date and time part (MM-DD HH:MM) for disambiguation
            # parts[0] is YYYY-MM-DD -> take MM-DD
            date_parts = parts[0].split('-')
            date_str = f"{date_parts[1]}-{date_parts[2]}" if len(date_parts) >= 3 else parts[0]

            # parts[1] is HH-MM-SS -> take HH-MM
            time_parts = parts[1].split('-')
            time_str = f"{time_parts[0]}:{time_parts[1]}" if len(time_parts) >= 2 else parts[1]

            timestamp = f"{date_str} {time_str}"

            # Join from index 2 onwards for clean name
            clean_name = "_".join(parts[2:])

        parsed_names.append({"original": name, "timestamp": timestamp, "clean": clean_name})

    # Find common prefix
    all_clean = [p["clean"] for p in parsed_names]
    if len(all_clean) > 1:
        prefix = os.path.commonprefix(all_clean)
        if prefix and prefix.endswith('_'):
            for p in parsed_names:
                p["short"] = p["clean"][len(prefix):]
        else:
            for p in parsed_names:
                p["short"] = p["clean"]
    else:
        for p in parsed_names:
            p["short"] = p["clean"]

    # Check for duplicates
    name_counts : defaultdict[str, int] = defaultdict(int)
    for p in parsed_names:
        name_counts[p["short"]] += 1

    # Finalize display names
    for p in parsed_names:
        if name_counts[p["short"]] > 1:
            # Ambiguous: prepend timestamp
            p["display"] = f"[{p['timestamp']}] {p['short']}"
        else:
            p["display"] = p["short"]

    # Sort by reversed display name
    # e.g. "mlp_medium" -> "muidem_plm" (sorts by suffix first)
    parsed_names.sort(key=lambda x: x["display"][::-1])

    # Reorder correlation matrix
    sorted_indices = [exp_names.index(p["original"]) for p in parsed_names]
    sorted_display_names = [p["display"] for p in parsed_names]

    sorted_correlation = correlation[sorted_indices, :][:, sorted_indices]

    return pd.DataFrame(sorted_correlation, index=sorted_display_names, columns=sorted_display_names)


def plot_training_curves(results: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """
    Generate training curve plots for specified experiments.

    Args:
        results: List of experiment dictionaries.
        output_dir: Output directory for plots.

    Returns:
        List of generated plot paths.
    """
    if not HAS_PLOTTING:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []

    for exp in results:
        exp_folder = Path(exp["experiment_folder"])
        history_file = exp_folder / "training_history.csv"

        if not history_file.exists():
            continue

        try:
            df = pd.read_csv(history_file)
            if len(df) == 0 or "epoch" not in df.columns:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Use concise title
            plot_title = f"{exp['model']} ({exp['dataset']})"

            # Loss plot
            if "train_loss" in df.columns and df["train_loss"].notna().any():
                axes[0].plot(df["epoch"], df["train_loss"], label="Train", marker="o", markersize=2)
            if "val_loss" in df.columns and df["val_loss"].notna().any():
                axes[0].plot(df["epoch"], df["val_loss"], label="Val", marker="o", markersize=2)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title(f"{plot_title}\nLoss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Accuracy plot
            if "train_accuracy" in df.columns:
                axes[1].plot(df["epoch"], df["train_accuracy"], label="Train", marker="o", markersize=2)
            if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
                axes[1].plot(df["epoch"], df["val_accuracy"], label="Val", marker="o", markersize=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title(f"Accuracy")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            plot_path = output_dir / f"{exp['folder_name']}.png"
            plt.savefig(plot_path, dpi=100)
            plt.close()

            plots.append(plot_path)

        except Exception as e:
            print(f"Warning: Could not plot {exp['folder_name']}: {e}")

    return plots


def plot_correlation_matrix(correlation_df: pd.DataFrame, output_path: Path) -> None:
    """Plot correlation heatmap."""
    if not HAS_PLOTTING or correlation_df.empty:
        return

    # Increased figure size for readability
    fig, ax = plt.subplots(figsize=(14, 12))

    sns.heatmap(
        correlation_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        square=True,
        annot_kws={"size": 8},  # Smaller font for values
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Model Error Correlation\n(Jaccard similarity of misclassified samples)")

    # Rotate x labels to prevent overlap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(
    output_dir: Path,
    dataset_name: str,
    leaderboard: pd.DataFrame,
    hard_samples: pd.DataFrame,
    hard_classes: pd.DataFrame,
    correlation: pd.DataFrame,
    training_curves: list[Path],
) -> Path:
    """
    Generate comprehensive markdown report for a dataset.

    Args:
        output_dir: Output directory.
        dataset_name: Name of the dataset (e.g. "network").
        leaderboard: Leaderboard DataFrame.
        hard_samples: Hard samples DataFrame.
        hard_classes: Hard classes DataFrame.
        correlation: Model correlation DataFrame.
        training_curves: List of training curve plot paths.

    Returns:
        Path to generated report.
    """
    report_path = output_dir / "report.md"
    dataset_title = dataset_name.capitalize()

    with open(report_path, "w") as f:
        # Header
        f.write(f"# Analysis Report: {dataset_title} Dataset\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Leaderboard
        f.write("## 📊 Model Leaderboard\n\n")
        f.write("Ranked by F1 (macro) score on test set.\n\n")

        if len(leaderboard) > 0:
            # Simplified table for markdown
            table_cols = ["Experiment Name", "Accuracy", "F1 (macro)", "Balanced Acc", "MCC", "Time (s)"]

            # Map dataframe cols to table cols
            if "Dataset" in leaderboard.columns:
                leaderboard = leaderboard.drop(columns=["Dataset"])

            f.write("| Rank | " + " | ".join(table_cols) + " |\n")
            f.write("|------|" + "|".join(["---" for _ in table_cols]) + "|\n")

            for rank, row in leaderboard.iterrows():
                values = [
                    f"`{row['Experiment']}`",
                    f"{row['Accuracy']:.4f}",
                    f"{row['F1 (macro)']:.4f}",
                    f"{row['Balanced Acc']:.4f}",
                    f"{row['MCC']:.4f}",
                    f"{row['Training Time (s)']:.1f}",
                ]
                f.write(f"| {rank} | " + " | ".join(values) + " |\n")
        else:
            f.write("*No experiments found for this dataset.*\n")

        f.write("\n---\n\n")

        # Hard Classes
        f.write("## 🎯 Hard Classes Analysis\n\n")
        f.write("Classes with highest error rates across all models.\n\n")

        if len(hard_classes) > 0:
            f.write("| Class | Total | Errors | Error Rate |\n")
            f.write("|-------|-------|--------|------------|\n")

            for _, row in hard_classes.iterrows():
                f.write(f"| {row['class']} | {row['total_predictions']} | {row['total_errors']} | {row['error_rate']:.2%} |\n")
        else:
            f.write("*No error data available.*\n")

        f.write("\n---\n\n")

        # Hard Samples
        f.write("## 🔍 Hardest Samples\n\n")
        f.write("Samples misclassified by most models (top 20).\n\n")

        if len(hard_samples) > 0:
            f.write("| Sample | True Label | Times Wrong | Common Prediction |\n")
            f.write("|--------|------------|-------------|-------------------|\n")

            for _, row in hard_samples.head(20).iterrows():
                f.write(f"| {row['sample_idx']} | {row['true_label']} | {row['times_wrong']} | {row['common_wrong_pred']} |\n")
        else:
            f.write("*No error data available.*\n")

        f.write("\n---\n\n")

        # Model Correlation
        f.write("## 🔗 Model Error Correlation\n\n")
        f.write("Which models make similar mistakes? (Jaccard similarity of error sets)\n\n")

        if len(correlation) > 0:
            # Text table
            f.write("**Correlation Matrix:**\n\n")
            f.write("| Model | " + " | ".join(correlation.columns) + " |\n")
            f.write("|-------|" + "|".join(["---" for _ in correlation.columns]) + "|\n")

            for idx, row in correlation.iterrows():
                values = [f"{v:.2f}" for v in row]
                f.write(f"| {idx} | " + " | ".join(values) + " |\n")

            f.write("\n")

            if (output_dir / "correlation" / "correlation_heatmap.png").exists():
                f.write("![Correlation Heatmap](correlation/correlation_heatmap.png)\n")
        else:
            f.write("*Need at least 2 experiments for correlation analysis.*\n")

        f.write("\n---\n\n")

        # Training Curves
        f.write("## 📈 Training Curves\n\n")

        if training_curves:
            for plot in training_curves:
                exp_name = plot.stem
                f.write(f"### {exp_name}\n\n")
                f.write(f"![{exp_name}](training_curves/{plot.name})\n\n")
        else:
            f.write("*No training curves available.*\n")

    return report_path


def run_analysis_for_dataset(
    dataset_name: str,
    results: list[dict[str, Any]],
    base_output_dir: Path
) -> None:
    """
    Run full analysis pipeline for a single dataset.
    """
    print(f"\n--- Analyzing Dataset: {dataset_name} ({len(results)} experiments) ---")

    # Create output directory for this dataset
    output_dir = base_output_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    curves_dir = output_dir / "training_curves"
    correlation_dir = output_dir / "correlation"
    curves_dir.mkdir(exist_ok=True)
    correlation_dir.mkdir(exist_ok=True)

    # 1. leaderboard
    print(f"Creating leaderboard for {dataset_name}...")
    leaderboard = create_leaderboard(results)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=True)

    # 2. errors
    print(f"Loading errors for {dataset_name}...")
    errors = load_error_files(results)

    # 3. hard samples
    print(f"Analyzing hard samples for {dataset_name}...")
    hard_samples, hard_classes = analyze_hard_samples(errors)
    if len(hard_samples) > 0:
        hard_samples.to_csv(output_dir / "hard_samples.csv", index=False)
    if len(hard_classes) > 0:
        hard_classes.to_csv(output_dir / "hard_classes.csv", index=False)

    # 4. correlation
    print(f"Computing correlation for {dataset_name}...")
    correlation = compute_model_correlation(errors)
    if len(correlation) > 0:
        correlation.to_csv(correlation_dir / "correlation_matrix.csv")
        plot_correlation_matrix(correlation, correlation_dir / "correlation_heatmap.png")

    # 5. curves
    print(f"Plotting curves for {dataset_name}...")
    training_curves = plot_training_curves(results, curves_dir)

    # 6. report
    print(f"Generating report for {dataset_name}...")
    report_path = generate_report(
        output_dir,
        dataset_name,
        leaderboard,
        hard_samples,
        hard_classes,
        correlation,
        training_curves,
    )
    print(f"✅ Report generated: {report_path}")


def analyze_results(
    experiments_dir: Path,
    output_dir: Path,
) -> None:
    """
    Run full analysis and generate reports.

    Args:
        experiments_dir: Directory containing experiments.
        output_dir: Output directory for analysis.
    """
    print("=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    print(f"Experiments directory: {experiments_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Load all results
    print("Loading all experiment results...")
    all_results = load_experiment_results(experiments_dir)
    print(f"  Found {len(all_results)} total experiments")

    if not all_results:
        print("No experiments found.")
        return

    # Group by dataset
    grouped_results: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for res in all_results:
        dataset = res.get("dataset", "unknown")
        grouped_results[dataset].append(res)

    print(f"  Datasets found: {list(grouped_results.keys())}")

    # Run analysis for each dataset
    for dataset, results in grouped_results.items():
        run_analysis_for_dataset(dataset, results, output_dir)

    print()
    print("=" * 60)
    print(f"✅ Analysis complete!")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze ML experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--experiments-dir", "-e",
        type=Path,
        default=None,
        help="Experiments directory (default: experiments/)",
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: results_analysis/)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    script_dir = Path(__file__).parent
    experiments_dir = args.experiments_dir or script_dir.parent / "experiments"
    output_dir = args.output_dir or script_dir.parent / "results_analysis"

    if not experiments_dir.exists():
        print(f"Error: Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    analyze_results(experiments_dir, output_dir)


if __name__ == "__main__":
    main()
