#!/usr/bin/env python3
"""
data_explore.py - Interactive Data Exploration Script

Command-line interface for exploring and analyzing the cyber-physical dataset.
Provides statistics, quality reports, label distributions, and data export.

Usage:
    python src/data_explore.py --physical --summary
    python src/data_explore.py --network --label-dist
    python src/data_explore.py --physical --quality --export data_quality.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from lib_data import (
    DataConfig,
    PhysicalDataLoader,
    NetworkDataLoader,
    DataPreprocessor,
    FeatureExtractor,
    DataFilter,
    get_label_distribution,
    get_data_summary,
)


# =============================================================================
# Display Utilities
# =============================================================================

def print_header(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    width = 60
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_table(df: pd.DataFrame, max_rows: int = 20) -> None:
    """Print a DataFrame as a formatted table."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", max_rows)
    print(df.to_string())


# =============================================================================
# Analysis Functions
# =============================================================================

def show_summary(df: pd.DataFrame, data_type: str) -> None:
    """Display statistical summary of the data."""
    print_header(f"{data_type.upper()} DATA SUMMARY")
    
    summary = get_data_summary(df)
    print(f"\nRows: {summary['n_rows']:,}")
    print(f"Columns: {summary['n_columns']}")
    print(f"Memory Usage: {summary['memory_mb']:.2f} MB")
    print(f"Missing Values: {summary['missing_values']}")
    
    print("\nColumn Types:")
    for dtype, count in summary["dtypes"].items():
        print(f"  - {dtype}: {count}")
    
    print("\nNumeric Columns Statistics:")
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print_table(numeric_df.describe().round(2))


def show_quality_report(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Display data quality report."""
    print_header(f"{data_type.upper()} DATA QUALITY REPORT")
    
    quality_data = []
    for col in df.columns:
        missing = df[col].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        unique = df[col].nunique()
        dtype = str(df[col].dtype)
        
        quality_data.append({
            "column": col,
            "dtype": dtype,
            "missing": missing,
            "missing_%": round(missing_pct, 2),
            "unique": unique,
        })
    
    quality_df = pd.DataFrame(quality_data)
    print_table(quality_df)
    
    # Highlight issues
    issues = quality_df[quality_df["missing"] > 0]
    if not issues.empty:
        print("\n⚠️  Columns with Missing Values:")
        for _, row in issues.iterrows():
            print(f"  - {row['column']}: {row['missing']} ({row['missing_%']}%)")
    else:
        print("\n✅ No missing values found!")
    
    return quality_df


def show_label_distribution(df: pd.DataFrame, data_type: str, label_col: str) -> pd.DataFrame:
    """Display label distribution analysis."""
    print_header(f"{data_type.upper()} LABEL DISTRIBUTION")
    
    if label_col not in df.columns:
        print(f"⚠️  Label column '{label_col}' not found.")
        return pd.DataFrame()
    
    dist_df = get_label_distribution(df, label_col)
    print_table(dist_df)
    
    # Imbalance analysis
    print("\nImbalance Analysis:")
    max_count = dist_df["count"].max()
    min_count = dist_df["count"].min()
    ratio = max_count / min_count if min_count > 0 else float("inf")
    print(f"  - Max/Min class ratio: {ratio:.2f}")
    
    if ratio > 10:
        print("  ⚠️  High imbalance detected! Consider using oversampling/undersampling.")
    elif ratio > 3:
        print("  ⚡ Moderate imbalance. May benefit from balancing strategies.")
    else:
        print("  ✅ Classes are relatively balanced.")
    
    return dist_df


def show_correlations(df: pd.DataFrame, data_type: str, top_n: int = 10) -> pd.DataFrame:
    """Display feature correlations (top N strongest)."""
    print_header(f"{data_type.upper()} FEATURE CORRELATIONS")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        print("No numeric columns found for correlation analysis.")
        return pd.DataFrame()
    
    corr_matrix = numeric_df.corr()
    
    # Get top correlations (excluding self-correlations)
    correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            correlations.append({
                "feature_1": col1,
                "feature_2": col2,
                "correlation": round(corr, 4),
            })
    
    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values("correlation", key=abs, ascending=False).head(top_n)
    
    print(f"Top {top_n} Strongest Correlations:")
    print_table(corr_df.reset_index(drop=True))
    
    return corr_matrix


def show_sample(df: pd.DataFrame, data_type: str, n: int = 10) -> None:
    """Display sample rows from the data."""
    print_header(f"{data_type.upper()} DATA SAMPLE ({n} rows)")
    print_table(df.head(n))


# =============================================================================
# Main CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Explore and analyze cyber-physical dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data_explore.py --physical --summary
  python src/data_explore.py --network --sample 50
  python src/data_explore.py --physical --label-dist --export labels.csv
  python src/data_explore.py --physical --all
        """
    )
    
    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--physical", "-p",
        action="store_true",
        help="Analyze physical sensor data"
    )
    data_group.add_argument(
        "--network", "-n",
        action="store_true",
        help="Analyze network traffic data"
    )
    
    # Analysis options
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Show statistical summary"
    )
    parser.add_argument(
        "--quality", "-q",
        action="store_true",
        help="Show data quality report"
    )
    parser.add_argument(
        "--label-dist", "-l",
        action="store_true",
        help="Show label distribution"
    )
    parser.add_argument(
        "--correlations", "-c",
        action="store_true",
        help="Show feature correlations"
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Show N sample rows"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all analyses"
    )
    
    # Data options
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["normal", "attack_1", "attack_2", "attack_3", "attack_4"],
        help="Specific datasets to load (default: all)"
    )
    parser.add_argument(
        "--nrows",
        type=int,
        help="Limit number of rows per file (useful for large network data)"
    )
    
    # Export
    parser.add_argument(
        "--export", "-e",
        type=str,
        metavar="PATH",
        help="Export processed data to file"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    config = DataConfig()
    data_type = "physical" if args.physical else "network"
    label_col = "Label" if args.physical else "label"
    
    # Load data
    print_header(f"LOADING {data_type.upper()} DATA")
    
    if args.physical:
        loader = PhysicalDataLoader(config)
        df = loader.load(datasets=args.datasets)
    else:
        loader = NetworkDataLoader(config)
        if args.nrows:
            df = loader.load(datasets=args.datasets, nrows=args.nrows)
        else:
            # For network, default to sampling unless nrows specified
            print("Note: Network data is large. Loading first 10000 rows per file.")
            print("Use --nrows to specify a different limit.")
            df = loader.load(datasets=args.datasets, nrows=10000)
    
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Run analyses
    any_analysis = args.summary or args.quality or args.label_dist or args.correlations or args.sample or args.all
    
    if not any_analysis:
        print("\nNo analysis specified. Use --help for options.")
        print("Common options: --summary, --quality, --label-dist, --sample N")
        return
    
    if args.all or args.summary:
        show_summary(df, data_type)
    
    if args.all or args.quality:
        quality_df = show_quality_report(df, data_type)
    
    if args.all or args.label_dist:
        show_label_distribution(df, data_type, label_col)
    
    if args.all or args.correlations:
        show_correlations(df, data_type)
    
    if args.sample:
        show_sample(df, data_type, args.sample)
    
    # Export
    if args.export:
        export_path = Path(args.export)
        df.to_csv(export_path, index=False)
        print(f"\n✅ Data exported to: {export_path.absolute()}")


if __name__ == "__main__":
    main()
