"""
lib_data.py - Data Utilities Library for Cyber-Physical Dataset Analysis

This module provides comprehensive utilities for loading, preprocessing,
filtering, and analyzing the physical sensor and network traffic datasets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Callable, Iterator
from datetime import datetime
import pickle
import hashlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for dataset paths and loading options."""

    base_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "dataset")

    # Physical dataset paths
    physical_dir: str = "Physical dataset"
    physical_files: dict[str, str] = field(default_factory=lambda: {
        "normal": "phy_norm.csv",
        "attack_1": "phy_att_1.csv",
        "attack_2": "phy_att_2.csv",
        "attack_3": "phy_att_3.csv",
        "attack_4": "phy_att_4.csv",
    })

    # Network dataset paths
    network_dir: str = "Network datatset/csv"
    network_files: dict[str, str] = field(default_factory=lambda: {
        "normal": "normal.csv",
        "attack_1": "attack_1.csv",
        "attack_2": "attack_2.csv",
        "attack_3": "attack_3.csv",
        "attack_4": "attack_4.csv",
    })

    # Column definitions for physical data
    physical_columns: dict[str, list[str]] = field(default_factory=lambda: {
        "tanks": [f"Tank_{i}" for i in range(1, 9)],
        "pumps": [f"Pump_{i}" for i in range(1, 7)],
        "flow_sensors": [f"Flow_sensor_{i}" for i in range(1, 5)],
        "valves": [f"Valv_{i}" for i in range(1, 23)],
        "labels": ["Label_n", "Label"],
    })

    # Memory settings for large files
    chunk_size: int = 50000
    low_memory: bool = True

    def get_physical_path(self, dataset: str) -> Path:
        """Get full path for a physical dataset file."""
        return self.base_path / self.physical_dir / self.physical_files[dataset]

    def get_network_path(self, dataset: str) -> Path:
        """Get full path for a network dataset file."""
        return self.base_path / self.network_dir / self.network_files[dataset]


# =============================================================================
# Data Loaders
# =============================================================================

class PhysicalDataLoader:
    """Loader for physical sensor data (tanks, pumps, valves, flow sensors)."""

    BOOL_COLUMNS = [f"Pump_{i}" for i in range(1, 7)] + [f"Valv_{i}" for i in range(1, 23)]

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def load(
        self,
        datasets: Optional[list[str]] = None,
        parse_dates: bool = True,
        convert_bools: bool = True,
    ) -> pd.DataFrame:
        """
        Load physical dataset(s) into a DataFrame.

        Args:
            datasets: List of datasets to load ['normal', 'attack_1', ...].
                     If None, loads all.
            parse_dates: Whether to parse the Time column as datetime.
            convert_bools: Whether to convert 'true'/'false' to 1/0.

        Returns:
            Combined DataFrame with all requested datasets.
        """
        if datasets is None:
            datasets = list(self.config.physical_files.keys())

        dfs = []
        for dataset in datasets:
            path = self.config.get_physical_path(dataset).resolve()
            # Binary read with explicit BOM handling
            import io
            raw_bytes = path.read_bytes()

            # Detect encoding and separator based on BOM
            if raw_bytes[:3] == b'\xef\xbb\xbf':
                # UTF-8 BOM - comma-separated
                content = raw_bytes[3:].decode('utf-8')
                sep = ","
            elif raw_bytes[:2] == b'\xff\xfe':
                # UTF-16 LE BOM - tab-separated
                content = raw_bytes[2:].decode('utf-16-le')
                sep = "\t"
            elif raw_bytes[:2] == b'\xfe\xff':
                # UTF-16 BE BOM - tab-separated
                content = raw_bytes[2:].decode('utf-16-be')
                sep = "\t"
            else:
                # Default: try UTF-16
                content = raw_bytes.decode('utf-16')
                sep = "\t"

            df = pd.read_csv(io.StringIO(content), sep=sep)
            df["source_file"] = dataset
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        if convert_bools:
            combined = self._convert_bool_columns(combined)

        if parse_dates:
            combined["Time"] = pd.to_datetime(combined["Time"], format="%d/%m/%Y %H:%M:%S")

        return combined

    def _convert_bool_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert 'true'/'false' string columns to 1/0."""
        for col in self.BOOL_COLUMNS:
            if col in df.columns:
                df[col] = df[col].map({"true": 1, "false": 0, True: 1, False: 0}).fillna(0).astype(int)
        return df

    def get_features_and_labels(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Split DataFrame into features (X), numeric labels, and string labels.

        Returns:
            (X, y_numeric, y_string) tuple
        """
        label_cols = ["Label_n", "Label", "Time", "source_file"]
        feature_cols = [c for c in df.columns if c not in label_cols]

        X = df[feature_cols].copy()
        y_numeric = df["Label_n"].copy()
        y_string = df["Label"].copy()

        return X, y_numeric, y_string


class NetworkDataLoader:
    """
    Loader for network traffic data with memory-efficient chunked reading.

    Network files are ~800MB+ each, so we provide chunked iteration and sampling.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    def load(
        self,
        datasets: Optional[list[str]] = None,
        nrows: Optional[int] = None,
        usecols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load network dataset(s) into a DataFrame.

        Args:
            datasets: List of datasets to load. If None, loads all.
            nrows: Maximum number of rows to load per file (for sampling).
            usecols: Specific columns to load (reduces memory).

        Returns:
            Combined DataFrame with all requested datasets.
        """
        if datasets is None:
            datasets = list(self.config.network_files.keys())

        dfs = []
        for dataset in datasets:
            path = self.config.get_network_path(dataset)
            df = pd.read_csv(
                path,
                nrows=nrows,
                usecols=usecols,
                low_memory=self.config.low_memory,
            )
            # Strip whitespace from column names (attack files have ' label' instead of 'label')
            df.columns = df.columns.str.strip()
            df["source_file"] = dataset
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def iter_chunks(
        self,
        dataset: str,
        chunk_size: Optional[int] = None,
        usecols: Optional[list[str]] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Iterate over chunks of a network dataset for memory-efficient processing.

        Args:
            dataset: Dataset name ('normal', 'attack_1', etc.)
            chunk_size: Number of rows per chunk.
            usecols: Specific columns to load.

        Yields:
            DataFrame chunks.
        """
        path = self.config.get_network_path(dataset)
        chunk_size = chunk_size or self.config.chunk_size

        for chunk in pd.read_csv(path, chunksize=chunk_size, usecols=usecols, low_memory=True):
            chunk["source_file"] = dataset
            yield chunk

    def sample(
        self,
        datasets: Optional[list[str]] = None,
        n_per_file: int = 1000,
        random_state: int = 42,
        stratify: bool = True,
        label_col: str = "label",
    ) -> pd.DataFrame:
        """
        Memory-efficient stratified random sample from each dataset file.

        When stratify=True, performs a two-pass approach:
        1. First pass: Read only label column to get class distribution
        2. Second pass: Sample proportionally from each class

        This ensures balanced class representation even when attack data
        appears later in time-ordered network files.

        Args:
            datasets: Dataset names to sample from.
            n_per_file: Number of rows to sample per file.
            random_state: Random seed for reproducibility.
            stratify: If True, sample proportionally from each class.
            label_col: Column name containing class labels.

        Returns:
            Combined sampled DataFrame with balanced class representation.
        """
        if datasets is None:
            datasets = list(self.config.network_files.keys())

        rng = np.random.RandomState(random_state)
        samples = []

        for dataset in datasets:
            path = self.config.get_network_path(dataset)

            if not stratify:
                # Original random sampling (non-stratified)
                total_rows = sum(1 for _ in open(path)) - 1
                if total_rows <= n_per_file:
                    df = pd.read_csv(path)
                else:
                    skip_idx = sorted(
                        rng.choice(
                            range(1, total_rows + 1),
                            size=total_rows - n_per_file,
                            replace=False
                        )
                    )
                    df = pd.read_csv(path, skiprows=skip_idx)
            else:
                # Stratified sampling: ensure proportional class representation
                # Pass 1: Read only label column to get class distribution
                # Try both clean and whitespace-prefixed column names
                try:
                    label_df = pd.read_csv(path, usecols=[label_col])
                except ValueError:
                    # Column might have leading whitespace (e.g., ' label')
                    try:
                        label_df = pd.read_csv(path, usecols=[f" {label_col}"])
                    except ValueError:
                        # Fall back to reading header and finding the right column
                        header = pd.read_csv(path, nrows=0)
                        matching_cols = [c for c in header.columns if c.strip() == label_col]
                        if matching_cols:
                            label_df = pd.read_csv(path, usecols=[matching_cols[0]])
                        else:
                            raise ValueError(f"Label column '{label_col}' not found in {path}")
                label_df.columns = label_df.columns.str.strip()

                if label_col not in label_df.columns:
                    # Try finding the column after stripping
                    possible_cols = [c for c in label_df.columns if c.strip() == label_col]
                    if possible_cols:
                        label_df = label_df.rename(columns={possible_cols[0]: label_col})

                labels = label_df[label_col].values if label_col in label_df.columns else label_df.iloc[:, 0].values
                total_rows = len(labels)

                if total_rows <= n_per_file:
                    # Load entire file if small enough
                    df = pd.read_csv(path)
                else:
                    # Get class distribution and sample proportionally
                    unique_classes, class_counts = np.unique(labels, return_counts=True)
                    class_ratios = class_counts / total_rows

                    # Calculate samples per class (at least 1 sample per class)
                    samples_per_class = np.maximum(1, (class_ratios * n_per_file).astype(int))

                    # Adjust to match n_per_file exactly
                    while samples_per_class.sum() > n_per_file:
                        max_idx = samples_per_class.argmax()
                        samples_per_class[max_idx] -= 1
                    while samples_per_class.sum() < n_per_file and samples_per_class.sum() < total_rows:
                        # Add to class with lowest current ratio
                        min_ratio_diff = np.inf
                        best_idx = 0
                        for i, (count, target) in enumerate(zip(class_counts, samples_per_class)):
                            if target < count:
                                ratio_diff = target / count if count > 0 else np.inf
                                if ratio_diff < min_ratio_diff:
                                    min_ratio_diff = ratio_diff
                                    best_idx = i
                        samples_per_class[best_idx] += 1

                    # Pass 2: Sample specific indices per class
                    selected_indices = []
                    for cls, n_samples in zip(unique_classes, samples_per_class):
                        class_indices = np.where(labels == cls)[0]
                        if len(class_indices) <= n_samples:
                            selected_indices.extend(class_indices.tolist())
                        else:
                            sampled = rng.choice(class_indices, size=n_samples, replace=False)
                            selected_indices.extend(sampled.tolist())

                    selected_indices = sorted(selected_indices)

                    # Create skiprows: all rows except selected (add 1 for header)
                    all_indices = set(range(total_rows))
                    skip_indices = all_indices - set(selected_indices)
                    skip_rows = sorted([i + 1 for i in skip_indices])  # +1 for header

                    df = pd.read_csv(path, skiprows=skip_rows)

            # Strip whitespace from column names (attack files have ' label')
            df.columns = df.columns.str.strip()
            df["source_file"] = dataset
            samples.append(df)

        return pd.concat(samples, ignore_index=True)

    def get_features_and_labels(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Split DataFrame into features (X), numeric labels, and string labels.
        """
        label_cols = ["label_n", "label", "Time", "source_file"]
        feature_cols = [c for c in df.columns if c not in label_cols]

        X = df[feature_cols].copy()
        y_numeric = df["label_n"].copy()
        y_string = df["label"].copy()

        return X, y_numeric, y_string


# =============================================================================
# Preprocessing
# =============================================================================

class DataPreprocessor:
    """Data cleaning and transformation utilities."""

    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.onehot_encoder: Optional[OneHotEncoder] = None

    def clean(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Clean the DataFrame by handling missing values and duplicates.

        Args:
            df: Input DataFrame.
            drop_na: Whether to drop rows with missing values.

        Returns:
            Cleaned DataFrame.
        """
        result = df.copy()

        # Remove duplicates
        result = result.drop_duplicates()

        # Handle missing values
        if drop_na:
            result = result.dropna()

        return result.reset_index(drop=True)

    def normalize(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Normalize numeric columns using StandardScaler.

        Args:
            df: Input DataFrame.
            columns: Columns to normalize. If None, all numeric columns.
            fit: Whether to fit the scaler or use existing.

        Returns:
            DataFrame with normalized columns.
        """
        result = df.copy()

        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()

        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            result[columns] = self.scaler.fit_transform(result[columns])
        else:
            result[columns] = self.scaler.transform(result[columns])

        return result

    def encode_labels(
        self,
        labels: pd.Series,
        fit: bool = True
    ) -> np.ndarray:
        """
        Encode string labels to integers.

        Args:
            labels: Series of string labels.
            fit: Whether to fit the encoder or use existing.

        Returns:
            Array of encoded labels.
        """
        if fit or self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            return self.label_encoder.fit_transform(labels)
        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded: np.ndarray) -> np.ndarray:
        """Convert encoded labels back to strings."""
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted. Call encode_labels first.")
        return self.label_encoder.inverse_transform(encoded)

    def onehot_encode(
        self,
        df: pd.DataFrame,
        columns: list[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        One-hot encode categorical columns.

        Args:
            df: Input DataFrame.
            columns: Columns to encode.
            fit: Whether to fit the encoder.

        Returns:
            DataFrame with one-hot encoded columns.
        """
        result = df.copy()

        if fit or self.onehot_encoder is None:
            self.onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = self.onehot_encoder.fit_transform(result[columns])
        else:
            encoded = self.onehot_encoder.transform(result[columns])

        # Create new column names
        new_cols = self.onehot_encoder.get_feature_names_out(columns)
        encoded_df = pd.DataFrame(encoded, columns=new_cols, index=result.index)

        # Drop original and add encoded
        result = result.drop(columns=columns)
        result = pd.concat([result, encoded_df], axis=1)

        return result


# =============================================================================
# Feature Extraction
# =============================================================================

class FeatureExtractor:
    """Feature engineering utilities for sensor and network data."""

    @staticmethod
    def add_time_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
        """
        Extract time-based features from a datetime column.

        Adds: hour, day_of_week, minute, is_weekend
        """
        result = df.copy()

        if time_col not in result.columns:
            return result

        dt = pd.to_datetime(result[time_col])
        result["hour"] = dt.dt.hour
        result["minute"] = dt.dt.minute
        result["day_of_week"] = dt.dt.dayofweek
        result["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

        return result

    @staticmethod
    def add_rolling_stats(
        df: pd.DataFrame,
        columns: list[str],
        windows: list[int] = [5, 10],
    ) -> pd.DataFrame:
        """
        Add rolling statistics (mean, std) for specified columns.

        Args:
            df: Input DataFrame (should be sorted by time).
            columns: Columns to compute rolling stats for.
            windows: List of window sizes.

        Returns:
            DataFrame with added rolling columns.
        """
        result = df.copy()

        for col in columns:
            if col not in result.columns:
                continue
            for w in windows:
                result[f"{col}_roll_mean_{w}"] = result[col].rolling(window=w, min_periods=1).mean()
                result[f"{col}_roll_std_{w}"] = result[col].rolling(window=w, min_periods=1).std().fillna(0)

        return result

    @staticmethod
    def add_diff_features(
        df: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        """Add difference (delta) features for specified columns."""
        result = df.copy()

        for col in columns:
            if col in result.columns:
                result[f"{col}_diff"] = result[col].diff().fillna(0)

        return result

    @staticmethod
    def add_aggregate_features(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
        """
        Add aggregate features for physical data:
        - Total tank level
        - Number of active pumps
        - Number of active valves
        """
        result = df.copy()

        tanks = [c for c in config.physical_columns["tanks"] if c in result.columns]
        pumps = [c for c in config.physical_columns["pumps"] if c in result.columns]
        valves = [c for c in config.physical_columns["valves"] if c in result.columns]

        if tanks:
            result["total_tank_level"] = result[tanks].sum(axis=1)
        if pumps:
            result["active_pumps"] = result[pumps].sum(axis=1)
        if valves:
            result["active_valves"] = result[valves].sum(axis=1)

        return result


# =============================================================================
# Filtering
# =============================================================================

class DataFilter:
    """Utilities for filtering data by various conditions."""

    @staticmethod
    def by_labels(
        df: pd.DataFrame,
        labels: list[str],
        label_col: str = "Label",
    ) -> pd.DataFrame:
        """Filter DataFrame to only include rows with specified labels."""
        return df[df[label_col].isin(labels)].reset_index(drop=True)

    @staticmethod
    def by_time_range(
        df: pd.DataFrame,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        time_col: str = "Time",
    ) -> pd.DataFrame:
        """Filter DataFrame to a specific time range."""
        result = df.copy()

        if start is not None:
            result = result[result[time_col] >= start]
        if end is not None:
            result = result[result[time_col] <= end]

        return result.reset_index(drop=True)

    @staticmethod
    def by_condition(
        df: pd.DataFrame,
        condition: Callable[[pd.DataFrame], pd.Series],
    ) -> pd.DataFrame:
        """Filter DataFrame using a custom condition function."""
        mask = condition(df)
        return df[mask].reset_index(drop=True)

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        columns: list[str],
        method: Literal["zscore", "iqr"] = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Remove outliers from specified columns.

        Args:
            df: Input DataFrame.
            columns: Columns to check for outliers.
            method: 'zscore' or 'iqr'.
            threshold: Threshold for outlier detection.

        Returns:
            DataFrame with outliers removed.
        """
        result = df.copy()
        mask = pd.Series(True, index=result.index)

        for col in columns:
            if col not in result.columns:
                continue

            if method == "zscore":
                z = np.abs((result[col] - result[col].mean()) / result[col].std())
                mask &= z < threshold
            else:  # IQR
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                mask &= (result[col] >= Q1 - threshold * IQR) & (result[col] <= Q3 + threshold * IQR)

        return result[mask].reset_index(drop=True)


# =============================================================================
# Data Balancing
# =============================================================================

class BalancingStrategy:
    """
    Utilities for handling class imbalance via oversampling/undersampling.

    Supports multiple strategies:
    - oversampling_copy: Random duplication of minority samples
    - oversampling_augmentation: Duplication with Gaussian noise
    - undersampling_standard: Random removal of majority samples
    - undersampling_easy_data: Remove easy-to-classify majority samples
    - smote: Synthetic minority oversampling
    """

    @staticmethod
    def oversampling_copy(
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Random oversampling by duplicating minority class samples.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_resampled, y_resampled) as numpy arrays.
        """
        from imblearn.over_sampling import RandomOverSampler

        # Convert to numpy if needed
        X_arr: np.ndarray = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr: np.ndarray = y.values if isinstance(y, pd.Series) else np.asarray(y)

        sampler = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_arr, y_arr)

        return X_resampled, y_resampled

    @staticmethod
    def oversampling_augmentation(
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        noise_std: float = 0.1,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Oversampling with Gaussian noise augmentation for minority classes.

        Generates synthetic samples by adding small Gaussian noise to duplicated
        minority samples, creating more diverse training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
            noise_std: Standard deviation of Gaussian noise (relative to feature std).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_resampled, y_resampled) as numpy arrays.
        """
        # Convert to numpy if needed
        X_arr: np.ndarray = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr: np.ndarray = y.values if isinstance(y, pd.Series) else np.asarray(y)

        rng = np.random.RandomState(random_state)

        # Get class counts
        unique_classes, class_counts = np.unique(y_arr, return_counts=True)
        max_count: int = int(class_counts.max())

        X_resampled_list: list[np.ndarray] = [X_arr]
        y_resampled_list: list[np.ndarray] = [y_arr]

        # Compute feature-wise std for noise scaling
        feature_std: np.ndarray = np.std(X_arr, axis=0) + 1e-8

        for cls, count in zip(unique_classes, class_counts):
            if count < max_count:
                # Get minority class samples
                cls_mask: np.ndarray = y_arr == cls
                X_cls: np.ndarray = X_arr[cls_mask]

                # Number of samples to generate
                n_generate: int = max_count - count

                # Random selection with replacement
                indices: np.ndarray = rng.choice(len(X_cls), size=n_generate, replace=True)
                X_new: np.ndarray = X_cls[indices].copy()

                # Add Gaussian noise scaled by feature std
                noise: np.ndarray = rng.normal(0, noise_std, X_new.shape) * feature_std
                X_new = X_new + noise

                X_resampled_list.append(X_new)
                y_resampled_list.append(np.full(n_generate, cls))

        X_resampled: np.ndarray = np.vstack(X_resampled_list)
        y_resampled: np.ndarray = np.concatenate(y_resampled_list)

        # Shuffle the resampled data
        shuffle_idx: np.ndarray = rng.permutation(len(y_resampled))

        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]

    @staticmethod
    def undersampling_standard(
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Random undersampling to balance classes.

        Reduces majority classes to match the minority class size by random removal.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_resampled, y_resampled) as numpy arrays.
        """
        from imblearn.under_sampling import RandomUnderSampler

        # Convert to numpy if needed
        X_arr: np.ndarray = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr: np.ndarray = y.values if isinstance(y, pd.Series) else np.asarray(y)

        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_arr, y_arr)

        return X_resampled, y_resampled

    @staticmethod
    def undersampling_easy_data(
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        n_neighbors: int = 3,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Undersampling by removing easy-to-classify majority samples.

        Uses Edited Nearest Neighbors (ENN) to remove majority class samples
        that are far from decision boundaries (easy samples), keeping harder
        samples that are more informative for learning.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
            n_neighbors: Number of neighbors for ENN algorithm.
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_resampled, y_resampled) as numpy arrays.
        """
        from imblearn.under_sampling import EditedNearestNeighbours

        # Convert to numpy if needed
        X_arr: np.ndarray = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr: np.ndarray = y.values if isinstance(y, pd.Series) else np.asarray(y)

        sampler = EditedNearestNeighbours(n_neighbors=n_neighbors)
        X_resampled, y_resampled = sampler.fit_resample(X_arr, y_arr)

        return X_resampled, y_resampled

    @staticmethod
    def smote(
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        SMOTE (Synthetic Minority Over-sampling Technique).

        Creates synthetic samples for minority classes by interpolating
        between existing minority samples and their nearest neighbors.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Label array of shape (n_samples,).
            random_state: Random seed for reproducibility.

        Returns:
            Tuple of (X_resampled, y_resampled) as numpy arrays.
        """
        from imblearn.over_sampling import SMOTE

        # Convert to numpy if needed
        X_arr: np.ndarray = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr: np.ndarray = y.values if isinstance(y, pd.Series) else np.asarray(y)

        sampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_arr, y_arr)

        return X_resampled, y_resampled

    @staticmethod
    def undersample(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Legacy method: Random undersampling returning DataFrames.

        Deprecated: Use undersampling_standard() for numpy arrays.
        """
        from imblearn.under_sampling import RandomUnderSampler

        sampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    @staticmethod
    def oversample(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Legacy method: Random oversampling returning DataFrames.

        Deprecated: Use oversampling_copy() for numpy arrays.
        """
        from imblearn.over_sampling import RandomOverSampler

        sampler = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    @staticmethod
    def get_class_weights(y: np.ndarray | pd.Series) -> dict[int, float]:
        """
        Compute class weights inversely proportional to class frequencies.

        Useful for weighted loss functions in ML models.

        Args:
            y: Label array.

        Returns:
            Dictionary mapping class labels to weights.
        """
        from sklearn.utils.class_weight import compute_class_weight

        y_arr: np.ndarray = y.values if isinstance(y, pd.Series) else np.asarray(y)
        classes: np.ndarray = np.unique(y_arr)
        weights: np.ndarray = compute_class_weight("balanced", classes=classes, y=y_arr)

        return dict(zip(classes.tolist(), weights.tolist()))


# =============================================================================
# Utility Functions
# =============================================================================

def get_label_distribution(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Get label distribution with counts and percentages."""
    counts = df[label_col].value_counts()
    percentages = df[label_col].value_counts(normalize=True) * 100

    return pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2),
    })


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get a comprehensive summary of the DataFrame."""
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values": df.isnull().sum().sum(),
        "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
    }


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split to maintain class distribution.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# =============================================================================
# ML-Ready Data Loading
# =============================================================================

@dataclass
class MLDataset:
    """
    Container for ML-ready dataset with train/val/test splits.

    All arrays are numpy ndarrays ready for model training.

    Attributes:
        X_train: Training features of shape (n_train, n_features).
        X_val: Validation features of shape (n_val, n_features).
        X_test: Test features of shape (n_test, n_features).
        y_train: Training labels of shape (n_train,).
        y_val: Validation labels of shape (n_val,).
        y_test: Test labels of shape (n_test,).
        feature_names: List of feature column names.
        label_encoder: Fitted LabelEncoder for label decoding.
        scaler: Fitted StandardScaler for feature denormalization.
        class_names: List of original class names in order.
        n_classes: Number of unique classes.
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    label_encoder: LabelEncoder
    scaler: StandardScaler | None
    class_names: list[str]
    n_classes: int


def load_ml_ready_data(
    dataset_type: Literal["physical", "network"] = "physical",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    balancing: Literal[
        "none",
        "oversampling_copy",
        "oversampling_augmentation",
        "undersampling_standard",
        "undersampling_easy_data",
        "smote",
    ] = "none",
    normalize: bool = True,
    n_samples: int | None = None,
    random_state: int = 42,
    noise_std: float = 0.1,
    datasets: list[str] | None = None,
    nrows: int | None = None,
    use_cache: bool = True,
    cache_dir: str | Path = "cached_datasets",
    show_progress: bool = True,
) -> MLDataset:
    """
    Load fully processed data ready for machine learning.

    Performs complete data pipeline:
    1. Check cache for preprocessed data (if use_cache=True)
    2. Load raw data from CSV files
    3. Clean and preprocess (handle missing values, encode labels)
    4. Split into train/val/test sets (stratified)
    5. Apply optional balancing strategy to training set only
    6. Normalize features (optional)
    7. Cache the result for future use
    8. Return numpy arrays ready for model training

    Args:
        dataset_type: Type of dataset to load ('physical' or 'network').
        train_ratio: Proportion of data for training (default 0.7).
        val_ratio: Proportion of data for validation (default 0.15).
        test_ratio: Proportion of data for testing (default 0.15).
        balancing: Balancing strategy to apply to training set.
        normalize: Whether to standardize features (fit on train, transform all).
        n_samples: Limit total number of samples (useful for quick experiments).
        random_state: Random seed for reproducibility.
        noise_std: Noise std for oversampling_augmentation strategy.
        datasets: Specific dataset files to load (e.g., ['normal', 'attack_1']).
        nrows: For network data, limit rows per file.
        use_cache: If True, cache preprocessed data and load from cache if available.
        cache_dir: Directory for cached datasets (default: 'cached_datasets').
        show_progress: If True, show tqdm progress bars during processing.

    Returns:
        MLDataset object containing all processed data and metadata.

    Raises:
        ValueError: If ratios don't sum to ~1.0 or invalid parameters.

    Example:
        >>> data = load_ml_ready_data(
        ...     dataset_type="physical",
        ...     balancing="oversampling_copy",
        ...     normalize=True,
        ... )
        >>> print(f"Training samples: {len(data.X_train)}")
        >>> print(f"Classes: {data.class_names}")
    """
    # Validate ratios
    ratio_sum: float = train_ratio + val_ratio + test_ratio
    if not (0.99 < ratio_sum < 1.01):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

    # Generate cache key based on all parameters
    cache_params = {
        "dataset_type": dataset_type,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "balancing": balancing,
        "normalize": normalize,
        "n_samples": n_samples,
        "random_state": random_state,
        "noise_std": noise_std,
        "datasets": sorted(datasets) if datasets else None,
        "nrows": nrows,
    }
    cache_hash = hashlib.md5(str(cache_params).encode()).hexdigest()[:8]
    nrows_str = str(nrows) if nrows else "all"
    cache_filename = f"{dataset_type}_{nrows_str}_{random_state}_{cache_hash}.pkl"
    cache_path = Path(cache_dir) / cache_filename

    result = None

    # Try to load from cache
    if use_cache and cache_path.exists():
        if show_progress:
            print(f"Loading from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            result = pickle.load(f)

    if result is None:
        # Progress wrapper for showing steps
        def progress_step(desc: str, total: int = 1):
            if show_progress:
                return tqdm(range(total), desc=desc, leave=False)
            return range(total)

        config = DataConfig()

        # Step 1: Load raw data
        for _ in progress_step("Loading raw data"):
            if dataset_type == "physical":
                loader = PhysicalDataLoader(config)
                df: pd.DataFrame = loader.load(datasets=datasets)
                label_col: str = "Label"
                exclude_cols: set[str] = {"Time", "Label", "Label_n", "Lable_n", "source_file"}
            else:
                loader_net = NetworkDataLoader(config)
                df = loader_net.load(datasets=datasets, nrows=nrows)
                label_col = "label"
                exclude_cols = {"Time", "label", "label_n", "source_file",
                                "mac_s", "mac_d", "ip_s", "ip_d", "proto",
                                "flags", "modbus_fn", "modbus_response"}

        # Step 2: Sample and clean
        for _ in progress_step("Sampling and cleaning data"):
            if n_samples is not None and len(df) > n_samples:
                df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
            df = df.dropna(subset=[label_col]).reset_index(drop=True)

        # Step 3: Extract features
        for _ in progress_step("Extracting features"):
            feature_cols: list[str] = [
                c for c in df.columns
                if c not in exclude_cols and df[c].dtype in [np.int64, np.float64, np.int32, np.float32]
            ]
            X: pd.DataFrame = df[feature_cols].copy()
            y: pd.Series = df[label_col].copy()
            X = X.fillna(0)

        # Step 4: Encode labels
        for _ in progress_step("Encoding labels"):
            label_encoder = LabelEncoder()
            y_encoded: np.ndarray = label_encoder.fit_transform(y)
            class_names: list[str] = label_encoder.classes_.tolist()
            n_classes: int = len(class_names)
            X_arr: np.ndarray = X.values.astype(np.float64)

        # Step 5: Stratified split
        for _ in progress_step("Stratified train/val/test split"):
            test_size_relative: float = test_ratio
            val_size_relative: float = val_ratio / (train_ratio + val_ratio)

            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X_arr, y_encoded,
                test_size=test_size_relative,
                random_state=random_state,
                stratify=y_encoded,
            )

            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval,
                test_size=val_size_relative,
                random_state=random_state,
                stratify=y_trainval,
            )

        # Step 6: Apply balancing
        for _ in progress_step(f"Applying balancing ({balancing})"):
            n_unique_classes: int = len(np.unique(y_train))
            if balancing != "none" and n_unique_classes > 1:
                balance_func = {
                    "oversampling_copy": BalancingStrategy.oversampling_copy,
                    "oversampling_augmentation": lambda X, y, rs: BalancingStrategy.oversampling_augmentation(
                        X, y, noise_std=noise_std, random_state=rs
                    ),
                    "undersampling_standard": BalancingStrategy.undersampling_standard,
                    "undersampling_easy_data": BalancingStrategy.undersampling_easy_data,
                    "smote": BalancingStrategy.smote,
                }

                if balancing in balance_func:
                    X_train, y_train = balance_func[balancing](X_train, y_train, random_state)
            elif balancing != "none" and n_unique_classes <= 1:
                import warnings
                warnings.warn(
                    f"Balancing strategy '{balancing}' skipped: only {n_unique_classes} class(es) in training data. "
                    "Increase n_samples or nrows_per_file to include more classes."
                )

        # Step 7: Normalize
        for _ in progress_step("Normalizing features"):
            scaler: StandardScaler | None = None
            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

        # Create result
        result = MLDataset(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_cols,
            label_encoder=label_encoder,
            scaler=scaler,
            class_names=class_names,
            n_classes=n_classes,
        )

        # Step 8: Cache result
        if use_cache:
            for _ in progress_step("Caching processed data"):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                if show_progress:
                    print(f"Cached to: {cache_path}")

    # Print dataset statistics
    if show_progress:
        print("\nDataset Split Statistics:")
        print("-" * 60)
        splits = [
            ("Train", result.y_train),
            ("Val", result.y_val),
            ("Test", result.y_test)
        ]

        for name, y_split in splits:
            total = len(y_split)
            print(f"{name} Set ({total} samples):")
            unique, counts = np.unique(y_split, return_counts=True)
            for label_idx, count in zip(unique, counts):
                 label_name = result.class_names[label_idx]
                 print(f"  - {label_name:<20}: {count:6d} ({count/total:.1%})")
            print()
        print("-" * 60)

    return result

