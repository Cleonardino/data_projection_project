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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


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
    ) -> pd.DataFrame:
        """
        Random sample from each dataset file.
        
        Args:
            datasets: Dataset names to sample from.
            n_per_file: Number of rows to sample per file.
            random_state: Random seed for reproducibility.
        
        Returns:
            Combined sampled DataFrame.
        """
        if datasets is None:
            datasets = list(self.config.network_files.keys())
        
        samples = []
        for dataset in datasets:
            path = self.config.get_network_path(dataset)
            # Get total rows (fast with pandas)
            total_rows = sum(1 for _ in open(path)) - 1  # subtract header
            
            if total_rows <= n_per_file:
                df = pd.read_csv(path)
            else:
                # Random skip rows
                skip_idx = sorted(
                    np.random.RandomState(random_state).choice(
                        range(1, total_rows + 1), 
                        size=total_rows - n_per_file,
                        replace=False
                    )
                )
                df = pd.read_csv(path, skiprows=skip_idx)
            
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
    """Utilities for handling class imbalance via oversampling/undersampling."""
    
    @staticmethod
    def undersample(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Random undersampling to balance classes.
        
        Reduces majority classes to match the minority class size.
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
        Random oversampling to balance classes.
        
        Increases minority classes to match the majority class size.
        """
        from imblearn.over_sampling import RandomOverSampler
        
        sampler = RandomOverSampler(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    @staticmethod
    def smote(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        SMOTE (Synthetic Minority Over-sampling Technique).
        
        Creates synthetic samples for minority classes.
        """
        from imblearn.over_sampling import SMOTE
        
        sampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    @staticmethod
    def get_class_weights(y: pd.Series) -> dict:
        """
        Compute class weights inversely proportional to class frequencies.
        
        Useful for weighted loss functions in ML models.
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes, weights))


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
