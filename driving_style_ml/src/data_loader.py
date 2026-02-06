"""
Data Loading Module for Driving Style ML Project
=================================================

This module handles loading, validation, and initial inspection of all datasets
used in the driving style classification and accident risk prediction pipeline.

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    DATASET_PATHS,
    ARCHIVE_DIRS,
    RANDOM_SEED,
)


class DataLoader:
    """
    A class for loading and validating driving behavior datasets.

    This class provides methods to:
    - Load individual CSV files
    - Load and combine multiple datasets
    - Validate data structure and quality
    - Print comprehensive dataset information

    Attributes:
        datasets (Dict): Dictionary storing loaded DataFrames
        metadata (Dict): Dictionary storing dataset metadata
    """

    def __init__(self):
        """Initialize the DataLoader with empty containers."""
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, dict] = {}

    def load_csv(
        self,
        file_path: str,
        dataset_name: str,
        nrows: Optional[int] = None,
        usecols: Optional[List[str]] = None,
        dtype: Optional[dict] = None,
        parse_dates: Optional[List[str]] = None,
        low_memory: bool = False,
    ) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame with validation.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file
        dataset_name : str
            Identifier name for the dataset
        nrows : int, optional
            Number of rows to read (useful for large files)
        usecols : list, optional
            Specific columns to load
        dtype : dict, optional
            Column data types specification
        parse_dates : list, optional
            Columns to parse as datetime
        low_memory : bool
            Whether to use low memory mode for large files

        Returns:
        --------
        pd.DataFrame
            Loaded and validated DataFrame
        """
        print(f"\n{'='*60}")
        print(f"Loading dataset: {dataset_name}")
        print(f"{'='*60}")

        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        # Get file size for information
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File path: {file_path}")
        print(f"File size: {file_size_mb:.2f} MB")

        # Load the data
        try:
            df = pd.read_csv(
                file_path,
                nrows=nrows,
                usecols=usecols,
                dtype=dtype,
                parse_dates=parse_dates,
                low_memory=low_memory,
            )
            print(f"✓ Successfully loaded {len(df):,} rows and {len(df.columns)} columns")

        except Exception as e:
            print(f"✗ Error loading dataset: {str(e)}")
            raise

        # Store dataset and metadata
        self.datasets[dataset_name] = df
        self.metadata[dataset_name] = {
            "file_path": file_path,
            "file_size_mb": file_size_mb,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }

        return df

    def load_us_accidents(
        self,
        nrows: Optional[int] = None,
        selected_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load the US Accidents dataset.

        This dataset contains accident records with various features including
        location, weather conditions, road features, and severity.

        Parameters:
        -----------
        nrows : int, optional
            Limit number of rows to load
        selected_columns : list, optional
            Specific columns to load

        Returns:
        --------
        pd.DataFrame
            US Accidents dataset
        """
        # Define relevant columns for driving style analysis
        default_columns = [
            "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
            "Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)",
            "Wind_Speed(mph)", "Weather_Condition", "Amenity", "Bump", "Crossing",
            "Junction", "Railway", "Station", "Stop", "Traffic_Signal",
            "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight",
        ]

        columns = selected_columns or default_columns

        df = self.load_csv(
            file_path=DATASET_PATHS["us_accidents"],
            dataset_name="us_accidents",
            nrows=nrows,
            usecols=columns if os.path.exists(DATASET_PATHS["us_accidents"]) else None,
            parse_dates=["Start_Time", "End_Time"] if "Start_Time" in columns else None,
            low_memory=False,
        )

        return df

    def load_carla_data(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the CARLA simulator driving data.

        This dataset contains simulated driving behavior data from the
        CARLA autonomous driving simulator.

        Parameters:
        -----------
        nrows : int, optional
            Limit number of rows to load

        Returns:
        --------
        pd.DataFrame
            CARLA driving data
        """
        df = self.load_csv(
            file_path=DATASET_PATHS["carla_data"],
            dataset_name="carla_data",
            nrows=nrows,
            low_memory=False,
        )

        return df

    def load_eco_driving(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the Eco Driving Score dataset.

        This dataset contains driving efficiency metrics and eco-driving
        behavior indicators.

        Parameters:
        -----------
        nrows : int, optional
            Limit number of rows to load

        Returns:
        --------
        pd.DataFrame
            Eco driving score data
        """
        df = self.load_csv(
            file_path=DATASET_PATHS["eco_driving"],
            dataset_name="eco_driving",
            nrows=nrows,
            low_memory=False,
        )

        return df

    def load_driver_behavior(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load the Driver Behavior Route Anomaly dataset.

        This dataset contains route information and anomaly detection
        features with derived behavioral indicators.

        Parameters:
        -----------
        nrows : int, optional
            Limit number of rows to load

        Returns:
        --------
        pd.DataFrame
            Driver behavior data with derived features
        """
        df = self.load_csv(
            file_path=DATASET_PATHS["driver_behavior"],
            dataset_name="driver_behavior",
            nrows=nrows,
            low_memory=False,
        )

        return df

    def load_all_datasets(
        self,
        nrows: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets.

        Parameters:
        -----------
        nrows : int, optional
            Limit rows for each dataset (useful for testing)

        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of all loaded datasets
        """
        print("\n" + "=" * 60)
        print("LOADING ALL DATASETS")
        print("=" * 60)

        loaded_datasets = {}

        # Try to load each dataset, skip if file not found
        dataset_loaders = {
            "us_accidents": self.load_us_accidents,
            "carla_data": self.load_carla_data,
            "eco_driving": self.load_eco_driving,
            "driver_behavior": self.load_driver_behavior,
        }

        for name, loader in dataset_loaders.items():
            try:
                loaded_datasets[name] = loader(nrows=nrows)
            except FileNotFoundError as e:
                print(f"\n⚠ Skipping {name}: {str(e)}")
            except Exception as e:
                print(f"\n✗ Error loading {name}: {str(e)}")

        print(f"\n{'='*60}")
        print(f"Successfully loaded {len(loaded_datasets)} datasets")
        print("=" * 60)

        return loaded_datasets

    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        required_columns: Optional[List[str]] = None,
        check_duplicates: bool = True,
    ) -> Dict[str, any]:
        """
        Validate a dataset and return quality metrics.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to validate
        dataset_name : str
            Name of the dataset for reporting
        required_columns : list, optional
            Columns that must be present
        check_duplicates : bool
            Whether to check for duplicate rows

        Returns:
        --------
        Dict
            Validation results and quality metrics
        """
        print(f"\n{'='*60}")
        print(f"VALIDATING: {dataset_name}")
        print("=" * 60)

        validation_results = {
            "dataset_name": dataset_name,
            "is_valid": True,
            "issues": [],
        }

        # Check for required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                validation_results["is_valid"] = False
                validation_results["issues"].append(
                    f"Missing required columns: {missing_cols}"
                )
                print(f"✗ Missing columns: {missing_cols}")
            else:
                print("✓ All required columns present")

        # Check for empty dataset
        if len(df) == 0:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Dataset is empty")
            print("✗ Dataset is empty")
        else:
            print(f"✓ Dataset has {len(df):,} rows")

        # Check for duplicates
        if check_duplicates:
            duplicates = df.duplicated().sum()
            validation_results["duplicate_rows"] = duplicates
            if duplicates > 0:
                validation_results["issues"].append(
                    f"Found {duplicates:,} duplicate rows"
                )
                print(f"⚠ Found {duplicates:,} duplicate rows ({duplicates/len(df)*100:.2f}%)")
            else:
                print("✓ No duplicate rows found")

        # Check missing values
        missing_counts = df.isnull().sum()
        missing_pct = (missing_counts / len(df)) * 100
        cols_with_missing = missing_pct[missing_pct > 0]

        validation_results["missing_values"] = cols_with_missing.to_dict()

        if len(cols_with_missing) > 0:
            print(f"\n⚠ Columns with missing values:")
            for col, pct in cols_with_missing.items():
                status = "⚠" if pct < 50 else "✗"
                print(f"  {status} {col}: {pct:.2f}% missing")
        else:
            print("✓ No missing values found")

        # Data type summary
        print(f"\nData types:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  - {dtype}: {count} columns")

        validation_results["dtype_counts"] = dtype_counts.to_dict()

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        validation_results["memory_mb"] = memory_mb
        print(f"\nMemory usage: {memory_mb:.2f} MB")

        return validation_results

    def print_dataset_info(self, df: pd.DataFrame, dataset_name: str) -> None:
        """
        Print comprehensive information about a dataset.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset to analyze
        dataset_name : str
            Name of the dataset
        """
        print(f"\n{'='*60}")
        print(f"DATASET INFO: {dataset_name}")
        print("=" * 60)

        # Basic info
        print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Column information
        print(f"\nColumn Details:")
        print("-" * 60)

        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            null_pct = (df[col].isna().sum() / len(df)) * 100
            unique = df[col].nunique()

            print(f"  {col}")
            print(f"    Type: {dtype} | Non-null: {non_null:,} | "
                  f"Missing: {null_pct:.1f}% | Unique: {unique:,}")

            # Sample values for object columns
            if dtype == "object" and unique <= 10:
                sample_vals = df[col].dropna().unique()[:5]
                print(f"    Values: {list(sample_vals)}")

        # Numeric column statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nNumeric Columns Summary:")
            print("-" * 60)
            print(df[numeric_cols].describe().round(2).to_string())

    def get_target_distribution(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> pd.Series:
        """
        Get the distribution of target variable classes.

        Parameters:
        -----------
        df : pd.DataFrame
            Dataset containing target column
        target_column : str
            Name of the target column

        Returns:
        --------
        pd.Series
            Value counts of target classes
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        distribution = df[target_column].value_counts()
        percentages = df[target_column].value_counts(normalize=True) * 100

        print(f"\nTarget Distribution: {target_column}")
        print("-" * 40)
        for cls in distribution.index:
            print(f"  {cls}: {distribution[cls]:,} ({percentages[cls]:.2f}%)")

        # Check for class imbalance
        min_pct = percentages.min()
        if min_pct < 20:
            print(f"\n⚠ Class imbalance detected! Minority class: {min_pct:.2f}%")

        return distribution

    def combine_datasets(
        self,
        datasets: List[pd.DataFrame],
        method: str = "concat",
        on: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Combine multiple datasets into one.

        Parameters:
        -----------
        datasets : List[pd.DataFrame]
            List of DataFrames to combine
        method : str
            Combination method: 'concat' or 'merge'
        on : list, optional
            Columns to merge on (for merge method)

        Returns:
        --------
        pd.DataFrame
            Combined dataset
        """
        if method == "concat":
            combined = pd.concat(datasets, ignore_index=True)
            print(f"Combined {len(datasets)} datasets: {len(combined):,} total rows")

        elif method == "merge":
            if on is None:
                raise ValueError("'on' parameter required for merge method")
            combined = datasets[0]
            for df in datasets[1:]:
                combined = combined.merge(df, on=on, how="outer")
            print(f"Merged {len(datasets)} datasets: {len(combined):,} total rows")

        else:
            raise ValueError(f"Unknown method: {method}")

        return combined


def create_sample_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    imbalance_ratio: Optional[List[float]] = None,
    include_time_series: bool = True,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Create a synthetic driving behavior dataset for testing.

    This function generates realistic-looking driving data that can be used
    for testing the ML pipeline when real data is not available.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of numerical features
    n_classes : int
        Number of driving style classes
    imbalance_ratio : list, optional
        Class proportions (should sum to 1)
    include_time_series : bool
        Whether to include simulated time-series features
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Synthetic driving behavior dataset
    """
    np.random.seed(random_state)

    print("\nGenerating synthetic driving dataset...")

    # Generate class labels with optional imbalance
    if imbalance_ratio is None:
        imbalance_ratio = [0.6, 0.25, 0.15]  # safe, normal, aggressive

    labels = np.random.choice(
        n_classes,
        size=n_samples,
        p=imbalance_ratio[:n_classes],
    )

    # Generate features based on driving style
    data = {}

    # Speed-related features
    base_speed = 30 + labels * 15  # Aggressive drivers tend to speed
    data["avg_speed"] = base_speed + np.random.normal(0, 5, n_samples)
    data["max_speed"] = data["avg_speed"] + np.random.exponential(10, n_samples)
    data["speed_variance"] = np.abs(np.random.normal(5, 3, n_samples) + labels * 3)

    # Acceleration features
    data["harsh_braking_count"] = np.random.poisson(labels + 1, n_samples)
    data["harsh_accel_count"] = np.random.poisson(labels + 1, n_samples)
    data["avg_acceleration"] = np.random.normal(0, 0.5, n_samples) + labels * 0.2

    # Behavioral features
    data["lane_changes"] = np.random.poisson(labels * 2 + 1, n_samples)
    data["following_distance"] = np.maximum(0.5, 3 - labels + np.random.normal(0, 0.5, n_samples))
    data["turn_signal_usage"] = np.clip(0.9 - labels * 0.2 + np.random.normal(0, 0.1, n_samples), 0, 1)

    # Time-series derived features
    if include_time_series:
        data["accel_x_mean"] = np.random.normal(0, 0.3, n_samples)
        data["accel_y_mean"] = np.random.normal(0, 0.3, n_samples)
        data["accel_z_mean"] = np.random.normal(9.8, 0.2, n_samples)
        data["gyro_x_std"] = np.abs(np.random.normal(0.1, 0.05, n_samples) + labels * 0.05)
        data["gyro_y_std"] = np.abs(np.random.normal(0.1, 0.05, n_samples) + labels * 0.05)
        data["gyro_z_std"] = np.abs(np.random.normal(0.1, 0.05, n_samples) + labels * 0.05)

    # Contextual features
    data["time_of_day"] = np.random.randint(0, 24, n_samples)
    data["day_of_week"] = np.random.randint(0, 7, n_samples)
    data["trip_duration"] = np.random.exponential(30, n_samples)
    data["trip_distance"] = data["trip_duration"] * data["avg_speed"] / 60

    # Target variables
    class_names = ["safe", "normal", "aggressive"]
    data["driving_style"] = [class_names[l] for l in labels]
    data["driving_style_encoded"] = labels

    # Accident probability (higher for aggressive drivers)
    accident_prob = 0.05 + labels * 0.1
    data["accident"] = np.random.binomial(1, accident_prob)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add some missing values (realistic scenario)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices[:len(missing_indices)//2], "following_distance"] = np.nan
    df.loc[missing_indices[len(missing_indices)//2:], "turn_signal_usage"] = np.nan

    print(f"✓ Generated {n_samples:,} samples with {len(df.columns)} features")
    print(f"  Classes: {dict(zip(class_names, imbalance_ratio[:n_classes]))}")
    print(f"  Accident rate: {df['accident'].mean()*100:.1f}%")

    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Demonstration of data loading capabilities
    print("\n" + "=" * 60)
    print("DATA LOADER DEMONSTRATION")
    print("=" * 60)

    loader = DataLoader()

    # Try to load real datasets
    try:
        datasets = loader.load_all_datasets(nrows=1000)  # Limit rows for demo

        for name, df in datasets.items():
            loader.print_dataset_info(df, name)
            loader.validate_dataset(df, name)

    except Exception as e:
        print(f"\nCould not load real datasets: {e}")
        print("\nGenerating synthetic dataset for demonstration...")

        # Generate and inspect synthetic data
        synthetic_df = create_sample_dataset(n_samples=1000)
        loader.datasets["synthetic"] = synthetic_df
        loader.print_dataset_info(synthetic_df, "synthetic")
        loader.validate_dataset(synthetic_df, "synthetic")
        loader.get_target_distribution(synthetic_df, "driving_style")
