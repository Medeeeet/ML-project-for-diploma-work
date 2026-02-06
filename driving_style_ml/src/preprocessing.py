"""
Data Preprocessing Module for Driving Style ML Project
=======================================================

This module handles all data preprocessing tasks including:
- Missing value imputation
- Feature scaling and normalization
- Label encoding
- Feature engineering (especially for time-series data)
- Outlier detection and handling

Author: [Your Name]
Project: Bachelor Diploma - Driving Style Assessment and Accident Risk Prediction
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import (
    PREPROCESSING_CONFIG,
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
    STYLE_LABEL_MAPPING,
)


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for driving behavior data.

    This class handles all preprocessing steps required to prepare raw driving
    data for machine learning models, including handling missing values,
    scaling features, encoding labels, and engineering new features.

    Attributes:
        config (dict): Preprocessing configuration parameters
        scalers (dict): Fitted scaler objects for each feature set
        encoders (dict): Fitted encoder objects for categorical features
        feature_names (list): Names of processed features
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the DataPreprocessor.

        Parameters:
        -----------
        config : dict, optional
            Custom preprocessing configuration. Uses default if not provided.
        """
        self.config = config or PREPROCESSING_CONFIG
        self.scalers: Dict[str, object] = {}
        self.encoders: Dict[str, object] = {}
        self.imputers: Dict[str, object] = {}
        self.feature_names: List[str] = []
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> "DataPreprocessor":
        """
        Fit the preprocessor on training data.

        Parameters:
        -----------
        df : pd.DataFrame
            Training data to fit on
        target_column : str, optional
            Name of target column to exclude from fitting

        Returns:
        --------
        self
            Fitted preprocessor instance
        """
        print("\n" + "=" * 60)
        print("FITTING PREPROCESSOR")
        print("=" * 60)

        # Identify column types
        self._identify_column_types(df, target_column)

        # Fit imputers for missing values
        self._fit_imputers(df)

        # Fit scalers for numeric features
        self._fit_scalers(df)

        # Fit encoders for categorical features
        self._fit_encoders(df)

        self._is_fitted = True
        print("\n✓ Preprocessor fitted successfully")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform

        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        print("\nTransforming data...")
        df_transformed = df.copy()

        # Handle missing values
        df_transformed = self._impute_missing(df_transformed)

        # Scale numeric features
        df_transformed = self._scale_features(df_transformed)

        # Encode categorical features
        df_transformed = self._encode_categoricals(df_transformed)

        print(f"✓ Transformed {len(df_transformed):,} samples")

        return df_transformed

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fit and transform data in one step.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to fit and transform
        target_column : str, optional
            Target column to exclude

        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        return self.fit(df, target_column).transform(df)

    def _identify_column_types(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> None:
        """Identify numeric and categorical columns."""
        exclude_cols = [target_column] if target_column else []

        self.numeric_columns = df.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.numeric_columns = [c for c in self.numeric_columns if c not in exclude_cols]

        self.categorical_columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.categorical_columns = [c for c in self.categorical_columns if c not in exclude_cols]

        print(f"  Numeric columns: {len(self.numeric_columns)}")
        print(f"  Categorical columns: {len(self.categorical_columns)}")

    def _fit_imputers(self, df: pd.DataFrame) -> None:
        """Fit imputers for missing value handling."""
        strategy = self.config.get("missing_value_strategy", "median")

        if len(self.numeric_columns) > 0:
            if strategy == "knn":
                self.imputers["numeric"] = KNNImputer(n_neighbors=5)
            else:
                self.imputers["numeric"] = SimpleImputer(strategy=strategy)

            self.imputers["numeric"].fit(df[self.numeric_columns])
            print(f"  Fitted numeric imputer (strategy: {strategy})")

        if len(self.categorical_columns) > 0:
            self.imputers["categorical"] = SimpleImputer(
                strategy="most_frequent"
            )
            self.imputers["categorical"].fit(df[self.categorical_columns])
            print("  Fitted categorical imputer (strategy: most_frequent)")

    def _fit_scalers(self, df: pd.DataFrame) -> None:
        """Fit scalers for numeric features."""
        method = self.config.get("scaling_method", "standard")

        if len(self.numeric_columns) == 0:
            return

        if method == "standard":
            self.scalers["numeric"] = StandardScaler()
        elif method == "minmax":
            self.scalers["numeric"] = MinMaxScaler()
        elif method == "robust":
            self.scalers["numeric"] = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # Handle missing values before fitting scaler
        df_numeric = df[self.numeric_columns].copy()
        if df_numeric.isnull().any().any():
            df_numeric = pd.DataFrame(
                self.imputers["numeric"].transform(df_numeric),
                columns=self.numeric_columns,
            )

        self.scalers["numeric"].fit(df_numeric)
        print(f"  Fitted scaler (method: {method})")

    def _fit_encoders(self, df: pd.DataFrame) -> None:
        """Fit encoders for categorical features."""
        for col in self.categorical_columns:
            encoder = LabelEncoder()
            # Handle missing values
            values = df[col].fillna("MISSING").astype(str)
            encoder.fit(values)
            self.encoders[col] = encoder
            print(f"  Fitted encoder for '{col}' ({len(encoder.classes_)} classes)")

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value imputation."""
        df = df.copy()

        if "numeric" in self.imputers and len(self.numeric_columns) > 0:
            cols_present = [c for c in self.numeric_columns if c in df.columns]
            if cols_present:
                df[cols_present] = self.imputers["numeric"].transform(df[cols_present])

        if "categorical" in self.imputers and len(self.categorical_columns) > 0:
            cols_present = [c for c in self.categorical_columns if c in df.columns]
            if cols_present:
                df[cols_present] = self.imputers["categorical"].transform(df[cols_present])

        return df

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling."""
        df = df.copy()

        if "numeric" in self.scalers and len(self.numeric_columns) > 0:
            cols_present = [c for c in self.numeric_columns if c in df.columns]
            if cols_present:
                df[cols_present] = self.scalers["numeric"].transform(df[cols_present])

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding."""
        df = df.copy()

        for col, encoder in self.encoders.items():
            if col in df.columns:
                values = df[col].fillna("MISSING").astype(str)
                # Handle unseen categories
                known_classes = set(encoder.classes_)
                values = values.apply(
                    lambda x: x if x in known_classes else "MISSING"
                )
                df[col] = encoder.transform(values)

        return df


# =============================================================================
# MISSING VALUE HANDLING FUNCTIONS
# =============================================================================

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    threshold: float = 0.5,
    numeric_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with missing values
    strategy : str
        Imputation strategy: 'mean', 'median', 'mode', 'drop', 'knn'
    threshold : float
        Drop columns with missing percentage above this threshold
    numeric_cols : list, optional
        Specific numeric columns to process
    categorical_cols : list, optional
        Specific categorical columns to process

    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
    print("\n" + "-" * 40)
    print("Handling Missing Values")
    print("-" * 40)

    df = df.copy()

    # Report initial missing values
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before: {missing_before:,}")

    # Drop columns with too many missing values
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >{threshold*100}% missing")
        df = df.drop(columns=cols_to_drop)

    # Identify column types if not provided
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Handle numeric columns
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "knn":
        if numeric_cols:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    else:
        if numeric_cols:
            imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Handle categorical columns (always use mode)
    if categorical_cols:
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown"
                df[col] = df[col].fillna(mode_value)

    missing_after = df.isnull().sum().sum()
    print(f"Total missing values after: {missing_after:,}")
    print(f"Rows remaining: {len(df):,}")

    return df


# =============================================================================
# FEATURE SCALING FUNCTIONS
# =============================================================================

def scale_features(
    df: pd.DataFrame,
    method: str = "standard",
    columns: Optional[List[str]] = None,
    return_scaler: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, object]]:
    """
    Scale numeric features using specified method.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    method : str
        Scaling method: 'standard', 'minmax', 'robust'
    columns : list, optional
        Specific columns to scale. If None, scales all numeric columns.
    return_scaler : bool
        Whether to return the fitted scaler object

    Returns:
    --------
    pd.DataFrame or Tuple[pd.DataFrame, scaler]
        Scaled DataFrame and optionally the scaler
    """
    print(f"\nScaling features using {method} method...")

    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(columns) == 0:
        print("No numeric columns to scale")
        return (df, None) if return_scaler else df

    # Select scaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    # Fit and transform
    df[columns] = scaler.fit_transform(df[columns])

    print(f"✓ Scaled {len(columns)} columns")

    if return_scaler:
        return df, scaler
    return df


# =============================================================================
# LABEL ENCODING FUNCTIONS
# =============================================================================

def encode_labels(
    df: pd.DataFrame,
    target_column: str,
    label_mapping: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Encode categorical labels to numeric values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Name of the target column to encode
    label_mapping : dict, optional
        Custom mapping from labels to integers

    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        DataFrame with encoded labels and the mapping used
    """
    print(f"\nEncoding labels for column: {target_column}")

    df = df.copy()

    if label_mapping is None:
        # Create automatic mapping
        unique_labels = df[target_column].unique()
        label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # Apply mapping
    encoded_col = f"{target_column}_encoded"
    df[encoded_col] = df[target_column].map(label_mapping)

    # Handle unmapped values
    unmapped = df[encoded_col].isnull().sum()
    if unmapped > 0:
        print(f"⚠ {unmapped} values could not be mapped")
        # Assign to a new class
        max_label = max(label_mapping.values())
        df[encoded_col] = df[encoded_col].fillna(max_label + 1)

    print(f"✓ Encoded {len(label_mapping)} classes: {label_mapping}")

    return df, label_mapping


def encode_categorical_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "label",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Encode categorical features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, optional
        Columns to encode. If None, encodes all object columns.
    method : str
        Encoding method: 'label' or 'onehot'

    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        Encoded DataFrame and encoder objects
    """
    df = df.copy()
    encoders = {}

    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\nEncoding {len(columns)} categorical columns using {method} encoding...")

    for col in columns:
        if method == "label":
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str))
            encoders[col] = encoder
        elif method == "onehot":
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            encoders[col] = list(dummies.columns)

    print(f"✓ Encoded {len(columns)} columns")

    return df, encoders


# =============================================================================
# FEATURE ENGINEERING FUNCTIONS
# =============================================================================

class FeatureEngineer:
    """
    Feature engineering class for creating derived features from driving data.

    This class provides methods to create meaningful features from raw sensor
    data, including statistical aggregations, time-series features, and
    domain-specific driving behavior indicators.
    """

    def __init__(self, window_sizes: List[int] = [5, 10, 20]):
        """
        Initialize the FeatureEngineer.

        Parameters:
        -----------
        window_sizes : list
            Window sizes for rolling statistics
        """
        self.window_sizes = window_sizes

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        window_sizes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Create rolling window statistics for time-series features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with time-series data
        columns : list
            Columns to compute rolling statistics for
        window_sizes : list, optional
            Window sizes to use

        Returns:
        --------
        pd.DataFrame
            DataFrame with added rolling features
        """
        df = df.copy()
        windows = window_sizes or self.window_sizes

        print(f"\nCreating rolling features for {len(columns)} columns...")

        for col in columns:
            if col not in df.columns:
                continue

            for window in windows:
                # Rolling mean
                df[f"{col}_rolling_mean_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).mean()
                )

                # Rolling standard deviation
                df[f"{col}_rolling_std_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).std()
                )

                # Rolling min and max
                df[f"{col}_rolling_min_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).min()
                )
                df[f"{col}_rolling_max_{window}"] = (
                    df[col].rolling(window=window, min_periods=1).max()
                )

        # Fill NaN values from rolling operations
        df = df.fillna(method="bfill").fillna(method="ffill")

        new_cols = len(df.columns) - len(columns)
        print(f"✓ Created {new_cols} rolling features")

        return df

    def create_statistical_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        group_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create statistical aggregation features.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        columns : list
            Columns to compute statistics for
        group_by : str, optional
            Column to group by (e.g., trip_id, driver_id)

        Returns:
        --------
        pd.DataFrame
            DataFrame with statistical features
        """
        df = df.copy()

        print(f"\nCreating statistical features for {len(columns)} columns...")

        if group_by and group_by in df.columns:
            # Compute per-group statistics
            agg_funcs = ["mean", "std", "min", "max", "median"]
            stats_df = df.groupby(group_by)[columns].agg(agg_funcs)
            stats_df.columns = [f"{col}_{stat}" for col, stat in stats_df.columns]
            df = df.merge(stats_df, on=group_by, how="left")
        else:
            # Compute global statistics as features
            for col in columns:
                if col not in df.columns:
                    continue
                df[f"{col}_zscore"] = stats.zscore(df[col].fillna(df[col].mean()))

        return df

    def create_acceleration_features(
        self,
        df: pd.DataFrame,
        accel_cols: List[str] = ["acceleration_x", "acceleration_y", "acceleration_z"],
    ) -> pd.DataFrame:
        """
        Create derived acceleration features.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with acceleration columns
        accel_cols : list
            Names of acceleration columns [x, y, z]

        Returns:
        --------
        pd.DataFrame
            DataFrame with derived acceleration features
        """
        df = df.copy()

        # Check if columns exist
        existing_cols = [c for c in accel_cols if c in df.columns]
        if len(existing_cols) < 3:
            print(f"⚠ Not all acceleration columns found. Found: {existing_cols}")
            return df

        print("\nCreating acceleration-derived features...")

        ax, ay, az = accel_cols

        # Total acceleration magnitude
        df["accel_magnitude"] = np.sqrt(
            df[ax]**2 + df[ay]**2 + df[az]**2
        )

        # Horizontal acceleration (driving plane)
        df["accel_horizontal"] = np.sqrt(df[ax]**2 + df[ay]**2)

        # Jerk (rate of change of acceleration)
        df["jerk_x"] = df[ax].diff()
        df["jerk_y"] = df[ay].diff()
        df["jerk_z"] = df[az].diff()
        df["jerk_magnitude"] = np.sqrt(
            df["jerk_x"]**2 + df["jerk_y"]**2 + df["jerk_z"]**2
        )

        # Harsh events detection
        accel_threshold = 3.0  # m/s^2
        df["harsh_event"] = (df["accel_magnitude"] > accel_threshold).astype(int)

        print("✓ Created acceleration-derived features")

        return df.fillna(0)

    def create_gyroscope_features(
        self,
        df: pd.DataFrame,
        gyro_cols: List[str] = ["gyroscope_x", "gyroscope_y", "gyroscope_z"],
    ) -> pd.DataFrame:
        """
        Create derived gyroscope features.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with gyroscope columns
        gyro_cols : list
            Names of gyroscope columns [x, y, z]

        Returns:
        --------
        pd.DataFrame
            DataFrame with derived gyroscope features
        """
        df = df.copy()

        existing_cols = [c for c in gyro_cols if c in df.columns]
        if len(existing_cols) < 3:
            print(f"⚠ Not all gyroscope columns found. Found: {existing_cols}")
            return df

        print("\nCreating gyroscope-derived features...")

        gx, gy, gz = gyro_cols

        # Total rotational velocity
        df["gyro_magnitude"] = np.sqrt(
            df[gx]**2 + df[gy]**2 + df[gz]**2
        )

        # Yaw rate (rotation around vertical axis)
        df["yaw_rate"] = df[gz].abs()

        # Sharp turn detection
        turn_threshold = 0.5  # rad/s
        df["sharp_turn"] = (df["yaw_rate"] > turn_threshold).astype(int)

        print("✓ Created gyroscope-derived features")

        return df.fillna(0)

    def create_speed_features(
        self,
        df: pd.DataFrame,
        speed_col: str = "speed",
        speed_limit: float = 120.0,
    ) -> pd.DataFrame:
        """
        Create speed-related features.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with speed column
        speed_col : str
            Name of speed column
        speed_limit : float
            Reference speed limit for violation detection

        Returns:
        --------
        pd.DataFrame
            DataFrame with speed-derived features
        """
        df = df.copy()

        if speed_col not in df.columns:
            print(f"⚠ Speed column '{speed_col}' not found")
            return df

        print("\nCreating speed-derived features...")

        # Speed change (acceleration proxy)
        df["speed_change"] = df[speed_col].diff()

        # Speeding indicator
        df["speeding"] = (df[speed_col] > speed_limit).astype(int)
        df["speed_over_limit"] = np.maximum(0, df[speed_col] - speed_limit)

        # Sudden speed changes
        speed_change_threshold = 10.0  # km/h or m/s
        df["sudden_speed_change"] = (
            df["speed_change"].abs() > speed_change_threshold
        ).astype(int)

        print("✓ Created speed-derived features")

        return df.fillna(0)

    def create_frequency_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_components: int = 5,
    ) -> pd.DataFrame:
        """
        Create frequency domain features using FFT.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with time-series data
        columns : list
            Columns to apply FFT
        n_components : int
            Number of frequency components to extract

        Returns:
        --------
        pd.DataFrame
            DataFrame with frequency features
        """
        df = df.copy()

        print(f"\nCreating frequency features for {len(columns)} columns...")

        for col in columns:
            if col not in df.columns:
                continue

            # Apply FFT
            signal = df[col].fillna(0).values
            fft_values = np.abs(fft(signal))

            # Extract top frequency components
            for i in range(n_components):
                df[f"{col}_freq_{i}"] = fft_values[i] if i < len(fft_values) else 0

            # Dominant frequency magnitude
            df[f"{col}_max_freq"] = np.max(fft_values[:len(fft_values)//2])

        print(f"✓ Created frequency features")

        return df

    def create_all_features(
        self,
        df: pd.DataFrame,
        sensor_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Apply all feature engineering methods.

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        sensor_columns : list, optional
            Sensor columns to process

        Returns:
        --------
        pd.DataFrame
            DataFrame with all engineered features
        """
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60)

        if sensor_columns is None:
            sensor_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Apply all feature engineering methods
        df = self.create_rolling_features(df, sensor_columns[:5])  # Limit for efficiency
        df = self.create_acceleration_features(df)
        df = self.create_gyroscope_features(df)
        df = self.create_speed_features(df)

        print(f"\n✓ Feature engineering complete. Total features: {len(df.columns)}")

        return df


# =============================================================================
# OUTLIER HANDLING FUNCTIONS
# =============================================================================

def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Detect outliers in numeric columns.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, optional
        Columns to check for outliers
    method : str
        Detection method: 'iqr', 'zscore'
    threshold : float
        IQR multiplier or z-score threshold

    Returns:
    --------
    pd.DataFrame
        Boolean DataFrame indicating outliers
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outliers = pd.DataFrame(index=df.index)

    for col in columns:
        if col not in df.columns:
            continue

        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers[col] = (
                (df[col] < Q1 - threshold * IQR) |
                (df[col] > Q3 + threshold * IQR)
            )
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))
            outliers[col] = z_scores > threshold

    return outliers


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "clip",
    detection_method: str = "iqr",
    threshold: float = 1.5,
) -> pd.DataFrame:
    """
    Handle outliers in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, optional
        Columns to process
    method : str
        Handling method: 'clip', 'remove', 'median'
    detection_method : str
        Outlier detection method
    threshold : float
        Detection threshold

    Returns:
    --------
    pd.DataFrame
        DataFrame with handled outliers
    """
    print(f"\nHandling outliers using {method} method...")

    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_mask = detect_outliers(df, columns, detection_method, threshold)
    total_outliers = outlier_mask.sum().sum()
    print(f"Detected {total_outliers:,} outliers across {len(columns)} columns")

    if method == "clip":
        for col in columns:
            if col not in df.columns:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)

    elif method == "remove":
        any_outlier = outlier_mask.any(axis=1)
        df = df[~any_outlier]
        print(f"Removed {any_outlier.sum():,} rows with outliers")

    elif method == "median":
        for col in columns:
            if col not in df.columns:
                continue
            median = df[col].median()
            df.loc[outlier_mask[col], col] = median

    print("✓ Outliers handled")

    return df


# =============================================================================
# DATA SPLITTING FUNCTIONS
# =============================================================================

def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = TEST_SIZE,
    val_size: float = VALIDATION_SIZE,
    stratify: bool = True,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_column : str
        Name of target column
    test_size : float
        Proportion for test set
    val_size : float
        Proportion of training data for validation
    stratify : bool
        Whether to use stratified splitting
    random_state : int
        Random seed

    Returns:
    --------
    Tuple
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n" + "-" * 40)
    print("Splitting Data")
    print("-" * 40)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First split: train+val vs test
    stratify_col = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state,
    )

    # Second split: train vs val
    stratify_col = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=stratify_col,
        random_state=random_state,
    )

    print(f"Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Check class distribution
    if stratify:
        print("\nClass distribution:")
        for name, data in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
            dist = data.value_counts(normalize=True)
            print(f"  {name}: {dict(dist.round(3))}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

def preprocess_pipeline(
    df: pd.DataFrame,
    target_column: str,
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, DataPreprocessor]:
    """
    Complete preprocessing pipeline.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw input data
    target_column : str
        Name of target column
    config : dict, optional
        Preprocessing configuration

    Returns:
    --------
    Tuple
        Preprocessed train, val, test data and fitted preprocessor
    """
    config = config or PREPROCESSING_CONFIG

    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Handle missing values
    df = handle_missing_values(
        df,
        strategy=config.get("missing_value_strategy", "median"),
        threshold=config.get("missing_threshold", 0.5),
    )

    # Step 2: Feature engineering
    fe = FeatureEngineer(window_sizes=config.get("rolling_window_sizes", [5, 10, 20]))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    df = fe.create_rolling_features(df, numeric_cols[:5])

    # Step 3: Handle outliers
    df = handle_outliers(
        df,
        method="clip",
        detection_method=config.get("outlier_method", "iqr"),
        threshold=config.get("outlier_threshold", 1.5),
    )

    # Step 4: Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, target_column
    )

    # Step 5: Fit preprocessor on training data and transform all sets
    preprocessor = DataPreprocessor(config)
    preprocessor.fit(X_train)

    X_train = preprocessor.transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Final feature count: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Demonstration
    print("\n" + "=" * 60)
    print("PREPROCESSING MODULE DEMONSTRATION")
    print("=" * 60)

    # Create sample data
    np.random.seed(RANDOM_SEED)
    n_samples = 1000

    sample_data = pd.DataFrame({
        "acceleration_x": np.random.normal(0, 1, n_samples),
        "acceleration_y": np.random.normal(0, 1, n_samples),
        "acceleration_z": np.random.normal(9.8, 0.5, n_samples),
        "speed": np.random.exponential(50, n_samples),
        "gyroscope_x": np.random.normal(0, 0.1, n_samples),
        "gyroscope_y": np.random.normal(0, 0.1, n_samples),
        "gyroscope_z": np.random.normal(0, 0.1, n_samples),
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "driving_style": np.random.choice(["safe", "normal", "aggressive"], n_samples, p=[0.6, 0.25, 0.15]),
    })

    # Add some missing values
    sample_data.loc[np.random.choice(n_samples, 50), "acceleration_x"] = np.nan
    sample_data.loc[np.random.choice(n_samples, 30), "speed"] = np.nan

    print(f"\nSample data shape: {sample_data.shape}")
    print(f"Missing values:\n{sample_data.isnull().sum()}")

    # Run preprocessing pipeline
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor = preprocess_pipeline(
        sample_data,
        target_column="driving_style",
    )

    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
