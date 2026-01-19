"""
Data preprocessing utilities for machine learning module.

This module handles data preparation, transformation, and balancing
for machine learning models.
"""

from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from fusion.modules.ml.constants import EXPECTED_ML_COLUMNS
from fusion.modules.ml.visualization import plot_data_distributions
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def process_training_data(simulation_dict: dict[str, Any], input_dataframe: pd.DataFrame, erlang: float) -> pd.DataFrame:
    """
    Process raw data for machine learning model training.

    Performs one-hot encoding and type conversions as needed.

    :param simulation_dict: Dictionary containing simulation parameters
    :type simulation_dict: Dict[str, Any]
    :param input_dataframe: Raw input DataFrame
    :type input_dataframe: pd.DataFrame
    :param erlang: Traffic volume value
    :type erlang: float
    :return: Processed DataFrame ready for training
    :rtype: pd.DataFrame

    Example:
        >>> sim_dict = {'train_file_path': 'experiment_001'}
        >>> raw_data = pd.DataFrame({'bandwidth': [50, 100], 'path_length': [10, 20]})
        >>> processed = process_training_data(sim_dict, raw_data, 1000.0)
    """
    # Generate visualizations of the input data
    plot_data_distributions(simulation_dict=simulation_dict, input_dataframe=input_dataframe, erlang=erlang)

    # One-hot encode categorical columns
    processed_dataframe = pd.get_dummies(input_dataframe, columns=["old_bandwidth"])

    # Convert boolean columns to integers
    for column in processed_dataframe.columns:
        if processed_dataframe[column].dtype == bool:
            processed_dataframe[column] = processed_dataframe[column].astype(int)

    logger.info(
        "Processed %d samples with %d features",
        len(processed_dataframe),
        len(processed_dataframe.columns),
    )

    return processed_dataframe


def balance_training_data(
    input_dataframe: pd.DataFrame,
    balance_per_slice: bool,
    erlang: float,
    simulation_dict: dict[str, Any],
) -> pd.DataFrame:
    """
    Balance training data to ensure representative sampling.

    Can balance equally across all classes or use weighted sampling.

    :param input_dataframe: Input DataFrame to balance
    :type input_dataframe: pd.DataFrame
    :param balance_per_slice: If True, balance equally across segments
    :type balance_per_slice: bool
    :param erlang: Traffic volume value
    :type erlang: float
    :param simulation_dict: Dictionary containing simulation parameters
    :type simulation_dict: Dict[str, Any]
    :return: Balanced DataFrame
    :rtype: pd.DataFrame

    Example:
        >>> data = pd.DataFrame({'num_segments': [1, 1, 2, 2, 4, 4, 8, 8]})
        >>> balanced = balance_training_data(data, True, 1000.0, sim_dict)
    """
    if "num_segments" not in input_dataframe.columns:
        logger.warning("Column 'num_segments' not found, returning unbalanced data")
        return process_training_data(simulation_dict, input_dataframe, erlang)

    if balance_per_slice:
        balanced_df = _balance_equally(input_dataframe)
    else:
        balanced_df = _balance_weighted(input_dataframe)

    return process_training_data(simulation_dict=simulation_dict, input_dataframe=balanced_df, erlang=erlang)


def _balance_equally(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Balance data with equal samples per segment class."""
    unique_segments = dataframe["num_segments"].unique()
    segment_dataframes = [dataframe[dataframe["num_segments"] == segment] for segment in unique_segments]

    # Find minimum class size
    min_size = min(len(df) for df in segment_dataframes)

    # Sample equal amounts from each class
    sampled_dataframes = [df.sample(n=min_size, random_state=42) for df in segment_dataframes]

    # Combine and shuffle
    balanced_dataframe = pd.concat(sampled_dataframes).sample(frac=1, random_state=42)

    logger.info("Balanced data equally: %d samples per segment class", min_size)

    return balanced_dataframe


def _balance_weighted(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Balance data with weighted sampling based on operational importance."""
    # Define sampling weights for different segment counts
    segment_weights = {
        1: 0.05,  # 5% - least common in practice
        2: 0.35,  # 35% - common
        4: 0.35,  # 35% - common
        8: 0.25,  # 25% - less common
    }

    segment_dataframes = {}
    for segments, _weight in segment_weights.items():
        segment_df = dataframe[dataframe["num_segments"] == segments]
        if len(segment_df) > 0:
            segment_dataframes[segments] = segment_df

    if not segment_dataframes:
        logger.warning("No valid segments found for weighted balancing")
        return dataframe

    # Calculate samples based on minimum class size
    min_size = min(len(df) for df in segment_dataframes.values())

    sampled_dataframes = []
    for segments, weight in segment_weights.items():
        if segments in segment_dataframes:
            sample_size = int(min_size * weight)
            if 0 < sample_size <= len(segment_dataframes[segments]):
                sampled_df = segment_dataframes[segments].sample(n=sample_size, random_state=42)
                sampled_dataframes.append(sampled_df)
                logger.debug("Sampled %d instances for %d segments", sample_size, segments)

    # Combine and shuffle
    balanced_dataframe = pd.concat(sampled_dataframes).sample(frac=1, random_state=42)

    total_samples = len(balanced_dataframe)
    logger.info("Balanced data with weighted sampling: %d total samples", total_samples)

    return balanced_dataframe


def prepare_prediction_features(
    raw_features: dict[str, Any],
    engine_properties: dict[str, Any],
    sdn_properties: object,
) -> pd.DataFrame:
    """
    Prepare features for model prediction from raw request data.

    Transforms raw features into the format expected by trained models.

    :param raw_features: Dictionary of raw feature values
    :type raw_features: Dict[str, Any]
    :param engine_properties: Engine configuration properties
    :type engine_properties: Dict[str, Any]
    :param sdn_properties: SDN controller properties object
    :type sdn_properties: object
    :return: DataFrame with properly formatted features
    :rtype: pd.DataFrame

    Example:
        >>> features = {'bandwidth': 100, 'path_length': 15, 'congestion': 0.3}
        >>> prepared = prepare_prediction_features(features, engine_props, sdn_props)
    """
    # Create DataFrame from single observation
    processed_dataframe = pd.DataFrame(raw_features, index=[0])

    # One-hot encode bandwidth
    processed_dataframe = pd.get_dummies(processed_dataframe, columns=["old_bandwidth"])

    # Convert boolean columns to integers
    for column in processed_dataframe.columns:
        if processed_dataframe[column].dtype == bool:
            processed_dataframe[column] = processed_dataframe[column].astype(int)

    # Ensure all expected bandwidth columns exist
    for bandwidth, percentage in engine_properties["request_distribution"].items():
        if percentage > 0:
            column_name = f"old_bandwidth_{bandwidth}"
            if bandwidth != getattr(sdn_properties, "bandwidth", None) and column_name not in processed_dataframe.columns:
                processed_dataframe[column_name] = 0

    # Only include columns that exist
    available_columns = [col for col in EXPECTED_ML_COLUMNS if col in processed_dataframe.columns]
    processed_dataframe = processed_dataframe.reindex(columns=available_columns)

    return processed_dataframe


def split_features_labels(dataframe: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and labels.

    :param dataframe: Complete DataFrame with features and target
    :type dataframe: pd.DataFrame
    :param target_column: Name of the target column
    :type target_column: str
    :return: Tuple of (features, labels)
    :rtype: Tuple[pd.DataFrame, pd.Series]
    :raises KeyError: If target column not found

    Example:
        >>> data = pd.DataFrame({'feature1': [1, 2], 'target': [0, 1]})
        >>> X, y = split_features_labels(data, 'target')
    """
    if target_column not in dataframe.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame")

    features = dataframe.drop(columns=[target_column])
    labels = dataframe[target_column]

    return features, labels


def normalize_features(features: pd.DataFrame, normalization_type: str = "standard") -> tuple[pd.DataFrame, Any]:
    """
    Normalize features using specified method.

    :param features: Feature DataFrame to normalize
    :type features: pd.DataFrame
    :param normalization_type: Type of normalization ('standard' or 'minmax')
    :type normalization_type: str
    :return: Tuple of (normalized features, scaler object)
    :rtype: Tuple[pd.DataFrame, Any]
    :raises ValueError: If normalization type not supported

    Example:
        >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [10, 20, 30]})
        >>> X_norm, scaler = normalize_features(X, 'standard')
    """
    if normalization_type == "standard":
        scaler = StandardScaler()
    elif normalization_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Normalization type '{normalization_type}' not supported. Use 'standard' or 'minmax'")

    # Fit and transform
    normalized_array = scaler.fit_transform(features)

    # Convert back to DataFrame with same columns
    normalized_features = pd.DataFrame(normalized_array, columns=features.columns, index=features.index)

    return normalized_features, scaler
