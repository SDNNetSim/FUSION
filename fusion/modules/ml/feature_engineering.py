"""
Feature engineering utilities for machine learning module.

This module handles feature extraction and creation from raw network data.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from fusion.modules.ml.constants import EXPECTED_ML_COLUMNS
from fusion.utils.logging_config import get_logger
from fusion.utils.network import find_core_congestion
from fusion.utils.network import find_path_length as find_path_len

logger = get_logger(__name__)


def extract_ml_features(
        request_dict: dict[str, Any],
        engine_properties: dict[str, Any],
        sdn_properties: object
) -> pd.DataFrame:
    """
    Extract machine learning features from a network request.

    Creates features including path length, congestion metrics, and
    bandwidth requirements formatted for ML model consumption.

    :param request_dict: Dictionary containing request information
    :type request_dict: Dict[str, Any]
    :param engine_properties: Engine configuration properties
    :type engine_properties: Dict[str, Any]
    :param sdn_properties: SDN controller properties object
    :type sdn_properties: object
    :return: DataFrame with extracted features
    :rtype: pd.DataFrame

    Example:
        >>> request = {'bandwidth': 100, 'mod_formats': {'QPSK': {'max_length': 2000}}}
        >>> features = extract_ml_features(request, engine_props, sdn_props)
        >>> print(features.columns.tolist())
        ['path_length', 'longest_reach', 'ave_cong', 'old_bandwidth_50', ...]
    """
    # Calculate path metrics
    path_length_km = find_path_len(
        path_list=getattr(sdn_properties, 'path_list', []),
        topology=engine_properties['topology']
    )

    # Calculate congestion across all cores
    congestion_metrics = _calculate_congestion_metrics(
        engine_properties=engine_properties,
        sdn_properties=sdn_properties
    )

    # Build feature dictionary
    feature_dict = {
        'old_bandwidth': request_dict['bandwidth'],
        'path_length': path_length_km,
        'longest_reach': request_dict['mod_formats']['QPSK']['max_length'],
        'ave_cong': congestion_metrics['average'],
    }

    # Convert to properly formatted DataFrame
    formatted_features = _format_features_for_prediction(
        feature_dict=feature_dict,
        engine_properties=engine_properties,
        sdn_properties=sdn_properties
    )

    return formatted_features


def _calculate_congestion_metrics(
        engine_properties: dict[str, Any],
        sdn_properties: object
) -> dict[str, float]:
    """
    Calculate congestion metrics across all cores.

    :param engine_properties: Engine configuration properties
    :type engine_properties: Dict[str, Any]
    :param sdn_properties: SDN controller properties object
    :type sdn_properties: object
    :return: Dictionary with congestion statistics
    :rtype: Dict[str, float]
    """
    congestion_array = np.array([])

    for core_number in range(engine_properties['cores_per_link']):
        current_congestion = find_core_congestion(
            core_index=core_number,
            network_spectrum=getattr(sdn_properties, 'network_spectrum', {}),
            path_list=getattr(sdn_properties, 'path_list', [])
        )
        congestion_array = np.append(congestion_array, current_congestion)

    congestion_metrics = {
        'average': float(np.mean(congestion_array)),
        'max': float(np.max(congestion_array)) if len(congestion_array) > 0 else 0.0,
        'min': float(np.min(congestion_array)) if len(congestion_array) > 0 else 0.0,
        'std': float(np.std(congestion_array)) if len(congestion_array) > 1 else 0.0
    }

    logger.debug("Calculated congestion metrics: avg=%.3f, max=%.3f",
                 congestion_metrics['average'], congestion_metrics['max'])

    return congestion_metrics


def _format_features_for_prediction(
        feature_dict: dict[str, Any],
        engine_properties: dict[str, Any],
        sdn_properties: object
) -> pd.DataFrame:
    """
    Format raw features into the structure expected by ML models.

    Handles one-hot encoding and ensures all expected columns are present.

    :param feature_dict: Dictionary of raw features
    :type feature_dict: Dict[str, Any]
    :param engine_properties: Engine configuration properties
    :type engine_properties: Dict[str, Any]
    :param sdn_properties: SDN controller properties object
    :type sdn_properties: object
    :return: Properly formatted DataFrame
    :rtype: pd.DataFrame
    """
    # Create DataFrame from features
    features_df = pd.DataFrame(feature_dict, index=[0])

    # One-hot encode bandwidth
    features_df = pd.get_dummies(features_df, columns=['old_bandwidth'])

    # Convert boolean columns to integers
    for column in features_df.columns:
        if features_df[column].dtype == bool:
            features_df[column] = features_df[column].astype(int)

    # Ensure all expected bandwidth columns exist
    for bandwidth, percentage in engine_properties['request_distribution'].items():
        if percentage > 0:
            column_name = f'old_bandwidth_{bandwidth}'
            if (bandwidth != getattr(sdn_properties, 'bandwidth', None) and
                    column_name not in features_df.columns):
                features_df[column_name] = 0

    # Reindex to expected columns
    features_df = features_df.reindex(columns=EXPECTED_ML_COLUMNS, fill_value=0)

    return features_df


def create_interaction_features(
        features: pd.DataFrame,
        interactions: list[tuple[str, str]] | None = None
) -> pd.DataFrame:
    """
    Create interaction features between existing features.

    :param features: Original features DataFrame
    :type features: pd.DataFrame
    :param interactions: List of tuples specifying interactions
    :type interactions: List[Tuple[str, str]]
    :return: DataFrame with interaction features added
    :rtype: pd.DataFrame

    Example:
        >>> features = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> interactions = [('A', 'B')]
        >>> new_features = create_interaction_features(features, interactions)
        >>> print('A_x_B' in new_features.columns)
        True
    """
    if interactions is None:
        # Default interactions that often improve model performance
        interactions = [
            ('path_length', 'ave_cong'),  # Length-congestion interaction
            ('path_length', 'longest_reach'),  # Length-reach constraint
        ]

    features_with_interactions = features.copy()

    for feature1, feature2 in interactions:
        if feature1 in features.columns and feature2 in features.columns:
            interaction_name = f"{feature1}_x_{feature2}"
            features_with_interactions[interaction_name] = (
                    features[feature1] * features[feature2]
            )
            logger.debug("Created interaction feature: %s", interaction_name)

    return features_with_interactions


def create_polynomial_features(
        features: pd.DataFrame,
        degree: int = 2,
        include_bias: bool = False
) -> pd.DataFrame:
    """
    Create polynomial features up to specified degree.

    :param features: Original features DataFrame
    :type features: pd.DataFrame
    :param degree: Maximum polynomial degree
    :type degree: int
    :param include_bias: Whether to include bias term
    :type include_bias: bool
    :return: DataFrame with polynomial features
    :rtype: pd.DataFrame

    Example:
        >>> features = pd.DataFrame({'x': [1, 2, 3]})
        >>> poly_features = create_polynomial_features(features, degree=2)
    """
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
    poly_array = poly.fit_transform(features)

    # Get feature names
    feature_names = poly.get_feature_names_out(features.columns)

    # Create DataFrame with new features
    poly_features = pd.DataFrame(
        poly_array,
        columns=feature_names,
        index=features.index
    )

    logger.info(
        "Created polynomial features: %d -> %d features",
        features.shape[1], poly_features.shape[1]
    )

    return poly_features


def engineer_network_features(
        request_dict: dict[str, Any],
        network_state: dict[str, Any]
) -> dict[str, float]:
    """
    Engineer advanced network-specific features.

    Creates features that capture network topology and state characteristics
    relevant for routing and spectrum assignment decisions.

    :param request_dict: Request information
    :type request_dict: Dict[str, Any]
    :param network_state: Current network state information
    :type network_state: Dict[str, Any]
    :return: Dictionary of engineered features
    :rtype: Dict[str, float]

    Example:
        >>> request = {'source': 'A', 'destination': 'B', 'bandwidth': 100}
        >>> state = {'link_utilization': {...}, 'active_paths': [...]}
        >>> features = engineer_network_features(request, state)
    """
    engineered_features = {}

    # Path diversity features
    if 'alternative_paths' in network_state:
        engineered_features['path_diversity'] = float(
            len(network_state['alternative_paths'])
        )
        engineered_features['avg_alternative_length'] = float(np.mean([
            len(path) for path in network_state['alternative_paths']
        ])) if network_state['alternative_paths'] else 0.0

    # Bottleneck features
    if 'link_utilization' in network_state:
        utilization_values = list(network_state['link_utilization'].values())
        engineered_features['max_link_utilization'] = float(
            max(utilization_values) if utilization_values else 0.0
        )
        engineered_features['utilization_variance'] = float(
            np.var(utilization_values) if len(utilization_values) > 1 else 0.0
        )

    # Temporal features (if available)
    if 'time_of_day' in network_state:
        hour = network_state['time_of_day']
        engineered_features['is_peak_hour'] = float(8 <= hour <= 18)
        engineered_features['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
        engineered_features['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))

    # Request-specific features
    if 'bandwidth' in request_dict:
        engineered_features['bandwidth_category'] = _categorize_bandwidth(
            request_dict['bandwidth']
        )

    return engineered_features


def _categorize_bandwidth(bandwidth: float) -> float:
    """
    Categorize bandwidth into predefined levels.

    :param bandwidth: Bandwidth value in Gbps
    :type bandwidth: float
    :return: Category as float (0-3)
    :rtype: float
    """
    if bandwidth <= 50:
        return 0.0  # Low
    if bandwidth <= 100:
        return 1.0  # Medium
    if bandwidth <= 200:
        return 2.0  # High
    return 3.0  # Very High
