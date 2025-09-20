"""
Constants and shared values for the ML module.

This module contains constants used across multiple ML components
to avoid code duplication and maintain consistency.
"""

# Standard column order for ML features
EXPECTED_ML_COLUMNS = [
    "path_length",
    "longest_reach",
    "ave_cong",
    "old_bandwidth_50",
    "old_bandwidth_100",
    "old_bandwidth_200",
    "old_bandwidth_400",
]

# Bandwidth categories for one-hot encoding
BANDWIDTH_CATEGORIES = [50, 100, 200, 400]
