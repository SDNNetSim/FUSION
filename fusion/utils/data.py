"""
Data manipulation utility functions for FUSION.

This module provides data structure manipulation utilities that are used
across different packages in FUSION. These functions are placed here to
avoid circular dependencies between fusion.core and fusion.sim packages.
"""

from typing import Any


def sort_dict_keys(dictionary: dict[str, Any]) -> dict[str, Any]:
    """
    Sort a dictionary by keys in descending order.

    :param dictionary: Dictionary to sort
    :type dictionary: dict[str, Any]
    :return: Newly sorted dictionary
    :rtype: dict[str, Any]
    """
    sorted_keys = sorted(map(int, dictionary.keys()), reverse=True)
    sorted_dict = {str(key): dictionary[str(key)] for key in sorted_keys}
    return sorted_dict


def sort_nested_dict_values(original_dict: dict[str, Any], nested_key: str) -> dict[str, Any]:
    """
    Sort a dictionary by a value which belongs to a nested key.

    :param original_dict: Original dictionary
    :type original_dict: dict[str, Any]
    :param nested_key: Nested key to sort by
    :type nested_key: str
    :return: Sorted dictionary, ascending
    :rtype: dict[str, Any]
    """
    sorted_items = sorted(original_dict.items(), key=lambda x: x[1][nested_key])
    return dict(sorted_items)
