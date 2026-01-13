"""
Data processing and matrix operation utilities.

This module provides functions for data transformation, matrix operations,
and statistical calculations used in simulations.
"""

from typing import Any

import numpy as np


def update_matrices(info_dict: dict[str, Any]) -> dict[str, list]:
    """
    Update network resource matrices from info dictionary.

    :param info_dict: Dictionary containing simulation information
    :type info_dict: dict[str, Any]
    :return: Dictionary containing updated matrices
    :rtype: dict[str, list]
    """
    response: dict[str, list] = {
        "times_matrix": [],
        "sims_matrix": [],
        "networks_matrix": [],
        "dates_matrix": [],
        "algorithms_matrix": [],
    }

    for current_time, obj in info_dict.items():
        response["times_matrix"].append([current_time])
        response["sims_matrix"].append(obj["sim_list"])
        response["networks_matrix"].append(obj["network_list"])
        response["dates_matrix"].append(obj["dates_list"])

        # Temporary patch for test_plot_helpers compatibility
        try:
            response["algorithms_matrix"].append(obj["algorithm_list"])
        except KeyError:
            response["dates_matrix"].append(obj["dates_list"])

    return response


def sort_nested_dict_values(
    original_dict: dict[str, Any], nested_key: str
) -> dict[str, Any]:
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


def dict_to_list(
    data_dict: dict,
    nested_key: str,
    path_list: list[str] | None = None,
    find_mean: bool = False,
) -> list[Any] | float:
    """
    Create a list from a dictionary taking values from a specified key.

    :param data_dict: Dictionary to search
    :type data_dict: dict
    :param nested_key: Where to take values from
    :type nested_key: str
    :param path_list: If the key is nested, the path is to that nested key
    :type path_list: Optional[list[str]]
    :param find_mean: Flag to return mean or not
    :type find_mean: bool
    :return: List or single value
    :rtype: list | float
    """
    if path_list is None:
        path_list = []

    extracted_list = []
    for value_dict in data_dict.values():
        for key in path_list:
            value_dict = value_dict.get(key, {})
        nested_value = value_dict.get(nested_key)
        if nested_value is not None:
            extracted_list.append(nested_value)

    if find_mean:
        return float(np.mean(extracted_list))

    return extracted_list


def calculate_matrix_statistics(input_dict: dict[str, list]) -> dict[str, list]:
    """
    Create a matrix based on dict values and take the min, max, and average of columns.

    :param input_dict: Input dict with values as lists
    :type input_dict: dict[str, list]
    :return: Min, max, and average of columns
    :rtype: dict[str, list]
    """
    response_dict = {}
    temp_matrix = np.array([])

    for episode, current_list in input_dict.items():
        if episode == "0":
            temp_matrix = np.array([current_list])
        else:
            temp_matrix = np.vstack((temp_matrix, current_list))

    response_dict["min"] = temp_matrix.min(axis=0, initial=np.inf).tolist()
    response_dict["max"] = temp_matrix.max(axis=0, initial=np.inf * -1.0).tolist()
    response_dict["average"] = temp_matrix.mean(axis=0).tolist()

    return response_dict


def min_max_scale(value: float, min_value: float, max_value: float) -> float:
    """
    Scale a value with respect to a min and a max value.

    :param value: Value to be scaled
    :type value: float
    :param min_value: Minimum value to scale by
    :type min_value: float
    :param max_value: Maximum value to scale by
    :type max_value: float
    :return: Final scaled value
    :rtype: float
    """
    return (value - min_value) / (max_value - min_value)


def update_dict_from_list(
    input_dict: dict[str, Any], updates_list: list[tuple[str, Any]]
) -> dict[str, Any]:
    """
    Update the input dictionary with values from the updates list.

    Keys are derived from the tuples in the list.

    :param input_dict: Dictionary to be updated
    :type input_dict: dict[str, Any]
    :param updates_list: List of tuples, each containing (key, value)
    :type updates_list: list[tuple[str, Any]]
    :return: Updated dictionary
    :rtype: dict[str, Any]
    """
    for key, value in updates_list:
        input_dict[key] = value

    return input_dict
