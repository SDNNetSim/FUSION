import json
import os
from typing import Any

from fusion.sim.utils.data import update_matrices

# NOTE: Consider moving these filter functions to a shared utilities module
# for reuse across different components (plotting, analysis, etc.)


def _not_filters(
    filter_dict: dict[str, list[list[Any]]], file_dict: dict[str, Any]
) -> bool:
    """
    Apply "not" filters to the file dictionary.

    Each entry in 'not_filter_list' is expected to be a list where all elements except
    the last represent a key path and the last element is the value to reject.

    :param filter_dict: Dictionary containing filter configurations
    :param file_dict: Dictionary to apply filters against
    :return: False if any "not" filter is triggered, True otherwise
    :rtype: bool
    """
    for flags_list in filter_dict.get("not_filter_list", []):
        keys_list = flags_list[:-1]
        check_value = flags_list[-1]

        # Traverse file_dict along the keys_list.
        file_value: dict[str, Any] | Any | None = file_dict
        for key in keys_list:
            if isinstance(file_value, dict):
                file_value = file_value.get(key)
            else:
                file_value = None
                break

        if file_value == check_value:
            return False  # Filter triggered, reject this config.
    return True


def _or_filters(
    filter_dict: dict[str, list[list[Any]]], file_dict: dict[str, Any]
) -> bool:
    """
    Apply "or" filters to the file dictionary.

    Each entry in 'or_filter_list' is expected to be a list where all elements except
    the last represent a key path and the last element is the value that, if matched,
    should accept the file.

    Returns True immediately if any "or" filter passes; if none pass, return False.
    If no 'or_filter_list' is provided, defaults to True.
    """
    or_filters = filter_dict.get("or_filter_list", [])
    if not or_filters:
        return True

    for flags_list in or_filters:
        keys_list = flags_list[:-1]
        check_value = flags_list[-1]

        file_value: dict[str, Any] | Any | None = file_dict
        for key in keys_list:
            if isinstance(file_value, dict):
                file_value = file_value.get(key)
            else:
                file_value = None
                break

        if file_value == check_value:
            return True  # At least one filter passed.
    return False


def _and_filters(
    filter_dict: dict[str, list[list[Any]]], file_dict: dict[str, Any]
) -> bool:
    """
    Apply "and" filters to the file dictionary.

    Each entry in 'and_filter_list' is expected to be a list where all elements
    except the last represent a key path and the last element is the value that
    must be matched.

    Returns False immediately if any "and" filter is not satisfied.
    """
    for flags_list in filter_dict.get("and_filter_list", []):
        keys_list = flags_list[:-1]
        check_value = flags_list[-1]

        file_value: dict[str, Any] | Any | None = file_dict
        for key in keys_list:
            if isinstance(file_value, dict):
                file_value = file_value.get(key)
            else:
                file_value = None
                break

        if file_value != check_value:
            return False  # This "and" condition failed.
    return True


def _check_filters(
    file_dict: dict[str, Any], filter_dict: dict[str, list[list[Any]]]
) -> bool:
    """
    Check all filters on a file dictionary.

    The file passes if it satisfies all "and" filters, at least one "or" filter
    (if provided), and none of the "not" filters.
    """
    if not _and_filters(filter_dict, file_dict):
        return False
    if not _or_filters(filter_dict, file_dict):
        return False
    return _not_filters(filter_dict, file_dict)


def find_times(
    dates_dict: dict[str, str], filter_dict: dict[str, list[list[Any]]]
) -> dict[str, list]:
    """
    Searches output directories based on filters and retrieves simulation
    directory information.

    :param dates_dict: Dictionary where keys are dates (e.g., '0227') and
        values are network names.
    :param filter_dict: Dictionary containing search filters with keys:
                        'not_filter_list', 'or_filter_list', and 'and_filter_list'.
    :return: A dictionary with lists for simulation times, simulation numbers,
        networks, and dates.
    """
    resp: dict[str, list[Any]] = {}
    info_dict: dict[str, dict[str, list[str]]] = {}

    for date, network in dates_dict.items():
        times_path = os.path.join("..", "..", "data", "input", network, date)
        if not os.path.isdir(times_path):
            continue

        times_list = [
            curr_dir
            for curr_dir in os.listdir(times_path)
            if os.path.isdir(os.path.join(times_path, curr_dir))
        ]

        for curr_time in times_list:
            sims_path = os.path.join(times_path, curr_time)
            input_file_list = [
                input_file
                for input_file in os.listdir(sims_path)
                if "sim" in input_file
            ]

            for input_file in input_file_list:
                file_path = os.path.join(sims_path, input_file)
                try:
                    with open(file_path, encoding="utf-8") as file_obj:
                        file_dict = json.load(file_obj)
                except json.JSONDecodeError:
                    print(f"Skipping file (incomplete write): {file_path}")
                    continue

                if not _check_filters(file_dict, filter_dict):
                    continue

                if curr_time not in info_dict:
                    info_dict[curr_time] = {
                        "sim_list": [],
                        "network_list": [],
                        "dates_list": [],
                        "algorithm_list": [],
                    }
                # Assuming the simulation number is the third element separated by
                # underscores.
                sim = input_file.split("_")[2].split(".")[0]
                algo_name = file_dict.get("path_algorithm", "Unknown Algorithm")
                info_dict[curr_time]["sim_list"].append(sim)
                info_dict[curr_time]["network_list"].append(network)
                info_dict[curr_time]["dates_list"].append(date)
                info_dict[curr_time]["algorithm_list"].append(algo_name)

    resp["algorithms_matrix"] = []
    resp = update_matrices(info_dict=info_dict)

    return resp
