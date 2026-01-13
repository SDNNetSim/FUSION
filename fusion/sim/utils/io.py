"""
File I/O operations and configuration utilities.

This module provides functions for reading and writing various file formats
used in simulations, including YAML and JSON files.
"""

import json
import os
from typing import Any

import yaml


def parse_yaml_file(yaml_file: str) -> Any:
    """
    Parse a YAML file.

    :param yaml_file: YAML file name
    :type yaml_file: str
    :return: YAML data as an object
    :rtype: Any
    """
    with open(yaml_file, encoding="utf-8") as file_obj:
        try:
            yaml_data = yaml.safe_load(file_obj)
            return yaml_data
        except yaml.YAMLError as exc:
            return exc


def modify_multiple_json_values(
    trial_num: int, file_path: str, update_list: list[tuple[str, Any]]
) -> None:
    """
    Open a JSON file, modify multiple key-value pairs, and save it back.

    Single process support only.

    :param trial_num: Trial number
    :type trial_num: int
    :param file_path: Path to the JSON file
    :type file_path: str
    :param update_list: List of tuples containing keys and their new values
                        Example: [('key1', 'new_value1'), ('key2', 'new_value2')]
    :type update_list: list[tuple[str, Any]]
    :raises KeyError: If key not found in the JSON file
    """
    open_filepath = os.path.join(file_path, "sim_input_s1.json")
    with open(open_filepath, encoding="utf-8") as json_file:
        data = json.load(json_file)

    for key, new_value in update_list:
        if key in data:
            data[key] = new_value
        else:
            raise KeyError(f"Key '{key}' not found in the JSON file.")

    save_filepath = os.path.join(file_path, f"sim_input_s{trial_num + 1}.json")
    with open(save_filepath, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=4)
