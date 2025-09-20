"""
Input setup utilities for FUSION simulations.

This module provides functions for creating and preparing simulation input data,
including network topology, bandwidth information, and physical topology setup.
"""

import copy
import json
import os
import time
from typing import Dict

from fusion.io.structure import create_network
from fusion.io.generate import create_bw_info, create_pt
from fusion.utils.os import create_directory, find_project_root

PROJECT_ROOT = find_project_root()


def create_input(base_fp: str, engine_props: Dict) -> Dict:
    """
    Create input data to run simulations.

    This function generates all necessary input files and data structures
    required for running FUSION simulations, including bandwidth information,
    network topology, and physical topology configuration.

    :param base_fp: The base file path to save input data
    :type base_fp: str
    :param engine_props: Input properties to engine containing simulation parameters
    :type engine_props: Dict
    :return: Engine props modified with network, physical topology, and bandwidth information
    :rtype: Dict
    :raises RuntimeError: If bandwidth info file is empty or invalid after multiple attempts
    """
    bw_info_dict = create_bw_info(
        mod_assumption=engine_props['mod_assumption'],
        mod_assumptions_path=engine_props['mod_assumption_path']
    )
    bw_file = f"bw_info_{engine_props['thread_num']}.json"
    save_input(base_fp=base_fp, properties=engine_props, file_name=bw_file, data_dict=bw_info_dict)

    save_path = os.path.join(PROJECT_ROOT, base_fp, 'input', engine_props['network'],
                             engine_props['date'], engine_props['sim_start'], bw_file)

    # Retry loop to ensure file is ready, used for Unity cluster runs
    max_attempts = 50
    for _ in range(max_attempts):
        try:
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                with open(save_path, 'r', encoding='utf-8') as file_object:
                    engine_props['mod_per_bw'] = json.load(file_object)
                break
        except json.JSONDecodeError:
            pass
        time.sleep(0.5)
    else:
        raise RuntimeError(f"File {save_path} is empty or invalid after multiple attempts")

    network_dict, core_nodes_list = create_network(
        base_fp=os.path.join(PROJECT_ROOT, base_fp),
        const_weight=engine_props['const_link_weight'],
        net_name=engine_props['network'],
        is_only_core_node=engine_props['is_only_core_node']
    )
    engine_props['topology_info'] = create_pt(
        cores_per_link=engine_props['cores_per_link'],
        network_spectrum_dict=network_dict
    )
    engine_props['core_nodes'] = core_nodes_list

    return engine_props


def save_input(base_fp: str, properties: Dict, file_name: str, data_dict: Dict) -> None:
    """
    Save simulation input data to file.

    This function saves simulation input data to a JSON file in the appropriate
    directory structure, ensuring proper serialization and file synchronization.

    :param base_fp: The base file path to save input
    :type base_fp: str
    :param properties: Properties of the simulation, used for directory structure
    :type properties: Dict
    :param file_name: The desired file name
    :type file_name: str
    :param data_dict: A dictionary containing the data to save
    :type data_dict: Dict
    """
    base_dir = os.path.join(PROJECT_ROOT, base_fp)
    path = os.path.join(base_dir, 'input', properties['network'], properties['date'], properties['sim_start'])
    create_directory(path)
    create_directory(os.path.join(PROJECT_ROOT, 'data', 'output'))

    save_path = os.path.join(path, file_name)

    save_dict = copy.deepcopy(data_dict)
    save_dict.pop('topology', None)
    save_dict.pop('callback', None)

    with open(save_path, 'w', encoding='utf-8') as file_path:
        json.dump(save_dict, file_path, indent=4)
        file_path.flush()
        os.fsync(file_path.fileno())

    time.sleep(0.1)
