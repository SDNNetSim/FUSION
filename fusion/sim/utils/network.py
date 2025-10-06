"""
Network analysis and path calculation utilities.

This module provides functions for network path analysis, including
path length calculations, congestion metrics, and fragmentation analysis.
"""

from typing import Any

import networkx as nx
import numpy as np


def find_path_length(path_list: list[int], topology: nx.Graph) -> float:
    """
    Find the length of a path in a physical topology.

    :param path_list: List of integers representing nodes in the path
    :type path_list: list[int]
    :param topology: Network topology
    :type topology: nx.Graph
    :return: Length of the path
    :rtype: float
    """
    path_length = 0
    for i in range(len(path_list) - 1):
        path_length += topology[path_list[i]][path_list[i + 1]]["length"]

    return path_length


def find_max_path_length(source: int, destination: int, topology: nx.Graph) -> float:
    """
    Find the maximum path length possible between nodes in the network.

    :param source: Source node
    :type source: int
    :param destination: Destination node
    :type destination: int
    :param topology: Network topology
    :type topology: nx.Graph
    :return: Length of the longest path possible
    :rtype: float
    """
    all_paths_list = list(nx.shortest_simple_paths(topology, source, destination))
    path_list = all_paths_list[-1]
    return find_path_length(path_list=path_list, topology=topology)


def get_path_modulation(
    modulation_formats: dict[str, dict[Any, Any]] | None = None,
    path_length: float | None = None,
    mods_dict: dict[str, dict[Any, Any]] | None = None,
    path_len: float | None = None,
) -> str | bool:
    """
    Choose a modulation format that will allocate a network request.

    :param modulation_formats: Information for maximum reach for each modulation format
    :type modulation_formats: dict[str, dict]
    :param path_length: Length of the path to be taken
    :type path_length: float
    :param mods_dict: Legacy parameter name for modulation_formats
    :type mods_dict: dict[str, dict]
    :param path_len: Legacy parameter name for path_length
    :type path_len: float
    :return: Chosen modulation format or False if path too long
    :rtype: str | bool
    """
    # Handle backward compatibility
    if mods_dict is not None:
        modulation_formats = mods_dict
    if path_len is not None:
        path_length = path_len

    if modulation_formats is None or path_length is None:
        raise ValueError("Must provide modulation_formats and path_length")

    qpsk_max = modulation_formats["QPSK"]["max_length"]
    qam16_max = modulation_formats["16-QAM"]["max_length"]
    qam64_max = modulation_formats["64-QAM"]["max_length"]

    if qpsk_max >= path_length > qam16_max:
        return "QPSK"
    elif qam16_max >= path_length > qam64_max:
        return "16-QAM"
    elif qam64_max >= path_length:
        return "64-QAM"
    else:
        return False


def find_path_congestion(
    path_list: list[int], network_spectrum: dict, band: str = "c"
) -> tuple[float, float]:
    """
    Compute average path congestion and scaled available capacity.

    Accounts for multiple cores per link.

    :param path_list: Sequence of nodes in the path
    :type path_list: list[int]
    :param network_spectrum: Current spectrum allocation info
    :type network_spectrum: dict
    :param band: Spectral band to evaluate
    :type band: str
    :return: (average congestion [0,1], scaled available capacity [0,1])
    :rtype: tuple[float, float]
    """
    link_congestion_list = []
    total_slots_available = 0.0

    for source, destination in zip(path_list, path_list[1:], strict=False):
        link_key = (source, destination)
        cores_matrix = network_spectrum[link_key]["cores_matrix"]
        band_cores_matrix = cores_matrix[band]

        num_cores = len(band_cores_matrix)
        num_slots_per_core = len(band_cores_matrix[0])

        slots_taken = 0.0
        for core_array in band_cores_matrix:
            core_slots_taken = float(np.count_nonzero(core_array))
            slots_taken += core_slots_taken

        total_slots = num_cores * num_slots_per_core
        slots_available = total_slots - slots_taken

        link_congestion_list.append(slots_taken / total_slots)
        total_slots_available += slots_available

    average_path_congestion = np.mean(link_congestion_list)
    scaled_available_capacity = total_slots_available

    return float(average_path_congestion), scaled_available_capacity


def find_path_fragmentation(
    path_list: list[int], network_spectrum: dict, band: str = "c"
) -> float:
    """
    Compute the average fragmentation ratio along a path.

    :param path_list: Sequence of nodes in the path
    :type path_list: list[int]
    :param network_spectrum: Spectrum allocation per link
    :type network_spectrum: dict
    :param band: Spectral band to use (e.g., 'c')
    :type band: str
    :return: Average fragmentation score [0,1] (higher = worse fragmentation)
    :rtype: float
    """
    fragmentation_ratios = []

    for source, destination in zip(path_list, path_list[1:], strict=False):
        link_key = (source, destination)
        cores_matrix = network_spectrum[link_key]["cores_matrix"]
        cores = cores_matrix[band]

        for core in cores:
            free_blocks = 0
            max_block = 0
            current_block = 0
            total_free = 0

            for slot in core:
                if slot == 0:
                    current_block += 1
                    total_free += 1
                else:
                    if current_block > 0:
                        free_blocks += 1
                        max_block = max(max_block, current_block)
                        current_block = 0

            if current_block > 0:  # Catch trailing free block
                free_blocks += 1
                max_block = max(max_block, current_block)

            if total_free == 0:
                fragmentation_ratio = 1.0  # fully occupied, max fragmentation
            else:
                fragmentation_ratio = 1 - (max_block / total_free)

            fragmentation_ratios.append(fragmentation_ratio)

    return float(np.mean(fragmentation_ratios)) if fragmentation_ratios else 1.0


def find_core_congestion(
    core_index: int, network_spectrum: dict, path_list: list[int]
) -> float:
    """
    Find the current percentage of congestion on a core along a path.

    :param core_index: Index of the core
    :type core_index: int
    :param network_spectrum: Network spectrum database
    :type network_spectrum: dict
    :param path_list: Current path
    :type path_list: list[int]
    :return: Average congestion percentage on the core
    :rtype: float
    """
    link_congestion_list = []

    for source, destination in zip(path_list, path_list[1:], strict=False):
        link_key = (source, destination)
        cores_matrix = network_spectrum[link_key]["cores_matrix"]
        total_slots = 0
        slots_taken = 0.0

        for band in cores_matrix:
            # Every core will have the same number of spectral slots
            total_slots += len(cores_matrix[band][0])
            core_slots_taken = float(
                len(np.where(cores_matrix[band][core_index] != 0.0)[0])
            )
            slots_taken += core_slots_taken

        link_congestion_list.append(slots_taken / total_slots)

    return float(np.mean(link_congestion_list))


def find_core_fragmentation_congestion(
    network_spectrum: dict, path: list[int], core: int, band: str
) -> tuple[float, float]:
    """
    Find the congestion and fragmentation scores for a specific request.

    :param network_spectrum: Current network spectrum database
    :type network_spectrum: dict
    :param path: Current path
    :type path: list[int]
    :param core: Current core
    :type core: int
    :param band: Current allocated band
    :type band: str
    :return: Fragmentation and congestion scores
    :rtype: tuple[float, float]
    """
    fragmentation_score = 0.0
    congestion_score = 0.0

    for source, destination in zip(path, path[1:], strict=False):
        link_key = (source, destination)
        core_array = network_spectrum[link_key]["cores_matrix"][band][core]

        if len(core_array) != 256:
            raise NotImplementedError("Only works for 256 spectral slots.")

        congestion_score += len(np.where(core_array != 0)[0])

        count = 0
        in_zero_group = False

        for number in core_array:
            if number == 0:
                if not in_zero_group:
                    in_zero_group = True
            else:
                if in_zero_group:
                    count += 1
                    in_zero_group = False

        fragmentation_score += count

    num_links = len(path) - 1
    # The lowest number of slots a request can take is 2, the max number of times
    # fragmentation can happen is 86 for 256 spectral slots
    fragmentation_score = fragmentation_score / 86.0 / num_links
    congestion_score = congestion_score / 256.0 / num_links

    return fragmentation_score, congestion_score


def classify_congestion(current_congestion: float, congestion_cutoff: float) -> int:
    """
    Classify congestion percentages to 'levels'.

    :param current_congestion: Current congestion percentage
    :type current_congestion: float
    :param congestion_cutoff: Conversion cutoff percentage
    :type congestion_cutoff: float
    :return: Congestion index or level
    :rtype: int
    """
    # Hard coded, only supports 2 path levels
    if current_congestion <= congestion_cutoff:
        return 0
    else:
        return 1
