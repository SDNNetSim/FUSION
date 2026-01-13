"""
Spectrum utility functions for FUSION.

This module provides spectrum-related utility functions that are used
across different packages in FUSION. These functions are placed here to
avoid circular dependencies between fusion.core and fusion.sim packages.
"""

import copy
from typing import Any

import numpy as np


def find_free_slots(
    network_spectrum: dict[Any, Any] | None = None,
    link_tuple: tuple[int, int] | None = None,
    network_spectrum_dict: dict[Any, Any] | None = None,
) -> dict:
    """
    Find every unallocated spectral slot for a given link.

    :param network_spectrum: Most updated network spectrum database
    :type network_spectrum: dict
    :param link_tuple: Link to find the free slots on
    :type link_tuple: tuple[int, int]
    :param network_spectrum_dict: Legacy parameter name for network_spectrum
    :type network_spectrum_dict: dict
    :return: Indexes of free spectral slots on the link for each core
    :rtype: dict
    """
    # Handle backward compatibility
    if network_spectrum_dict is not None:
        network_spectrum = network_spectrum_dict

    if network_spectrum is None or link_tuple is None:
        raise ValueError("Must provide network_spectrum and link_tuple")

    response: dict[str, dict[int, Any]] = {}
    for band in network_spectrum[link_tuple]["cores_matrix"].keys():
        response[band] = {}

        num_cores = len(network_spectrum[link_tuple]["cores_matrix"][band])
        for core_num in range(num_cores):
            cores_matrix = network_spectrum[link_tuple]["cores_matrix"][band]
            free_slots_list = np.where(cores_matrix[core_num] == 0)[0]
            response[band].update({core_num: free_slots_list})

    return response


def find_free_channels(
    network_spectrum: dict[Any, Any] | None = None,
    slots_needed: int | None = None,
    link_tuple: tuple[int, int] | None = None,
    network_spectrum_dict: dict[Any, Any] | None = None,
) -> dict:
    """
    Find the free super-channels on a given link.

    :param network_spectrum: Most updated network spectrum database
    :type network_spectrum: dict
    :param slots_needed: Number of slots needed for the request
    :type slots_needed: int
    :param link_tuple: Link to search on
    :type link_tuple: tuple[int, int]
    :param network_spectrum_dict: Legacy parameter name for network_spectrum
    :type network_spectrum_dict: dict
    :return: Available super-channels for every core
    :rtype: dict
    """
    # Handle backward compatibility
    if network_spectrum_dict is not None:
        network_spectrum = network_spectrum_dict

    if network_spectrum is None or link_tuple is None or slots_needed is None:
        raise ValueError("Must provide network_spectrum, link_tuple, and slots_needed")

    response: dict[str, dict[int, list]] = {}
    for band in network_spectrum[link_tuple]["cores_matrix"].keys():
        cores_matrix = copy.deepcopy(network_spectrum[link_tuple]["cores_matrix"][band])
        response.update({band: {}})

        for core_num, link_list in enumerate(cores_matrix):
            indexes = np.where(link_list == 0)[0]
            channels_list = []
            current_channel_list = []

            for i, free_index in enumerate(indexes):
                if i == 0:
                    current_channel_list.append(free_index)
                    if len(current_channel_list) == slots_needed:
                        channels_list.append(current_channel_list.copy())
                        current_channel_list.pop(0)
                elif free_index == indexes[i - 1] + 1:
                    current_channel_list.append(free_index)
                    if len(current_channel_list) == slots_needed:
                        channels_list.append(current_channel_list.copy())
                        current_channel_list.pop(0)
                else:
                    current_channel_list = [free_index]

            response[band].update({core_num: channels_list})

    return response


def find_taken_channels(
    network_spectrum: dict[Any, Any] | None = None,
    link_tuple: tuple[int, int] | None = None,
    network_spectrum_dict: dict[Any, Any] | None = None,
) -> dict:
    """
    Find the taken super-channels on a given link.

    :param network_spectrum: Most updated network spectrum database
    :type network_spectrum: dict
    :param link_tuple: Link to search on
    :type link_tuple: tuple[int, int]
    :param network_spectrum_dict: Legacy parameter name for network_spectrum
    :type network_spectrum_dict: dict
    :return: Unavailable super-channels for every core
    :rtype: dict
    """
    # Handle backward compatibility
    if network_spectrum_dict is not None:
        network_spectrum = network_spectrum_dict

    if network_spectrum is None or link_tuple is None:
        raise ValueError("Must provide network_spectrum and link_tuple")

    response: dict[str, dict[int, list]] = {}
    for band in network_spectrum[link_tuple]["cores_matrix"].keys():
        response.update({band: {}})
        cores_matrix = copy.deepcopy(network_spectrum[link_tuple]["cores_matrix"][band])

        for core_num, link_list in enumerate(cores_matrix):
            channels_list = []
            current_channel_list = []

            for value in link_list:
                if value > 0:
                    current_channel_list.append(value)
                elif value < 0 and current_channel_list:
                    channels_list.append(current_channel_list)
                    current_channel_list = []

            if current_channel_list:
                channels_list.append(current_channel_list)

            response[band][core_num] = channels_list

    return response


def get_channel_overlaps(free_channels_dict: dict, free_slots_dict: dict) -> dict:
    """
    Find the number of overlapping and non-overlapping channels between adjacent cores.

    :param free_channels_dict: Free super-channels found on a path
    :type free_channels_dict: dict
    :param free_slots_dict: Free slots found on the given path
    :type free_slots_dict: dict
    :return: Overlapping and non-overlapping channels for every core
    :rtype: dict
    """
    response: dict[Any, dict[str, dict]] = {}

    for link in free_channels_dict.keys():
        response.update({link: {"overlapped_dict": {}, "non_over_dict": {}}})

        for band, free_channels in free_channels_dict[link].items():
            num_cores = int(len(free_channels.keys()))
            response[link]["overlapped_dict"][band] = {}
            response[link]["non_over_dict"][band] = {}

            for core_num, channels_list in free_channels.items():
                response[link]["overlapped_dict"][band][core_num] = []
                response[link]["non_over_dict"][band][core_num] = []

                for current_channel in channels_list:
                    for sub_core, slots_dict in free_slots_dict[link][band].items():
                        if sub_core == core_num:
                            continue

                        # The final core overlaps with all other cores
                        if core_num == num_cores - 1:
                            result_array = np.isin(
                                current_channel, slots_dict[sub_core]
                            )
                        else:
                            # Only certain cores neighbor each other on a fiber
                            first_neighbor = 5 if core_num == 0 else core_num - 1
                            second_neighbor = 0 if core_num == 5 else core_num + 1

                            result_array = np.isin(
                                current_channel,
                                free_slots_dict[link][band][first_neighbor],
                            )
                            result_array = np.append(
                                result_array,
                                np.isin(
                                    current_channel,
                                    free_slots_dict[link][band][second_neighbor],
                                ),
                            )
                            result_array = np.append(
                                result_array,
                                np.isin(
                                    current_channel,
                                    free_slots_dict[link][band][num_cores - 1],
                                ),
                            )

                        if result_array is False:
                            response[link]["overlapped_dict"][band][core_num].append(
                                current_channel
                            )
                            break

                    response[link]["non_over_dict"][band][core_num].append(
                        current_channel
                    )

    return response


def find_common_channels_on_paths(
    network_spectrum_dict: dict[Any, Any],
    paths: list[list[int]],
    slots_needed: int,
    band: str,
    core: int,
) -> list[int]:
    """
    Find slot indices available on ALL paths simultaneously.

    This function finds starting slot indices where contiguous spectrum
    is available across all provided paths. Useful for 1+1 protection
    where spectrum must be reserved on both primary and backup paths.

    :param network_spectrum_dict: Network spectrum database
    :type network_spectrum_dict: dict[Any, Any]
    :param paths: List of paths, where each path is a list of node IDs
    :type paths: list[list[int]]
    :param slots_needed: Number of contiguous slots needed
    :type slots_needed: int
    :param band: Spectrum band identifier
    :type band: str
    :param core: Core number
    :type core: int
    :return: Sorted list of starting slot indices available on all paths
    :rtype: list[int]

    Example:
        >>> primary_path = [0, 1, 2]
        >>> backup_path = [0, 3, 2]
        >>> common = find_common_channels_on_paths(
        ...     spectrum_dict,
        ...     [primary_path, backup_path],
        ...     slots_needed=4,
        ...     band="c",
        ...     core=0
        ... )
        >>> print(common)
        [10, 20, 35]  # Starting indices where 4 slots are free on both paths
    """
    if not paths:
        return []

    common_starts: set[int] | None = None

    for path in paths:
        if len(path) < 2:
            return []

        path_starts: set[int] | None = None

        # Check each link in the path
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])

            if link not in network_spectrum_dict:
                return []

            # Find free channels on this link using existing utility
            channels = find_free_channels(
                network_spectrum_dict=network_spectrum_dict,
                slots_needed=slots_needed,
                link_tuple=link,
            )

            # Extract starting indices for the specified band/core
            if band not in channels or core not in channels[band]:
                return []

            # Get set of starting slot indices from channel lists
            link_starts = {ch[0] for ch in channels[band][core] if len(ch) > 0}

            # Intersection with previous links in this path
            if path_starts is None:
                path_starts = link_starts
            else:
                path_starts &= link_starts

            # Early exit if no common slots on this path
            if not path_starts:
                return []

        # Intersection with previous paths
        if common_starts is None:
            common_starts = path_starts
        else:
            if path_starts is not None:
                common_starts &= path_starts

    return sorted(common_starts) if common_starts else []


def adjacent_core_indices(core_id: int, cores_per_link: int) -> list[int]:
    """
    Return list of adjacent core indices based on known core layout.

    :param core_id: Core ID to find adjacencies for
    :type core_id: int
    :param cores_per_link: Total number of cores per link
    :type cores_per_link: int
    :return: List of adjacent core indices
    :rtype: list[int]
    """
    if cores_per_link == 7:
        # 7-core layout: ring of 6 cores (0-5) plus center core (6)
        if core_id == 6:
            # Center core is adjacent to all outer cores
            return list(range(6))
        # Outer cores: adjacent to prev, next (in ring), and center
        before = 5 if core_id == 0 else core_id - 1
        after = 0 if core_id == 5 else core_id + 1
        return [before, after, 6]
    elif cores_per_link == 4:
        # 4-core layout: 2x2 grid
        adjacency_map = {
            0: [1, 2],    # top-left
            1: [0, 3],    # top-right
            2: [0, 3],    # bottom-left
            3: [1, 2],    # bottom-right
        }
        return adjacency_map.get(core_id, [])
    elif cores_per_link == 13:
        # 13-core layout: 2 hexagonal rings
        adjacency_map = {
            0: [1, 5, 6],
            1: [0, 2, 6],
            2: [1, 3, 6],
            3: [2, 4, 6],
            4: [3, 5, 6],
            5: [0, 4, 6],
            6: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],  # center of inner ring
            7: [6, 8, 12],
            8: [6, 7, 9],
            9: [6, 8, 10],
            10: [6, 9, 11],
            11: [6, 10, 12],
            12: [6, 7, 11],
        }
        return adjacency_map.get(core_id, [])
    elif cores_per_link == 19:
        # 19-core layout: 3 hexagonal rings
        adjacency_map = {
            0: [1, 5, 6],
            1: [0, 2, 6],
            2: [1, 3, 6],
            3: [2, 4, 6],
            4: [3, 5, 6],
            5: [0, 4, 6],
            6: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],  # center of inner ring
            7: [6, 8, 12, 13],
            8: [6, 7, 9, 13],
            9: [6, 8, 10, 14],
            10: [6, 9, 11, 15],
            11: [6, 10, 12, 16],
            12: [6, 7, 11, 17],
            13: [7, 8, 14, 18],
            14: [8, 9, 13, 15],
            15: [9, 10, 14, 16],
            16: [10, 11, 15, 17],
            17: [11, 12, 16, 18],
            18: [12, 13, 17],
        }
        return adjacency_map.get(core_id, [])
    else:
        # Default: no adjacency for unknown layouts
        return []


def edge_set(path: list, bidirectional: bool = True) -> set[tuple]:
    """
    Return normalized set of links from a path.

    Always normalizes edges by sorting node IDs to ensure (A,B) == (B,A).
    This matches V5 behavior where edges are always normalized regardless
    of the bidirectional parameter.

    :param path: List of node IDs representing a path
    :type path: list
    :param bidirectional: Whether links are bidirectional (kept for API compat, always normalizes)
    :type bidirectional: bool
    :return: Set of link tuples (always sorted)
    :rtype: set[tuple]
    """
    # Always normalize edges like V5 - sort to collapse direction
    return {tuple(sorted((u, v))) for u, v in zip(path, path[1:])}


def get_overlapping_lightpaths(
    new_lp: dict,
    lp_list: list[dict],
    *,
    cores_per_link: int,
    include_adjacent_cores: bool = True,
    include_all_bands: bool = True,
    bidirectional_links: bool = True,
) -> list[dict]:
    """
    Return lightpaths that overlap with a new lightpath.

    Two lightpaths overlap if they:
    1. Share at least one link
    2. Use the same or adjacent cores (if include_adjacent_cores=True)
    3. May overlap spectrally (spectrum overlap check not included)

    :param new_lp: New lightpath dict with keys: path, core, spectrum, band
    :type new_lp: dict
    :param lp_list: List of existing lightpath dicts
    :type lp_list: list[dict]
    :param cores_per_link: Number of cores per fiber link
    :type cores_per_link: int
    :param include_adjacent_cores: Whether to include adjacent cores
    :type include_adjacent_cores: bool
    :param include_all_bands: Whether to include all bands
    :type include_all_bands: bool
    :param bidirectional_links: Whether links are bidirectional
    :type bidirectional_links: bool
    :return: List of overlapping lightpaths
    :rtype: list[dict]
    """
    new_edges = edge_set(new_lp["path"], bidirectional_links)
    new_core = new_lp["core"]

    # Get adjacent cores if requested
    adj_cores = adjacent_core_indices(new_core, cores_per_link) if include_adjacent_cores else []

    affected = []

    for lp in lp_list:
        # Check link overlap
        lp_edges = edge_set(lp["path"], bidirectional_links)
        intersection = lp_edges & new_edges

        if not intersection:
            continue

        # Check core overlap
        lp_core = lp.get("core")
        if not (lp_core == new_core or lp_core in adj_cores):
            continue

        affected.append(lp)

    return affected
