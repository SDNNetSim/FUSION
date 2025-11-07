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
