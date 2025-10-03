"""
Spectrum allocation and channel management utilities.

This module provides functions for finding and managing spectral resources,
including free/taken channels, super-channels, and fragmentation metrics.
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


def get_super_channels(input_array: np.ndarray, slots_needed: int) -> np.ndarray:
    """
    Get available super-channels with respect to the current request's needs.

    :param input_array: Current spectrum (a single core)
    :type input_array: np.ndarray
    :param slots_needed: Slots needed by the request
    :type slots_needed: int
    :return: Matrix of positions of available super-channels
    :rtype: np.ndarray
    """
    potential_super_channels = []
    consecutive_zeros = 0

    for i in range(len(input_array)):
        if input_array[i] == 0:
            consecutive_zeros += 1
            # Plus one to account for the guard band
            if consecutive_zeros >= (slots_needed + 1):
                start_position = i - slots_needed
                end_position = i

                if start_position == end_position:
                    potential_super_channels.append([start_position])
                else:
                    potential_super_channels.append([start_position, end_position])
        else:
            consecutive_zeros = 0

    return np.array(potential_super_channels)


def combine_and_one_hot(array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
    """
    Perform OR operation of two arrays to find overlaps.

    :param array1: First input array
    :type array1: np.ndarray
    :param array2: Second input array
    :type array2: np.ndarray
    :return: Output of the OR operation
    :rtype: np.ndarray
    :raises ValueError: If arrays have different lengths
    """
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length.")

    one_hot_array1 = (array1 != 0).astype(int)
    one_hot_array2 = (array2 != 0).astype(int)

    return np.array(one_hot_array1 | one_hot_array2)


def _get_hfrag_score(
    super_channel_index_matrix: np.ndarray, spectral_slots: int
) -> float:
    """
    Calculate Shannon entropy fragmentation score.

    Private helper function for get_shannon_entropy_fragmentation.

    :param super_channel_index_matrix: Matrix of super-channel indices
    :type super_channel_index_matrix: np.ndarray
    :param spectral_slots: Number of spectral slots
    :type spectral_slots: int
    :return: Shannon entropy fragmentation score
    :rtype: float
    """
    big_n = len(super_channel_index_matrix) * -1.0
    if big_n == 0.0:
        return float(np.inf)

    channel_length = len(super_channel_index_matrix[0])
    response_score = (
        big_n
        * (channel_length / spectral_slots)
        * np.log(channel_length / spectral_slots)
    )
    return float(response_score)


def get_shannon_entropy_fragmentation(
    path_list: list[int],
    core_num: int,
    band: str,
    slots_needed: int,
    spectral_slots: int,
    network_spectrum: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the Shannon entropy fragmentation scores for allocating a request.

    :param path_list: Current path
    :type path_list: list[int]
    :param core_num: Core number
    :type core_num: int
    :param band: Current allocated band
    :type band: str
    :param slots_needed: Slots needed by the request
    :type slots_needed: int
    :param spectral_slots: Number of spectral slots on a single core
    :type spectral_slots: int
    :param network_spectrum: Up-to-date network spectrum database
    :type network_spectrum: dict
    :return: Array with all Shannon entropy fragmentation scores
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    path_allocation_array = np.zeros(spectral_slots)
    response_fragmentation_array = np.ones(spectral_slots)

    # First fit for core, testing use only
    if core_num is None:
        core_num = 0

    for source, destination in zip(path_list, path_list[1:], strict=False):
        link_key = (source, destination)
        core_array = network_spectrum[link_key]["cores_matrix"][band][core_num]
        path_allocation_array = combine_and_one_hot(path_allocation_array, core_array)

    super_channel_index_matrix = get_super_channels(
        input_array=path_allocation_array, slots_needed=slots_needed
    )
    hfrag_before = _get_hfrag_score(
        super_channel_index_matrix=super_channel_index_matrix,
        spectral_slots=spectral_slots,
    )

    for super_channel in super_channel_index_matrix:
        mock_allocation_array = copy.deepcopy(path_allocation_array)
        for index in super_channel:
            mock_allocation_array[index] = 1

        temp_super_channel_matrix = get_super_channels(
            input_array=mock_allocation_array, slots_needed=slots_needed
        )
        hfrag_after = _get_hfrag_score(
            super_channel_index_matrix=temp_super_channel_matrix,
            spectral_slots=spectral_slots,
        )
        delta_hfrag = hfrag_before - hfrag_after
        start_index = super_channel[0]
        response_fragmentation_array[start_index] = np.round(delta_hfrag, 3)

    response_fragmentation_array = np.where(
        response_fragmentation_array == 1, np.inf, response_fragmentation_array
    )

    return super_channel_index_matrix, response_fragmentation_array
