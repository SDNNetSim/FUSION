import copy
from datetime import datetime

import networkx as nx
import numpy as np


def get_path_mod(mods_dict: dict, path_len: int):
    """
    Choose a modulation format that will allocate a network request.

    :param mods_dict: Information for maximum reach for each modulation format.
    :param path_len: The length of the path to be taken.
    :return: The chosen modulation format.
    :rtype: str
    """
    # Pycharm auto-formats it like this for comparisons...I'd rather this look weird than look at PyCharm warnings
    if mods_dict['QPSK']['max_length'] >= path_len > mods_dict['16-QAM']['max_length']:
        resp = 'QPSK'
    elif mods_dict['16-QAM']['max_length'] >= path_len > mods_dict['64-QAM']['max_length']:
        resp = '16-QAM'
    elif mods_dict['64-QAM']['max_length'] >= path_len:
        resp = '64-QAM'
    else:
        return False

    return resp


def find_max_path_len(source: int, destination: int, topology: nx.Graph):
    """
    Find the maximum path length possible of a path in the network.

    :param source: The source node.
    :param destination: The destination node.
    :param topology: The network topology.
    :return: The length of the longest path possible.
    :rtype: float
    """
    all_paths_list = list(nx.shortest_simple_paths(topology, source, destination))
    path_list = all_paths_list[-1]
    resp = find_path_len(path_list=path_list, topology=topology)

    return resp


def sort_nested_dict_vals(original_dict: dict, nested_key: str):
    """
    Sort a dictionary by a value which belongs to a nested key.

    :param original_dict: The original dictionary.
    :param nested_key: The nested key to sort by.
    :return: The sorted dictionary, ascending.
    :rtype: dict
    """
    sorted_items = sorted(original_dict.items(), key=lambda x: x[1][nested_key])
    sorted_dict = dict(sorted_items)
    return sorted_dict


def sort_dict_keys(dictionary: dict):
    """
    Sort a dictionary by keys in descending order.

    :param dictionary: The dictionary to sort.
    :return: The newly sorted dictionary.
    :rtype: dict
    """
    sorted_keys = sorted(map(int, dictionary.keys()), reverse=True)
    sorted_dict = {str(key): dictionary[str(key)] for key in sorted_keys}

    return sorted_dict


def find_path_len(path_list: list, topology: nx.Graph):
    """
    Finds the length of a path in a physical topology.

    :param path_list: A list of integers representing the nodes in the path.
    :param topology: The network topology.
    :return: The length of the path.
    """
    path_len = 0
    for i in range(len(path_list) - 1):
        path_len += topology[path_list[i]][path_list[i + 1]]['length']

    return path_len


def find_path_cong(path_list: list, net_spec_dict: dict):
    """
    Finds the average percentage of congestion for a given path.

    :param path_list: The path to be analyzed.
    :param net_spec_dict: The current up-to-date network spectrum database.
    :return: The average congestion as a decimal.
    :rtype: float
    """
    # Divide by the total length of that array
    links_cong_list = list()
    for src, dest in zip(path_list, path_list[1:]):
        src_dest = (src, dest)
        cores_matrix = net_spec_dict[src_dest]['cores_matrix']
        cores_per_link = float(len(cores_matrix))

        # Every core will have the same number of spectral slots
        total_slots = len(cores_matrix[0])
        slots_taken = 0
        for curr_core in cores_matrix:
            core_slots_taken = float(len(np.where(curr_core != 0.0)[0]))
            slots_taken += core_slots_taken

        links_cong_list.append(slots_taken / (total_slots * cores_per_link))

    average_path_cong = np.mean(links_cong_list)
    return average_path_cong


def find_core_frag_cong(net_spec_db, path: list, core: int):
    frag_resp = 0.0
    cong_resp = 0.0
    for src, dest in zip(path, path[1:]):
        src_dest = (src, dest)
        core_arr = net_spec_db[src_dest]['cores_matrix'][core]

        if len(core_arr) != 256:
            raise NotImplementedError('Only works for 256 spectral slots.')

        cong_resp += len(np.where(core_arr != 0)[0])

        count = 0
        in_zero_group = False

        for number in core_arr:
            if number == 0:
                if not in_zero_group:
                    in_zero_group = True
            else:
                if in_zero_group:
                    count += 1
                    in_zero_group = False

        frag_resp += count

    num_links = len(path) - 1
    # The lowest number of slots a request can take is 2, the max number of times [1, 1, 0, 2, 2, 0, ..., 5, 5, 0]
    # fragmentation can happen is 43 for 128 spectral slots and 86 for 256 spectral slots (Rounded)
    frag_resp = (frag_resp / 86.0 / num_links)
    cong_resp = (cong_resp / 256.0 / num_links)
    return frag_resp, cong_resp


def get_channel_overlaps(free_channels_dict: dict, free_slots_dict: dict):
    """
    Find the number of overlapping and non-overlapping channels between adjacent cores.

    :param free_channels_dict: The free super-channels found on a path.
    :param free_slots_dict: The free slots found on the given path.
    :return: The overlapping and non-overlapping channels for every core.
    :rtype: dict
    """
    resp = {'overlapped_dict': {}, 'non_over_dict': {}}
    num_cores = int(len(free_channels_dict.keys()))

    for core_num, channels_list in free_channels_dict.items():
        resp['overlapped_dict'][core_num] = list()
        resp['non_over_dict'][core_num] = list()

        for curr_channel in channels_list:
            is_overlapped = False
            for sub_core in range(0, num_cores):
                if sub_core == core_num:
                    continue

                for _, slots_dict in free_slots_dict.items():
                    # The final core overlaps with all other cores
                    if core_num == num_cores - 1:
                        result = np.isin(curr_channel, slots_dict[sub_core])
                    else:
                        # Only certain cores neighbor each other on a fiber
                        first_neighbor = 5 if core_num == 0 else core_num - 1
                        second_neighbor = 0 if core_num == 5 else core_num + 1

                        result = np.isin(curr_channel, slots_dict[first_neighbor])
                        result = np.append(result, np.isin(curr_channel, slots_dict[second_neighbor]))
                        result = np.append(result, np.isin(curr_channel, slots_dict[num_cores - 1]))

                    if np.any(result):
                        resp['overlapped_dict'][core_num].append(curr_channel)
                        is_overlapped = True
                        break

                    resp['non_over_dict'][core_num].append(curr_channel)

                # No need to check other cores, we already determined this channel overlaps with other channels
                if is_overlapped:
                    break

    return resp


def find_free_slots(net_spec_dict: dict, link_tuple: tuple):
    """
    Find every unallocated spectral slot for a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param link_tuple: The link to find the free slots on.
    :return: The indexes of the free spectral slots on the link for each core.
    :rtype: dict
    """
    link = net_spec_dict[link_tuple]['cores_matrix']
    resp = {}
    for core_num in range(len(link)):  # pylint: disable=consider-using-enumerate
        free_slots_list = np.where(link[core_num] == 0)[0]
        resp.update({core_num: free_slots_list})

    return resp


def find_free_channels(net_spec_dict: dict, slots_needed: int, link_tuple: tuple):
    """
    Finds the free super-channels on a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param slots_needed: The number of slots needed for the request.
    :param link_tuple: The link to search on.
    :return: Available super-channels for every core.
    :rtype: dict
    """
    resp = {}
    cores_matrix = copy.deepcopy(net_spec_dict[link_tuple]['cores_matrix'])
    for core_num, link_list in enumerate(cores_matrix):
        indexes = np.where(link_list == 0)[0]
        channels_list = []
        curr_channel_list = []

        for i, free_index in enumerate(indexes):
            if i == 0:
                curr_channel_list.append(free_index)
            elif free_index == indexes[i - 1] + 1:
                curr_channel_list.append(free_index)
                if len(curr_channel_list) == slots_needed:
                    channels_list.append(curr_channel_list.copy())
                    curr_channel_list.pop(0)
            else:
                curr_channel_list = [free_index]

        resp.update({core_num: channels_list})

    return resp


def find_taken_channels(net_spec_dict: dict, link_tuple: tuple):
    """
    Finds the taken super-channels on a given link.

    :param net_spec_dict: The most updated network spectrum database.
    :param link_tuple: The link to search on.
    :return: Unavailable super-channels for every core.
    :rtype: dict
    """
    resp = {}
    cores_matrix = copy.deepcopy(net_spec_dict[link_tuple]['cores_matrix'])
    for core_num, link_list in enumerate(cores_matrix):
        channels_list = []
        curr_channel_list = []

        for value in link_list:
            if value > 0:
                curr_channel_list.append(value)
            elif value < 0 and curr_channel_list:
                channels_list.append(curr_channel_list)
                curr_channel_list = []

        if curr_channel_list:
            channels_list.append(curr_channel_list)

        resp[core_num] = channels_list

    return resp


def snake_to_title(snake_str: str):
    """
    Converts a snake string to a title string.

    :param snake_str: The string to convert in snake case.
    :return: Another string in title case.
    :rtype: str
    """
    words_list = snake_str.split('_')
    title_str = ' '.join(word.capitalize() for word in words_list)
    return title_str


def int_to_string(number: int):
    """
    Converts an integer to a string.

    :param number: The number to convert.
    :return: The original number as a string.
    :rtype: str
    """
    return '{:,}'.format(number)  # pylint: disable=consider-using-f-string


def dict_to_list(data_dict: dict, nested_key: str, path_list: list = None, find_mean: bool = False):
    """
    Creates a list from a dictionary taken values from a specified key.

    :param data_dict: The dictionary to search.
    :param nested_key: Where to take values from.
    :param path_list: If the key is nested, the path is to that nested key.
    :param find_mean: Flag to return mean or not.
    :return: A list or single value.
    :rtype: list or float
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
        return np.mean(extracted_list)

    return np.array(extracted_list)


def list_to_title(input_list: list):
    """
    Converts a list to a title case.

    :param input_list: The input list to convert, each element is a word.
    :return: A title string.
    :rtype: str
    """
    if not input_list:
        return ""

    unique_list = list()
    for item in input_list:
        if item[0] not in unique_list:
            unique_list.append(item[0])

    if len(unique_list) > 1:
        return ", ".join(unique_list[:-1]) + " & " + unique_list[-1]

    return unique_list[0]


def calc_matrix_stats(input_dict: dict):
    """
    Creates a matrix based on dict values and takes the min, max, and average of columns.
    :param input_dict: The input dict with values as lists.
    :return: The min, max, and average of columns.
    :rtype: dict
    """
    resp_dict = dict()
    tmp_matrix = np.array([])
    for episode, curr_list in input_dict.items():
        if episode == '0':
            tmp_matrix = np.array([curr_list])
        else:
            tmp_matrix = np.vstack((tmp_matrix, curr_list))

    resp_dict['min'] = tmp_matrix.min(axis=0, initial=np.inf).tolist()
    resp_dict['max'] = tmp_matrix.max(axis=0, initial=np.inf * -1.0).tolist()
    resp_dict['average'] = tmp_matrix.mean(axis=0).tolist()

    return resp_dict


def combine_and_one_hot(arr1: np.array, arr2: np.array):
    """
    Or operation of two arrays to find overlaps.
    :param arr1: The first input array.
    :param arr2: The second input array.
    :return: The output of the or operation.
    :rtype: np.array
    """
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same length.")

    one_hot_arr1 = (arr1 != 0).astype(int)
    one_hot_arr2 = (arr2 != 0).astype(int)

    result = one_hot_arr1 | one_hot_arr2
    return result


def get_start_time(sim_dict: dict):
    """
    Gets the start time of a simulation.
    :param sim_dict: Holds the simulation parameters.
    """
    sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
    sim_dict['s1']['date'] = sim_start.split('_')[0]
    tmp_list = sim_start.split('_')

    time_string = f'{tmp_list[1]}_{tmp_list[2]}_{tmp_list[3]}_{tmp_list[4]}'
    sim_dict['s1']['sim_start'] = time_string


def min_max_scale(value: float, min_value: float, max_value: float):
    return (value - min_value) / (max_value - min_value)


def get_super_channels(input_arr: np.array, slots_needed: int):
    potential_super_channels = []
    current_start = 0
    consecutive_zeros = 0

    for i in range(len(input_arr)):
        if input_arr[i] == 0:
            consecutive_zeros += 1
        else:
            consecutive_zeros = 0
            current_start = i + 1

        if consecutive_zeros >= slots_needed:
            potential_super_channels.append([current_start, i])
            consecutive_zeros -= 1
            current_start += 1

    return np.array(potential_super_channels)


# TODO: Add reference
# Please refer to this paper for the formulation:
def _get_hfrag_score(sc_index_mat: np.array, spectral_slots: int):
    big_n = len(sc_index_mat) * -1.0
    if big_n == 0.0:
        return np.inf

    channel_len = len(sc_index_mat[0])
    resp_score = big_n * (channel_len / spectral_slots) * np.log((channel_len / spectral_slots))
    return resp_score


def get_hfrag(path_list: list, core_num: int, slots_needed: int, spectral_slots: int, net_spec_dict: dict):
    path_alloc_arr = np.zeros(spectral_slots)
    resp_frag_arr = np.ones(spectral_slots)
    if core_num is None:
        core_num = 0

    for source, dest in zip(path_list, path_list[1:]):
        core_arr = net_spec_dict[(source, dest)]['cores_matrix'][core_num]
        path_alloc_arr = combine_and_one_hot(path_alloc_arr, core_arr)

    sc_index_mat = get_super_channels(input_arr=path_alloc_arr, slots_needed=slots_needed)
    hfrag_before = _get_hfrag_score(sc_index_mat=sc_index_mat, spectral_slots=spectral_slots)
    for super_channel in sc_index_mat:
        mock_alloc_arr = copy.deepcopy(path_alloc_arr)
        for index in super_channel:
            mock_alloc_arr[index] = 1

        tmp_sc_mat = get_super_channels(input_arr=mock_alloc_arr, slots_needed=slots_needed)
        hfrag_after = _get_hfrag_score(sc_index_mat=tmp_sc_mat, spectral_slots=spectral_slots)
        delta_hfrag = hfrag_before - hfrag_after
        start_index = super_channel[0]
        resp_frag_arr[start_index] = np.round(delta_hfrag, 3)

    resp_frag_arr = np.where(resp_frag_arr == 1, np.inf, resp_frag_arr)

    return sc_index_mat, resp_frag_arr
