import os

import numpy as np


#TODO Ensure Geometry data matches requirements for 13 and 19 cores
def get_loaded_files(core_num: int, cores_per_link: int, file_mapping_dict: dict, network: str):
    """
    Fetch the appropriate modulation format and GSNR files based on core_num and cores_per_link.

    :param core_num: The core number being used.
    :param cores_per_link: The total number of cores per link.
    :param file_mapping_dict: A dictionary mapping (core_num, cores_per_link) to file paths.
    :param network: The current network.
    :return: The loaded modulation format and GSNR data.
    :rtype: tuple
    """
    if core_num == 0:
        key = 'multi_fiber'
    else:
        key = (core_num, cores_per_link)

    base_path = os.path.join('data', 'pre_calc', network)
    file_mapping = file_mapping_dict[0][network]

    if key in file_mapping:
        mf_path = os.path.join(base_path, 'modulations', file_mapping[key]['mf'])
        gsnr_path = os.path.join(base_path, 'snr', file_mapping[key]['gsnr'])
        return (
            np.load(mf_path, allow_pickle=True),
            np.load(gsnr_path, allow_pickle=True),
        )
    raise ValueError(f"No matching file found for core_num={core_num}, cores_per_link={cores_per_link}")


def get_slot_index(curr_band, start_slot, engine_props):
    """
    Compute the slot index based on the current band and start slot.

    :param curr_band: The current band ('l', 'c', or 's').
    :param start_slot: The starting slot index.
    :param engine_props: The engine properties containing band offsets.
    :return: The computed slot index.
    :rtype: int
    """
    band_offset = {
        'l': 0,
        'c': engine_props['l_band'],
        's': engine_props['l_band'] + engine_props['c_band'],
    }
    if curr_band not in band_offset:
        raise ValueError(f"Unexpected band: {curr_band}")
    return band_offset[curr_band] + start_slot


def compute_response(mod_format, snr_props, spectrum_props, sdn_props):
    """
    Compute whether the SNR threshold can be met and validate modulation.

    :param mod_format: The modulation format retrieved from the data.
    :param snr_props: The SNR properties.
    :param spectrum_props: The spectrum properties.
    :param sdn_props: The SDN properties containing bandwidth.
    :return: Whether the SNR threshold is met.
    :rtype: bool
    """
    is_valid_modulation = (
            snr_props.mod_format_mapping_dict[mod_format] == spectrum_props.modulation
    )
    meets_bw_requirements = (
            snr_props.bw_mapping_dict[spectrum_props.modulation] >= int(sdn_props.bandwidth)
    )
    return mod_format != 0 and is_valid_modulation and meets_bw_requirements


def adjacent_core_indices(core_id: int, cores_per_link: int) -> list[int]:
    """Returns a list of adjacent core indices, based on known layout."""
    if cores_per_link == 7:
        if core_id == 6:
            return list(range(6))
        before = 5 if core_id == 0 else core_id - 1
        after = 0 if core_id == 5 else core_id + 1
        return [before, after, 6]

    if cores_per_link == 4:
        return {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}.get(core_id, [])

    if cores_per_link == 13:
        # Assuming hexagonal layout
        adjacency_map = {
            0: [1, 5, 6], 1: [0, 2, 6, 7], 2: [1, 3, 7, 8], 3: [2, 4, 8, 9], 4: [3, 5, 9, 10], 5: [0, 4, 10, 11],
            6: [0, 1, 7, 11, 12], 7: [1, 2, 6, 8, 12], 8: [2, 3, 7, 9, 12], 9: [3, 4, 8, 10, 12],
            10: [4, 5, 9, 11, 12], 11: [5, 6, 10, 12], 12: list(range(6, 12))
        }
        return adjacency_map.get(core_id, [])

    if cores_per_link == 19:
        # Placeholder implementation - NEEDS REAL GEOMETRY DATA
        if core_id < 6:  # Inner ring
            return [core_id - 1 if core_id > 0 else 5, (core_id + 1) % 6, core_id + 6, 18]
        elif 6 <= core_id < 18:  # Outer ring
            return [core_id - 6, core_id - 1 if core_id > 6 else 17, (core_id + 1) if core_id < 17 else 6, 18]
        else:  # Center core
            return list(range(18))

    return []

def edge_set(path: list[int], bidirectional: bool = True) -> set:
    """
    Return a normalized set of links from a path.
    If bidirectional, uses frozenset to collapse direction.
    """
    return {tuple(sorted((u, v))) for u, v in zip(path, path[1:])}


def get_overlapping_lightpaths(new_lp: dict, lp_list: list[dict], *, cores_per_link: int,
                               include_adjacent_cores: bool = True,
                               include_all_bands: bool = True,
                               bidirectional_links: bool = True) -> list[dict]:
    """
    Return LPs that overlap with a new LP in terms of link, slots, core, and band.
    """

    new_edges = edge_set(new_lp["path"], bidirectional_links)
    new_start, new_end = new_lp["spectrum"]
    new_core = new_lp["core"]
    new_band = new_lp.get("band")
    adj_cores = adjacent_core_indices(new_core, cores_per_link) if include_adjacent_cores else []

    affected = []

    for lp in lp_list:
        # For link overlap
        lp_edges = edge_set(lp["path"], bidirectional_links)
        if not (lp_edges & new_edges):
            continue

        # For core overlap
        lp_core = lp.get("core")
        if not (lp_core == new_core or lp_core in adj_cores):
            continue

        # for band overlap
        # lp_band = lp.get("band")
        # if not (include_all_bands or lp_band == new_band):
        #     continue

        # for slot interval overlap
        # lp_start, lp_end = lp["start_slot"], lp["end_slot"]
        # if new_end < lp_start or new_start > lp_end:
        #     continue

        affected.append(lp)

    return affected
