import os

import numpy as np


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
    file_mapping = file_mapping_dict[network]

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
