import math

from arg_scripts.data_args import YUE_MOD_ASSUMPTIONS, ARASH_MOD_ASSUMPTIONS


def create_pt(cores_per_link: int, net_spec_dict: dict):
    """
    Generates information relevant to the physical topology of the network.

    :param cores_per_link: The number of cores in each fiber's link.
    :param net_spec_dict: The network spectrum database.
    :return: Physical layer information topology of the network.
    :rtype: dict
    """
    fiber_props_dict = {
        'attenuation': 0.2 / 4.343 * 1e-3,
        'non_linearity': 1.3e-3,
        'dispersion': (16e-6 * 1550e-9 ** 2) / (2 * math.pi * 3e8),
        'num_cores': cores_per_link,
        'fiber_type': 0,
        'bending_radius': 0.05,
        'mode_coupling_co': 4.0e-4,
        'propagation_const': 4e6,
        'core_pitch': 4e-5,
    }

    topology_dict = {
        'nodes': {node: {'type': 'CDC'} for nodes in net_spec_dict for node in nodes},
        'links': {},
    }

    for link_num, (source_node, destination_node) in enumerate(net_spec_dict, 1):
        link_props_dict = {
            'fiber': fiber_props_dict,
            'length': net_spec_dict[(source_node, destination_node)],
            'source': source_node,
            'destination': destination_node,
            'span_length': 100,
        }
        topology_dict['links'][link_num] = link_props_dict
    return topology_dict


def create_bw_info(sim_type: str):
    """
    Determines reach and slots needed for each bandwidth and modulation format.

    :param sim_type: Controls which assumptions to be used.
    :return: The number of spectral slots needed for each bandwidth and modulation format pair.
    :rtype: dict
    """
    if sim_type == 'yue':
        bw_mod_dict = YUE_MOD_ASSUMPTIONS
    elif sim_type == 'arash':
        bw_mod_dict = ARASH_MOD_ASSUMPTIONS
    else:
        raise NotImplementedError(f"Invalid simulation type '{sim_type}'")

    return bw_mod_dict
