import math
import json
import os


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


def create_bw_info(mod_assumption: str, mod_assumptions_path: str = None):
    """
    Determines reach and slots needed for each bandwidth and modulation format.

    :param mod_assumption: Controls which assumptions to be used.
    :param mod_assumptions_path: Path to modulation assumptions file.
    :return: The number of spectral slots needed for each bandwidth and modulation format pair.
    :rtype: dict
    """
    if mod_assumptions_path is None or mod_assumptions_path == 'None':
        base_fp = os.path.join('data', 'json_input', 'run_mods')
        mod_assumptions_path = os.path.join(base_fp, 'mod_formats.json')

    try:
        mod_assumptions_path = os.path.join(mod_assumptions_path)
        with open(mod_assumptions_path, 'r', encoding='utf-8') as mod_assumptions_fp:
            mod_formats_obj = json.load(mod_assumptions_fp)

        if mod_assumption in mod_formats_obj.keys():
            return mod_formats_obj[mod_assumption]
    except json.JSONDecodeError as json_decode_error:
        raise FileExistsError(f"Could not parse: {json_decode_error.doc}")  # pylint: disable=raise-missing-from
    except FileNotFoundError as file_not_found:
        raise FileNotFoundError(f"Could not find: {file_not_found.strerror}: {file_not_found.filename}") # pylint: disable=raise-missing-from

    raise NotImplementedError(f"Unknown modulation assumption '{mod_assumption}'")
