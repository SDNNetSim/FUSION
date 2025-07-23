from pathlib import Path

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


def create_bw_info(mod_assumption: str, mod_assumptions_path: str = None) -> dict:
    """
    Determines reach and slots needed for each bandwidth and modulation format.

    :param mod_assumption: Controls which assumptions to be used.
    :param mod_assumptions_path: Path to modulation assumptions file.
    :return: The number of spectral slots needed for each bandwidth and modulation format pair.
    :rtype: dict
    """
    # Set default path if none provided
    if not mod_assumptions_path or mod_assumptions_path == "None":
        mod_assumptions_path = Path("data/json_input/run_mods/mod_formats.json")
    else:
        mod_assumptions_path = Path(mod_assumptions_path)

    # Resolve to absolute path
    if not mod_assumptions_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]  # Adjust if needed
        mod_assumptions_path = project_root / mod_assumptions_path

    try:
        with mod_assumptions_path.open("r", encoding="utf-8") as mod_fp:
            mod_formats_obj = json.load(mod_fp)

        if mod_assumption in mod_formats_obj:
            return mod_formats_obj[mod_assumption]

    except json.JSONDecodeError as json_err:
        raise FileExistsError(f"[ERROR] Could not parse JSON: {json_err.doc}") from json_err
    except FileNotFoundError as file_err:
        raise FileNotFoundError(f"[ERROR] File not found: {mod_assumptions_path}") from file_err

    raise NotImplementedError(f"[ERROR] Unknown modulation assumption '{mod_assumption}'")
