import json
import math
from pathlib import Path
from typing import Dict, Optional


def create_pt(cores_per_link: int, network_spectrum_dict: Dict[tuple, float]) -> Dict[str, Dict]:
    """Generate information relevant to the physical topology of the network.
    
    :param cores_per_link: The number of cores in each fiber's link
    :type cores_per_link: int
    :param network_spectrum_dict: The network spectrum database mapping node pairs to lengths
    :type network_spectrum_dict: Dict[tuple, float]
    :return: Physical layer information topology of the network
    :rtype: Dict[str, Dict]
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
        'nodes': {node: {'type': 'CDC'} for nodes in network_spectrum_dict for node in nodes},
        'links': {},
    }

    for link_num, (source_node, destination_node) in enumerate(network_spectrum_dict, 1):
        link_props_dict = {
            'fiber': fiber_props_dict,
            'length': network_spectrum_dict[(source_node, destination_node)],
            'source': source_node,
            'destination': destination_node,
            'span_length': 100,
        }
        topology_dict['links'][link_num] = link_props_dict

    # Validation check to ensure we have nodes
    if not topology_dict['nodes']:
        raise ValueError(
            f"create_pt generated empty nodes dictionary. Input network_spectrum_dict had {len(network_spectrum_dict)} links: {list(network_spectrum_dict.keys())[:5]}...")

    return topology_dict


def create_bw_info(mod_assumption: str, mod_assumptions_path: Optional[str] = None) -> Dict[str, Dict]:
    """Determine reach and slots needed for each bandwidth and modulation format.
    
    :param mod_assumption: Controls which assumptions to be used
    :type mod_assumption: str
    :param mod_assumptions_path: Path to modulation assumptions file
    :type mod_assumptions_path: Optional[str]
    :return: The number of spectral slots needed for each bandwidth and modulation format pair
    :rtype: Dict[str, Dict]
    :raises FileNotFoundError: If modulation assumptions file is not found
    :raises NotImplementedError: If unknown modulation assumption is provided
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
        with mod_assumptions_path.open("r", encoding="utf-8") as modulation_file:
            modulation_formats_dict = json.load(modulation_file)

        if mod_assumption in modulation_formats_dict:
            return modulation_formats_dict[mod_assumption]

    except json.JSONDecodeError as json_error:
        raise FileExistsError(f"Could not parse JSON: {json_error.doc}") from json_error
    except FileNotFoundError as file_error:
        raise FileNotFoundError(f"File not found: {mod_assumptions_path}") from file_error

    raise NotImplementedError(f"Unknown modulation assumption '{mod_assumption}'")
