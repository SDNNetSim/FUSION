import json
from pathlib import Path

from fusion.utils.os import find_project_root


def create_pt(
    cores_per_link: int, network_spectrum_dict: dict[tuple, float]
) -> dict[str, dict]:
    """Generate information relevant to the physical topology of the network.

    :param cores_per_link: The number of cores in each fiber's link
    :type cores_per_link: int
    :param network_spectrum_dict: The network spectrum database mapping
        node pairs to lengths
    :type network_spectrum_dict: Dict[tuple, float]
    :return: Physical layer information topology of the network
    :rtype: Dict[str, Dict]
    """
    fiber_props_dict = {
        "attenuation": 0.2 / 4.343 * 1e-3,
        "non_linearity": 1.3e-3,
        "dispersion": -21.3e-27,
        "num_cores": cores_per_link,
        "fiber_type": 0,
        "bending_radius": 0.05,
        "mode_coupling_co": 4.0e-4,
        "propagation_const": 4e6,
        "core_pitch": 4e-5,
        # Optical band frequency ranges (in Hz) - standard ITU-T specifications
        "frequency_start_c": 3e8 / 1565e-9,  # C-band start from 1565nm wavelength (~191.69 THz)
        "frequency_end_c": (3e8 / 1565e-9) + 6.0e12,  # C-band end (match v5)
        "frequency_start_l": 3e8 / 1620e-9,  # L-band start: from 1620nm wavelength
        "frequency_end_l": (3e8 / 1620e-9) + 6.0e12,  # L-band end (match v5)
        "frequency_start_s": 185.0e12,  # S-band start: 185.0 THz (~1460 nm)
        "frequency_end_s": 190.0e12,    # S-band end: 190.0 THz (~1530 nm)
        "c_band_bw": 6.0e12,
        # Multi-band GSNR parameters (ported from v5 for C+L band calculations)
        "raman_gain_slope": 0.028 / 1e3 / 1e12,  # C_r Raman gain slope
        "gvd": -22.6 * (1e-12 * 1e-12) / 1e3,    # beta2 - Group velocity dispersion
        "gvd_slope": 0.14 * (1e-12 * 1e-12 * 1e-12) / 1e3,  # beta3 - GVD slope
    }

    topology_dict = {
        "nodes": {
            node: {"type": "CDC"} for nodes in network_spectrum_dict for node in nodes
        },
        "links": {},
    }

    for link_num, (source_node, destination_node) in enumerate(
        network_spectrum_dict, 1
    ):
        link_props_dict = {
            "fiber": fiber_props_dict,
            "length": network_spectrum_dict[(source_node, destination_node)],
            "source": source_node,
            "destination": destination_node,
            "span_length": 100,
        }
        topology_dict["links"][link_num] = link_props_dict

    # Validation check to ensure we have nodes
    if not topology_dict["nodes"]:
        keys_sample = list(network_spectrum_dict.keys())[:5]
        raise ValueError(
            f"create_pt generated empty nodes dictionary. Input "
            f"network_spectrum_dict had {len(network_spectrum_dict)} "
            f"links: {keys_sample}..."
        )

    return topology_dict


def create_bw_info(
    mod_assumption: str, mod_assumptions_path: str | None = None
) -> dict[str, dict]:
    """Determine reach and slots needed for each bandwidth and modulation format.

    :param mod_assumption: Controls which assumptions to be used
    :type mod_assumption: str
    :param mod_assumptions_path: Path to modulation assumptions file
    :type mod_assumptions_path: Optional[str]
    :return: The number of spectral slots needed for each bandwidth and
        modulation format pair
    :rtype: Dict[str, Dict]
    :raises FileNotFoundError: If modulation assumptions file is not found
    :raises NotImplementedError: If unknown modulation assumption is provided
    """
    # Set default path if none provided
    if not mod_assumptions_path or mod_assumptions_path == "None":
        project_root = Path(find_project_root())
        resolved_path = project_root / "data/json_input/run_mods/mod_formats.json"
    else:
        resolved_path = Path(mod_assumptions_path)

    # Resolve to absolute path
    if not resolved_path.is_absolute():
        project_root = Path(find_project_root())
        resolved_path = project_root / resolved_path

    try:
        with resolved_path.open("r", encoding="utf-8") as modulation_file:
            modulation_formats_dict = json.load(modulation_file)

        if mod_assumption in modulation_formats_dict:
            result: dict[str, dict] = modulation_formats_dict[mod_assumption]
            return result

    except json.JSONDecodeError as json_error:
        raise FileNotFoundError(
            f"Could not parse JSON: {json_error.doc}"
        ) from json_error
    except FileNotFoundError as file_error:
        raise FileNotFoundError(f"File not found: {resolved_path}") from file_error

    raise NotImplementedError(f"Unknown modulation assumption '{mod_assumption}'")
