from pathlib import Path

from fusion.utils.os import find_project_root


def assign_link_lengths(
    network_fp: Path,
    node_pairs_dict: dict[str, str],
    constant_weight: bool = False
) -> dict[tuple[str, str], float]:
    """Assign length to each link in a given topology.

    :param network_fp: Path to the network topology file
    :type network_fp: Path
    :param node_pairs_dict: Dictionary mapping node names to alternative names
    :type node_pairs_dict: Dict[str, str]
    :param constant_weight: If True, assign constant weight of 1.0 to all links
    :type constant_weight: bool
    :return: Dictionary mapping node pairs to link lengths
    :rtype: Dict[Tuple[str, str], float]
    """
    link_lengths_dict = {}
    with network_fp.open('r', encoding='utf-8') as file_obj:
        for line in file_obj:
            src, dest, link_len_str = line.strip().split('\t')
            link_len = float(link_len_str) if not constant_weight else 1.0

            src_dest_tuple = (
                node_pairs_dict.get(src, src),
                node_pairs_dict.get(dest, dest)
            )

            if src_dest_tuple not in link_lengths_dict:
                link_lengths_dict[src_dest_tuple] = link_len

    return link_lengths_dict


def assign_core_nodes(core_nodes_fp: Path) -> list[str]:
    """Determine which nodes are core nodes in the network.

    :param core_nodes_fp: Path to the core nodes file
    :type core_nodes_fp: Path
    :return: List of core node identifiers
    :rtype: List[str]
    """
    core_nodes_list = []
    with core_nodes_fp.open('r', encoding='utf-8') as file_obj:
        for line in file_obj:
            core_nodes_list.append(line.strip().split('\t')[0])
    return core_nodes_list


def create_network(
    net_name: str,
    base_fp: str | None = None,
    const_weight: bool = False,
    is_only_core_node: bool = False
) -> tuple[dict[tuple[str, str], float], list[str]]:
    """Build a physical network structure from topology files.

    Resolves all paths safely using project-root-relative logic.

    :param net_name: Name of the network topology to load
    :type net_name: str
    :param base_fp: Base file path for topology files
    :type base_fp: str
    :param const_weight: If True, assign constant weight to all links
    :type const_weight: bool
    :param is_only_core_node: If True, only load core nodes
    :type is_only_core_node: bool
    :return: Tuple of (network dictionary, core nodes list)
    :rtype: Tuple[Dict[Tuple[str, str], float], List[str]]
    :raises NotImplementedError: If unknown network name is provided
    """
    core_nodes_list = []

    # Resolve base_fp to absolute path
    project_root = Path(find_project_root())
    if base_fp is None:
        base_path = project_root / "data/raw"
    else:
        base_path = Path(base_fp) / "raw"
        if not base_path.is_absolute():
            base_path = project_root / base_path

    # Map network names to file paths
    network_files = {
        'USNet': 'us_network.txt',
        'NSFNet': 'nsf_network.txt',
        'Pan-European': 'europe_network.txt',
        'USbackbone60': 'USB6014.txt',
        'Spainbackbone30': 'SPNB3014.txt',
        'geant': 'geant.txt',
        'toy_network': 'toy_network.txt',
        'metro_net': 'metro_net.txt',
        'dt_network': 'dt_network.txt',
    }

    if net_name not in network_files:
        raise NotImplementedError(f"Unknown network name: '{net_name}'")

    network_fp = base_path / network_files[net_name]

    if net_name == 'USbackbone60' and not is_only_core_node:
        core_nodes_fp = base_path / 'USB6014_core_nodes.txt'
        core_nodes_list = assign_core_nodes(core_nodes_fp)

    # Future: Add other core node files here if needed

    network_dict = assign_link_lengths(
        network_fp=network_fp,
        node_pairs_dict={},
        constant_weight=const_weight
    )

    return network_dict, core_nodes_list
