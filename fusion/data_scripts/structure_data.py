from pathlib import Path


def assign_link_lengths(network_fp: Path, node_pairs_dict: dict, constant_weight: bool = False):
    response_dict = {}
    with network_fp.open('r', encoding='utf-8') as file_obj:
        for line in file_obj:
            src, dest, link_len_str = line.strip().split('\t')
            link_len = float(link_len_str) if not constant_weight else 1.0

            src_dest_tuple = (
                node_pairs_dict.get(src, src),
                node_pairs_dict.get(dest, dest)
            )

            if src_dest_tuple not in response_dict:
                response_dict[src_dest_tuple] = link_len

    return response_dict


def assign_core_nodes(core_nodes_fp: Path):
    response_list = []
    with core_nodes_fp.open('r', encoding='utf-8') as file_obj:
        for line in file_obj:
            response_list.append(line.strip().split('\t')[0])
    return response_list


def create_network(net_name: str, base_fp: str = None, const_weight: bool = False, is_only_core_node: bool = False):
    """
    Builds a physical network structure from topology files.
    Resolves all paths safely using project-root-relative logic.
    """
    core_nodes_list = []

    # Resolve base_fp to absolute path
    project_root = Path(__file__).resolve().parents[2]
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
