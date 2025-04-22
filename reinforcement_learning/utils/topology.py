import torch
import networkx as nx


def convert_networkx_topo(graph: nx.Graph, as_directed: bool = True):
    """
    Converts a NetworkX graph into GNN-ready PyTorch tensors.

    Args:
        graph: NetworkX Graph or DiGraph with edge attributes 'capacity' and 'length'.
        as_directed: If True, treat each edge as two directed edges.

    Returns:
        edge_index: LongTensor of shape (2, E)
        edge_attr:  FloatTensor of shape (E, 2) with columns [capacity, length]
        node_feats: FloatTensor of shape (N, 1), all ones by default
    """
    # 1) Node indexing
    nodes = list(graph.nodes())
    nodes.sort()
    id2idx = {nid: i for i, nid in enumerate(nodes)}
    num_nodes = len(nodes)

    # 2) Edge list and attributes
    edge_list = []
    attr_list = []
    for u, v, data in graph.edges(data=True):
        ui, vi = id2idx[u], id2idx[v]
        # TODO: (drl_path_agents) Need to update cap I believe
        cap = float(data.get('capacity', 1.0))
        length = float(data.get('length', 1.0))
        # u -> v
        edge_list.append([ui, vi])
        attr_list.append([cap, length])
        if as_directed:
            # v -> u
            edge_list.append([vi, ui])
            attr_list.append([cap, length])

    # 3) Tensors
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attr_list, dtype=torch.float32)
    node_feats = torch.ones((num_nodes, 1), dtype=torch.float32)

    return edge_index, edge_attr, node_feats


def load_topology_from_graph(graph: nx.Graph, **kwargs):
    """
    Shortcut to get (edge_index, edge_attr, node_feats) from a NetworkX graph.
    """
    return convert_networkx_topo(graph, **kwargs)
