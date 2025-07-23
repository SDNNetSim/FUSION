import torch
import networkx as nx


def convert_networkx_topo(graph: nx.Graph, as_directed: bool = True):
    """
    Converts a networkx topology to a tensor.
    """
    nodes = list(graph.nodes())
    nodes.sort()
    id2idx = {nid: i for i, nid in enumerate(nodes)}
    num_nodes = len(nodes)

    edge_betweenness = nx.edge_betweenness_centrality(graph)
    edge_list = []
    attr_list = []
    for u, v, _ in graph.edges(data=True):
        ui, vi = id2idx[u], id2idx[v]
        betweenness = edge_betweenness.get((u, v), 0.0)

        edge_list.append([ui, vi])
        attr_list.append([betweenness])

        if as_directed:
            edge_list.append([vi, ui])
            attr_list.append([betweenness])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(attr_list, dtype=torch.float32)
    node_feats = torch.ones((num_nodes, 1), dtype=torch.float32)

    # Remember that networkx using strings to sort, so nodes are: 0, 10, ... NOT 0, 1, ...
    return edge_index, edge_attr, node_feats, id2idx


def load_topology_from_graph(graph: nx.Graph, **kwargs):
    """
    Shortcut to get (edge_index, edge_attr, node_feats) from a NetworkX graph.
    """
    return convert_networkx_topo(graph, **kwargs)
