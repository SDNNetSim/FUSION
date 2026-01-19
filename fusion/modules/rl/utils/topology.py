from typing import Any

import networkx as nx
import torch


def convert_networkx_topo(graph: nx.Graph, as_directed: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[Any, int]]:
    """
    Converts a networkx topology to a tensor.

    :param graph: NetworkX graph to convert
    :type graph: nx.Graph
    :param as_directed: Whether to treat the graph as directed
    :type as_directed: bool
    :return: Tuple containing edge index tensor, edge attributes tensor,
        node features tensor, and node ID to index mapping
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[Any, int]]
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

    # Remember that networkx using strings to sort, so nodes are:
    # 0, 10, ... NOT 0, 1, ...
    return edge_index, edge_attr, node_feats, id2idx


def load_topology_from_graph(graph: nx.Graph, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[Any, int]]:
    """
    Shortcut to get (edge_index, edge_attr, node_feats) from a NetworkX graph.

    :param graph: NetworkX graph to convert
    :type graph: nx.Graph
    :param kwargs: Additional keyword arguments passed to convert_networkx_topo
    :type kwargs: Any
    :return: Tuple containing edge index tensor, edge attributes tensor,
        node features tensor, and node ID to index mapping
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[Any, int]]
    """
    return convert_networkx_topo(graph, **kwargs)
