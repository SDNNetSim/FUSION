"""
Failure type implementations for network failures.
"""

from typing import Any

import networkx as nx

from .errors import FailureConfigError


def fail_link(
    topology: nx.Graph, link_id: tuple[Any, Any], t_fail: float, t_repair: float
) -> dict[str, Any]:
    """
    Fail a single link (F1).

    :param topology: Network topology
    :type topology: nx.Graph
    :param link_id: Link tuple (src, dst)
    :type link_id: tuple[Any, Any]
    :param t_fail: Failure time
    :type t_fail: float
    :param t_repair: Repair time
    :type t_repair: float
    :return: Failure event details
    :rtype: dict[str, Any]
    :raises FailureConfigError: If link does not exist

    Example:
        >>> event = fail_link(G, link_id=(0, 1), t_fail=10.0, t_repair=20.0)
        >>> print(event['failed_links'])
        [(0, 1)]
    """
    # Validate link exists
    if not topology.has_edge(*link_id):
        # Try reverse direction
        reverse_link = (link_id[1], link_id[0])
        if not topology.has_edge(*reverse_link):
            raise FailureConfigError(f"Link {link_id} does not exist in topology")
        link_id = reverse_link

    return {
        "failure_type": "link",
        "t_fail": t_fail,
        "t_repair": t_repair,
        "failed_links": [link_id],
        "meta": {"link_id": link_id},
    }


def fail_node(
    topology: nx.Graph, node_id: Any, t_fail: float, t_repair: float
) -> dict[str, Any]:
    """
    Fail a node and all adjacent links (F2).

    :param topology: Network topology
    :type topology: nx.Graph
    :param node_id: Node ID to fail
    :type node_id: Any
    :param t_fail: Failure time
    :type t_fail: float
    :param t_repair: Repair time
    :type t_repair: float
    :return: Failure event details
    :rtype: dict[str, Any]
    :raises FailureConfigError: If node does not exist

    Example:
        >>> event = fail_node(G, node_id=1, t_fail=10.0, t_repair=20.0)
        >>> print(len(event['failed_links']))
        3  # Node 1 has 3 adjacent links
    """
    # Validate node exists
    if node_id not in topology.nodes:
        raise FailureConfigError(f"Node {node_id} does not exist in topology")

    # Get all adjacent links
    failed_links = list(topology.edges(node_id))

    if not failed_links:
        raise FailureConfigError(f"Node {node_id} has no adjacent links")

    return {
        "failure_type": "node",
        "t_fail": t_fail,
        "t_repair": t_repair,
        "failed_links": failed_links,
        "meta": {"node_id": node_id},
    }


def fail_srlg(
    topology: nx.Graph,
    srlg_links: list[tuple[Any, Any]],
    t_fail: float,
    t_repair: float,
) -> dict[str, Any]:
    """
    Fail all links in a Shared Risk Link Group (F3).

    :param topology: Network topology
    :type topology: nx.Graph
    :param srlg_links: List of link tuples in SRLG
    :type srlg_links: list[tuple[Any, Any]]
    :param t_fail: Failure time
    :type t_fail: float
    :param t_repair: Repair time
    :type t_repair: float
    :return: Failure event details
    :rtype: dict[str, Any]
    :raises FailureConfigError: If SRLG list is empty or contains invalid links

    Example:
        >>> srlg = [(0, 1), (2, 3), (4, 5)]
        >>> event = fail_srlg(G, srlg_links=srlg, t_fail=10.0, t_repair=20.0)
        >>> print(len(event['failed_links']))
        3
    """
    if not srlg_links:
        raise FailureConfigError("SRLG link list cannot be empty")

    # Validate all links exist
    validated_links = []
    for link_id in srlg_links:
        if not topology.has_edge(*link_id):
            # Try reverse direction
            reverse_link = (link_id[1], link_id[0])
            if not topology.has_edge(*reverse_link):
                raise FailureConfigError(
                    f"SRLG link {link_id} does not exist in topology"
                )
            validated_links.append(reverse_link)
        else:
            validated_links.append(link_id)

    return {
        "failure_type": "srlg",
        "t_fail": t_fail,
        "t_repair": t_repair,
        "failed_links": validated_links,
        "meta": {"srlg_size": len(validated_links)},
    }


def fail_geo(
    topology: nx.Graph,
    center_node: Any,
    hop_radius: int,
    t_fail: float,
    t_repair: float,
) -> dict[str, Any]:
    """
    Fail all links within hop_radius of center_node (F4).

    Uses NetworkX shortest path to determine hop distance.

    :param topology: Network topology
    :type topology: nx.Graph
    :param center_node: Center node of disaster
    :type center_node: Any
    :param hop_radius: Hop radius for failure region
    :type hop_radius: int
    :param t_fail: Failure time
    :type t_fail: float
    :param t_repair: Repair time
    :type t_repair: float
    :return: Failure event details
    :rtype: dict[str, Any]
    :raises FailureConfigError: If center_node invalid or radius non-positive

    Example:
        >>> event = fail_geo(G, center_node=5, hop_radius=2, t_fail=10.0, t_repair=20.0)
        >>> print(event['meta']['affected_nodes'])
        [5, 4, 6, 3, 7, 2, 8]
    """
    # Validate inputs
    if center_node not in topology.nodes:
        raise FailureConfigError(
            f"Center node {center_node} does not exist in topology"
        )

    if hop_radius <= 0:
        raise FailureConfigError(f"Hop radius must be positive, got {hop_radius}")

    # Find all nodes within hop_radius
    affected_nodes = set()
    affected_nodes.add(center_node)

    # BFS from center node up to hop_radius
    try:
        shortest_paths = nx.single_source_shortest_path_length(
            topology, center_node, cutoff=hop_radius
        )
        affected_nodes.update(shortest_paths.keys())
    except nx.NetworkXError as e:
        raise FailureConfigError(f"Error computing geographic failure: {e}") from e

    # Find all links with at least one endpoint in affected region
    failed_links = []
    for u, v in topology.edges():
        if u in affected_nodes or v in affected_nodes:
            failed_links.append((u, v))

    if not failed_links:
        raise FailureConfigError(
            f"No links found within radius {hop_radius} of node {center_node}"
        )

    return {
        "failure_type": "geo",
        "t_fail": t_fail,
        "t_repair": t_repair,
        "failed_links": failed_links,
        "meta": {
            "center_node": center_node,
            "hop_radius": hop_radius,
            "affected_nodes": list(affected_nodes),
        },
    }
