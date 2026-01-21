"""API routes for network topology visualization."""

import math
from pathlib import Path

import networkx as nx
from fastapi import APIRouter, HTTPException

from fusion.api.schemas.topology import (
    TopologyLink,
    TopologyListItem,
    TopologyListResponse,
    TopologyNode,
    TopologyResponse,
)

router = APIRouter(prefix="/topology", tags=["topology"])

# Map of topology names to file paths
TOPOLOGY_FILES = {
    "USNet": "us_network.txt",
    "NSFNet": "nsf_network.txt",
    "Pan-European": "europe_network.txt",
    "USbackbone60": "USB6014.txt",
    "Spainbackbone30": "SPNB3014.txt",
    "geant": "geant.txt",
    "toy_network": "toy_network.txt",
    "metro_net": "metro_net.txt",
    "dt_network": "dt_network.txt",
}


def _get_raw_data_path() -> Path:
    """Get the path to the raw topology data directory."""
    # Navigate from fusion/api/routes to data/raw
    return Path(__file__).parent.parent.parent.parent / "data" / "raw"


def _load_topology_graph(name: str) -> nx.Graph:
    """
    Load a topology from file into a NetworkX graph.

    :param name: Name of the topology.
    :returns: NetworkX graph with nodes and edges.
    :raises HTTPException: If topology not found.
    """
    if name not in TOPOLOGY_FILES:
        raise HTTPException(status_code=404, detail=f"Topology not found: {name}")

    file_path = _get_raw_data_path() / TOPOLOGY_FILES[name]
    if not file_path.exists():
        raise HTTPException(
            status_code=404, detail=f"Topology file not found: {file_path}"
        )

    graph = nx.Graph()

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                source, target, length = parts[0], parts[1], float(parts[2])
                graph.add_edge(source, target, length=length)

    return graph


def _compute_layout(graph: nx.Graph, scale: float = 400) -> dict[str, tuple[float, float]]:
    """
    Compute node positions using spring layout algorithm.

    :param graph: NetworkX graph.
    :param scale: Scale factor for positions.
    :returns: Dictionary mapping node IDs to (x, y) coordinates.
    """
    # Use spring layout with edge weights based on link length
    # Shorter links should have stronger attraction
    max_length = max(
        (d.get("length", 100) for _, _, d in graph.edges(data=True)), default=100
    )

    # Create weight dict (inverse of length, normalized)
    weights = {}
    for u, v, d in graph.edges(data=True):
        length = d.get("length", 100)
        # Shorter links = higher weight = closer nodes
        weights[(u, v)] = max_length / max(length, 1)

    nx.set_edge_attributes(graph, weights, "weight")

    # Compute layout
    pos = nx.spring_layout(
        graph,
        k=2 / math.sqrt(graph.number_of_nodes()),  # Optimal distance
        iterations=100,
        weight="weight",
        seed=42,  # Reproducible layout
    )

    # Scale and center
    scaled_pos = {}
    for node, (x, y) in pos.items():
        scaled_pos[node] = (x * scale, y * scale)

    return scaled_pos


@router.get("", response_model=TopologyListResponse)
def list_topologies() -> TopologyListResponse:
    """
    List all available network topologies.

    :returns: List of topology summaries.
    """
    topologies = []
    raw_path = _get_raw_data_path()

    for name, filename in TOPOLOGY_FILES.items():
        file_path = raw_path / filename
        if not file_path.exists():
            continue

        # Quick count of nodes and links
        nodes = set()
        link_count = 0

        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    nodes.add(parts[0])
                    nodes.add(parts[1])
                    link_count += 1

        # Divide by 2 since edges are listed twice (A->B and B->A)
        topologies.append(
            TopologyListItem(
                name=name,
                node_count=len(nodes),
                link_count=link_count // 2,
            )
        )

    return TopologyListResponse(topologies=topologies)


@router.get("/{name}", response_model=TopologyResponse)
def get_topology(name: str) -> TopologyResponse:
    """
    Get topology data with node positions.

    :param name: Name of the topology.
    :returns: Topology with nodes and links.
    """
    graph = _load_topology_graph(name)
    positions = _compute_layout(graph)

    nodes = [
        TopologyNode(
            id=str(node),
            label=f"Node {node}",
            x=positions[node][0],
            y=positions[node][1],
        )
        for node in graph.nodes()
    ]

    links = []
    seen_edges = set()
    for u, v, data in graph.edges(data=True):
        # Avoid duplicate edges (since graph is undirected)
        edge_key = tuple(sorted([str(u), str(v)]))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        links.append(
            TopologyLink(
                id=f"{u}-{v}",
                source=str(u),
                target=str(v),
                length_km=data.get("length", 0),
            )
        )

    return TopologyResponse(name=name, nodes=nodes, links=links)
