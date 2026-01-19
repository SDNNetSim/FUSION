"""
Disjoint path finding algorithms for 1+1 protection.

This module provides algorithms for finding link-disjoint and node-disjoint
path pairs required for 1+1 dedicated protection.

Phase: P5.4 - Protection Pipeline
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DisjointnessType(Enum):
    """Type of path disjointness."""

    LINK = "link"
    NODE = "node"


class DisjointPathFinder:
    """
    Finds disjoint path pairs for 1+1 protection.

    Supports two disjointness modes:
    - LINK: Paths share no common edges (may share intermediate nodes)
    - NODE: Paths share no common intermediate nodes (stronger guarantee)

    For link-disjoint paths, wraps the existing OnePlusOneProtection
    algorithm (Suurballe's via NetworkX edge_disjoint_paths).
    For node-disjoint paths, uses node removal in residual graph.

    Attributes:
        disjointness: Type of disjointness (LINK or NODE)

    Example:
        >>> finder = DisjointPathFinder(DisjointnessType.LINK)
        >>> paths = finder.find_disjoint_pair(topology, "A", "D")
        >>> if paths:
        ...     primary, backup = paths
    """

    def __init__(self, disjointness: DisjointnessType = DisjointnessType.LINK) -> None:
        """
        Initialize DisjointPathFinder.

        Args:
            disjointness: Type of disjointness to enforce
        """
        self.disjointness = disjointness

    def find_disjoint_pair(
        self,
        topology: nx.Graph,
        source: str,
        destination: str,
    ) -> tuple[list[str], list[str]] | None:
        """
        Find a disjoint path pair between source and destination.

        Args:
            topology: Network topology graph
            source: Source node identifier
            destination: Destination node identifier

        Returns:
            Tuple of (primary_path, backup_path) or None if not possible
        """
        if self.disjointness == DisjointnessType.LINK:
            return self._find_link_disjoint(topology, source, destination)
        else:
            return self._find_node_disjoint(topology, source, destination)

    def find_all_disjoint_paths(
        self,
        topology: nx.Graph,
        source: str,
        destination: str,
        max_paths: int = 10,
    ) -> list[list[str]]:
        """
        Find all disjoint paths between source and destination.

        Args:
            topology: Network topology graph
            source: Source node
            destination: Destination node
            max_paths: Maximum paths to return

        Returns:
            List of disjoint paths
        """
        if self.disjointness == DisjointnessType.LINK:
            return self._find_all_link_disjoint(
                topology, source, destination, max_paths
            )
        else:
            return self._find_all_node_disjoint(
                topology, source, destination, max_paths
            )

    def _find_link_disjoint(
        self,
        topology: nx.Graph,
        source: str,
        destination: str,
    ) -> tuple[list[str], list[str]] | None:
        """
        Find link-disjoint path pair using edge-disjoint shortest paths.

        Uses NetworkX's edge_disjoint_paths which implements Suurballe's
        algorithm for finding edge-disjoint paths.

        Algorithm:
        1. Use NetworkX edge_disjoint_paths to find all edge-disjoint paths
        2. Return first two paths as (primary, backup)
        """
        try:
            paths = list(
                nx.edge_disjoint_paths(
                    topology,
                    source,
                    destination,
                    flow_func=None,
                )
            )
            if len(paths) >= 2:
                return ([str(n) for n in paths[0]], [str(n) for n in paths[1]])
            logger.debug(
                f"No link-disjoint backup for {source}->{destination}"
            )
            return None
        except (nx.NetworkXNoPath, nx.NetworkXError):
            logger.debug(f"No path exists from {source} to {destination}")
            return None

    def _find_node_disjoint(
        self,
        topology: nx.Graph,
        source: str,
        destination: str,
    ) -> tuple[list[str], list[str]] | None:
        """
        Find node-disjoint path pair.

        Algorithm:
        1. Find shortest path (primary)
        2. Remove intermediate nodes of primary from graph
        3. Find shortest path in residual graph (backup)
        """
        try:
            # Find primary path
            primary = nx.shortest_path(
                topology, source, destination, weight="weight"
            )
            primary = [str(n) for n in primary]

            # Create residual graph without intermediate nodes
            residual = topology.copy()
            for node in primary[1:-1]:  # Exclude source and destination
                residual.remove_node(node)

            # Find backup in residual
            try:
                backup = nx.shortest_path(
                    residual, source, destination, weight="weight"
                )
                backup = [str(n) for n in backup]
                return (primary, backup)
            except nx.NetworkXNoPath:
                logger.debug(
                    f"No node-disjoint backup for {source}->{destination}"
                )
                return None

        except nx.NetworkXNoPath:
            logger.debug(f"No path exists from {source} to {destination}")
            return None

    def _find_all_link_disjoint(
        self,
        topology: nx.Graph,
        source: str,
        destination: str,
        max_paths: int,
    ) -> list[list[str]]:
        """Find all link-disjoint paths using NetworkX edge_disjoint_paths."""
        try:
            paths = list(
                nx.edge_disjoint_paths(
                    topology,
                    source,
                    destination,
                    flow_func=None,
                )
            )
            return [[str(n) for n in path] for path in paths[:max_paths]]
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return []

    def _find_all_node_disjoint(
        self,
        topology: nx.Graph,
        source: str,
        destination: str,
        max_paths: int,
    ) -> list[list[str]]:
        """Find all node-disjoint paths by iterative removal."""
        paths: list[list[str]] = []
        residual = topology.copy()

        while len(paths) < max_paths:
            try:
                path = nx.shortest_path(
                    residual, source, destination, weight="weight"
                )
                path = [str(n) for n in path]
                paths.append(path)

                # Remove intermediate nodes
                for node in path[1:-1]:
                    if residual.has_node(node):
                        residual.remove_node(node)

            except nx.NetworkXNoPath:
                break

        return paths

    def are_link_disjoint(
        self, path1: list[str], path2: list[str]
    ) -> bool:
        """
        Check if two paths are link-disjoint.

        Args:
            path1: First path as list of node IDs
            path2: Second path as list of node IDs

        Returns:
            True if paths share no common edges (in either direction)
        """
        edges1 = {(path1[i], path1[i + 1]) for i in range(len(path1) - 1)}
        edges1.update((path1[i + 1], path1[i]) for i in range(len(path1) - 1))
        edges2 = {(path2[i], path2[i + 1]) for i in range(len(path2) - 1)}
        edges2.update((path2[i + 1], path2[i]) for i in range(len(path2) - 1))
        return not edges1.intersection(edges2)

    def are_node_disjoint(
        self, path1: list[str], path2: list[str]
    ) -> bool:
        """
        Check if two paths are node-disjoint (except endpoints).

        Args:
            path1: First path as list of node IDs
            path2: Second path as list of node IDs

        Returns:
            True if paths share no common intermediate nodes
        """
        nodes1 = set(path1[1:-1])
        nodes2 = set(path2[1:-1])
        return not nodes1.intersection(nodes2)
