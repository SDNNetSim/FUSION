"""
K-Path cache for pre-computed shortest paths.
"""

import sys
from typing import Any

import networkx as nx


class KPathCache:
    """
    Pre-compute and cache K shortest paths for all (src, dst) pairs.

    Uses Yen's K-shortest paths algorithm to compute alternatives,
    ordered by a configurable criterion (hops, length, latency).

    :param topology: Network topology
    :type topology: nx.Graph
    :param k: Number of paths to compute
    :type k: int
    :param ordering: Path ordering criterion (hops, length, latency)
    :type ordering: str
    """

    def __init__(self, topology: nx.Graph, k: int = 4, ordering: str = "hops") -> None:
        """
        Initialize K-Path cache.

        :param topology: Network topology
        :type topology: nx.Graph
        :param k: Number of paths per pair
        :type k: int
        :param ordering: Ordering criterion
        :type ordering: str
        """
        self.topology = topology
        self.k = k
        self.ordering = ordering
        self.cache: dict[tuple[Any, Any], list[list[int]]] = {}
        self._precompute_paths()

    def _precompute_paths(self) -> None:
        """
        Pre-compute K shortest paths for all node pairs.

        Uses NetworkX's k_shortest_paths or custom implementation
        based on Yen's algorithm. Paths are stored ordered by
        the specified criterion.

        :raises ValueError: If topology is invalid or k is non-positive
        """
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")

        if not self.topology.nodes:
            raise ValueError("Topology has no nodes")

        # Weight function based on ordering
        if self.ordering == "hops":
            weight = None  # Unweighted = hop count
        elif self.ordering == "length":
            weight = "length"
        elif self.ordering == "latency":
            weight = "latency"
        else:
            raise ValueError(f"Unknown ordering: {self.ordering}")

        # Compute K paths for all node pairs
        nodes = list(self.topology.nodes())
        for src in nodes:
            for dst in nodes:
                if src == dst:
                    continue

                try:
                    # Use NetworkX k_shortest_paths
                    paths = list(nx.shortest_simple_paths(self.topology, src, dst, weight=weight))

                    # Take up to K paths
                    self.cache[(src, dst)] = paths[: self.k]

                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # No path exists
                    self.cache[(src, dst)] = []

    def get_k_paths(self, source: Any, destination: Any) -> list[list[int]]:
        """
        Get K paths from cache.

        :param source: Source node
        :type source: Any
        :param destination: Destination node
        :type destination: Any
        :return: List of K paths (may be fewer if not enough exist)
        :rtype: list[list[int]]

        Example:
            >>> cache = KPathCache(topology, k=4)
            >>> paths = cache.get_k_paths(0, 5)
            >>> print(len(paths))
            4
            >>> print(paths[0])
            [0, 1, 3, 5]
        """
        return self.cache.get((source, destination), [])

    def get_path_features(
        self,
        path: list[int],
        network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]],
        failure_manager: Any = None,
    ) -> dict[str, Any]:
        """
        Extract features for a candidate path.

        Computes features needed for RL policy decisions and heuristics:
        - path_hops: Number of hops in the path
        - min_residual_slots: Minimum contiguous free slots along path (bottleneck)
        - frag_indicator: Fragmentation proxy (1 - largest_contig / total_free)
        - failure_mask: Whether path uses any failed link
        - dist_to_disaster_centroid: Hops to failure center (0 if no failure)

        :param path: Path node list
        :type path: list[int]
        :param network_spectrum_dict: Current spectrum state
        :type network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]]
        :param failure_manager: Optional failure manager for failure_mask
        :type failure_manager: Any
        :return: Path features dict
        :rtype: dict[str, Any]

        Example:
            >>> features = cache.get_path_features(
            ...     path=[0, 1, 2, 3],
            ...     network_spectrum_dict=spectrum,
            ...     failure_manager=None
            ... )
            >>> print(features)
            {
                'path_hops': 3,
                'min_residual_slots': 15,
                'frag_indicator': 0.23,
                'failure_mask': 0,
                'dist_to_disaster_centroid': 0
            }
        """
        # Compute hop count
        path_hops = len(path) - 1

        # Compute min_residual_slots (bottleneck link)
        min_residual = float("inf")
        total_free = 0
        largest_contig = 0

        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            reverse_link = (path[i + 1], path[i])

            # Get link spectrum (try both directions)
            link_spectrum = network_spectrum_dict.get(link, network_spectrum_dict.get(reverse_link, {}))

            if not link_spectrum:
                # Link not in spectrum dict (shouldn't happen)
                continue

            # Compute contiguous free slots
            slots = link_spectrum.get("slots", [])
            free_blocks = self._find_free_blocks(slots)

            if free_blocks:
                link_total_free = sum(block[1] - block[0] for block in free_blocks)
                link_largest_contig = max(block[1] - block[0] for block in free_blocks)

                total_free += link_total_free
                largest_contig = max(largest_contig, link_largest_contig)
                min_residual = min(min_residual, link_largest_contig)
            else:
                min_residual = 0

        # Handle case where no free slots found
        if min_residual == float("inf"):
            min_residual = 0

        # Compute fragmentation indicator
        if total_free > 0:
            frag_indicator = 1.0 - (largest_contig / total_free)
        else:
            frag_indicator = 1.0  # Fully fragmented (no free slots)

        # Compute failure mask
        failure_mask = 0
        if failure_manager and not failure_manager.is_path_feasible(path):
            failure_mask = 1

        # Compute distance to disaster centroid
        dist_to_disaster = 0
        if failure_manager and failure_manager.active_failures:
            # Find center of failed region (simplified: use first failed link)
            failed_links = list(failure_manager.active_failures)
            if failed_links:
                center_node = failed_links[0][0]  # Use one endpoint as center
                try:
                    # Distance from path to center (min distance from any path node)
                    distances = []
                    for node in path:
                        if node == center_node:
                            distances.append(0)
                        else:
                            try:
                                dist = nx.shortest_path_length(self.topology, node, center_node)
                                distances.append(dist)
                            except nx.NetworkXNoPath:
                                pass

                    if distances:
                        dist_to_disaster = min(distances)
                except Exception:  # nosec B110  # Defensive: defaults to 0 if topology inconsistent
                    pass

        return {
            "path_hops": path_hops,
            "min_residual_slots": int(min_residual),
            "frag_indicator": round(frag_indicator, 4),
            "failure_mask": failure_mask,
            "dist_to_disaster_centroid": dist_to_disaster,
        }

    def _find_free_blocks(self, slots: list[int]) -> list[tuple[int, int]]:
        """
        Find contiguous free blocks in slot array.

        :param slots: Slot occupancy array (0 = free, >0 = occupied)
        :type slots: list[int]
        :return: List of (start, end) tuples for free blocks
        :rtype: list[tuple[int, int]]

        Example:
            >>> slots = [0, 0, 1, 1, 0, 0, 0]
            >>> blocks = cache._find_free_blocks(slots)
            >>> print(blocks)
            [(0, 2), (4, 7)]
        """
        blocks = []
        start = None

        for i, slot in enumerate(slots):
            if slot == 0:  # Free
                if start is None:
                    start = i
            else:  # Occupied
                if start is not None:
                    blocks.append((start, i))
                    start = None

        # Handle trailing free block
        if start is not None:
            blocks.append((start, len(slots)))

        return blocks

    def get_cache_size(self) -> int:
        """
        Get number of cached path pairs.

        :return: Number of (src, dst) pairs cached
        :rtype: int
        """
        return len(self.cache)

    def get_memory_estimate_mb(self) -> float:
        """
        Estimate memory usage in MB.

        Rough estimate based on number of paths and average path length.

        :return: Estimated memory usage in MB
        :rtype: float
        """
        total_size = 0
        for paths in self.cache.values():
            for path in paths:
                # Size of list + size of integers
                total_size += sys.getsizeof(path)
                total_size += sum(sys.getsizeof(node) for node in path)

        return total_size / (1024 * 1024)
