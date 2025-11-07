"""
1+1 disjoint protection routing implementation.

This module provides a routing algorithm that computes link-disjoint primary
and backup paths for survivability. On failure, traffic switches to the backup
path with fixed protection switchover latency.
"""

import logging
from typing import Any

import networkx as nx

from fusion.core.properties import SDNProps
from fusion.interfaces.router import AbstractRoutingAlgorithm

logger = logging.getLogger(__name__)


class OnePlusOneProtection(AbstractRoutingAlgorithm):
    """
    1+1 disjoint protection routing with automatic switchover.

    Computes link-disjoint primary and backup paths at setup time.
    On failure detection, switches to backup with fixed protection
    switchover latency (default: 50ms).

    Key features:
    - Link-disjoint path computation (Suurballe's algorithm or K-SP)
    - Simultaneous spectrum reservation on both paths
    - Fast switchover on failure (protection)
    - Optional revert-to-primary after repair

    :param engine_props: Engine configuration
    :type engine_props: dict[str, Any]
    :param sdn_props: SDN controller properties
    :type sdn_props: SDNProps
    """

    def __init__(self, engine_props: dict[str, Any], sdn_props: SDNProps) -> None:
        """
        Initialize 1+1 protection router.

        :param engine_props: Engine configuration
        :type engine_props: dict[str, Any]
        :param sdn_props: SDN properties
        :type sdn_props: SDNProps
        """
        super().__init__(engine_props, sdn_props)
        self.topology = engine_props.get("topology", sdn_props.topology)
        self.protection_switchover_ms = engine_props.get("protection_settings", {}).get(
            "protection_switchover_ms", 50.0
        )
        self.revert_to_primary = engine_props.get("protection_settings", {}).get(
            "revert_to_primary", False
        )
        self._disjoint_paths_found = 0
        self._disjoint_paths_failed = 0

    @property
    def algorithm_name(self) -> str:
        """
        Get the name of the routing algorithm.

        :return: The algorithm name '1plus1_protection'.
        :rtype: str
        """
        return "1plus1_protection"

    @property
    def supported_topologies(self) -> list[str]:
        """
        Get the list of supported topology types.

        :return: List of supported topology names.
        :rtype: list[str]
        """
        return ["NSFNet", "USBackbone60", "Pan-European", "Generic"]

    def validate_environment(self, topology: nx.Graph) -> bool:
        """
        Validate that the routing algorithm can work with the given topology.

        1+1 protection requires a connected graph with sufficient edge
        connectivity for disjoint paths.

        :param topology: NetworkX graph representing the network topology.
        :type topology: nx.Graph
        :return: True if the algorithm can route in this environment.
        :rtype: bool
        """
        try:
            # Check if graph is connected
            if not nx.is_connected(topology):
                logger.warning("Topology is not connected")
                return False

            # Check if graph has sufficient edge connectivity
            # At least 2-edge-connected for disjoint paths
            edge_connectivity = nx.edge_connectivity(topology)
            if edge_connectivity < 2:
                logger.warning(
                    f"Topology edge connectivity ({edge_connectivity}) "
                    "is too low for 1+1 protection (need >= 2)"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating environment: {e}")
            return False

    def route(self, source: Any, destination: Any, request: Any) -> None:
        """
        Find link-disjoint primary and backup paths.

        Stores primary and backup paths in SDN properties (sdn_props.primary_path,
        sdn_props.backup_path). Consumers should check these attributes for results.

        :param source: Source node ID
        :type source: Any
        :param destination: Destination node ID
        :type destination: Any
        :param request: Request details (optional)
        :type request: Any

        Example:
            >>> router = OnePlusOneProtection(props, sdn_props)
            >>> router.route(0, 5)
            >>> print(sdn_props.primary_path)
            [0, 1, 3, 5]
            >>> print(sdn_props.backup_path)
            [0, 2, 4, 5]
        """
        primary, backup = self.find_disjoint_paths(source, destination)

        if primary is None or backup is None:
            logger.warning(
                f"Could not find disjoint paths for {source} -> {destination}"
            )
            self._disjoint_paths_failed += 1
            return

        # Store paths in SDN properties
        self.sdn_props.primary_path = primary
        self.sdn_props.backup_path = backup
        self.sdn_props.is_protected = True
        self.sdn_props.active_path = "primary"
        self.sdn_props.protection_mode = "1plus1"

        self._disjoint_paths_found += 1

        logger.debug(
            f"1+1 protection: Primary={len(primary)} hops, Backup={len(backup)} hops"
        )

    def find_disjoint_paths(
        self, source: Any, destination: Any
    ) -> tuple[list[int] | None, list[int] | None]:
        """
        Find link-disjoint primary and backup paths.

        Uses NetworkX's edge_disjoint_paths function for optimal disjoint path finding.

        :param source: Source node
        :type source: Any
        :param destination: Destination node
        :type destination: Any
        :return: (primary_path, backup_path) or (None, None)
        :rtype: tuple[list[int] | None, list[int] | None]

        Example:
            >>> primary, backup = router.find_disjoint_paths(0, 5)
            >>> # Verify disjointness
            >>> primary_links = set(zip(primary[:-1], primary[1:]))
            >>> backup_links = set(zip(backup[:-1], backup[1:]))
            >>> assert primary_links.isdisjoint(backup_links)
        """
        try:
            # Use NetworkX edge_disjoint_paths (uses max-flow algorithm internally)
            paths = list(
                nx.edge_disjoint_paths(
                    self.topology,
                    source,
                    destination,
                    flow_func=None,  # Use default (shortest augmenting path)
                )
            )

            if len(paths) >= 2:
                return list(paths[0]), list(paths[1])

            return None, None

        except (AttributeError, nx.NetworkXNoPath, nx.NetworkXError):
            return None, None

    def find_disjoint_paths_k_shortest(
        self, source: Any, destination: Any, k: int = 10
    ) -> tuple[list[int] | None, list[int] | None]:
        """
        Find disjoint paths using K-shortest paths (alternative method).

        This is an alternative to edge_disjoint_paths. It finds the shortest path
        as primary, then finds the shortest path on a graph with primary links removed.

        Note: This method is retained for potential future use or comparison, but
        find_disjoint_paths() should be preferred as it uses a max-flow algorithm
        which is more robust.

        :param source: Source node
        :type source: Any
        :param destination: Destination node
        :type destination: Any
        :param k: Max paths to consider
        :type k: int
        :return: (primary, backup) paths
        :rtype: tuple[list[int] | None, list[int] | None]
        """
        # Find K shortest paths
        try:
            k_paths = list(nx.shortest_simple_paths(self.topology, source, destination))
        except nx.NetworkXNoPath:
            return None, None

        if len(k_paths) < 2:
            return None, None

        # Take first path as primary
        primary = k_paths[0]
        primary_links = set(zip(primary[:-1], primary[1:], strict=False))

        # Find first path that is link-disjoint with primary
        for candidate in k_paths[1:k]:
            candidate_links = set(zip(candidate[:-1], candidate[1:], strict=False))

            # Check for link-disjointness (both directions)
            is_disjoint = True
            for link in candidate_links:
                if link in primary_links or (link[1], link[0]) in primary_links:
                    is_disjoint = False
                    break

            if is_disjoint:
                return primary, candidate

        return None, None

    def handle_failure(
        self, current_time: float, affected_requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Handle failure by switching protected requests to backup.

        :param current_time: Current simulation time
        :type current_time: float
        :param affected_requests: Requests on failed links
        :type affected_requests: list[dict[str, Any]]
        :return: Recovery actions performed
        :rtype: list[dict[str, Any]]

        Example:
            >>> actions = router.handle_failure(100.0, affected_requests)
            >>> for action in actions:
            ...     print(f"Request {action['request_id']}: "
            ...           f"switched in {action['recovery_time_ms']}ms")
        """
        recovery_actions = []

        for request in affected_requests:
            if request.get("is_protected", False):
                # Switch to backup
                recovery_time_ms = self.protection_switchover_ms

                recovery_actions.append(
                    {
                        "request_id": request["id"],
                        "action": "switchover",
                        "recovery_time_ms": recovery_time_ms,
                        "from_path": "primary",
                        "to_path": "backup",
                    }
                )

                logger.info(
                    f"Request {request['id']}: 1+1 switchover ({recovery_time_ms}ms)"
                )

        return recovery_actions

    def get_paths(self, source: Any, destination: Any, k: int = 2) -> list[list[Any]]:
        """
        Get k shortest paths between source and destination.

        For 1+1 protection, this returns the primary and backup paths.

        :param source: Source node identifier
        :type source: Any
        :param destination: Destination node identifier
        :type destination: Any
        :param k: Number of paths to return (default 2 for 1+1)
        :type k: int
        :return: List of k paths, where each path is a list of nodes
        :rtype: list[list[Any]]
        """
        primary, backup = self.find_disjoint_paths(source, destination)

        # Return both paths if available
        if primary and backup:
            return [primary, backup]
        if primary:
            return [primary]
        return []

    def update_weights(self, topology: nx.Graph) -> None:
        """
        Update edge weights based on current network state.

        For 1+1 protection, weights are typically uniform (hop count).

        :param topology: NetworkX graph to update weights for
        :type topology: nx.Graph
        """
        # Set uniform weights (hop count)
        for u, v in topology.edges():
            topology[u][v]["weight"] = 1.0

    def get_metrics(self) -> dict[str, Any]:
        """
        Get routing algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics
        :rtype: dict[str, Any]
        """
        total_attempts = self._disjoint_paths_found + self._disjoint_paths_failed
        success_rate = (
            self._disjoint_paths_found / total_attempts if total_attempts > 0 else 0.0
        )

        return {
            "algorithm": self.algorithm_name,
            "disjoint_paths_found": self._disjoint_paths_found,
            "disjoint_paths_failed": self._disjoint_paths_failed,
            "success_rate": success_rate,
            "protection_switchover_ms": self.protection_switchover_ms,
            "revert_to_primary": self.revert_to_primary,
        }

    def reset(self) -> None:
        """
        Reset the routing algorithm state.

        Clears metrics and counters.
        """
        self._disjoint_paths_found = 0
        self._disjoint_paths_failed = 0
