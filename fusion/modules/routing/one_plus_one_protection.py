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
from fusion.utils.data import sort_nested_dict_values
from fusion.utils.network import find_path_length, get_path_modulation

logger = logging.getLogger(__name__)


class OnePlusOneProtection(AbstractRoutingAlgorithm):
    """
    Traditional 1+1 disjoint protection routing for optical networks.

    Implements standard 1+1 protection as used in optical networking:
    - Each request gets ONE pair of link-disjoint paths (primary + backup)
    - Uses max-flow algorithm (Suurballe's) for optimal disjoint pair finding
    - Spectrum allocated on BOTH paths simultaneously with same slots
    - Traffic transmitted on both paths, receiver uses primary signal
    - Fast switchover on failure (protection switchover latency: default 50ms)

    Traditional Flow:
    1. Find ONE optimal disjoint pair using max-flow algorithm
    2. Validate BOTH paths are feasible (modulation format constraints)
    3. Allocate spectrum on BOTH paths (shared protection spectrum)
    4. Transmit on both, receiver monitors primary, switches to backup on failure

    Key features:
    - Max-flow based optimal disjoint path computation
    - Link-disjoint path validation (no shared links)
    - Modulation format feasibility check for both paths
    - Simultaneous dual allocation (shared spectrum)
    - Automatic failure detection and switchover
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
        Find ONE link-disjoint path pair for traditional 1+1 protection.

        Traditional 1+1 protection flow:
        1. Use max-flow algorithm to find optimal disjoint pair
        2. Validate BOTH paths are feasible (modulation format check)
        3. Pass the single pair to SDN controller
        4. SDN controller allocates spectrum on BOTH paths simultaneously
        5. Traffic transmitted on both paths, receiver uses primary (fast switchover)

        :param source: Source node ID
        :type source: Any
        :param destination: Destination node ID
        :type destination: Any
        :param request: Request details (optional)
        :type request: Any

        Example:
            >>> router = OnePlusOneProtection(props, sdn_props)
            >>> router.route(0, 5)
            >>> # route_props contains ONE path pair
            >>> # SDN controller will allocate on BOTH primary and backup paths
        """
        # Clear previous route properties
        self.route_props.paths_matrix = []
        self.route_props.modulation_formats_matrix = []
        self.route_props.weights_list = []
        self.route_props.backup_paths_matrix = []
        self.route_props.backup_modulation_formats_matrix = []

        # Find all disjoint paths using max-flow algorithm (Suurballe's)
        all_disjoint_paths = self.find_all_disjoint_paths(source, destination)

        if len(all_disjoint_paths) < 2:
            logger.warning(
                f"1+1 protection: Could not find at least 2 disjoint paths for "
                f"{source} -> {destination}"
            )
            self._disjoint_paths_failed += 1
            return

        # Try all possible pairs from disjoint paths, find first feasible pair
        # Suurballe's returns paths in optimal order (shortest first)
        # We try pairs in order: (P1,P2), (P1,P3), (P2,P3), ... until both are feasible
        primary_path = None
        backup_path = None
        primary_mods = None
        backup_mods = None

        for i in range(len(all_disjoint_paths)):
            for j in range(i + 1, len(all_disjoint_paths)):
                candidate_primary = all_disjoint_paths[i]
                candidate_backup = all_disjoint_paths[j]

                # Calculate modulation formats for BOTH paths in this pair
                temp_primary_mods = self._get_modulation_formats_for_path(
                    candidate_primary
                )
                temp_backup_mods = self._get_modulation_formats_for_path(
                    candidate_backup
                )

                # Check if BOTH paths have feasible modulation formats
                primary_feasible = any(
                    mod and mod is not False for mod in temp_primary_mods
                )
                backup_feasible = any(
                    mod and mod is not False for mod in temp_backup_mods
                )

                if primary_feasible and backup_feasible:
                    # Found a feasible pair!
                    primary_path = candidate_primary
                    backup_path = candidate_backup
                    primary_mods = temp_primary_mods
                    backup_mods = temp_backup_mods
                    logger.debug(
                        f"1+1 protection: Found feasible pair (paths {i + 1}, {j + 1}) "
                        f"from {len(all_disjoint_paths)} disjoint paths"
                    )
                    break

            if primary_path is not None:
                break  # Found feasible pair, exit outer loop

        # Check if we found a feasible pair
        if primary_path is None or backup_path is None:
            logger.warning(
                f"1+1 protection: Found {len(all_disjoint_paths)} disjoint paths but "
                f"no pair where BOTH paths are feasible for {source} -> {destination}"
            )
            self._disjoint_paths_failed += 1
            return

        # Store paths in SDN properties
        self.sdn_props.primary_path = primary_path
        self.sdn_props.backup_path = backup_path
        self.sdn_props.is_protected = True
        self.sdn_props.active_path = "primary"

        # Populate route_props with the feasible pair
        # SDN controller will allocate spectrum on BOTH paths
        self.route_props.paths_matrix.append(primary_path)
        self.route_props.backup_paths_matrix.append(backup_path)
        self.route_props.modulation_formats_matrix.append(primary_mods)
        self.route_props.backup_modulation_formats_matrix.append(backup_mods)
        self.route_props.weights_list.append(len(primary_path) - 1)

        self._disjoint_paths_found += 1

        logger.debug(
            f"1+1 protection: Using feasible pair for {source}->{destination}:\n"
            f"  Primary: {primary_path} ({len(primary_path) - 1} hops, mods={primary_mods})\n"
            f"  Backup:  {backup_path} ({len(backup_path) - 1} hops, mods={backup_mods})"
        )

    def find_all_disjoint_paths(self, source: Any, destination: Any) -> list[list[int]]:
        """
        Find all link-disjoint paths using max-flow algorithm.

        Uses NetworkX's edge_disjoint_paths which implements Suurballe's algorithm
        to find all edge-disjoint paths between source and destination.

        :param source: Source node
        :type source: Any
        :param destination: Destination node
        :type destination: Any
        :return: List of all disjoint paths found (empty if none exist)
        :rtype: list[list[int]]

        Example:
            >>> all_paths = router.find_all_disjoint_paths(0, 5)
            >>> # Returns: [[0,1,5], [0,2,5], [0,3,4,5], ...]
        """
        try:
            paths = list(
                nx.edge_disjoint_paths(
                    self.topology,
                    source,
                    destination,
                    flow_func=None,  # Use default (shortest augmenting path)
                )
            )
            return [list(path) for path in paths]
        except (AttributeError, nx.NetworkXNoPath, nx.NetworkXError):
            return []

    def find_disjoint_paths(
        self, source: Any, destination: Any
    ) -> tuple[list[int] | None, list[int] | None]:
        """
        Find link-disjoint primary and backup paths.

        Uses NetworkX's edge_disjoint_paths function for optimal disjoint path finding.
        Returns the first two paths found.

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
        all_paths = self.find_all_disjoint_paths(source, destination)

        if len(all_paths) >= 2:
            return all_paths[0], all_paths[1]

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

    def select_best_path_pair(
        self, path_pairs: list[tuple[list[int], list[int]]]
    ) -> tuple[list[int], list[int]]:
        """
        Select the best path pair from multiple options.

        Currently selects based on shortest combined path length (primary + backup).
        This minimizes total resource usage while maintaining protection.

        :param path_pairs: List of (primary, backup) path tuples
        :type path_pairs: list[tuple[list[int], list[int]]]
        :return: Best (primary, backup) path pair
        :rtype: tuple[list[int], list[int]]

        Example:
            >>> pairs = [([0,1,2], [0,3,2]), ([0,4,5,2], [0,3,2])]
            >>> best = router.select_best_path_pair(pairs)
            >>> print(best)
            ([0, 1, 2], [0, 3, 2])  # Shortest combined length
        """
        if not path_pairs:
            raise ValueError("No path pairs to select from")

        # Select pair with shortest combined length
        best_pair = min(path_pairs, key=lambda pair: len(pair[0]) + len(pair[1]))

        return best_pair

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

    def _get_modulation_formats_for_path(self, path: list[Any]) -> list[str]:
        """
        Get modulation formats for a given path.

        Determines appropriate modulation formats based on path length,
        bandwidth requirements, and available modulation format configurations.

        :param path: List of nodes representing the path
        :type path: list[Any]
        :return: List of modulation format strings
        :rtype: list[str]
        """
        path_length = find_path_length(path_list=path, topology=self.topology)

        chosen_bandwidth = getattr(self.sdn_props, "bandwidth", None)
        if chosen_bandwidth and not self.engine_props.get(
            "pre_calc_mod_selection", False
        ):
            # Use mod_per_bw if available
            if (
                "mod_per_bw" in self.engine_props
                and chosen_bandwidth in self.engine_props["mod_per_bw"]
            ):
                modulation_format = get_path_modulation(
                    mods_dict=self.engine_props["mod_per_bw"][chosen_bandwidth],
                    path_len=path_length,
                )
                return [str(modulation_format)]
            else:
                # Fallback to mod_formats
                modulation_formats = getattr(self.sdn_props, "mod_formats", {})
                modulation_format = get_path_modulation(modulation_formats, path_length)
                return [str(modulation_format)]
        else:
            # Use all modulation formats sorted by max_length
            has_mod_dict = hasattr(self.sdn_props, "modulation_formats_dict")
            if has_mod_dict and self.sdn_props.modulation_formats_dict is not None:
                modulation_formats_dict = sort_nested_dict_values(
                    original_dict=self.sdn_props.modulation_formats_dict,
                    nested_key="max_length",
                )
                # Ensure all keys are strings
                return [str(key) for key in modulation_formats_dict.keys()][::-1]
            else:
                # Fallback to simple list
                return ["QPSK"]

    def reset(self) -> None:
        """
        Reset the routing algorithm state.

        Clears metrics and counters.
        """
        self._disjoint_paths_found = 0
        self._disjoint_paths_failed = 0
