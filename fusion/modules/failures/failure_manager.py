"""
Failure manager for network failure injection and tracking.
"""

from typing import Any

import networkx as nx

from .errors import FailureConfigError
from .registry import get_failure_handler


class FailureManager:
    """
    Manages failure injection and tracking for network simulations.

    Tracks active failures, maintains failure history, and provides
    path feasibility checking based on current network state.

    :param engine_props: Engine configuration properties
    :type engine_props: dict[str, Any]
    :param topology: Network topology graph
    :type topology: nx.Graph
    """

    def __init__(self, engine_props: dict[str, Any], topology: nx.Graph) -> None:
        """
        Initialize FailureManager.

        :param engine_props: Engine configuration
        :type engine_props: dict[str, Any]
        :param topology: Network topology
        :type topology: nx.Graph
        """
        self.engine_props = engine_props
        self.topology = topology
        self.active_failures: set[tuple[Any, Any]] = set()  # Currently failed links
        self.failure_history: list[dict[str, Any]] = []  # Historical failure events
        self.scheduled_repairs: dict[
            float, list[tuple[Any, Any]]
        ] = {}  # Repair schedule

    def inject_failure(
        self, failure_type: str, t_fail: float, t_repair: float, **kwargs: Any
    ) -> dict[str, Any]:
        """
        Inject a failure event into the network.

        :param failure_type: Type of failure (link, node, srlg, geo)
        :type failure_type: str
        :param t_fail: Failure occurrence time
        :type t_fail: float
        :param t_repair: Repair completion time
        :type t_repair: float
        :param kwargs: Additional failure-specific parameters
        :type kwargs: Any
        :return: Failure event details
        :rtype: dict[str, Any]
        :raises FailureConfigError: If failure configuration is invalid
        :raises InvalidFailureTypeError: If failure type is unknown

        Example:
            >>> manager = FailureManager(props, topology)
            >>> event = manager.inject_failure(
            ...     'link',
            ...     t_fail=10.0,
            ...     t_repair=20.0,
            ...     link_id=(0, 1)
            ... )
            >>> print(event['failed_links'])
            [(0, 1)]
        """
        # Validate timing
        if t_repair <= t_fail:
            raise FailureConfigError(
                f"Repair time ({t_repair}) must be after failure time ({t_fail})"
            )

        # Get failure handler from registry
        handler = get_failure_handler(failure_type)

        # Execute failure
        event = handler(
            topology=self.topology, t_fail=t_fail, t_repair=t_repair, **kwargs
        )

        # Track active failures
        for link in event["failed_links"]:
            self.active_failures.add(link)

        # Schedule repairs
        if t_repair not in self.scheduled_repairs:
            self.scheduled_repairs[t_repair] = []
        self.scheduled_repairs[t_repair].extend(event["failed_links"])

        # Record in history
        self.failure_history.append(event)

        return event

    def is_path_feasible(self, path: list[int]) -> bool:
        """
        Check if path is feasible given active failures.

        A path is infeasible if any of its links are currently failed.

        :param path: List of node IDs forming the path
        :type path: list[int]
        :return: True if path has no failed links, False otherwise
        :rtype: bool

        Example:
            >>> manager = FailureManager(props, topology)
            >>> manager.active_failures = {(0, 1)}
            >>> manager.is_path_feasible([0, 1, 2])
            False
            >>> manager.is_path_feasible([0, 3, 2])
            True
        """
        if not self.active_failures:
            return True

        # Check each link in the path
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            reverse_link = (path[i + 1], path[i])

            # Check both directions (undirected graph)
            if link in self.active_failures or reverse_link in self.active_failures:
                return False

        return True

    def get_affected_links(self) -> list[tuple[Any, Any]]:
        """
        Get list of currently failed links.

        :return: List of failed link tuples
        :rtype: list[tuple[Any, Any]]
        """
        return list(self.active_failures)

    def repair_failures(self, current_time: float) -> list[tuple[Any, Any]]:
        """
        Repair all failures scheduled for repair at current_time.

        :param current_time: Current simulation time
        :type current_time: float
        :return: List of repaired link tuples
        :rtype: list[tuple[Any, Any]]

        Example:
            >>> manager = FailureManager(props, topology)
            >>> manager.inject_failure(
            ...     'link', t_fail=10.0, t_repair=20.0, link_id=(0, 1)
            ... )
            >>> repaired = manager.repair_failures(20.0)
            >>> print(repaired)
            [(0, 1)]
            >>> print(manager.active_failures)
            set()
        """
        if current_time not in self.scheduled_repairs:
            return []

        # Get links to repair
        links_to_repair = self.scheduled_repairs[current_time]

        # Remove from active failures
        for link in links_to_repair:
            self.active_failures.discard(link)

        # Remove from schedule
        del self.scheduled_repairs[current_time]

        return links_to_repair

    def get_failure_count(self) -> int:
        """
        Get number of currently active failures.

        :return: Number of failed links
        :rtype: int
        """
        return len(self.active_failures)

    def clear_all_failures(self) -> None:
        """
        Clear all active failures (for testing or reset).

        This removes all active failures and clears the repair schedule.
        """
        self.active_failures.clear()
        self.scheduled_repairs.clear()
