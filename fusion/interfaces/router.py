"""
Abstract base class for routing algorithms in FUSION.
"""

from abc import ABC, abstractmethod
from typing import Any


class AbstractRoutingAlgorithm(ABC):
    """
    Base class for all routing algorithms in FUSION.

    This interface defines the contract that all routing algorithms must follow
    to ensure compatibility with the FUSION simulation framework.
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        """
        Initialize the routing algorithm.

        :param engine_props: Dictionary containing engine configuration and properties
        :type engine_props: dict
        :param sdn_props: Object containing SDN controller properties and network state
        :type sdn_props: object
        """
        self.engine_props = engine_props
        self.sdn_props = sdn_props

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """
        Return the name of the routing algorithm.

        :return: String identifier for this routing algorithm
        :rtype: str
        """

    @property
    @abstractmethod
    def supported_topologies(self) -> list[str]:
        """
        Return list of supported topology types.

        :return: List of topology names this algorithm supports
                (e.g., ['NSFNet', 'USBackbone'])
        :rtype: List[str]
        """

    @abstractmethod
    def validate_environment(self, topology: Any) -> bool:
        """
        Validate that the routing algorithm can work with the given environment.

        :param topology: NetworkX graph representing the network topology
        :type topology: Any
        :return: True if the algorithm can route in this environment, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def route(self, source: Any, destination: Any, request: Any) -> list[Any] | None:
        """
        Find a route from source to destination for the given request.

        :param source: Source node identifier
        :type source: Any
        :param destination: Destination node identifier
        :type destination: Any
        :param request: Request object containing traffic demand details
        :type request: Any
        :return: List representing the path from source to destination,
                or None if no path found
        :rtype: Optional[List[Any]]
        """

    @abstractmethod
    def get_paths(self, source: Any, destination: Any, k: int = 1) -> list[list[Any]]:
        """
        Get k shortest paths between source and destination.

        :param source: Source node identifier
        :type source: Any
        :param destination: Destination node identifier
        :type destination: Any
        :param k: Number of paths to return
        :type k: int
        :return: List of k paths, where each path is a list of nodes
        :rtype: List[List[Any]]
        """

    @abstractmethod
    def update_weights(self, topology: Any) -> None:
        """
        Update edge weights based on current network state.

        :param topology: NetworkX graph to update weights for
        :type topology: Any
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """
        Get routing algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics
        :rtype: Dict[str, Any]
        """

    def reset(self) -> None:  # noqa: B027
        """
        Reset the routing algorithm state.

        This method can be overridden by subclasses that maintain state.
        """
        pass
