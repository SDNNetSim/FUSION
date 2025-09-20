"""
Abstract base class for routing algorithms in FUSION.
"""
# pylint: disable=duplicate-code

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AbstractRoutingAlgorithm(ABC):
    """Base class for all routing algorithms in FUSION.
    
    This interface defines the contract that all routing algorithms must follow
    to ensure compatibility with the FUSION simulation framework.
    """

    def __init__(self, engine_props: dict, sdn_props: object):
        """Initialize the routing algorithm.
        
        Args:
            engine_props: Dictionary containing engine configuration and properties
            sdn_props: Object containing SDN controller properties and network state
        """
        self.engine_props = engine_props
        self.sdn_props = sdn_props

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the name of the routing algorithm.
        
        Returns:
            String identifier for this routing algorithm
        """

    @property
    @abstractmethod
    def supported_topologies(self) -> List[str]:
        """Return list of supported topology types.
        
        Returns:
            List of topology names this algorithm supports (e.g., ['NSFNet', 'USBackbone'])
        """

    @abstractmethod
    def validate_environment(self, topology: Any) -> bool:
        """Validate that the routing algorithm can work with the given environment.
        
        Args:
            topology: NetworkX graph representing the network topology
            
        Returns:
            True if the algorithm can route in this environment, False otherwise
        """

    @abstractmethod
    def route(self, source: Any, destination: Any, request: Any) -> Optional[List[Any]]:
        """Find a route from source to destination for the given request.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            request: Request object containing traffic demand details
            
        Returns:
            List representing the path from source to destination, or None if no path found
        """

    @abstractmethod
    def get_paths(self, source: Any, destination: Any, k: int = 1) -> List[List[Any]]:
        """Get k shortest paths between source and destination.
        
        Args:
            source: Source node identifier
            destination: Destination node identifier
            k: Number of paths to return
            
        Returns:
            List of k paths, where each path is a list of nodes
        """

    @abstractmethod
    def update_weights(self, topology: Any) -> None:
        """Update edge weights based on current network state.
        
        Args:
            topology: NetworkX graph to update weights for
        """

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing algorithm performance metrics.
        
        Returns:
            Dictionary containing algorithm-specific metrics
        """

    def reset(self) -> None:
        """Reset the routing algorithm state.
        
        This method can be overridden by subclasses that maintain state.
        """
