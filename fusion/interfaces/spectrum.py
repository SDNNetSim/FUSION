"""
Abstract base class for spectrum assignment algorithms in FUSION.
"""
# pylint: disable=duplicate-code

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class AbstractSpectrumAssigner(ABC):
    """Base class for all spectrum assignment algorithms in FUSION.
    
    This interface defines the contract that all spectrum assignment algorithms must follow
    to ensure compatibility with the FUSION simulation framework.
    """

    def __init__(self, engine_props: dict, sdn_props: object, route_props: object):
        """Initialize the spectrum assignment algorithm.
        
        Args:
            engine_props: Dictionary containing engine configuration and properties
            sdn_props: Object containing SDN controller properties and network state
            route_props: Object containing routing properties and selected path
        """
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.route_props = route_props

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the name of the spectrum assignment algorithm.
        
        Returns:
            String identifier for this spectrum assignment algorithm
        """

    @property
    @abstractmethod
    def supports_multiband(self) -> bool:
        """Indicate whether this algorithm supports multi-band assignment.
        
        Returns:
            True if the algorithm supports multi-band, False otherwise
        """

    @abstractmethod
    def assign(self, path: List[Any], request: Any) -> Optional[Dict[str, Any]]:
        """Assign spectrum resources along the given path for the request.
        
        Args:
            path: List of nodes representing the path
            request: Request object containing traffic demand and spectrum requirements
            
        Returns:
            Dictionary containing spectrum assignment details:
                - 'start_slot': Starting slot index
                - 'end_slot': Ending slot index
                - 'core_num': Core number (for multi-core fibers)
                - 'band': Band identifier
                - 'is_free': Whether assignment was successful
            Returns None if assignment fails
        """

    @abstractmethod
    def check_spectrum_availability(self, path: List[Any], start_slot: int,
                                    end_slot: int, core_num: int, band: str) -> bool:
        """Check if spectrum slots are available along the entire path.
        
        Args:
            path: List of nodes representing the path
            start_slot: Starting slot index
            end_slot: Ending slot index (inclusive)
            core_num: Core number to check
            band: Band identifier
            
        Returns:
            True if all slots are available, False otherwise
        """

    @abstractmethod
    def allocate_spectrum(self, path: List[Any], start_slot: int,
                          end_slot: int, core_num: int, band: str, request_id: Any) -> bool:
        """Allocate spectrum resources along the path.
        
        Args:
            path: List of nodes representing the path
            start_slot: Starting slot index
            end_slot: Ending slot index (inclusive)
            core_num: Core number to allocate
            band: Band identifier
            request_id: Unique identifier for the request
            
        Returns:
            True if allocation successful, False otherwise
        """

    @abstractmethod
    def deallocate_spectrum(self, path: List[Any], start_slot: int,
                            end_slot: int, core_num: int, band: str) -> bool:
        """Deallocate spectrum resources along the path.
        
        Args:
            path: List of nodes representing the path
            start_slot: Starting slot index
            end_slot: Ending slot index (inclusive)
            core_num: Core number to deallocate
            band: Band identifier
            
        Returns:
            True if deallocation successful, False otherwise
        """

    @abstractmethod
    def get_fragmentation_metric(self, path: List[Any]) -> float:
        """Calculate fragmentation metric for the given path.
        
        Args:
            path: List of nodes representing the path
            
        Returns:
            Fragmentation value (0.0 = no fragmentation, 1.0 = maximum fragmentation)
        """

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get spectrum assignment algorithm performance metrics.
        
        Returns:
            Dictionary containing algorithm-specific metrics
        """

    def reset(self) -> None:
        """Reset the spectrum assignment algorithm state.
        
        This method can be overridden by subclasses that maintain state.
        """
