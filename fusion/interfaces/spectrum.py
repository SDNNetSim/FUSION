"""
Abstract base class for spectrum assignment algorithms in FUSION.
"""

from abc import ABC, abstractmethod
from typing import Any

from fusion.core.properties import SDNProps


class AbstractSpectrumAssigner(ABC):
    """
    Base class for all spectrum assignment algorithms in FUSION.

    This interface defines the contract that all spectrum assignment algorithms
    must follow
    to ensure compatibility with the FUSION simulation framework.
    """

    def __init__(self, engine_props: dict, sdn_props: SDNProps, route_props: object):
        """
        Initialize the spectrum assignment algorithm.

        :param engine_props: Dictionary containing engine configuration and properties
        :type engine_props: dict
        :param sdn_props: Object containing SDN controller properties and network state
        :type sdn_props: SDNProps
        :param route_props: Object containing routing properties and selected path
        :type route_props: object
        """
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.route_props = route_props

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """
        Return the name of the spectrum assignment algorithm.

        :return: String identifier for this spectrum assignment algorithm
        :rtype: str
        """

    @property
    @abstractmethod
    def supports_multiband(self) -> bool:
        """
        Indicate whether this algorithm supports multi-band assignment.

        :return: True if the algorithm supports multi-band, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def assign(self, path: list[Any], request: Any) -> dict[str, Any] | None:
        """
        Assign spectrum resources along the given path for the request.

        :param path: List of nodes representing the path
        :type path: List[Any]
        :param request: Request object containing traffic demand and spectrum
            requirements
        :type request: Any
        :return: Dictionary containing spectrum assignment details
            (start_slot, end_slot, core_num, band, is_free),
            or None if assignment fails.
        :rtype: Optional[Dict[str, Any]]
        """

    @abstractmethod
    def check_spectrum_availability(
        self, path: list[Any], start_slot: int, end_slot: int, core_num: int, band: str
    ) -> bool:
        """
        Check if spectrum slots are available along the entire path.

        :param path: List of nodes representing the path
        :type path: List[Any]
        :param start_slot: Starting slot index
        :type start_slot: int
        :param end_slot: Ending slot index (inclusive)
        :type end_slot: int
        :param core_num: Core number to check
        :type core_num: int
        :param band: Band identifier
        :type band: str
        :return: True if all slots are available, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def allocate_spectrum(
        self,
        path: list[Any],
        start_slot: int,
        end_slot: int,
        core_num: int,
        band: str,
        request_id: Any,
    ) -> bool:
        """
        Allocate spectrum resources along the path.

        :param path: List of nodes representing the path
        :type path: List[Any]
        :param start_slot: Starting slot index
        :type start_slot: int
        :param end_slot: Ending slot index (inclusive)
        :type end_slot: int
        :param core_num: Core number to allocate
        :type core_num: int
        :param band: Band identifier
        :type band: str
        :param request_id: Unique identifier for the request
        :type request_id: Any
        :return: True if allocation successful, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def deallocate_spectrum(
        self, path: list[Any], start_slot: int, end_slot: int, core_num: int, band: str
    ) -> bool:
        """
        Deallocate spectrum resources along the path.

        :param path: List of nodes representing the path
        :type path: List[Any]
        :param start_slot: Starting slot index
        :type start_slot: int
        :param end_slot: Ending slot index (inclusive)
        :type end_slot: int
        :param core_num: Core number to deallocate
        :type core_num: int
        :param band: Band identifier
        :type band: str
        :return: True if deallocation successful, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def get_fragmentation_metric(self, path: list[Any]) -> float:
        """
        Calculate fragmentation metric for the given path.

        :param path: List of nodes representing the path
        :type path: List[Any]
        :return: Fragmentation value (0.0 = no fragmentation,
                1.0 = maximum fragmentation)
        :rtype: float
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """
        Get spectrum assignment algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics
        :rtype: Dict[str, Any]
        """

    def reset(self) -> None:  # noqa: B027
        """
        Reset the spectrum assignment algorithm state.

        This method can be overridden by subclasses that maintain state.
        """
        pass
