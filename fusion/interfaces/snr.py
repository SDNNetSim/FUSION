"""
Abstract base class for SNR measurement algorithms in FUSION.
"""

from abc import ABC, abstractmethod
from typing import Any


class AbstractSNRMeasurer(ABC):
    """
    Base class for all SNR measurement algorithms in FUSION.

    This interface defines the contract that all SNR measurement algorithms must follow
    to ensure compatibility with the FUSION simulation framework.
    """

    def __init__(
        self,
        engine_props: dict,
        sdn_props: object,
        spectrum_props: object,
        route_props: object,
    ):
        """
        Initialize the SNR measurement algorithm.

        :param engine_props: Dictionary containing engine configuration and properties
        :type engine_props: dict
        :param sdn_props: Object containing SDN controller properties and network state
        :type sdn_props: object
        :param spectrum_props: Object containing spectrum assignment properties
        :type spectrum_props: object
        :param route_props: Object containing routing properties and selected path
        :type route_props: object
        """
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.spectrum_props = spectrum_props
        self.route_props = route_props

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """
        Return the name of the SNR measurement algorithm.

        :return: String identifier for this SNR measurement algorithm
        :rtype: str
        """

    @property
    @abstractmethod
    def supports_multicore(self) -> bool:
        """
        Indicate whether this algorithm supports multi-core fiber measurements.

        :return: True if the algorithm supports multi-core, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def calculate_snr(self, path: list[Any], spectrum_info: dict[str, Any]) -> float:
        """
        Calculate the SNR for a given path and spectrum assignment.

        :param path: List of nodes representing the path
        :type path: List[Any]
        :param spectrum_info: Dictionary containing spectrum assignment details:
                - 'start_slot': Starting slot index
                - 'end_slot': Ending slot index
                - 'core_num': Core number (for multi-core fibers)
                - 'band': Band identifier
                - 'modulation': Modulation format
        :type spectrum_info: Dict[str, Any]
        :return: SNR value in dB
        :rtype: float
        """

    @abstractmethod
    def calculate_link_snr(self, source: Any, destination: Any, spectrum_info: dict[str, Any]) -> float:
        """
         Calculate the SNR for a single link.

        :param source: Source node identifier
        :type source: Any
        :param destination: Destination node identifier
        :type destination: Any
        :param spectrum_info: Dictionary containing spectrum assignment details
        :type spectrum_info: Dict[str, Any]
        :return: SNR value in dB for the link
        :rtype: float
        """

    @abstractmethod
    def calculate_crosstalk(self, path: list[Any], core_num: int, spectrum_info: dict[str, Any]) -> float:
        """
        Calculate crosstalk noise for the given path and core.

        :param path: List of nodes representing the path
        :type path: List[Any]
        :param core_num: Core number to calculate crosstalk for
        :type core_num: int
        :param spectrum_info: Dictionary containing spectrum assignment details
        :type spectrum_info: Dict[str, Any]
        :return: Crosstalk noise power in linear units
        :rtype: float
        """

    @abstractmethod
    def calculate_nonlinear_noise(self, path: list[Any], spectrum_info: dict[str, Any]) -> dict[str, float]:
        """
        Calculate nonlinear noise components.

        :param path: List of nodes representing the path
        :type path: List[Any]
        :param spectrum_info: Dictionary containing spectrum assignment details
        :type spectrum_info: Dict[str, Any]
        :return: Dictionary containing nonlinear noise components:
                - 'sci': Self-channel interference
                - 'xci': Cross-channel interference
                - 'xpm': Cross-phase modulation
                - 'fwm': Four-wave mixing (if applicable)
        :rtype: Dict[str, float]
        """

    @abstractmethod
    def get_required_snr_threshold(self, modulation: str, reach: float) -> float:
        """
        Get the required SNR threshold for a given modulation format and reach.

        :param modulation: Modulation format identifier
        :type modulation: str
        :param reach: Transmission reach in km
        :type reach: float
        :return: Required SNR threshold in dB
        :rtype: float
        """

    @abstractmethod
    def is_snr_acceptable(self, calculated_snr: float, required_snr: float, margin: float = 0.0) -> bool:
        """
        Check if calculated SNR meets the requirement with optional margin.

        :param calculated_snr: Calculated SNR value in dB
        :type calculated_snr: float
        :param required_snr: Required SNR threshold in dB
        :type required_snr: float
        :param margin: Additional margin in dB (default: 0.0)
        :type margin: float
        :return: True if SNR is acceptable, False otherwise
        :rtype: bool
        """

    @abstractmethod
    def update_link_state(self, source: Any, destination: Any, spectrum_info: dict[str, Any]) -> None:
        """
        Update link state based on new spectrum allocation.

        :param source: Source node identifier
        :type source: Any
        :param destination: Destination node identifier
        :type destination: Any
        :param spectrum_info: Dictionary containing spectrum assignment details
        :type spectrum_info: Dict[str, Any]
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """
        Get SNR measurement algorithm performance metrics.

        :return: Dictionary containing algorithm-specific metrics
        :rtype: Dict[str, Any]
        """

    def reset(self) -> None:  # noqa: B027
        """
        Reset the SNR measurement algorithm state.

        This method can be overridden by subclasses that maintain state.
        """
        pass
