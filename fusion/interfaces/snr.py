"""
Abstract base class for SNR measurement algorithms in FUSION.
"""

# pylint: disable=duplicate-code

from abc import ABC, abstractmethod
from typing import Any


class AbstractSNRMeasurer(ABC):
    """Base class for all SNR measurement algorithms in FUSION.

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
        """Initialize the SNR measurement algorithm.

        Args:
            engine_props: Dictionary containing engine configuration and properties
            sdn_props: Object containing SDN controller properties and network state
            spectrum_props: Object containing spectrum assignment properties
            route_props: Object containing routing properties and selected path
        """
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.spectrum_props = spectrum_props
        self.route_props = route_props

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the name of the SNR measurement algorithm.

        Returns:
            String identifier for this SNR measurement algorithm
        """

    @property
    @abstractmethod
    def supports_multicore(self) -> bool:
        """Indicate whether this algorithm supports multi-core fiber measurements.

        Returns:
            True if the algorithm supports multi-core, False otherwise
        """

    @abstractmethod
    def calculate_snr(self, path: list[Any], spectrum_info: dict[str, Any]) -> float:
        """Calculate the SNR for a given path and spectrum assignment.

        Args:
            path: List of nodes representing the path
            spectrum_info: Dictionary containing spectrum assignment details:
                - 'start_slot': Starting slot index
                - 'end_slot': Ending slot index
                - 'core_num': Core number (for multi-core fibers)
                - 'band': Band identifier
                - 'modulation': Modulation format

        Returns:
            SNR value in dB
        """

    @abstractmethod
    def calculate_link_snr(
        self, source: Any, destination: Any, spectrum_info: dict[str, Any]
    ) -> float:
        """Calculate the SNR for a single link.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            spectrum_info: Dictionary containing spectrum assignment details

        Returns:
            SNR value in dB for the link
        """

    @abstractmethod
    def calculate_crosstalk(
        self, path: list[Any], core_num: int, spectrum_info: dict[str, Any]
    ) -> float:
        """Calculate crosstalk noise for the given path and core.

        Args:
            path: List of nodes representing the path
            core_num: Core number to calculate crosstalk for
            spectrum_info: Dictionary containing spectrum assignment details

        Returns:
            Crosstalk noise power in linear units
        """

    @abstractmethod
    def calculate_nonlinear_noise(
        self, path: list[Any], spectrum_info: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate nonlinear noise components.

        Args:
            path: List of nodes representing the path
            spectrum_info: Dictionary containing spectrum assignment details

        Returns:
            Dictionary containing nonlinear noise components:
                - 'sci': Self-channel interference
                - 'xci': Cross-channel interference
                - 'xpm': Cross-phase modulation
                - 'fwm': Four-wave mixing (if applicable)
        """

    @abstractmethod
    def get_required_snr_threshold(self, modulation: str, reach: float) -> float:
        """Get the required SNR threshold for a given modulation format and reach.

        Args:
            modulation: Modulation format identifier
            reach: Transmission reach in km

        Returns:
            Required SNR threshold in dB
        """

    @abstractmethod
    def is_snr_acceptable(
        self, calculated_snr: float, required_snr: float, margin: float = 0.0
    ) -> bool:
        """Check if calculated SNR meets the requirement with optional margin.

        Args:
            calculated_snr: Calculated SNR value in dB
            required_snr: Required SNR threshold in dB
            margin: Additional margin in dB (default: 0.0)

        Returns:
            True if SNR is acceptable, False otherwise
        """

    @abstractmethod
    def update_link_state(
        self, source: Any, destination: Any, spectrum_info: dict[str, Any]
    ) -> None:
        """Update link state based on new spectrum allocation.

        Args:
            source: Source node identifier
            destination: Destination node identifier
            spectrum_info: Dictionary containing spectrum assignment details
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get SNR measurement algorithm performance metrics.

        Returns:
            Dictionary containing algorithm-specific metrics
        """
