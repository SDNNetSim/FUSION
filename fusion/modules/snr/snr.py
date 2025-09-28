"""
Standard SNR measurement algorithm implementation.
"""

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except ImportError:
        np = None

from fusion.core.properties import SNRProps
from fusion.interfaces.snr import AbstractSNRMeasurer

# Removed unused imports


class StandardSNRMeasurer(AbstractSNRMeasurer):
    """
    Standard SNR measurement algorithm.

    This algorithm calculates signal-to-noise ratio considering linear noise (ASE),
    nonlinear impairments (SCI, XCI), and optionally cross-talk in multi-core fibers.
    """

    def __init__(
        self,
        engine_props: dict,
        sdn_props: object,
        spectrum_props: object,
        route_props: object,
    ):
        """
        Initialize SNR measurement algorithm.

        :param engine_props: Dictionary containing engine configuration
        :param sdn_props: Object containing SDN controller properties
        :param spectrum_props: Object containing spectrum assignment properties
        :param route_props: Object containing routing properties
        """
        super().__init__(engine_props, sdn_props, spectrum_props, route_props)
        self.snr_props = SNRProps()
        self._calculations_performed = 0
        self._total_snr_computed = 0.0

        # Initialize calculation state
        self.channels_list: list | None = None
        self.link_id: Any | None = None
        self.num_slots: int | None = None

    @property
    def algorithm_name(self) -> str:
        """Return the name of the SNR measurement algorithm."""
        return "standard_snr"

    @property
    def supports_multicore(self) -> bool:
        """Indicate whether this algorithm supports multi-core fiber measurements."""
        return True

    def calculate_snr(self, path: list[Any], spectrum_info: dict[str, Any]) -> float:
        """
        Calculate the SNR for a given path and spectrum assignment.

        :param path: List of nodes representing the path
        :param spectrum_info: Dictionary containing spectrum assignment details
        :return: SNR value in dB
        :raises ValueError: If path is empty or spectrum_info is invalid
        """
        # Validate inputs
        if not path:
            raise ValueError("Path cannot be empty")
        if not spectrum_info:
            raise ValueError("Spectrum info cannot be empty")
        if "start_slot" not in spectrum_info or "end_slot" not in spectrum_info:
            raise ValueError("Spectrum info must contain 'start_slot' and 'end_slot'")

        # Store path and spectrum info
        if hasattr(self.spectrum_props, "path_list"):
            self.spectrum_props.path_list = path
        if hasattr(self.spectrum_props, "start_slot"):
            self.spectrum_props.start_slot = spectrum_info.get("start_slot", 0)
        if hasattr(self.spectrum_props, "end_slot"):
            self.spectrum_props.end_slot = spectrum_info.get("end_slot", 0)
        if hasattr(self.spectrum_props, "core_number"):
            self.spectrum_props.core_number = spectrum_info.get("core_number", 0)
        if hasattr(self.spectrum_props, "current_band"):
            self.spectrum_props.current_band = spectrum_info.get("band", "c")

        # Calculate parameters for SNR computation
        self._setup_snr_calculation(spectrum_info)

        # Calculate total SNR across the path
        total_snr_linear = 0.0

        for i in range(len(path) - 1):
            link_snr = self.calculate_link_snr(path[i], path[i + 1], spectrum_info)
            # Convert to linear scale and accumulate (SNR adds inversely)
            if link_snr > 0:
                total_snr_linear += 1.0 / (10 ** (link_snr / 10.0))

        # Convert back to dB
        if total_snr_linear > 0:
            total_snr_db = 10 * math.log10(1.0 / total_snr_linear)
        else:
            total_snr_db = float("inf")  # Perfect SNR

        self._calculations_performed += 1
        self._total_snr_computed += total_snr_db

        return total_snr_db

    def _setup_snr_calculation(self, spectrum_info: dict[str, Any]) -> None:
        """
        Setup parameters for SNR calculation.

        :param spectrum_info: Dictionary containing spectrum assignment details
                              including start_slot, end_slot, core_number, and band
        """
        # Calculate center frequency and bandwidth
        start_slot = spectrum_info.get("start_slot", 0)
        end_slot = spectrum_info.get("end_slot", 0)

        self.num_slots = end_slot - start_slot + 1
        slot_width = self.engine_props.get("bw_per_slot", 12.5e9)  # Hz

        self.snr_props.bandwidth = self.num_slots * slot_width
        if self.num_slots is not None:
            self.snr_props.center_frequency = (
                start_slot + self.num_slots / 2
            ) * slot_width
        else:
            self.snr_props.center_frequency = start_slot * slot_width

        # Calculate power spectral density
        input_power = self.engine_props.get("input_power", 1e-3)  # W
        self.snr_props.center_psd = input_power / self.snr_props.bandwidth

    def calculate_link_snr(
        self, source: Any, destination: Any, spectrum_info: dict[str, Any]
    ) -> float:
        """
        Calculate the SNR for a single link.

        :param source: Source node identifier
        :param destination: Destination node identifier
        :param spectrum_info: Dictionary containing spectrum assignment details
        :return: SNR value in dB for the link
        """
        link_key = (source, destination)

        # Validate inputs
        if source is None or destination is None:
            raise ValueError("Source and destination cannot be None")

        if (
            not hasattr(self.sdn_props, "network_spectrum_dict")
            or link_key not in self.sdn_props.network_spectrum_dict
        ):
            return 0.0

        # Get link properties
        if hasattr(self.sdn_props, "topology") and self.sdn_props.topology.has_edge(
            source, destination
        ):
            link_length = self.sdn_props.topology[source][destination].get(
                "length", 100
            )  # km
        else:
            link_length = 100  # Default link length

        self.snr_props.link_dictionary = {
            "length": link_length,
            "attenuation": self.engine_props.get("fiber_attenuation", 0.2),  # dB/km
            "dispersion": self.engine_props.get("fiber_dispersion", 16.7),  # ps/nm/km
            "nonlinear_coeff": self.engine_props.get(
                "nonlinear_coefficient", 1.3e-3
            ),  # 1/W/km
            "bending_radius": self.engine_props.get("bending_radius", 7.5e-3),  # m
        }

        # Calculate linear noise (ASE)
        ase_noise = self._calculate_ase_noise(link_length)

        # Calculate nonlinear noise
        nonlinear_noise = self._calculate_nonlinear_noise(
            source, destination, spectrum_info
        )

        # Calculate cross-talk noise (if multi-core)
        xt_noise = 0.0
        if self.supports_multicore and spectrum_info.get("core_num", 0) is not None:
            xt_noise = self._calculate_crosstalk_noise(
                source, destination, spectrum_info
            )

        # Total noise
        total_noise = (
            ase_noise + nonlinear_noise["sci"] + nonlinear_noise["xci"] + xt_noise
        )

        # Signal power
        signal_power = self.engine_props.get("input_power", 1e-3)  # W

        # SNR in dB
        if total_noise > 0:
            snr_db = 10 * math.log10(signal_power / total_noise)
        else:
            snr_db = float("inf")

        return snr_db

    def _calculate_ase_noise(self, link_length: float) -> float:
        """
        Calculate amplified spontaneous emission noise.

        :param link_length: Physical length of the fiber link in kilometers
        :return: ASE noise power in watts
        """
        # Simplified ASE noise calculation
        h = 6.626e-34  # Planck constant
        freq = 193.1e12  # Optical frequency (Hz) - C band center
        noise_figure = self.engine_props.get("edfa_noise_figure", 4.5)  # dB

        # Convert noise figure to linear
        nf_linear = 10 ** (noise_figure / 10.0)

        # ASE noise power
        ase_power = h * freq * nf_linear * self.snr_props.bandwidth

        # Account for number of amplifiers (assuming EDFA every 80 km)
        num_amps = max(1, int(link_length / 80))

        return float(ase_power * num_amps)

    def calculate_nonlinear_noise(
        self, path: list[Any], spectrum_info: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate nonlinear noise components.

        :param path: List of nodes representing the path
        :param spectrum_info: Dictionary containing spectrum assignment details
        :return: Dictionary containing nonlinear noise components
        """
        # Store path for internal calculations
        if hasattr(self.spectrum_props, "path_list"):
            self.spectrum_props.path_list = path
        if hasattr(self.spectrum_props, "start_slot"):
            self.spectrum_props.start_slot = spectrum_info.get("start_slot", 0)
        if hasattr(self.spectrum_props, "end_slot"):
            self.spectrum_props.end_slot = spectrum_info.get("end_slot", 0)
        if hasattr(self.spectrum_props, "core_number"):
            self.spectrum_props.core_number = spectrum_info.get("core_number", 0)
        if hasattr(self.spectrum_props, "current_band"):
            self.spectrum_props.current_band = spectrum_info.get("band", "c")

        # Setup calculation parameters
        self._setup_snr_calculation(spectrum_info)

        total_sci = 0.0
        total_xci = 0.0

        # Calculate nonlinear noise for each link in the path
        for i in range(len(path) - 1):
            link_noise = self._calculate_nonlinear_noise(
                path[i], path[i + 1], spectrum_info
            )
            total_sci += link_noise["sci"]
            total_xci += link_noise["xci"]

        return {
            "sci": total_sci,
            "xci": total_xci,
            "xpm": 0.0,  # Cross-phase modulation (simplified)
            "fwm": 0.0,  # Four-wave mixing (simplified)
        }

    def _calculate_nonlinear_noise(
        self, _source: Any, _destination: Any, _spectrum_info: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate nonlinear noise components for a single link.

        :param _source: Source node identifier (unused)
        :param _destination: Destination node identifier (unused)
        :param _spectrum_info: Dictionary containing spectrum assignment details
        :return: Dictionary containing nonlinear noise components (SCI, XCI, XPM, FWM)
        """
        # Self-channel interference (SCI)
        sci_noise = self._calculate_sci_psd()

        # Cross-channel interference (XCI)
        xci_noise = self._calculate_xci(0)  # Link index 0 for single link

        return {
            "sci": sci_noise,
            "xci": xci_noise,
            "xpm": 0.0,  # Cross-phase modulation (simplified)
            "fwm": 0.0,  # Four-wave mixing (simplified)
        }

    def _calculate_sci_psd(self) -> float:
        """
        Calculate the self-channel interference power spectral density.

        :return: SCI power spectral density in watts/Hz
        """
        if self.snr_props.link_dictionary is not None:
            dispersion = self.snr_props.link_dictionary.get("dispersion", 16.7)
            attenuation = self.snr_props.link_dictionary.get("attenuation", 0.2)
            rho_param = (math.pi**2) * abs(dispersion)
            rho_param /= 2 * attenuation

            center_psd = getattr(self.snr_props, "center_psd", 1e-6)
            bandwidth = getattr(self.snr_props, "bandwidth", 12.5e9)
            sci_psd = center_psd**2
            sci_psd *= math.asinh(rho_param * (bandwidth**2))
        else:
            sci_psd = 0.0

        return sci_psd

    def _calculate_xci(self, link_num: int) -> float:
        """
        Calculate cross-channel interference noise from other channels.

        :param link_num: Index of the link in the path (0-based)
        :return: Total XCI noise power in watts
        """
        self.channels_list = []
        xci_noise = 0.0

        # Get spectrum state for the link
        if not hasattr(self.spectrum_props, "path_list") or not hasattr(
            self.sdn_props, "network_spectrum_dict"
        ):
            return 0.0

        path_list = getattr(self.spectrum_props, "path_list", [])
        if link_num >= len(path_list) - 1:
            return 0.0

        link_tuple = (
            path_list[link_num],
            path_list[link_num + 1],
        )

        network_spectrum_dict = getattr(self.sdn_props, "network_spectrum_dict", {})
        if link_tuple not in network_spectrum_dict:
            return 0.0

        curr_link = network_spectrum_dict[link_tuple]
        band = getattr(self.spectrum_props, "current_band", "c")
        core_num = getattr(self.spectrum_props, "core_num", 0)

        if band not in curr_link.get("cores_matrix", {}):
            return 0.0

        core_array = curr_link["cores_matrix"][band][core_num]

        # Check for interfering channels
        c_band_slots = self.engine_props.get("c_band", 320)  # Number of slots in C-band

        for slot_index in range(min(len(core_array), c_band_slots)):
            req_id = core_array[slot_index]

            # Spectrum is occupied by another request
            if (
                req_id > 0
                and self.channels_list is not None
                and req_id not in self.channels_list
            ):
                self.channels_list.append(req_id)
                xci_noise = self._update_link_xci(
                    req_id, core_array, slot_index, xci_noise
                )

        return xci_noise

    def _update_link_xci(
        self, req_id: float, curr_link: Any, slot_index: int, curr_xci: float
    ) -> float:
        """
        Update cross-channel interference from a specific interfering channel.

        :param req_id: Request ID of the interfering channel
        :param curr_link: Current link spectrum state array
        :param slot_index: Slot index of the interfering channel
        :param curr_xci: Current accumulated XCI noise power
        :return: Updated XCI noise power in watts
        """
        # Calculate interfering channel bandwidth and frequency
        core_num = getattr(self.spectrum_props, "core_num", 0)
        if np is not None:
            # Use numpy if available
            channel_bw = len(np.where(req_id == curr_link[core_num])[0])
        else:
            # Fallback implementation without numpy
            core_array = curr_link[core_num]
            channel_bw = sum(1 for x in core_array if x == req_id)

        channel_bw *= self.engine_props.get("bw_per_slot", 12.5e9)

        channel_freq = (slot_index * self.engine_props.get("bw_per_slot", 12.5e9)) + (
            channel_bw / 2
        )

        channel_psd = self.engine_props.get("input_power", 1e-3) / channel_bw

        if self.snr_props.center_frequency != channel_freq:
            log_term = abs(self.snr_props.center_frequency - channel_freq) + (
                channel_bw / 2
            )
            log_term /= abs(self.snr_props.center_frequency - channel_freq) - (
                channel_bw / 2
            )

            calculated_xci = (channel_psd**2) * math.log(abs(log_term))
            new_xci = curr_xci + calculated_xci
        else:
            new_xci = curr_xci

        return float(new_xci)

    def calculate_crosstalk(
        self, path: list[Any], core_num: int, spectrum_info: dict[str, Any]
    ) -> float:
        """Calculate crosstalk noise for the given path and core."""
        total_xt = 0.0

        for i in range(len(path) - 1):
            link_xt = self._calculate_crosstalk_noise(
                path[i], path[i + 1], spectrum_info
            )
            total_xt += link_xt

        return float(total_xt)

    def _calculate_crosstalk_noise(
        self, _source: Any, _destination: Any, spectrum_info: dict[str, Any]
    ) -> float:
        """
        Calculate crosstalk noise for a single link in multi-core fiber.

        :param _source: Source node identifier (unused)
        :param _destination: Destination node identifier (unused)
        :param spectrum_info: Dictionary containing spectrum assignment details
                              including core_num for multi-core calculations
        :return: Cross-talk noise power in watts
        """
        if not self.supports_multicore:
            return 0.0

        # Get number of adjacent cores
        cores_per_link = self.engine_props.get("cores_per_link", 7)
        core_num = spectrum_info.get("core_num", 0)

        # Simplified: assume all cores except center have 6 neighbors,
        # center has all as neighbors
        if core_num == 0:  # Center core
            num_adjacent = cores_per_link - 1
        else:
            num_adjacent = min(6, cores_per_link - 1)  # Outer cores

        return float(self._calculate_pxt(num_adjacent))

    def _calculate_pxt(self, num_adjacent: int) -> float:
        """
        Calculate cross-talk noise power based on adjacent cores.

        :param num_adjacent: Number of adjacent cores contributing to cross-talk
        :return: Total cross-talk noise power in watts
        """
        # Statistical mean of cross-talk
        link_dict = getattr(self.snr_props, "link_dictionary", {})
        if link_dict is not None:
            _bending_radius = link_dict.get("bending_radius", 7.5e-3)
        else:
            _bending_radius = 7.5e-3
        # Cross-talk coefficient calculation uses bending radius

        # Cross-talk power (simplified model)
        xt_coefficient = self.engine_props.get("xt_coefficient", -40)  # dB
        xt_linear = 10 ** (xt_coefficient / 10.0)

        # Scale by number of adjacent cores
        total_xt = num_adjacent * xt_linear * self.engine_props.get("input_power", 1e-3)

        return float(total_xt)

    def get_required_snr_threshold(self, modulation: str, reach: float) -> float:
        """
        Get the required SNR threshold for a given modulation format and reach.

        :param modulation: Modulation format (e.g., 'QPSK', '16QAM', '64QAM')
        :param reach: Transmission reach in kilometers
        :return: Required SNR threshold in dB
        """
        # SNR thresholds for different modulations (simplified)
        thresholds = {
            "BPSK": 6.0,  # dB
            "QPSK": 9.0,  # dB
            "8QAM": 12.0,  # dB
            "16QAM": 15.0,  # dB
            "32QAM": 18.0,  # dB
            "64QAM": 21.0,  # dB
        }

        base_threshold = thresholds.get(modulation, 12.0)  # Default to QPSK

        # Add reach penalty (simplified: 0.1 dB per 100 km)
        reach_penalty = (reach / 100.0) * 0.1

        return float(base_threshold + reach_penalty)

    def is_snr_acceptable(
        self, calculated_snr: float, required_snr: float, margin: float = 0.0
    ) -> bool:
        """Check if calculated SNR meets the requirement with optional margin."""
        return calculated_snr >= (required_snr + margin)

    def update_link_state(
        self, source: Any, destination: Any, spectrum_info: dict[str, Any]
    ) -> None:
        """
        Update link state based on new spectrum allocation.

        :param source: Source node identifier
        :param destination: Destination node identifier
        :param spectrum_info: Dictionary containing spectrum assignment details

        Note:
            This implementation doesn't maintain dynamic state.
            Subclasses can override for adaptive algorithms.
        """
        # This implementation doesn't maintain dynamic state
        # Subclasses can override for adaptive algorithms

    def get_metrics(self) -> dict[str, Any]:
        """Get SNR measurement algorithm performance metrics."""
        avg_snr = (
            self._total_snr_computed / self._calculations_performed
            if self._calculations_performed > 0
            else 0
        )

        return {
            "algorithm": self.algorithm_name,
            "calculations_performed": self._calculations_performed,
            "average_snr_computed": avg_snr,
            "supports_multicore": self.supports_multicore,
            "noise_models": ["ASE", "SCI", "XCI", "XT"],
        }

    def reset(self) -> None:
        """Reset the SNR measurement algorithm state."""
        self._calculations_performed = 0
        self._total_snr_computed = 0.0
        self.snr_props = SNRProps()
