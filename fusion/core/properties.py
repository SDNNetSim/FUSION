"""
Core simulation properties and data classes.

Consolidated from various CLI argument files for better organization.
"""

from typing import Any, Optional

# Physical constants
PLANCK_CONSTANT = 6.62607004e-34  # Planck's constant in J⋅s
LIGHT_FREQUENCY_CENTER = 1.9341e14  # Center light frequency in Hz

# Network constants
DEFAULT_INPUT_POWER = 1e-3  # Default input power in Watts
DEFAULT_FREQUENCY_SPACING = 12.5e9  # Default frequency spacing in Hz
DEFAULT_SPAN_LENGTH = 100.0  # Default span length in km
WORST_CASE_MCI = 6.3349755556585961e-27  # Worst-case mutual coupling interference

# Statistics tracking constants
SNAP_KEYS_LIST = [
    "occupied_slots",
    "guard_slots",
    "active_requests",
    "blocking_prob",
    "num_segments",
]


# pylint: disable=too-few-public-methods
class RoutingProps:
    """
    Main properties used for routing operations.

    This class encapsulates all properties required for path computation
    and routing decisions in the optical network simulation.
    """

    def __init__(self) -> None:
        """Initialize routing properties with default values."""
        # Path computation properties - can contain path lists or path data dicts
        self.paths_matrix: list[Any] = []
        self.modulation_formats_matrix: list[list[str]] = []
        self.weights_list: list[float] = []
        self.path_index_list: list[int] = []

        # Backup paths for 1+1 protection (corresponds to paths_matrix)
        self.backup_paths_matrix: list[list[int] | None] = []
        self.backup_modulation_formats_matrix: list[list[str]] = []

        # Physical layer parameters
        self.input_power: float = DEFAULT_INPUT_POWER
        self.frequency_spacing: float = DEFAULT_FREQUENCY_SPACING
        self.mci_worst: float = WORST_CASE_MCI
        self.span_length: float = DEFAULT_SPAN_LENGTH

        # Network constraints
        self.max_link_length: float | None = None
        self.max_span: int | None = None

        # Routing state
        self.connection_index: int | None = None
        self.path_index: int | None = None

    def __repr__(self) -> str:
        """
        Return string representation of RoutingProps.

        :return: String representation with all properties
        :rtype: str
        """
        return f"RoutingProps({self.__dict__})"


class SpectrumProps:
    """
    Main properties used for spectrum assignment operations.

    This class manages all properties related to spectrum allocation,
    including slot assignment, core selection, and modulation format.
    """

    def __init__(self) -> None:
        """Initialize spectrum properties with default values."""
        # Path and resource requirements
        self.path_list: list[int] | None = None
        self.backup_path: list[int] | None = None  # For 1+1 protection
        self.slots_needed: int | None = None
        self.modulation: str | None = None

        # Core assignment properties
        self.forced_core: int | None = None
        self.core_number: int | None = None
        self.cores_matrix: dict[str, Any] | None = None
        self.reverse_cores_matrix: dict[str, Any] | None = None

        # Spectrum assignment properties
        self.is_free: bool = False
        self.forced_index: int | None = None
        self.start_slot: int | None = None
        self.end_slot: int | None = None

        # Multi-band support
        self.forced_band: str | None = None
        self.current_band: str | None = None

        # Cost metrics
        self.crosstalk_cost: float | None = None

        # Lightpath tracking (for grooming)
        self.lightpath_id: int | None = None
        self.lightpath_bandwidth: float | None = None

    def __repr__(self) -> str:
        """
        Return string representation of SpectrumProps.

        :return: String representation with all properties
        :rtype: str
        """
        return f"SpectrumProps({self.__dict__})"


class SNRProps:
    """
    Main properties used for SNR measurements.

    This class contains all parameters required for Signal-to-Noise Ratio
    calculations and modulation format selection based on link quality.
    """

    def __init__(self) -> None:
        """Initialize SNR properties with default values."""
        # Physical constants
        self.light_frequency: float = LIGHT_FREQUENCY_CENTER
        self.planck_constant: float = PLANCK_CONSTANT

        # System parameters
        self.request_bit_rate: float = 12.5  # Gb/s
        self.request_snr: float = 8.5  # dB
        self.noise_spectral_density: float = 1.8

        # Current request parameters
        self.center_frequency: float | None = None
        self.bandwidth: float | None = None
        self.center_psd: float | None = None
        self.mu_parameter: float | None = None
        self.self_channel_interference_psd: float | None = None
        self.cross_channel_interference_psd: float | None = None
        self.length: float | None = None
        self.number_of_spans: int | None = None

        # Link information
        self.link_dictionary: dict[str, Any] | None = None

        # Modulation format mappings
        self.modulation_format_mapping_dict: dict[int, str] = {
            6: "64-QAM",
            5: "32-QAM",
            4: "16-QAM",
            3: "8-QAM",
            2: "QPSK",
            1: "BPSK",
        }

        self.bandwidth_mapping_dict: dict[str, int] = {
            "64-QAM": 600,
            "32-QAM": 500,
            "16-QAM": 400,
            "8-QAM": 300,
            "QPSK": 200,
            "BPSK": 100,
        }

        # Pre-calculated SNR file mappings
        self._initialize_file_mappings()

    def _initialize_file_mappings(self) -> None:
        """Initialize the pre-calculated SNR file mappings."""
        self.file_mapping_dict: dict[str, dict[str | tuple, dict[str, str]]] = {
            "USbackbone60": {
                "multi_fiber": {
                    "mf": "MF-USB6014-MF.npy",
                    "gsnr": "GSNR-USB6014-MF.npy",
                },
                (2, 4): {
                    "mf": "MF-USB6014-MCF4-C2.npy",
                    "gsnr": "GSNR-USB6014-MCF4-C2.npy",
                },
                (6, 7): {
                    "mf": "MF-USB6014-MCF7-C6.npy",
                    "gsnr": "GSNR-USB6014-MCF7-C6.npy",
                },
                (3, 7): {
                    "mf": "MF-USB6014-MCF7-C3.npy",
                    "gsnr": "GSNR-USB6014-MCF7-C3.npy",
                },
                (2, 13): {
                    "mf": "MF-USB6014-MCF13-C2.npy",
                    "gsnr": "GSNR-USB6014-MCF13-C2.npy",
                },
                (5, 13): {
                    "mf": "MF-USB6014-MCF13-C5.npy",
                    "gsnr": "GSNR-USB6014-MCF13-C5.npy",
                },
                (6, 13): {
                    "mf": "MF-USB6014-MCF13-C6.npy",
                    "gsnr": "GSNR-USB6014-MCF13-C6.npy",
                },
                (6, 19): {
                    "mf": "MF-USB6014-MCF19-C6.npy",
                    "gsnr": "GSNR-USB6014-MCF19-C6.npy",
                },
                (3, 19): {
                    "mf": "MF-USB6014-MCF19-C3.npy",
                    "gsnr": "GSNR-USB6014-MCF19-C3.npy",
                },
                (4, 19): {
                    "mf": "MF-USB6014-MCF19-C4.npy",
                    "gsnr": "GSNR-USB6014-MCF19-C4.npy",
                },
            },
            "Spainbackbone30": {
                "multi_fiber": {
                    "mf": "MF-SPNB3014-MF.npy",
                    "gsnr": "GSNR-SPNB3014-MF.npy",
                },
                (2, 4): {
                    "mf": "MF-SPNB3014-MCF4-C2.npy",
                    "gsnr": "GSNR-SPNB3014-MCF4-C2.npy",
                },
                (6, 7): {
                    "mf": "MF-SPNB3014-MCF7-C6.npy",
                    "gsnr": "GSNR-SPNB3014-MCF7-C6.npy",
                },
                (3, 7): {
                    "mf": "MF-SPNB3014-MCF7-C3.npy",
                    "gsnr": "GSNR-SPNB3014-MCF7-C3.npy",
                },
                (2, 13): {
                    "mf": "MF-SPNB3014-MCF13-C2.npy",
                    "gsnr": "GSNR-SPNB3014-MCF13-C2.npy",
                },
                (5, 13): {
                    "mf": "MF-SPNB3014-MCF13-C5.npy",
                    "gsnr": "GSNR-SPNB3014-MCF13-C5.npy",
                },
                (6, 13): {
                    "mf": "MF-SPNB3014-MCF13-C6.npy",
                    "gsnr": "GSNR-SPNB3014-MCF13-C6.npy",
                },
                (6, 19): {
                    "mf": "MF-SPNB3014-MCF19-C6.npy",
                    "gsnr": "GSNR-SPNB3014-MCF19-C6.npy",
                },
                (3, 19): {
                    "mf": "MF-SPNB3014-MCF19-C3.npy",
                    "gsnr": "GSNR-SPNB3014-MCF19-C3.npy",
                },
                (4, 19): {
                    "mf": "MF-SPNB3014-MCF19-C4.npy",
                    "gsnr": "GSNR-SPNB3014-MCF19-C4.npy",
                },
            },
        }

    def __repr__(self) -> str:
        """
        Return string representation of SNRProps.

        :return: String representation with all properties
        :rtype: str
        """
        return f"SNRProps({self.__dict__})"


class GroomingProps:
    """
    Main properties used for traffic grooming operations.

    This class manages properties specific to traffic grooming,
    including grooming type selection and lightpath status tracking.
    """

    def __init__(self) -> None:
        """Initialize grooming properties with default values."""
        self.grooming_type: str | None = None  # Grooming method selection
        self.lightpath_status_dict: dict | None = None  # Established lightpath status

    def __repr__(self) -> str:
        """
        Return string representation of GroomingProps.

        :return: String representation with all properties
        :rtype: str
        """
        return f"GroomingProps({self.__dict__})"


class SDNProps:
    """
    Main properties used for SDN controller operations.

    This class manages all properties related to Software-Defined Networking
    control, including request management, path selection, and resource tracking.
    """

    def __init__(self) -> None:
        """Initialize SDN properties with default values."""
        # Network state
        self.topology: Any | None = None  # NetworkX graph
        self.network_spectrum_dict: dict[tuple[Any, Any], dict[str, Any]] | None = None

        # Current request properties
        self.request_id: int | None = None
        self.source: str | None = None  # Node IDs are strings in the topology
        self.destination: str | None = None  # Node IDs are strings in the topology
        self.bandwidth: float | None = None
        self.arrive: float | None = None
        self.depart: float | None = None
        self.request_type: str | None = None
        self.slots_needed: int | None = None

        # Path information
        self.path_list: list[int] | None = None
        self.path_index: int | None = None
        self.was_routed: bool | None = None

        # Resource allocation lists (for light-segment slicing)
        self.bandwidth_list: list[float] = []
        self.modulation_list: list[str] = []
        self.core_list: list[int] = []
        self.band_list: list[str] = []
        self.crosstalk_list: list[float] = []
        self.start_slot_list: list[int] = []
        self.end_slot_list: list[int] = []

        # Configuration and results
        self.number_of_transponders: int | None = None
        self.single_core: bool = False
        self.block_reason: str | None = None
        self.modulation_formats_dict: dict[str, Any] | None = None

        # Additional attributes needed by SDN controller
        self.route_time: float | None = None
        self.path_weight: float | None = None
        self.spectrum_object: SpectrumProps | None = None
        self.is_sliced: bool | None = None

        # Lightpath tracking dictionaries
        self.lightpath_status_dict: dict | None = None
        self.transponder_usage_dict: dict | None = None
        self.lp_bw_utilization_dict: dict | None = None

        # Grooming state flags
        self.was_groomed: bool | None = None
        self.was_partially_groomed: bool = False
        self.was_partially_routed: bool = False
        self.was_new_lp_established: list[int] = []

        # Lightpath resource tracking
        self.lightpath_id_list: list[int] = []
        self.lightpath_bandwidth_list: list[float] = []
        self.remaining_bw: int | float | str | None = None

        # Lightpath ID counter
        self.lightpath_counter: int = 0

        # Statistical tracking keys
        self.stat_key_list: list[str] = [
            "modulation_list",
            "crosstalk_list",
            "core_list",
            "band_list",
            "start_slot_list",
            "end_slot_list",
            "lightpath_bandwidth_list",
            "lightpath_id_list",
        ]

        # 1+1 Protection attributes
        # Note: Protection is enabled via route_method=1plus1_protection
        self.primary_path: list[int] | None = None
        self.backup_path: list[int] | None = None
        self.is_protected: bool = False
        self.active_path: str = "primary"  # "primary" or "backup"

        # Protection timing (milliseconds)
        self.protection_switchover_ms: float = 50.0
        self.restoration_latency_ms: float = 100.0

        # Switchover tracking
        self.switchover_count: int = 0
        self.last_switchover_time: float | None = None

        # Recovery tracking
        self.recovery_start_time: float | None = None
        self.recovery_end_time: float | None = None
        self.recovery_type: str | None = None  # "protection" or "restoration"

        # Request tracking for failure handling
        # Maps request_id -> request info dict with paths, protection status, etc.
        self.allocated_requests: dict[int, dict[str, Any]] = {}

    def update_params(
        self,
        key: str,
        spectrum_key: str | None,
        spectrum_obj: Optional["SpectrumProps"],
        value: Any | None = None,
    ) -> None:
        """
        Update lists to track statistics of routed requests or general network metrics.

        This method provides a flexible way to update various tracking lists
        based on spectrum assignment results or direct values.

        :param key: Property key to update
        :type key: str
        :param spectrum_key: Property key from spectrum object to retrieve value
        :type spectrum_key: str
        :param spectrum_obj: Spectrum assignment object containing spectrum_props
        :type spectrum_obj: Optional[SpectrumProps]
        :param value: Direct value to set (used when spectrum_key is None)
        :type value: Optional[Any]
        :raises AttributeError: If key doesn't exist in SDNProps

        Example:
            >>> sdn_props.update_params('core_list', 'core_number', spectrum_obj)
            >>> sdn_props.update_params('block_reason', None, None, 'congestion')
        """
        if hasattr(self, key):
            if spectrum_key and spectrum_obj is not None:
                spectrum_value = getattr(spectrum_obj, spectrum_key)
                current_value = getattr(self, key)
                if isinstance(current_value, list):
                    current_value.append(spectrum_value)
                else:
                    setattr(self, key, spectrum_value)
            else:
                setattr(self, key, value)

    def get_lightpath_id(self) -> int:
        """
        Generate and return a new unique lightpath ID.

        :return: New lightpath ID
        :rtype: int
        """
        self.lightpath_counter += 1
        return self.lightpath_counter

    def reset_lightpath_id_counter(self) -> None:
        """
        Reset the lightpath ID counter to zero.

        Called at the start of each simulation iteration.
        """
        self.lightpath_counter = 0

    def reset_params(self) -> None:
        """
        Reset statistical tracking lists to empty state.

        This method clears all lists used for tracking per-request statistics,
        typically called before processing a new request.
        """
        # Existing resets
        self.modulation_list = []
        self.crosstalk_list = []
        self.core_list = []
        self.band_list = []
        self.start_slot_list = []
        self.end_slot_list = []
        self.bandwidth_list = []

        # Grooming-related resets
        self.lightpath_bandwidth_list = []
        self.lightpath_id_list = []
        self.path_list = []
        self.was_new_lp_established = []
        self.lp_bw_utilization_dict = {}

        # Reset flags
        self.was_routed = None
        self.was_groomed = None
        self.was_partially_groomed = False
        self.was_partially_routed = False
        self.remaining_bw = None
        self.is_sliced = None
        self.path_weight = None

    def get_data(self, key: str) -> Any:
        """
        Retrieve a property value from the SDNProps object.

        :param key: The property name to retrieve
        :type key: str
        :return: The value of the requested property
        :rtype: Any
        :raises AttributeError: If the property doesn't exist

        Example:
            >>> request_id = sdn_props.get_data('request_id')
            >>> bandwidth = sdn_props.get_data('bandwidth')
        """
        if hasattr(self, key):
            return getattr(self, key)

        raise AttributeError(f"'SDNProps' object has no attribute '{key}'")

    def __repr__(self) -> str:
        """
        Return string representation of SDNProps.

        :return: String representation with all properties
        :rtype: str
        """
        return f"SDNProps({self.__dict__})"


class StatsProps:
    """
    Main properties used for statistics collection and tracking.

    This class aggregates all simulation statistics, including blocking
    probabilities, resource utilization, and performance metrics.
    """

    def __init__(self) -> None:
        """Initialize statistics properties with empty collections."""
        # Snapshot tracking
        self.snapshots_dict: dict[int, dict[str, Any]] = {}

        # Resource usage tracking
        self.cores_dict: dict[int, int] = {}
        self.weights_dict: dict[str, Any] = {}
        self.modulations_used_dict: dict[str, Any] = {}
        self.bandwidth_blocking_dict: dict[str | float, int] = {}
        self.link_usage_dict: dict[str, dict[str, Any]] = {}

        # Blocking reasons
        self.block_reasons_dict: dict[str, int | float | None] = {
            "distance": None,
            "congestion": None,
            "xt_threshold": None,
            "failure": None,
        }

        # Per-simulation metrics
        self.simulation_blocking_list: list[float] = []
        self.simulation_bitrate_blocking_list: list[float] = []
        self.transponders_list: list[int | float] = []
        self.hops_list: list[float] = []
        self.lengths_list: list[float] = []
        self.route_times_list: list[float] = []
        self.crosstalk_list: list[float] = []

        # Resource allocation tracking
        self.bands_list: list[int] = []
        self.start_slot_list: list[int] = []
        self.end_slot_list: list[int] = []
        self.modulation_list: list[str] = []
        self.bandwidth_list: list[float] = []
        self.path_index_list: list[int] = []

        # Protection and survivability metrics
        self.protection_switchovers: int = 0  # Number of successful switchovers
        self.protection_failures: int = 0  # Requests where both paths failed
        self.failure_induced_blocks: int = 0  # Total requests dropped due to failures
        self.switchover_times: list[float] = []  # Times when switchovers occurred

    def __repr__(self) -> str:
        """
        Return string representation of StatsProps.

        :return: String representation with all properties
        :rtype: str
        """
        return f"StatsProps({self.__dict__})"
