"""
SimulationConfig - Immutable simulation configuration.

This module defines the SimulationConfig frozen dataclass that replaces
the legacy engine_props dictionary with a typed, immutable structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# =============================================================================
# Physical Constants (from properties.py)
# =============================================================================
DEFAULT_PLANCK_CONSTANT = 6.62607004e-34  # Planck's constant in J*s
DEFAULT_LIGHT_FREQUENCY = 1.9341e14  # Center light frequency in Hz
DEFAULT_INPUT_POWER = 1e-3  # Default input power in Watts
DEFAULT_FREQUENCY_SPACING = 12.5e9  # Default frequency spacing in Hz
DEFAULT_SPAN_LENGTH = 100.0  # Default span length in km
DEFAULT_MCI_WORST = 6.3349755556585961e-27  # Worst-case mutual coupling interference

# Default noise spontaneous parameters per band
DEFAULT_NSP_PER_BAND: dict[str, float] = {
    "c": 1.77,  # C-band EDFA noise figure
    "l": 1.99,  # L-band EDFA noise figure
    "s": 2.0,   # S-band amplifier noise figure
    "o": 2.0,   # O-band amplifier noise figure
    "e": 2.0,   # E-band amplifier noise figure
}

# Default modulation format mappings
DEFAULT_MOD_FORMAT_MAP: dict[int, str] = {
    6: "64-QAM",
    5: "32-QAM",
    4: "16-QAM",
    3: "8-QAM",
    2: "QPSK",
    1: "BPSK",
}

DEFAULT_BANDWIDTH_MAP: dict[str, int] = {
    "64-QAM": 600,
    "32-QAM": 500,
    "16-QAM": 400,
    "8-QAM": 300,
    "QPSK": 200,
    "BPSK": 100,
}


@dataclass(frozen=True)
class SimulationConfig:
    """
    Immutable simulation configuration.

    This dataclass captures all simulation parameters in a typed,
    frozen structure. Once created, the configuration cannot be modified.

    Attributes:
        Network Configuration:
            network_name: Network topology identifier (e.g., "USbackbone60")
            cores_per_link: Number of cores per fiber link (MCF support)
            band_list: Available frequency bands as immutable tuple
            band_slots: Slot count per band (e.g., {"c": 320, "l": 320})
            guard_slots: Guard band slots between allocations

        Topology Constraints:
            span_length: Default span length in km
            max_link_length: Maximum link length constraint (None = no limit)
            max_span: Maximum spans per link (None = no limit)
            max_transponders: Maximum transponders per node (None = no limit)
            single_core: Force single core allocation

        Traffic Configuration:
            num_requests: Total requests to simulate
            erlang: Traffic intensity (arrival_rate * holding_time)
            holding_time: Mean request duration

        Routing Configuration:
            route_method: Routing algorithm name
            k_paths: Number of candidate paths to compute
            allocation_method: Spectrum allocation strategy

        Feature Flags:
            grooming_enabled: Enable traffic grooming
            grooming_type: Grooming algorithm type (None if disabled)
            slicing_enabled: Enable lightpath slicing
            max_slices: Maximum slices per request
            snr_enabled: Enable SNR validation
            snr_type: SNR calculation method or None
            snr_recheck: Re-validate SNR after allocation
            can_partially_serve: Allow partial bandwidth fulfillment

        Protection Configuration:
            protection_switchover_ms: Time to switch to backup path (ms)
            restoration_latency_ms: Time to restore after failure (ms)

        Physical Layer Parameters:
            input_power: Optical input power in Watts
            frequency_spacing: Channel spacing in Hz
            light_frequency: Center light frequency in Hz
            planck_constant: Planck's constant in J*s
            noise_spectral_density: Noise spectral density
            mci_worst: Worst-case mutual coupling interference
            nsp_per_band: Noise spontaneous parameter per band

        SNR Configuration:
            request_bit_rate: Default request bit rate in Gb/s
            request_snr: Default requested SNR in dB
            snr_thresholds: SNR thresholds per modulation format

        Modulation Configuration:
            modulation_formats: Available modulation format definitions
            mod_per_bw: Modulation formats available per bandwidth
            mod_format_map: Mapping of format ID to format name
            bandwidth_map: Mapping of format name to bandwidth capacity

    Example:
        >>> config = SimulationConfig.from_engine_props(engine_props)
        >>> config.k_paths
        3
        >>> config.grooming_enabled
        True
        >>> legacy = config.to_engine_props()
    """

    # =========================================================================
    # Network Configuration (Required)
    # =========================================================================
    network_name: str
    cores_per_link: int
    band_list: tuple[str, ...]  # Immutable tuple: ("c",) or ("c", "l")
    band_slots: dict[str, int]  # {"c": 320, "l": 320, "s": 320}
    guard_slots: int

    # =========================================================================
    # Traffic Configuration (Required)
    # =========================================================================
    num_requests: int
    erlang: float
    holding_time: float

    # =========================================================================
    # Routing Configuration (Required)
    # =========================================================================
    route_method: str  # "k_shortest_path", "1plus1_protection", etc.
    k_paths: int
    allocation_method: str  # "first_fit", "best_fit", "last_fit"

    # =========================================================================
    # Topology Constraints (from RoutingProps, SDNProps)
    # =========================================================================
    span_length: float = DEFAULT_SPAN_LENGTH
    max_link_length: float | None = None
    max_span: int | None = None
    max_transponders: int | None = None
    single_core: bool = False

    # =========================================================================
    # Topology Info (physical topology from create_pt)
    # Contains link/node info needed by SNR calculations
    # =========================================================================
    topology_info: dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # Feature Flags
    # =========================================================================
    grooming_enabled: bool = False
    grooming_type: str | None = None  # From GroomingProps
    slicing_enabled: bool = False
    max_slices: int = 1
    snr_enabled: bool = False
    snr_type: str | None = None  # "snr_e2e", "snr_segment", or None
    snr_recheck: bool = False
    can_partially_serve: bool = False
    fixed_grid: bool = True  # True for fixed grid, False for flexi-grid
    spectrum_priority: str | None = None  # Band selection priority: "BSC", "CSB", or None
    multi_fiber: bool = False  # True for multi-fiber (MF), False for multi-core fiber (MCF)
    dynamic_lps: bool = False  # True for dynamic lightpath slicing mode

    # =========================================================================
    # Protection Configuration (from SDNProps)
    # =========================================================================
    protection_switchover_ms: float = 50.0
    restoration_latency_ms: float = 100.0

    # =========================================================================
    # Physical Layer Parameters (from RoutingProps, SNRProps)
    # =========================================================================
    bw_per_slot: float = 12.5  # Bandwidth per slot in GHz
    input_power: float = DEFAULT_INPUT_POWER
    frequency_spacing: float = DEFAULT_FREQUENCY_SPACING
    light_frequency: float = DEFAULT_LIGHT_FREQUENCY
    planck_constant: float = DEFAULT_PLANCK_CONSTANT
    noise_spectral_density: float = 1.8
    mci_worst: float = DEFAULT_MCI_WORST
    nsp_per_band: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_NSP_PER_BAND)
    )

    # =========================================================================
    # SNR Configuration (from SNRProps)
    # =========================================================================
    request_bit_rate: float = 12.5  # Gb/s
    request_snr: float = 8.5  # dB
    snr_thresholds: dict[str, float] = field(default_factory=dict)
    phi: dict[str, float] = field(default_factory=dict)  # SNR phi per modulation
    egn_model: bool = False  # Use EGN model for SNR
    xt_type: str | None = None  # Crosstalk type
    beta: float = 0.5  # SNR beta parameter
    theta: float = 0.0  # SNR theta parameter
    bi_directional: bool = True  # Bi-directional SNR calculation
    xt_noise: bool = False  # Include crosstalk noise
    requested_xt: dict[str, float] = field(default_factory=dict)  # Requested crosstalk

    # =========================================================================
    # Modulation Configuration
    # =========================================================================
    modulation_formats: dict[str, Any] = field(default_factory=dict)
    mod_per_bw: dict[str, Any] = field(default_factory=dict)
    mod_format_map: dict[int, str] = field(
        default_factory=lambda: dict(DEFAULT_MOD_FORMAT_MAP)
    )
    bandwidth_map: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_BANDWIDTH_MAP)
    )

    # =========================================================================
    # Validation
    # =========================================================================
    def __post_init__(self) -> None:
        """Validate configuration after creation."""
        # Validate network config
        if self.cores_per_link < 1:
            raise ValueError("cores_per_link must be >= 1")
        if not self.band_list:
            raise ValueError("band_list cannot be empty")
        if self.guard_slots < 0:
            raise ValueError("guard_slots must be >= 0")

        # Validate traffic config
        if self.num_requests < 1:
            raise ValueError("num_requests must be >= 1")
        if self.erlang <= 0:
            raise ValueError("erlang must be > 0")
        if self.holding_time <= 0:
            raise ValueError("holding_time must be > 0")

        # Validate routing config
        if self.k_paths < 1:
            raise ValueError("k_paths must be >= 1")

        # Validate feature flags
        if self.max_slices < 1:
            raise ValueError("max_slices must be >= 1")

        # Validate band_slots matches band_list
        for band in self.band_list:
            if band not in self.band_slots:
                raise ValueError(f"band_slots missing entry for band '{band}'")

        # Validate physical layer parameters
        if self.input_power <= 0:
            raise ValueError("input_power must be > 0")
        if self.span_length <= 0:
            raise ValueError("span_length must be > 0")

    # =========================================================================
    # Computed Properties
    # =========================================================================
    @property
    def total_slots(self) -> int:
        """Total spectrum slots across all bands."""
        return sum(self.band_slots.values())

    @property
    def arrival_rate(self) -> float:
        """Computed arrival rate from erlang and holding_time."""
        return self.erlang / self.holding_time

    @property
    def is_multiband(self) -> bool:
        """True if using multiple frequency bands."""
        return len(self.band_list) > 1

    @property
    def is_multicore(self) -> bool:
        """True if using multiple cores per fiber."""
        return self.cores_per_link > 1

    @property
    def protection_enabled(self) -> bool:
        """True if protection routing is enabled."""
        return self.route_method == "1plus1_protection"

    # =========================================================================
    # Legacy Adapters
    # =========================================================================
    @classmethod
    def from_engine_props(cls, engine_props: dict[str, Any]) -> SimulationConfig:
        """
        Create SimulationConfig from legacy engine_props dictionary.

        Args:
            engine_props: Legacy configuration dictionary

        Returns:
            New SimulationConfig instance

        Raises:
            KeyError: If required fields are missing
            ValueError: If field values are invalid
        """
        # Extract band list (convert list to tuple for immutability)
        band_list_raw = engine_props.get("band_list", ["c"])
        if isinstance(band_list_raw, list):
            band_list = tuple(band_list_raw)
        else:
            band_list = (band_list_raw,)

        # Build band_slots dict from individual band slot counts
        band_slots: dict[str, int] = {}
        band_key_map = {
            "c": "c_band",
            "l": "l_band",
            "s": "s_band",
            "o": "o_band",
            "e": "e_band",
        }
        for band in band_list:
            key = band_key_map.get(band, f"{band}_band")
            band_slots[band] = engine_props.get(key, 320)  # Default 320 slots

        # Compute erlang from arrival_rate and holding_time
        arrival_rate = engine_props.get("arrival_rate", 1.0)
        holding_time = engine_props.get("holding_time", 1.0)
        erlang = arrival_rate * holding_time

        # Extract SNR configuration
        snr_type = engine_props.get("snr_type")
        snr_enabled = snr_type is not None and snr_type != ""

        # Extract modulation configuration
        mod_per_bw = engine_props.get("mod_per_bw", {})
        modulation_formats = engine_props.get("modulation_formats", {})
        snr_thresholds = engine_props.get("snr_thresholds", {})
        mod_format_map = engine_props.get(
            "modulation_format_mapping_dict", dict(DEFAULT_MOD_FORMAT_MAP)
        )
        bandwidth_map = engine_props.get(
            "bandwidth_mapping_dict", dict(DEFAULT_BANDWIDTH_MAP)
        )

        # Extract physical layer parameters
        nsp_per_band = engine_props.get("nsp", dict(DEFAULT_NSP_PER_BAND))

        return cls(
            # Network
            network_name=engine_props["network"],
            cores_per_link=engine_props.get("cores_per_link", 1),
            band_list=band_list,
            band_slots=band_slots,
            guard_slots=engine_props.get("guard_slots", 1),
            # Topology constraints
            span_length=engine_props.get("span_length", DEFAULT_SPAN_LENGTH),
            max_link_length=engine_props.get("max_link_length"),
            max_span=engine_props.get("max_span"),
            max_transponders=engine_props.get("number_of_transponders"),
            single_core=engine_props.get("single_core", False),
            # Topology info (physical topology for SNR calculations)
            topology_info=engine_props.get("topology_info", {}),
            # Traffic
            num_requests=engine_props.get("num_requests", 1000),
            erlang=erlang,
            holding_time=holding_time,
            # Routing
            route_method=engine_props.get("route_method", "k_shortest_path"),
            k_paths=engine_props.get("k_paths", 3),
            allocation_method=engine_props.get("allocation_method", "first_fit"),
            # Features
            grooming_enabled=engine_props.get("is_grooming_enabled", False),
            grooming_type=engine_props.get("grooming_type"),
            slicing_enabled=engine_props.get("max_segments", 1) > 1,
            max_slices=engine_props.get("max_segments", 1),
            snr_enabled=snr_enabled,
            snr_type=snr_type if snr_enabled else None,
            snr_recheck=engine_props.get("snr_recheck", False),
            can_partially_serve=engine_props.get("can_partially_serve", False),
            fixed_grid=engine_props.get("fixed_grid", True),
            spectrum_priority=engine_props.get("spectrum_priority"),
            multi_fiber=engine_props.get("multi_fiber", False),
            dynamic_lps=engine_props.get("dynamic_lps", False),
            # Protection
            protection_switchover_ms=engine_props.get(
                "protection_switchover_ms", 50.0
            ),
            restoration_latency_ms=engine_props.get("restoration_latency_ms", 100.0),
            # Physical layer
            bw_per_slot=engine_props.get("bw_per_slot", 12.5),
            input_power=engine_props.get("input_power", DEFAULT_INPUT_POWER),
            frequency_spacing=engine_props.get(
                "frequency_spacing", DEFAULT_FREQUENCY_SPACING
            ),
            light_frequency=engine_props.get(
                "light_frequency", DEFAULT_LIGHT_FREQUENCY
            ),
            planck_constant=engine_props.get(
                "planck_constant", DEFAULT_PLANCK_CONSTANT
            ),
            noise_spectral_density=engine_props.get("noise_spectral_density", 1.8),
            mci_worst=engine_props.get("mci_worst", DEFAULT_MCI_WORST),
            nsp_per_band=nsp_per_band,
            # SNR
            request_bit_rate=engine_props.get("request_bit_rate", 12.5),
            request_snr=engine_props.get("request_snr", 8.5),
            snr_thresholds=snr_thresholds,
            phi=engine_props.get("phi", {}),
            egn_model=engine_props.get("egn_model", False),
            xt_type=engine_props.get("xt_type"),
            beta=engine_props.get("beta", 0.5),
            theta=engine_props.get("theta", 0.0),
            bi_directional=engine_props.get("bi_directional", True),
            xt_noise=engine_props.get("xt_noise", False),
            requested_xt=engine_props.get("requested_xt", {}),
            # Modulation
            modulation_formats=modulation_formats,
            mod_per_bw=mod_per_bw,
            mod_format_map=mod_format_map,
            bandwidth_map=bandwidth_map,
        )

    def to_engine_props(self) -> dict[str, Any]:
        """
        Convert to legacy engine_props dictionary format.

        Returns:
            Dictionary compatible with legacy engine_props consumers
        """
        props: dict[str, Any] = {
            # Network
            "network": self.network_name,
            "cores_per_link": self.cores_per_link,
            "band_list": list(self.band_list),  # Convert tuple back to list
            "guard_slots": self.guard_slots,
            # Topology constraints
            "span_length": self.span_length,
            "max_link_length": self.max_link_length,
            "max_span": self.max_span,
            "number_of_transponders": self.max_transponders,
            "single_core": self.single_core,
            # Topology info (physical topology for SNR calculations)
            "topology_info": self.topology_info,
            # Traffic
            "num_requests": self.num_requests,
            "arrival_rate": self.arrival_rate,  # Computed property
            "holding_time": self.holding_time,
            # Routing
            "route_method": self.route_method,
            "k_paths": self.k_paths,
            "allocation_method": self.allocation_method,
            # Features
            "is_grooming_enabled": self.grooming_enabled,
            "grooming_type": self.grooming_type,
            "max_segments": self.max_slices,
            "snr_type": self.snr_type,
            "snr_recheck": self.snr_recheck,
            "can_partially_serve": self.can_partially_serve,
            "fixed_grid": self.fixed_grid,
            "spectrum_priority": self.spectrum_priority,
            "multi_fiber": self.multi_fiber,
            "dynamic_lps": self.dynamic_lps,
            # Protection
            "protection_switchover_ms": self.protection_switchover_ms,
            "restoration_latency_ms": self.restoration_latency_ms,
            # Physical layer
            "bw_per_slot": self.bw_per_slot,
            "input_power": self.input_power,
            "frequency_spacing": self.frequency_spacing,
            "light_frequency": self.light_frequency,
            "planck_constant": self.planck_constant,
            "noise_spectral_density": self.noise_spectral_density,
            "mci_worst": self.mci_worst,
            "nsp": self.nsp_per_band,
            # SNR
            "request_bit_rate": self.request_bit_rate,
            "request_snr": self.request_snr,
            "snr_thresholds": self.snr_thresholds,
            "phi": self.phi,
            "egn_model": self.egn_model,
            "xt_type": self.xt_type,
            "beta": self.beta,
            "theta": self.theta,
            "bi_directional": self.bi_directional,
            "xt_noise": self.xt_noise,
            "requested_xt": self.requested_xt,
            # Modulation
            "modulation_formats": self.modulation_formats,
            "mod_per_bw": self.mod_per_bw,
            "modulation_format_mapping_dict": self.mod_format_map,
            "bandwidth_mapping_dict": self.bandwidth_map,
        }

        # Add individual band slot counts
        band_key_map = {
            "c": "c_band",
            "l": "l_band",
            "s": "s_band",
            "o": "o_band",
            "e": "e_band",
        }
        for band, slots in self.band_slots.items():
            key = band_key_map.get(band, f"{band}_band")
            props[key] = slots

        return props
