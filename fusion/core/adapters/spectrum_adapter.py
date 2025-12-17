"""
SpectrumAdapter - Adapts legacy SpectrumAssignment to SpectrumPipeline protocol.

ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
It will be replaced with a clean implementation in Phase 4.

Phase: P2.4 - Legacy Adapters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fusion.interfaces.pipelines import SpectrumPipeline

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.network_state import NetworkState
    from fusion.domain.results import SpectrumResult

logger = logging.getLogger(__name__)


@dataclass
class SDNPropsProxyForSpectrum:
    """
    Minimal proxy for SDNProps to satisfy legacy SpectrumAssignment.

    Only implements attributes that SpectrumAssignment actually reads.
    """

    topology: Any
    source: str = ""
    destination: str = ""
    bandwidth: float = 0.0
    request_id: int = 0
    path_index: int = 0
    modulation_formats_dict: dict[str, Any] = field(default_factory=dict)
    network_spectrum_dict: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    lightpath_status_dict: dict[tuple[str, str], dict[int, dict[str, Any]]] = field(
        default_factory=dict
    )
    block_reason: str | None = None

    # Grooming fields
    was_partially_groomed: bool = False
    remaining_bw: float | None = None
    bandwidth_list: list[float] = field(default_factory=list)

    # Path/lightpath tracking fields
    path_list: list[str] | None = None
    path_weight: float = 0.0
    arrive: float = 0.0  # Arrival time for lightpath tracking

    # 1+1 Protection fields
    backup_path: list[str] | None = None

    @classmethod
    def from_network_state(
        cls,
        network_state: NetworkState,
        source: str,
        destination: str,
        bandwidth: float,
        modulation_formats: dict[str, Any] | None = None,
        backup_path: list[str] | None = None,
        path_index: int = 0,
    ) -> SDNPropsProxyForSpectrum:
        """Create proxy from NetworkState with request context."""
        return cls(
            topology=network_state.topology,
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            request_id=0,
            path_index=path_index,
            modulation_formats_dict=modulation_formats or {},
            network_spectrum_dict=network_state.network_spectrum_dict,
            lightpath_status_dict=network_state.lightpath_status_dict,
            backup_path=backup_path,
        )


@dataclass
class RoutePropsProxy:
    """
    Minimal proxy for RoutingProps to satisfy legacy SpectrumAssignment.
    """

    paths_matrix: list[list[str]] = field(default_factory=list)
    modulation_formats_matrix: list[list[str]] = field(default_factory=list)
    weights_list: list[float] = field(default_factory=list)
    path_index_list: list[int] = field(default_factory=list)
    backup_paths_matrix: list[list[str] | None] = field(default_factory=list)
    connection_index: int | None = None


class SpectrumAdapter(SpectrumPipeline):
    """
    Adapts legacy SpectrumAssignment to SpectrumPipeline protocol.

    ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
    It will be replaced with a clean implementation in Phase 4.

    The adapter:
    1. Creates proxy objects from NetworkState
    2. Calls legacy SpectrumAssignment.get_spectrum()
    3. Converts spectrum_props to SpectrumResult

    Removal Checklist:
    [ ] Clean SpectrumPipeline implementation exists
    [ ] All callers migrated to clean implementation
    [ ] run_comparison.py passes without this adapter
    [ ] grep 'SpectrumAdapter' returns only this definition

    Example:
        >>> adapter = SpectrumAdapter(config)
        >>> result = adapter.find_spectrum(path, modulation, 100, network_state)
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize adapter with configuration.

        Args:
            config: SimulationConfig for creating legacy engine_props
        """
        self._config = config
        self._engine_props = config.to_engine_props()

    def find_spectrum(
        self,
        path: list[str],
        modulation: str | list[str],
        bandwidth_gbps: int,
        network_state: NetworkState,
        *,
        connection_index: int | None = None,
        path_index: int = 0,
        use_dynamic_slicing: bool = False,
        snr_bandwidth: int | None = None,
    ) -> SpectrumResult:
        """
        Find available spectrum along a path.

        Args:
            path: Ordered list of node IDs forming the route
            modulation: Modulation format name (e.g., "QPSK", "16-QAM")
            bandwidth_gbps: Required bandwidth in Gbps
            network_state: Current network state
            connection_index: External routing index for pre-calculated SNR lookup
            path_index: Index of which k-path is being tried (0, 1, 2...)
            snr_bandwidth: Bandwidth to use for SNR checks (if different from bandwidth_gbps).
                Used for partial grooming where slots are for remaining_bw but SNR check
                should use original request bandwidth.

        Returns:
            SpectrumResult with slot allocation or is_free=False on failure
        """
        from fusion.domain.results import SpectrumResult

        # Validate inputs
        if not path or len(path) < 2:
            slots_needed = self._calculate_slots_needed(modulation, bandwidth_gbps)
            return SpectrumResult.not_found(slots_needed)

        try:
            # Get source and destination from path
            source = str(path[0])
            destination = str(path[-1])

            # Look up modulation formats from mod_per_bw for this bandwidth
            # mod_per_bw structure: {bandwidth_str: {modulation: {slots_needed: X, ...}}}
            # IMPORTANT: Use snr_bandwidth (original request bandwidth) for lookup if provided.
            # For partial grooming, remaining_bw is passed as bandwidth_gbps, but modulation
            # formats should be looked up using the ORIGINAL request bandwidth (like legacy).
            mod_per_bw = self._config.mod_per_bw
            lookup_bw = snr_bandwidth if snr_bandwidth is not None else bandwidth_gbps
            bw_key = str(lookup_bw)
            modulation_formats = mod_per_bw.get(bw_key, {})

            # Create proxies
            # Use snr_bandwidth for SNR checks if provided (partial grooming case),
            # otherwise use bandwidth_gbps
            proxy_bandwidth = snr_bandwidth if snr_bandwidth is not None else bandwidth_gbps
            sdn_props = SDNPropsProxyForSpectrum.from_network_state(
                network_state=network_state,
                source=source,
                destination=destination,
                bandwidth=float(proxy_bandwidth),
                modulation_formats=modulation_formats,
                path_index=path_index,
            )

            # Handle both single modulation and list of modulations
            mod_list = [modulation] if isinstance(modulation, str) else list(modulation)
            route_props = RoutePropsProxy(
                paths_matrix=[list(path)],
                modulation_formats_matrix=[mod_list],
                weights_list=[0.0],  # Weight not needed for spectrum search
                connection_index=connection_index,
            )

            # Make engine_props copy with updated topology
            engine_props = dict(self._engine_props)
            engine_props["topology"] = network_state.topology

            # Instantiate legacy SpectrumAssignment
            from fusion.core.spectrum_assignment import SpectrumAssignment

            legacy_spectrum = SpectrumAssignment(
                engine_props=engine_props,
                sdn_props=sdn_props,
                route_props=route_props,
            )

            # Set path in spectrum_props
            legacy_spectrum.spectrum_props.path_list = list(path)

            # NOTE: Don't pre-set lightpath_bandwidth here - get_spectrum may clear it.
            # We set it AFTER get_spectrum as a fallback (matching legacy sdn_controller behavior).

            # DEBUG: Show spectrum state when find_spectrum is called during slicing
            # Check if this might be request 5 (we don't have request_id here, but can check network state)
            if use_dynamic_slicing and len(path) >= 2:
                link = (path[0], path[1])
                if link in network_state.network_spectrum_dict:
                    cores_matrix = network_state.network_spectrum_dict[link]["cores_matrix"]
                    for bnd, bnd_cores in cores_matrix.items():
                        if len(bnd_cores) > 0:
                            slots_preview = list(bnd_cores[0][:10])
                            occupied = [i for i, v in enumerate(slots_preview) if v != 0]
                            if occupied:
                                print(f"[V5_ADAPTER_SPECTRUM] dynamic_slice link={link} band={bnd} core=0 slots[0:10]={slots_preview} occupied={occupied}")
                            break  # Just show first band

            # Choose method based on whether we're in slicing mode
            # Only use get_spectrum_dynamic_slicing when explicitly in slicing stage
            if use_dynamic_slicing and self._config.dynamic_lps:
                # Dynamic slicing mode: spectrum determines modulation/bandwidth
                result_mod, result_bw = legacy_spectrum.get_spectrum_dynamic_slicing(
                    _mod_format_list=[modulation] if modulation else [],
                    path_index=path_index,
                )
                # Update modulation from dynamic result
                if result_mod and result_mod is not False:
                    modulation = str(result_mod)
                # Store achieved bandwidth in spectrum_props for later retrieval
                if result_bw and result_bw is not False:
                    legacy_spectrum.spectrum_props.lightpath_bandwidth = int(result_bw)
            else:
                # Standard mode: modulation/bandwidth specified upfront
                # Support both single modulation and list of modulations (like legacy)
                mod_list = [modulation] if isinstance(modulation, str) else list(modulation)
                legacy_spectrum.get_spectrum(
                    mod_format_list=mod_list,
                )

            # LEGACY fallback: If lightpath_bandwidth not set by get_spectrum (e.g., fixed grid),
            # use proxy_bandwidth (original request bw for partial grooming, or bandwidth_gbps).
            # This matches sdn_controller.py lines 1110-1113 fallback behavior.
            if legacy_spectrum.spectrum_props.lightpath_bandwidth is None:
                legacy_spectrum.spectrum_props.lightpath_bandwidth = proxy_bandwidth

            # Convert results
            return self._convert_spectrum_props(
                legacy_spectrum.spectrum_props,
                sdn_props,
                modulation,
            )

        except Exception as e:
            import traceback
            print(f"[SPECTRUM-ERROR] find_spectrum exception: {e}")
            traceback.print_exc()
            logger.warning("SpectrumAdapter.find_spectrum failed: %s", e)
            slots_needed = self._calculate_slots_needed(modulation, bandwidth_gbps)
            return SpectrumResult.not_found(slots_needed)

    def find_protected_spectrum(
        self,
        primary_path: list[str],
        backup_path: list[str],
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """
        Find spectrum for both primary and backup paths (1+1 protection).

        Args:
            primary_path: Primary route node sequence
            backup_path: Backup route node sequence (should be disjoint)
            modulation: Modulation format name
            bandwidth_gbps: Required bandwidth
            network_state: Current network state

        Returns:
            SpectrumResult with both primary and backup allocations
        """
        from fusion.domain.results import SpectrumResult

        # Validate inputs
        if not primary_path or len(primary_path) < 2:
            slots_needed = self._calculate_slots_needed(modulation, bandwidth_gbps)
            return SpectrumResult.not_found(slots_needed)

        if not backup_path or len(backup_path) < 2:
            slots_needed = self._calculate_slots_needed(modulation, bandwidth_gbps)
            return SpectrumResult.not_found(slots_needed)

        try:
            source = str(primary_path[0])
            destination = str(primary_path[-1])

            # Look up modulation formats from mod_per_bw for this bandwidth
            mod_per_bw = self._config.mod_per_bw
            bw_key = str(bandwidth_gbps)
            modulation_formats = mod_per_bw.get(bw_key, {})

            # Create proxies with backup path
            sdn_props = SDNPropsProxyForSpectrum.from_network_state(
                network_state=network_state,
                source=source,
                destination=destination,
                bandwidth=float(bandwidth_gbps),
                modulation_formats=modulation_formats,
                backup_path=list(backup_path),
            )

            route_props = RoutePropsProxy(
                paths_matrix=[list(primary_path)],
                modulation_formats_matrix=[[modulation]],
                weights_list=[0.0],
                backup_paths_matrix=[list(backup_path)],
            )

            # Make engine_props copy
            engine_props = dict(self._engine_props)
            engine_props["topology"] = network_state.topology

            # Instantiate legacy SpectrumAssignment
            from fusion.core.spectrum_assignment import SpectrumAssignment

            legacy_spectrum = SpectrumAssignment(
                engine_props=engine_props,
                sdn_props=sdn_props,
                route_props=route_props,
            )

            # Set paths in spectrum_props
            legacy_spectrum.spectrum_props.path_list = list(primary_path)
            legacy_spectrum.spectrum_props.backup_path = list(backup_path)

            # Call legacy get_spectrum with backup modulation
            legacy_spectrum.get_spectrum(
                mod_format_list=[modulation],
                backup_mod_format_list=[modulation],
            )

            # Convert results
            return self._convert_spectrum_props(
                legacy_spectrum.spectrum_props,
                sdn_props,
                modulation,
            )

        except Exception as e:
            import traceback
            print(f"[SPECTRUM-ERROR] find_protected_spectrum exception: {e}")
            traceback.print_exc()
            logger.warning("SpectrumAdapter.find_protected_spectrum failed: %s", e)
            slots_needed = self._calculate_slots_needed(modulation, bandwidth_gbps)
            return SpectrumResult.not_found(slots_needed)

    def _calculate_slots_needed(
        self,
        modulation: str | list[str],
        bandwidth_gbps: float,
    ) -> int:
        """Calculate number of spectrum slots needed."""
        # Handle list of modulations - use first one
        mod_str = modulation[0] if isinstance(modulation, list) else modulation
        # Get modulation info from config
        mod_info = self._config.modulation_formats.get(mod_str, {})

        if isinstance(mod_info, dict):
            bits_per_symbol = mod_info.get("bits_per_symbol", 2)  # Default QPSK
        else:
            bits_per_symbol = 2

        # Calculate based on bandwidth per slot (typically 12.5 GHz)
        bw_per_slot = self._engine_props.get("bw_per_slot", 12.5)

        # Slots = bandwidth / (bits_per_symbol * bw_per_slot)
        slots = int((bandwidth_gbps / (bits_per_symbol * bw_per_slot)) + 0.5)

        # Add guard slots
        guard_slots = self._config.guard_slots
        return max(1, slots) + guard_slots

    def _convert_spectrum_props(
        self,
        spectrum_props: Any,
        sdn_props: SDNPropsProxyForSpectrum,
        modulation: str,
    ) -> SpectrumResult:
        """Convert legacy spectrum_props to SpectrumResult.

        Important: Legacy end_slot semantics depend on guard_slots:
        - end_index = start + slots_needed + guard_slots - 1 (last slot, inclusive)
        - end_slot = end_index + guard_slots

        When guard_slots > 0: end_slot is effectively exclusive (end_index + guard > end_index)
        When guard_slots == 0: end_slot is inclusive (end_index + 0 = end_index)

        New domain objects use EXCLUSIVE end_slot (Python slice notation).
        """
        from fusion.domain.results import SpectrumResult

        # Calculate slots needed for empty result
        slots_needed = spectrum_props.slots_needed or 0

        # Check if allocation succeeded
        if not spectrum_props.is_free:
            return SpectrumResult.not_found(slots_needed)

        # Extract primary allocation
        start_slot = spectrum_props.start_slot
        legacy_end_slot = spectrum_props.end_slot
        core = spectrum_props.core_number
        band = spectrum_props.current_band
        mod = spectrum_props.modulation or modulation

        if start_slot is None or legacy_end_slot is None:
            return SpectrumResult.not_found(slots_needed)

        # Get guard_slots to determine end_slot semantics
        guard_slots = self._config.guard_slots

        # Legacy end_slot conversion:
        # - When guard_slots > 0: end_slot = end_index + guard_slots (already exclusive)
        # - When guard_slots == 0: end_slot = end_index (inclusive, need +1)
        if guard_slots == 0:
            # Inclusive end_slot - convert to exclusive
            end_slot = legacy_end_slot + 1
        else:
            # Already exclusive - no conversion needed
            end_slot = legacy_end_slot

        # Handle backup allocation (if 1+1 protection)
        backup_start = getattr(spectrum_props, "backup_start_slot", None)
        legacy_backup_end = getattr(spectrum_props, "backup_end_slot", None)
        backup_core = getattr(spectrum_props, "backup_core_number", None)
        backup_band = getattr(spectrum_props, "backup_band", None)

        # Backup end_slot uses same convention as primary
        if legacy_backup_end is not None:
            if guard_slots == 0:
                backup_end = legacy_backup_end + 1
            else:
                backup_end = legacy_backup_end
        else:
            backup_end = None

        # Get achieved bandwidth from dynamic slicing (may be less than requested)
        achieved_bw = getattr(spectrum_props, "lightpath_bandwidth", None)
        achieved_bandwidth_gbps = int(achieved_bw) if achieved_bw is not None else None

        # Get SNR value from spectrum assignment (crosstalk_cost)
        # This is calculated during spectrum assignment, not after lightpath creation
        snr_db = getattr(spectrum_props, "crosstalk_cost", None)
        if snr_db is not None:
            snr_db = float(snr_db)

        return SpectrumResult(
            is_free=True,
            start_slot=start_slot,
            end_slot=end_slot,
            core=core if core is not None else 0,
            band=band if band is not None else "c",
            modulation=mod,
            slots_needed=slots_needed if slots_needed > 0 else (end_slot - start_slot),
            achieved_bandwidth_gbps=achieved_bandwidth_gbps,
            snr_db=snr_db,
            # Backup fields
            backup_start_slot=backup_start,
            backup_end_slot=backup_end,
            backup_core=backup_core,
            backup_band=backup_band,
        )
