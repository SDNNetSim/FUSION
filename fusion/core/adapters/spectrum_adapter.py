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
    ) -> SDNPropsProxyForSpectrum:
        """Create proxy from NetworkState with request context."""
        return cls(
            topology=network_state.topology,
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            request_id=0,
            path_index=0,
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
        modulation: str,
        bandwidth_gbps: int,
        network_state: NetworkState,
    ) -> SpectrumResult:
        """
        Find available spectrum along a path.

        Args:
            path: Ordered list of node IDs forming the route
            modulation: Modulation format name (e.g., "QPSK", "16-QAM")
            bandwidth_gbps: Required bandwidth in Gbps
            network_state: Current network state

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

            # Create proxies
            sdn_props = SDNPropsProxyForSpectrum.from_network_state(
                network_state=network_state,
                source=source,
                destination=destination,
                bandwidth=float(bandwidth_gbps),
                modulation_formats=self._config.modulation_formats,
            )

            route_props = RoutePropsProxy(
                paths_matrix=[list(path)],
                modulation_formats_matrix=[[modulation]],
                weights_list=[0.0],  # Weight not needed for spectrum search
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

            # Call legacy get_spectrum
            legacy_spectrum.get_spectrum(
                mod_format_list=[modulation],
            )

            # Convert results
            return self._convert_spectrum_props(
                legacy_spectrum.spectrum_props,
                sdn_props,
                modulation,
            )

        except Exception as e:
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

            # Create proxies with backup path
            sdn_props = SDNPropsProxyForSpectrum.from_network_state(
                network_state=network_state,
                source=source,
                destination=destination,
                bandwidth=float(bandwidth_gbps),
                modulation_formats=self._config.modulation_formats,
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
            logger.warning("SpectrumAdapter.find_protected_spectrum failed: %s", e)
            slots_needed = self._calculate_slots_needed(modulation, bandwidth_gbps)
            return SpectrumResult.not_found(slots_needed)

    def _calculate_slots_needed(
        self,
        modulation: str,
        bandwidth_gbps: float,
    ) -> int:
        """Calculate number of spectrum slots needed."""
        # Get modulation info from config
        mod_info = self._config.modulation_formats.get(modulation, {})

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
        """Convert legacy spectrum_props to SpectrumResult."""
        from fusion.domain.results import SpectrumResult

        # Calculate slots needed for empty result
        slots_needed = spectrum_props.slots_needed or 0

        # Check if allocation succeeded
        if not spectrum_props.is_free:
            return SpectrumResult.not_found(slots_needed)

        # Extract primary allocation
        start_slot = spectrum_props.start_slot
        end_slot = spectrum_props.end_slot
        core = spectrum_props.core_number
        band = spectrum_props.current_band
        mod = spectrum_props.modulation or modulation

        if start_slot is None or end_slot is None:
            return SpectrumResult.not_found(slots_needed)

        # Handle backup allocation (if 1+1 protection)
        backup_start = getattr(spectrum_props, "backup_start_slot", None)
        backup_end = getattr(spectrum_props, "backup_end_slot", None)
        backup_core = getattr(spectrum_props, "backup_core_number", None)
        backup_band = getattr(spectrum_props, "backup_band", None)

        return SpectrumResult(
            is_free=True,
            start_slot=start_slot,
            end_slot=end_slot,
            core=core if core is not None else 0,
            band=band if band is not None else "c",
            modulation=mod,
            slots_needed=slots_needed if slots_needed > 0 else (end_slot - start_slot),
            # Backup fields
            backup_start_slot=backup_start,
            backup_end_slot=backup_end,
            backup_core=backup_core,
            backup_band=backup_band,
        )
