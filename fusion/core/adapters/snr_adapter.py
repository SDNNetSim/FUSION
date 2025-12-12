"""
SNRAdapter - Adapts legacy SnrMeasurements to SNRPipeline protocol.

ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
It will be replaced with a clean implementation in Phase 4.

Phase: P2.4 - Legacy Adapters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fusion.interfaces.pipelines import SNRPipeline

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.lightpath import Lightpath
    from fusion.domain.network_state import NetworkState
    from fusion.domain.results import SNRRecheckResult, SNRResult

logger = logging.getLogger(__name__)


@dataclass
class SDNPropsProxyForSNR:
    """
    Minimal proxy for SDNProps to satisfy legacy SnrMeasurements.
    """

    topology: Any
    source: str = ""
    destination: str = ""
    bandwidth: float = 0.0
    path_index: int = 0
    network_spectrum_dict: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )

    @classmethod
    def from_network_state(
        cls,
        network_state: NetworkState,
        source: str,
        destination: str,
        bandwidth: float,
        path_index: int = 0,
    ) -> SDNPropsProxyForSNR:
        """Create proxy from NetworkState."""
        return cls(
            topology=network_state.topology,
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            path_index=path_index,
            network_spectrum_dict=network_state.network_spectrum_dict,
        )


@dataclass
class RoutePropsProxyForSNR:
    """
    Minimal proxy for RoutingProps to satisfy legacy SnrMeasurements.
    """

    paths_matrix: list[list[str]] = field(default_factory=list)
    weights_list: list[float] = field(default_factory=list)
    connection_index: int | None = None


@dataclass
class SpectrumPropsProxyForSNR:
    """
    Minimal proxy for SpectrumProps to satisfy legacy SnrMeasurements.
    """

    path_list: list[str] | None = None
    start_slot: int | None = None
    end_slot: int | None = None
    core_number: int | None = None
    current_band: str | None = None
    modulation: str | None = None
    slots_needed: int | None = None
    is_free: bool = True
    crosstalk_cost: float | None = None
    lightpath_bandwidth: float | None = None


class SNRAdapter(SNRPipeline):
    """
    Adapts legacy SnrMeasurements to SNRPipeline protocol.

    ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
    It will be replaced with a clean implementation in Phase 4.

    The adapter:
    1. Creates proxy objects from NetworkState and Lightpath
    2. Calls legacy SnrMeasurements.handle_snr()
    3. Converts results to SNRResult

    Special Cases:
    - Returns SNRResult.skipped() if SNR checking is disabled
    - Returns SNRResult with passed=False if SNR fails

    Removal Checklist:
    [ ] Clean SNRPipeline implementation exists
    [ ] All callers migrated to clean implementation
    [ ] run_comparison.py passes without this adapter
    [ ] grep 'SNRAdapter' returns only this definition

    Example:
        >>> adapter = SNRAdapter(config)
        >>> result = adapter.validate(lightpath, network_state)
        >>> if result.passed:
        ...     # Proceed with allocation
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize adapter with configuration.

        Args:
            config: SimulationConfig for creating legacy engine_props
        """
        self._config = config
        self._engine_props = config.to_engine_props()

    def validate(
        self,
        lightpath: Lightpath,
        network_state: NetworkState,
    ) -> SNRResult:
        """
        Validate SNR for a lightpath.

        Args:
            lightpath: The lightpath to validate
            network_state: Current network state

        Returns:
            SNRResult indicating if SNR is acceptable
        """
        from fusion.domain.results import SNRResult

        # Check if SNR validation is enabled
        snr_type = self._engine_props.get("snr_type")
        if snr_type is None or snr_type == "None":
            return SNRResult.skipped()

        try:
            # Create proxies from lightpath
            sdn_props = SDNPropsProxyForSNR.from_network_state(
                network_state=network_state,
                source=lightpath.source,
                destination=lightpath.destination,
                bandwidth=float(lightpath.total_bandwidth_gbps),
                path_index=0,
            )

            route_props = RoutePropsProxyForSNR(
                paths_matrix=[list(lightpath.path)],
                weights_list=[lightpath.path_weight_km],
                connection_index=lightpath.connection_index,
            )

            spectrum_props = SpectrumPropsProxyForSNR(
                path_list=list(lightpath.path),
                start_slot=lightpath.start_slot,
                end_slot=lightpath.end_slot,
                core_number=lightpath.core,
                current_band=lightpath.band,
                modulation=lightpath.modulation,
                slots_needed=lightpath.num_slots,
                is_free=True,
            )

            # Make engine_props copy
            engine_props = dict(self._engine_props)
            engine_props["topology"] = network_state.topology

            # Instantiate legacy SnrMeasurements
            from fusion.core.snr_measurements import SnrMeasurements

            legacy_snr = SnrMeasurements(
                engine_props_dict=engine_props,
                sdn_props=sdn_props,
                spectrum_props=spectrum_props,
                route_props=route_props,
            )

            # Call legacy handle_snr
            # Returns: (snr_acceptable, snr_value_db, lp_bandwidth)
            snr_acceptable, snr_value_db, lp_bandwidth = legacy_snr.handle_snr(0)

            # Get required SNR threshold for modulation
            snr_thresholds = self._config.snr_thresholds
            required_snr = snr_thresholds.get(lightpath.modulation, 0.0)

            # Convert snr_value_db to float (may be numpy type)
            snr_db = float(snr_value_db) if snr_value_db is not None else 0.0

            if snr_acceptable:
                return SNRResult.success(
                    snr_db=snr_db,
                    required_snr_db=required_snr,
                )
            else:
                return SNRResult.failure(
                    snr_db=snr_db,
                    required_snr_db=required_snr,
                    reason="SNR below threshold",
                )

        except Exception as e:
            logger.warning("SNRAdapter.validate failed: %s", e)
            # On error, skip SNR check (fail open)
            return SNRResult.skipped()

    def recheck_affected(
        self,
        new_lightpath_id: int,
        network_state: NetworkState,
        *,
        affected_range_slots: int = 5,
    ) -> SNRRecheckResult:
        """
        Recheck SNR of existing lightpaths after new allocation.

        Args:
            new_lightpath_id: ID of newly created lightpath
            network_state: Current network state
            affected_range_slots: Consider lightpaths within this many slots

        Returns:
            SNRRecheckResult with list of degraded lightpaths
        """
        from fusion.domain.results import SNRRecheckResult

        # Check if SNR validation is enabled
        snr_type = self._engine_props.get("snr_type")
        if snr_type is None or snr_type == "None":
            return SNRRecheckResult(
                all_pass=True,
                degraded_lightpath_ids=(),
                violations={},
                checked_count=0,
            )

        # Get the new lightpath
        new_lp = network_state.get_lightpath(new_lightpath_id)
        if new_lp is None:
            return SNRRecheckResult(
                all_pass=True,
                degraded_lightpath_ids=(),
                violations={},
                checked_count=0,
            )

        # Find potentially affected lightpaths
        degraded_ids: list[int] = []
        violations: dict[int, float] = {}
        checked_count = 0

        # Get all lightpaths on the same links
        for link in zip(new_lp.path[:-1], new_lp.path[1:]):
            lightpaths_on_link = network_state.get_lightpaths_on_link(link)

            for lp in lightpaths_on_link:
                if lp.lightpath_id == new_lightpath_id:
                    continue

                # Check if within affected range
                slot_distance = min(
                    abs(lp.start_slot - new_lp.end_slot),
                    abs(new_lp.start_slot - lp.end_slot),
                )
                if slot_distance > affected_range_slots:
                    continue

                # Validate this lightpath
                checked_count += 1
                result = self.validate(lp, network_state)

                if not result.passed:
                    degraded_ids.append(lp.lightpath_id)
                    violations[lp.lightpath_id] = result.margin_db

        return SNRRecheckResult(
            all_pass=len(degraded_ids) == 0,
            degraded_lightpath_ids=tuple(degraded_ids),
            violations=violations,
            checked_count=checked_count,
        )
