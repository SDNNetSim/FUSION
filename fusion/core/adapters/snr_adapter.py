"""
SNRAdapter - Adapts legacy SnrMeasurements to SNRPipeline protocol.

ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
It will be replaced with a clean implementation in v6.1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

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
    lightpath_status_dict: dict[tuple[str, str], dict[int, dict[str, Any]]] = field(
        default_factory=dict
    )
    # Additional fields for snr_recheck_after_allocation
    lightpath_id_list: list[int] = field(default_factory=list)
    request_id: int | None = None
    path_list: list[str] = field(default_factory=list)
    # Required for dynamic modulation selection in SNR recheck
    modulation_formats_dict: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_network_state(
        cls,
        network_state: NetworkState,
        source: str,
        destination: str,
        bandwidth: float,
        path_index: int = 0,
        lightpath_id: int | None = None,
        path_list: list[str] | None = None,
        request_id: int | None = None,
        modulation_formats_dict: dict[str, dict[str, Any]] | None = None,
    ) -> SDNPropsProxyForSNR:
        """
        Create proxy from NetworkState.

        :param network_state: Current network state
        :type network_state: NetworkState
        :param source: Source node identifier
        :type source: str
        :param destination: Destination node identifier
        :type destination: str
        :param bandwidth: Requested bandwidth in Gbps
        :type bandwidth: float
        :param path_index: Index of path in paths matrix
        :type path_index: int
        :param lightpath_id: Optional lightpath ID for recheck operations
        :type lightpath_id: int | None
        :param path_list: Optional explicit path list
        :type path_list: list[str] | None
        :param request_id: Optional request ID
        :type request_id: int | None
        :param modulation_formats_dict: Modulation format configurations
        :type modulation_formats_dict: dict[str, dict[str, Any]] | None
        :return: Proxy instance populated from network state
        :rtype: SDNPropsProxyForSNR
        """
        return cls(
            topology=network_state.topology,
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            path_index=path_index,
            network_spectrum_dict=network_state.network_spectrum_dict,
            lightpath_status_dict=network_state.lightpath_status_dict,
            lightpath_id_list=[lightpath_id] if lightpath_id is not None else [],
            request_id=request_id,
            path_list=list(path_list) if path_list else [],
            modulation_formats_dict=modulation_formats_dict or {},
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
    slicing_flag: bool = False


class SNRAdapter(SNRPipeline):
    """
    Adapts legacy SnrMeasurements to SNRPipeline protocol.

    ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
    It will be replaced with a clean implementation in v6.1.0.

    The adapter:
    1. Creates proxy objects from NetworkState and Lightpath
    2. Calls legacy SnrMeasurements.handle_snr()
    3. Converts results to SNRResult

    Special Cases:
    - Returns SNRResult.skipped() if SNR checking is disabled
    - Returns SNRResult with passed=False if SNR fails

    Example:
        >>> adapter = SNRAdapter(config)
        >>> result = adapter.validate(lightpath, network_state)
        >>> if result.passed:
        ...     # Proceed with allocation
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize adapter with configuration.

        :param config: SimulationConfig for creating legacy engine_props
        :type config: SimulationConfig
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

        :param lightpath: The lightpath to validate
        :type lightpath: Lightpath
        :param network_state: Current network state
        :type network_state: NetworkState
        :return: SNRResult indicating if SNR is acceptable
        :rtype: SNRResult
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
            from fusion.core.properties import RoutingProps, SDNProps, SpectrumProps
            from fusion.core.snr_measurements import SnrMeasurements

            # Cast proxy objects to satisfy mypy - proxies implement same interface
            legacy_snr = SnrMeasurements(
                engine_props_dict=engine_props,
                sdn_props=cast(SDNProps, sdn_props),
                spectrum_props=cast(SpectrumProps, spectrum_props),
                route_props=cast(RoutingProps, route_props),
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
        slicing_flag: bool = False,
    ) -> SNRRecheckResult:
        """
        Recheck SNR of existing lightpaths after new allocation.

        Delegates to legacy SnrMeasurements.snr_recheck_after_allocation()
        to ensure identical behavior.

        :param new_lightpath_id: ID of newly created lightpath
        :type new_lightpath_id: int
        :param network_state: Current network state
        :type network_state: NetworkState
        :param affected_range_slots: Consider lightpaths within this many slots
            (unused, legacy uses overlap)
        :type affected_range_slots: int
        :param slicing_flag: Whether the allocation was a slicing allocation
            (affects SNR check behavior)
        :type slicing_flag: bool
        :return: SNRRecheckResult with list of degraded lightpaths
        :rtype: SNRRecheckResult
        """
        from fusion.domain.results import SNRRecheckResult

        # Check if SNR recheck is enabled (legacy checks this internally too)
        if not self._engine_props.get("snr_recheck", False):
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

        try:
            # Build new_lp_info dict matching legacy format
            new_lp_info = {
                "id": new_lightpath_id,
                "path": list(new_lp.path),
                "spectrum": (new_lp.start_slot, new_lp.end_slot),
                "core": new_lp.core,
                "band": new_lp.band,
                "mod_format": new_lp.modulation,
            }

            # Create proxies for legacy SnrMeasurements
            # Get modulation_formats_dict from mod_per_bw (pick any bandwidth's formats)
            mod_per_bw = self._engine_props.get('mod_per_bw', {})
            # Use first available bandwidth's modulation formats
            modulation_formats_dict = {}
            if mod_per_bw:
                first_bw = next(iter(mod_per_bw.keys()), None)
                if first_bw:
                    modulation_formats_dict = mod_per_bw[first_bw]
            sdn_props = SDNPropsProxyForSNR.from_network_state(
                network_state=network_state,
                source=new_lp.source,
                destination=new_lp.destination,
                bandwidth=float(new_lp.total_bandwidth_gbps),
                path_index=0,
                lightpath_id=new_lightpath_id,
                path_list=list(new_lp.path),
                request_id=None,  # Not needed for recheck
                modulation_formats_dict=modulation_formats_dict,
            )

            route_props = RoutePropsProxyForSNR(
                paths_matrix=[list(new_lp.path)],
                weights_list=[new_lp.path_weight_km],
            )

            spectrum_props = SpectrumPropsProxyForSNR(
                path_list=list(new_lp.path),
                start_slot=new_lp.start_slot,
                end_slot=new_lp.end_slot,
                core_number=new_lp.core,
                current_band=new_lp.band,
                modulation=new_lp.modulation,
                slots_needed=new_lp.num_slots,
                is_free=True,
                # Match legacy: spectrum_props.slicing_flag should match the allocation type.
                # For standard allocations, slicing_flag=False causes check_gsnr to do
                # bandwidth validation that can return resp=False.
                # For slicing allocations, slicing_flag=True returns modulation string.
                slicing_flag=slicing_flag,
            )

            # Make engine_props copy with topology
            engine_props = dict(self._engine_props)
            engine_props["topology"] = network_state.topology

            # Instantiate legacy SnrMeasurements
            from fusion.core.properties import RoutingProps, SDNProps, SpectrumProps
            from fusion.core.snr_measurements import SnrMeasurements

            # Cast proxy objects to satisfy mypy - proxies implement same interface
            legacy_snr = SnrMeasurements(
                engine_props_dict=engine_props,
                sdn_props=cast(SDNProps, sdn_props),
                spectrum_props=cast(SpectrumProps, spectrum_props),
                route_props=cast(RoutingProps, route_props),
            )

            # Call legacy snr_recheck_after_allocation
            all_pass, violations_list = legacy_snr.snr_recheck_after_allocation(new_lp_info)

            # Convert violations_list to dict format: [(lp_id, observed_snr, required_snr), ...]
            violations_dict: dict[int, float] = {}
            degraded_ids: list[int] = []
            for lp_id, observed_snr, required_snr in violations_list:
                degraded_ids.append(lp_id)
                # Store margin (observed - required)
                violations_dict[lp_id] = float(observed_snr) - float(required_snr)

            return SNRRecheckResult(
                all_pass=all_pass,
                degraded_lightpath_ids=tuple(degraded_ids),
                violations=violations_dict,
                checked_count=len(violations_list) if violations_list else 0,
            )

        except Exception as e:
            import traceback
            logger.warning("SNRAdapter.recheck_affected failed: %s", e)
            traceback.print_exc()
            # On error, assume no violations (fail open)
            return SNRRecheckResult(
                all_pass=True,
                degraded_lightpath_ids=(),
                violations={},
                checked_count=0,
            )
