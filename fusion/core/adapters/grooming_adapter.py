"""
GroomingAdapter - Adapts legacy Grooming to GroomingPipeline protocol.

ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
It will be replaced with a clean implementation in Phase 4.

IMPORTANT: This adapter has SIDE EFFECTS. Unlike other adapters,
grooming modifies lightpath bandwidth in the underlying NetworkState.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fusion.interfaces.pipelines import GroomingPipeline

if TYPE_CHECKING:
    from fusion.domain.config import SimulationConfig
    from fusion.domain.network_state import NetworkState
    from fusion.domain.request import Request
    from fusion.domain.results import GroomingResult

logger = logging.getLogger(__name__)


@dataclass
class SDNPropsProxyForGrooming:
    """
    Minimal proxy for SDNProps to satisfy legacy Grooming class.

    This proxy tracks state changes made by Grooming to report back.

    WARNING: Grooming has side effects - it modifies lightpath bandwidth.
    The proxy captures these modifications for NetworkState synchronization.
    """

    topology: Any
    source: str = ""
    destination: str = ""
    bandwidth: float = 0.0
    request_id: int = 0
    arrive: float = 0.0
    network_spectrum_dict: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    lightpath_status_dict: dict[tuple[str, str], dict[int, dict[str, Any]]] = field(default_factory=dict)

    # Lists populated by Grooming during allocation
    bandwidth_list: list[str] = field(default_factory=list)
    core_list: list[int] = field(default_factory=list)
    band_list: list[str] = field(default_factory=list)
    start_slot_list: list[int] = field(default_factory=list)
    end_slot_list: list[int] = field(default_factory=list)
    modulation_list: list[str] = field(default_factory=list)
    snr_list: list[float | None] = field(default_factory=list)
    xt_list: list[float | None] = field(default_factory=list)
    lightpath_bandwidth_list: list[float] = field(default_factory=list)
    lightpath_id_list: list[int] = field(default_factory=list)

    # State flags set by Grooming
    was_routed: bool = False
    was_groomed: bool = False
    was_partially_groomed: bool = False
    is_sliced: bool = False
    number_of_transponders: int = 0
    was_new_lp_established: list[int] = field(default_factory=list)
    remaining_bw: int | float | str = 0
    path_list: list[str] | None = None
    path_weight: float = 0.0

    @classmethod
    def from_network_state(
        cls,
        network_state: NetworkState,
        source: str,
        destination: str,
        bandwidth: float,
        request_id: int,
        arrive_time: float,
    ) -> SDNPropsProxyForGrooming:
        """
        Create proxy from NetworkState with request context.

        :param network_state: Current network state
        :type network_state: NetworkState
        :param source: Source node identifier
        :type source: str
        :param destination: Destination node identifier
        :type destination: str
        :param bandwidth: Requested bandwidth in Gbps
        :type bandwidth: float
        :param request_id: Unique request identifier
        :type request_id: int
        :param arrive_time: Request arrival time
        :type arrive_time: float
        :return: Proxy instance populated from network state
        :rtype: SDNPropsProxyForGrooming
        """
        return cls(
            topology=network_state.topology,
            source=source,
            destination=destination,
            bandwidth=bandwidth,
            request_id=request_id,
            arrive=arrive_time,
            network_spectrum_dict=network_state.network_spectrum_dict,
            lightpath_status_dict=network_state.lightpath_status_dict,
        )


class GroomingAdapter(GroomingPipeline):
    """
    Adapts legacy Grooming to GroomingPipeline protocol.

    ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
    It will be replaced with a clean implementation in v6.1.0.

    WARNING: SIDE EFFECTS
    Unlike other adapters, GroomingAdapter has side effects:
    - try_groom() modifies lightpath bandwidth via lightpath_status_dict
    - These changes persist in NetworkState

    The adapter:
    1. Creates proxy from NetworkState
    2. Calls legacy Grooming.handle_grooming()
    3. Converts results to GroomingResult
    4. Side effects are applied through lightpath_status_dict reference

    Example:
        >>> adapter = GroomingAdapter(config)
        >>> result = adapter.try_groom(request, network_state)
        >>> if result.fully_groomed:
        ...     print(f"Groomed onto lightpath {result.lightpaths_used}")
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize adapter with configuration.

        :param config: SimulationConfig for creating legacy engine_props
        :type config: SimulationConfig
        """
        self._config = config
        self._engine_props = config.to_engine_props()

    def try_groom(
        self,
        request: Request,
        network_state: NetworkState,
    ) -> GroomingResult:
        """
        Attempt to groom request onto existing lightpaths.

        WARNING: This method has SIDE EFFECTS. If grooming succeeds,
        the lightpath bandwidth is modified in NetworkState.

        :param request: Request object with source, destination, bandwidth
        :type request: Request
        :param network_state: Current network state (may be modified!)
        :type network_state: NetworkState
        :return: GroomingResult indicating success/partial/failure
        :rtype: GroomingResult
        """
        from fusion.domain.results import GroomingResult

        # Check if grooming is enabled
        if not self._engine_props.get("is_grooming_enabled", False):
            return GroomingResult.no_grooming(request.bandwidth_gbps)

        try:
            # Create proxy with references to NetworkState's dictionaries
            sdn_props = SDNPropsProxyForGrooming.from_network_state(
                network_state=network_state,
                source=str(request.source),
                destination=str(request.destination),
                bandwidth=float(request.bandwidth_gbps),
                request_id=request.request_id,
                arrive_time=request.arrive_time,
            )

            # Make engine_props copy
            engine_props = dict(self._engine_props)
            engine_props["topology"] = network_state.topology

            # Instantiate legacy Grooming
            from fusion.core.grooming import Grooming

            legacy_grooming = Grooming(
                engine_props=engine_props,
                sdn_props=sdn_props,
            )

            # Call legacy handle_grooming for arrival
            # handle_grooming returns bool for arrivals, list[int] for releases
            grooming_result = legacy_grooming.handle_grooming("arrival")
            was_fully_groomed = bool(grooming_result)  # Always bool for arrivals

            # CRITICAL: Sync grooming changes back to actual Lightpath objects
            # The legacy grooming code modifies lightpath_status_dict in place,
            # but in new architecture this is a rebuilt property. We must sync
            # remaining_bandwidth and requests_dict back to the domain model.
            if was_fully_groomed or sdn_props.was_partially_groomed:
                self._sync_grooming_changes(network_state, sdn_props, request.request_id)

            # Convert results
            return self._convert_grooming_result(sdn_props, was_fully_groomed, request.bandwidth_gbps)

        except Exception as e:
            logger.warning("GroomingAdapter.try_groom failed: %s", e)
            return GroomingResult.no_grooming(request.bandwidth_gbps)

    def rollback_groom(
        self,
        request: Request,
        lightpath_ids: list[int],
        network_state: NetworkState,
    ) -> None:
        """
        Rollback grooming allocations (e.g., after downstream failure).

        :param request: The request that was groomed
        :type request: Request
        :param lightpath_ids: Lightpath IDs to rollback
        :type lightpath_ids: list[int]
        :param network_state: Current network state
        :type network_state: NetworkState
        """
        try:
            # Create proxy
            sdn_props = SDNPropsProxyForGrooming.from_network_state(
                network_state=network_state,
                source=str(request.source),
                destination=str(request.destination),
                bandwidth=float(request.bandwidth_gbps),
                request_id=request.request_id,
                arrive_time=request.arrive_time,
            )

            # Make engine_props copy
            engine_props = dict(self._engine_props)
            engine_props["topology"] = network_state.topology

            # Instantiate legacy Grooming
            from fusion.core.grooming import Grooming

            legacy_grooming = Grooming(
                engine_props=engine_props,
                sdn_props=sdn_props,
            )

            # Call legacy handle_grooming for release
            legacy_grooming.handle_grooming("release")

        except Exception as e:
            logger.warning("GroomingAdapter.rollback_groom failed: %s", e)

    def _convert_grooming_result(
        self,
        sdn_props: SDNPropsProxyForGrooming,
        was_fully_groomed: bool,
        original_bandwidth: int,
    ) -> GroomingResult:
        """
        Convert SDN props state to GroomingResult.

        :param sdn_props: SDN properties proxy with grooming state
        :type sdn_props: SDNPropsProxyForGrooming
        :param was_fully_groomed: Whether request was fully groomed
        :type was_fully_groomed: bool
        :param original_bandwidth: Original requested bandwidth in Gbps
        :type original_bandwidth: int
        :return: Converted grooming result
        :rtype: GroomingResult
        """
        from fusion.domain.results import GroomingResult

        # Extract SNR and modulation lists for Legacy compatibility
        # These include values from grooming attempts (even if grooming fails)
        snr_list = [v for v in sdn_props.snr_list if v is not None]
        modulation_list = list(sdn_props.modulation_list)

        # Determine grooming outcome
        if was_fully_groomed:
            # Fully groomed - no new lightpath needed
            bandwidth_groomed = sum(int(float(bw)) for bw in sdn_props.bandwidth_list) if sdn_props.bandwidth_list else original_bandwidth

            return GroomingResult.full(
                bandwidth_gbps=bandwidth_groomed,
                lightpath_ids=list(sdn_props.lightpath_id_list),
                snr_list=snr_list,
                modulation_list=modulation_list,
            )

        elif sdn_props.was_partially_groomed:
            # Partially groomed - some bandwidth still needs allocation
            remaining = sdn_props.remaining_bw
            if isinstance(remaining, str):
                remaining = int(float(remaining)) if remaining else 0
            else:
                remaining = int(remaining) if remaining else 0

            bandwidth_groomed = original_bandwidth - remaining

            forced_path = None
            if sdn_props.path_list:
                forced_path = [str(n) for n in sdn_props.path_list]

            return GroomingResult.partial(
                bandwidth_groomed=bandwidth_groomed,
                remaining=remaining,
                lightpath_ids=list(sdn_props.lightpath_id_list),
                forced_path=forced_path,
                snr_list=snr_list,
                modulation_list=modulation_list,
            )

        else:
            # Not groomed - no existing lightpaths available
            return GroomingResult.no_grooming(original_bandwidth)

    def _sync_grooming_changes(
        self,
        network_state: NetworkState,
        sdn_props: SDNPropsProxyForGrooming,
        request_id: int,
    ) -> None:
        """
        Sync grooming changes back to actual Lightpath objects.

        The legacy grooming code modifies lightpath_status_dict in place.
        Since NetworkState.lightpath_status_dict is a property that rebuilds
        from _lightpaths, those changes are lost. This method syncs the
        changes back to the actual Lightpath domain objects.

        :param network_state: Network state containing lightpath objects
        :type network_state: NetworkState
        :param sdn_props: SDN properties proxy with modified state
        :type sdn_props: SDNPropsProxyForGrooming
        :param request_id: Request ID being groomed
        :type request_id: int
        """
        # Iterate through lightpaths that were used for grooming
        for lp_id in sdn_props.lightpath_id_list:
            # Find the lightpath in network_state
            lightpath = network_state.get_lightpath(lp_id)
            if lightpath is None:
                continue

            # Find the modified entry in lightpath_status_dict
            # The dict was modified in place by legacy grooming code
            for _light_id, lp_dict in sdn_props.lightpath_status_dict.items():
                if lp_id in lp_dict:
                    lp_info = lp_dict[lp_id]

                    # Sync remaining_bandwidth
                    new_remaining = float(lp_info.get("remaining_bandwidth", 0))
                    lightpath.remaining_bandwidth_gbps = int(new_remaining)

                    # Sync requests_dict (which requests are using this LP)
                    requests_dict = lp_info.get("requests_dict", {})
                    if request_id in requests_dict:
                        bw_used = int(float(requests_dict[request_id]))
                        lightpath.request_allocations[request_id] = bw_used

                    # Sync time_bw_usage for utilization tracking
                    time_bw_usage = lp_info.get("time_bw_usage", {})
                    if time_bw_usage:
                        lightpath.time_bw_usage.update(time_bw_usage)

                    break
