"""
GroomingAdapter - Adapts legacy Grooming to GroomingPipeline protocol.

ADAPTER: This class is a TEMPORARY MIGRATION LAYER.
It will be replaced with a clean implementation in Phase 4.

IMPORTANT: This adapter has SIDE EFFECTS. Unlike other adapters,
grooming modifies lightpath bandwidth in the underlying NetworkState.

Phase: P2.4 - Legacy Adapters
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
    network_spectrum_dict: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    lightpath_status_dict: dict[tuple[str, str], dict[int, dict[str, Any]]] = field(
        default_factory=dict
    )

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
        """Create proxy from NetworkState with request context."""
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
    It will be replaced with a clean implementation in Phase 4.

    WARNING: SIDE EFFECTS
    Unlike other adapters, GroomingAdapter has side effects:
    - try_groom() modifies lightpath bandwidth via lightpath_status_dict
    - These changes persist in NetworkState

    The adapter:
    1. Creates proxy from NetworkState
    2. Calls legacy Grooming.handle_grooming()
    3. Converts results to GroomingResult
    4. Side effects are applied through lightpath_status_dict reference

    Removal Checklist:
    [ ] Clean GroomingPipeline implementation exists
    [ ] All callers migrated to clean implementation
    [ ] run_comparison.py passes without this adapter
    [ ] grep 'GroomingAdapter' returns only this definition

    Example:
        >>> adapter = GroomingAdapter(config)
        >>> result = adapter.try_groom(request, network_state)
        >>> if result.fully_groomed:
        ...     print(f"Groomed onto lightpath {result.lightpaths_used}")
    """

    def __init__(self, config: SimulationConfig) -> None:
        """
        Initialize adapter with configuration.

        Args:
            config: SimulationConfig for creating legacy engine_props
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

        Args:
            request: Request object with source, destination, bandwidth
            network_state: Current network state (may be modified!)

        Returns:
            GroomingResult indicating success/partial/failure
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
            was_fully_groomed = legacy_grooming.handle_grooming("arrival")

            # Convert results
            return self._convert_grooming_result(
                sdn_props, was_fully_groomed, request.bandwidth_gbps
            )

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

        Args:
            request: The request that was groomed
            lightpath_ids: Lightpath IDs to rollback
            network_state: Current network state
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
        """Convert SDN props state to GroomingResult."""
        from fusion.domain.results import GroomingResult

        # Determine grooming outcome
        if was_fully_groomed:
            # Fully groomed - no new lightpath needed
            bandwidth_groomed = sum(
                int(float(bw)) for bw in sdn_props.bandwidth_list
            ) if sdn_props.bandwidth_list else original_bandwidth

            return GroomingResult.full(
                bandwidth_gbps=bandwidth_groomed,
                lightpath_ids=list(sdn_props.lightpath_id_list),
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
            )

        else:
            # Not groomed - no existing lightpaths available
            return GroomingResult.no_grooming(original_bandwidth)
