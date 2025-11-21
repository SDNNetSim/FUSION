"""
Software-defined network controller module for managing network requests.

This module provides the main SDN controller functionality for routing and spectrum
allocation in software-defined optical networks.
"""

import time
from typing import Any

import numpy as np

from fusion.core.grooming import Grooming
from fusion.core.properties import SDNProps
from fusion.core.routing import Routing
from fusion.core.spectrum_assignment import SpectrumAssignment
from fusion.modules.ml import get_ml_obs
from fusion.modules.spectrum.light_path_slicing import LightPathSlicingManager
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


class SDNController:

    """
    Software-defined network controller for managing network requests.

    This class provides functionality for routing, spectrum allocation, and resource
    management in software-defined optical networks. It handles request allocation,
    release, and various slicing strategies.

    :param engine_props: Engine configuration properties
    :type engine_props: dict[str, Any]
    """

    def __init__(self, engine_props: dict[str, Any]) -> None:
        self.engine_props = engine_props
        self.sdn_props = SDNProps()

        self.ai_obj = None
        self.route_obj = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )
        self.spectrum_obj = SpectrumAssignment(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            route_props=self.route_obj.route_props,
        )
        self.slicing_manager = LightPathSlicingManager(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_obj=self.spectrum_obj,
        )
        self.grooming_obj = Grooming(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )
        self.stats = None

        # FailureManager reference for path feasibility checking
        # (set by SimulationEngine)
        self.failure_manager: Any | None = None

    def _update_throughput(self) -> None:
        """
        Update throughput statistics for the current request.

        Should be called once per request release, before releasing individual lightpaths.
        This prevents overcounting throughput when multiple lightpaths are used for a
        single groomed request.
        """
        try:
            if (
                self.sdn_props.depart is None
                or self.sdn_props.arrive is None
                or self.sdn_props.bandwidth is None
                or self.sdn_props.path_list is None
                or self.sdn_props.network_spectrum_dict is None
            ):
                logger.warning("Missing data for throughput calculation")
                return

            duration = self.sdn_props.depart - self.sdn_props.arrive  # seconds
            bandwidth = int(self.sdn_props.bandwidth)  # Gbps
            data_transferred = bandwidth * duration  # Gbps·s

            for source, dest in zip(
                self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False
            ):
                self.sdn_props.network_spectrum_dict[(source, dest)]["throughput"] += (
                    data_transferred
                )
                self.sdn_props.network_spectrum_dict[(dest, source)]["throughput"] += (
                    data_transferred
                )
        except (TypeError, ValueError) as e:
            logger.warning("Throughput update skipped: %s", e)

    def release(
        self, lightpath_id: int | None = None, slicing_flag: bool = False
    ) -> None:
        """
        Remove a previously allocated request from the network.

        :param lightpath_id: Specific lightpath ID to release (for grooming)
        :type lightpath_id: int | None
        :param slicing_flag: If True, only release spectrum, not transponders
        :type slicing_flag: bool
        """
        if lightpath_id is None:
            raise ValueError('Lightpath ID is none')

        for source, dest in zip(
            self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False
        ):
            for band in self.engine_props["band_list"]:
                for core_num in range(self.engine_props["cores_per_link"]):
                    if self.sdn_props.network_spectrum_dict is None:
                        continue
                    core_array = self.sdn_props.network_spectrum_dict[(source, dest)][
                        "cores_matrix"
                    ][band][core_num]

                    request_id_indices = np.where(core_array == lightpath_id)
                    guard_band_indices = np.where(core_array == (lightpath_id * -1))

                    for request_index in request_id_indices[0]:
                        self.sdn_props.network_spectrum_dict[(source, dest)][
                            "cores_matrix"
                        ][band][core_num][request_index] = 0
                        self.sdn_props.network_spectrum_dict[(dest, source)][
                            "cores_matrix"
                        ][band][core_num][request_index] = 0

                    for guard_band_index in guard_band_indices[0]:
                        self.sdn_props.network_spectrum_dict[(source, dest)][
                            "cores_matrix"
                        ][band][core_num][guard_band_index] = 0
                        self.sdn_props.network_spectrum_dict[(dest, source)][
                            "cores_matrix"
                        ][band][core_num][guard_band_index] = 0

        # Handle grooming-specific cleanup and dynamic slicing bandwidth tracking
        if lightpath_id is not None:
            self._release_lightpath_resources(lightpath_id)

    def _release_lightpath_resources(self, lightpath_id: int) -> None:
        """
        Release transponders and update lightpath status dict.

        :param lightpath_id: ID of lightpath to release
        :type lightpath_id: int
        """
        # Early return only if path_list or lightpath_status_dict are None
        # (transponder_usage_dict is optional and only needed for transponder tracking)
        if self.sdn_props.path_list is None or self.sdn_props.lightpath_status_dict is None:
            return

        # Always update transponders
        # for node in [self.sdn_props.source, self.sdn_props.destination]:
        #     if node not in self.sdn_props.transponder_usage_dict:
        #         logger.warning("Node %s not in transponder usage dict", node)
        #         continue
        #     self.sdn_props.transponder_usage_dict[node]["available_transponder"] += 1
        # Update transponders if tracking is enabled and dict is available
        if (self.engine_props.get("transponder_usage_per_node", None) and
            self.sdn_props.transponder_usage_dict is not None):
            for node in [self.sdn_props.source, self.sdn_props.destination]:
                if node not in self.sdn_props.transponder_usage_dict:
                    raise KeyError(f"Node '{node}' not found in transponder usage dictionary.")
                self.sdn_props.transponder_usage_dict[node]["available_transponder"] += 1

        light_id = tuple(
            sorted([self.sdn_props.path_list[0], self.sdn_props.path_list[-1]])
        )


        # Check if light_id exists
        if light_id not in self.sdn_props.lightpath_status_dict:
            return

        # Handle lightpath status dict
        if (
            light_id in self.sdn_props.lightpath_status_dict
            and lightpath_id in self.sdn_props.lightpath_status_dict[light_id]
        ):
            # Calculate bandwidth utilization stats
            try:
                if self.sdn_props.lp_bw_utilization_dict is None:
                    return

                lp_status = self.sdn_props.lightpath_status_dict[light_id][lightpath_id]

                # Debug: Track lightpath details before utilization calculation
                lp_bw = lp_status.get("lightpath_bandwidth", "N/A")
                remaining_bw = lp_status.get("remaining_bandwidth", "N/A")
                time_bw_usage = lp_status.get("time_bw_usage", {})

                average_bw_usage = 0.0
                # Note: average_bandwidth_usage may not exist yet
                # Skip if not available
                try:
                    from fusion.utils.network import (  # type: ignore[attr-defined]
                        average_bandwidth_usage,
                    )
                    if self.sdn_props.depart is not None:
                        average_bw_usage = average_bandwidth_usage(
                            bw_dict=lp_status["time_bw_usage"],
                            departure_time=self.sdn_props.depart,
                        )
                except (ImportError, AttributeError):
                    pass

                # Debug: Track final utilization dict entry

                self.sdn_props.lp_bw_utilization_dict.update(
                    {
                        lightpath_id: {
                            "band": lp_status["band"],
                            "core": lp_status["core"],
                            "bit_rate": lp_status["lightpath_bandwidth"],
                            "utilization": average_bw_usage,
                        }
                    }
                )
            except (TypeError, ValueError, KeyError) as e:
                logger.warning("Average BW update skipped: %s", e)

            # Grooming validation - ensure no active requests
            if (
                self.sdn_props.lightpath_status_dict[light_id][lightpath_id][
                    "requests_dict"
                ]
                and self.engine_props["is_grooming_enabled"]
            ):
                raise ValueError(f"Lightpath {lightpath_id} still has active requests")

            # Remove from status dict
            self.sdn_props.lightpath_status_dict[light_id].pop(lightpath_id)

    def _allocate_guard_band(
        self,
        band: str,
        core_matrix: dict[str, Any],
        reverse_core_matrix: dict[str, Any],
        core_num: int,
        end_slot: int,
        lightpath_id: int,
    ) -> None:
        """
        Allocate guard band slots for spectrum isolation.

        :param band: Spectral band identifier
        :type band: str
        :param core_matrix: Core matrix for forward direction
        :type core_matrix: list[Any]
        :param reverse_core_matrix: Core matrix for reverse direction
        :type reverse_core_matrix: list[Any]
        :param core_num: Core number to allocate on
        :type core_num: int
        :param end_slot: End slot position for guard band
        :type end_slot: int
        :param lightpath_id: Lightpath ID to use for guard band marking
        :type lightpath_id: int
        :raises BufferError: If attempting to allocate already taken spectrum
        """
        if (
            core_matrix[band][core_num][end_slot] != 0.0
            or reverse_core_matrix[band][core_num][end_slot] != 0.0
        ):
            raise BufferError("Attempted to allocate a taken spectrum.")

        # Use lightpath_id with negative sign for guard bands
        core_matrix[band][core_num][end_slot] = lightpath_id * -1
        reverse_core_matrix[band][core_num][end_slot] = lightpath_id * -1

    def _allocate_on_path(self, path: list[int]) -> None:
        """
        Allocate spectrum on a specific path (bidirectionally).

        This helper method contains the core allocation logic for a single path.
        It validates spectrum availability, updates usage counts, allocates slots
        on both forward and reverse links, and handles guard bands.

        :param path: List of node IDs representing the path
        :type path: list[int]
        :raises BufferError: If attempting to allocate already taken spectrum
        :raises ValueError: If no spectrum is detected during allocation
        """
        # Get allocation parameters from spectrum_props
        start_slot = self.spectrum_obj.spectrum_props.start_slot
        end_slot = self.spectrum_obj.spectrum_props.end_slot
        core_num = self.spectrum_obj.spectrum_props.core_number
        band = self.spectrum_obj.spectrum_props.current_band
        lightpath_id = self.spectrum_obj.spectrum_props.lightpath_id

        # Lightpath ID must always be set before allocation
        if lightpath_id is None:
            raise ValueError(
                "lightpath_id must be set in spectrum_props before calling allocate(). "
                "This is a programming error - ensure get_lightpath_id() is called "
                "and assigned to spectrum_props.lightpath_id before allocation."
            )

        # Validate all required parameters are present
        if start_slot is None or end_slot is None or core_num is None or band is None:
            raise ValueError("Missing required spectrum allocation parameters")
        if self.sdn_props.network_spectrum_dict is None:
            raise ValueError("Network spectrum dictionary is None")

        logger.debug(
            f"Attempting allocation on path {path}: "
            f"slots [{start_slot}:{end_slot}], band={band}, core={core_num}"
        )

        # Guard slot adjustment
        if self.engine_props["guard_slots"] != 0:
            end_slot = end_slot - 1
        else:
            end_slot = end_slot + 1

        # Allocate on each link in the path
        for link_tuple in zip(path, path[1:], strict=False):
            link_dict = self.sdn_props.network_spectrum_dict[
                (link_tuple[0], link_tuple[1])
            ]
            reverse_link_dict = self.sdn_props.network_spectrum_dict[
                (link_tuple[1], link_tuple[0])
            ]

            # Validate spectrum is free on both directions
            spectrum_slots_set = set(
                link_dict["cores_matrix"][band][core_num][start_slot:end_slot]
            )
            reverse_spectrum_slots_set = set(
                reverse_link_dict["cores_matrix"][band][core_num][start_slot:end_slot]
            )

            if spectrum_slots_set == {} or reverse_spectrum_slots_set == {}:
                raise ValueError("Nothing detected on the spectrum when allocating.")

            if spectrum_slots_set != {0.0} or reverse_spectrum_slots_set != {0.0}:
                logger.error(
                    f"Spectrum already taken on link {link_tuple}: "
                    f"forward={spectrum_slots_set}, reverse={reverse_spectrum_slots_set}. "
                    f"Path: {path}, slots [{start_slot}:{end_slot}], band={band}, core={core_num}"
                )
                raise BufferError("Attempted to allocate a taken spectrum.")

            # Update usage counts
            if link_tuple in self.sdn_props.network_spectrum_dict:
                self.sdn_props.network_spectrum_dict[link_tuple]["usage_count"] += 1
            self.sdn_props.network_spectrum_dict[(link_tuple[1], link_tuple[0])][
                "usage_count"
            ] += 1

            # Allocate spectrum on both directions
            core_matrix = link_dict["cores_matrix"]
            reverse_core_matrix = reverse_link_dict["cores_matrix"]

            # Use lightpath_id instead of request_id
            core_matrix[band][core_num][start_slot:end_slot] = lightpath_id
            reverse_core_matrix[band][core_num][start_slot:end_slot] = lightpath_id

            # Handle guard bands
            if self.engine_props["guard_slots"]:
                self._allocate_guard_band(
                    band=band,
                    core_matrix=core_matrix,
                    reverse_core_matrix=reverse_core_matrix,
                    core_num=core_num,
                    end_slot=end_slot,
                    lightpath_id=lightpath_id,
                )

    def allocate(self) -> None:
        """
        Allocate spectrum resources for a network request.

        Assigns spectrum slots to the current request across all links in the path,
        including guard bands if configured. For 1+1 protected requests, allocates
        on both primary and backup paths.

        :raises BufferError: If attempting to allocate already taken spectrum
        :raises ValueError: If no spectrum is detected during allocation
        """
        if (
            self.spectrum_obj.spectrum_props.start_slot is None
            or self.spectrum_obj.spectrum_props.end_slot is None
            or self.spectrum_obj.spectrum_props.core_number is None
            or self.spectrum_obj.spectrum_props.current_band is None
            or self.sdn_props.path_list is None
            or self.sdn_props.network_spectrum_dict is None
        ):
            raise ValueError("Missing required spectrum or path information")

        # Allocate on primary path
        self._allocate_on_path(self.sdn_props.path_list)

        # If protected, also allocate on backup path
        backup_path = self.spectrum_obj.spectrum_props.backup_path
        if backup_path is not None:
            logger.debug(
                f"1+1 protection: Allocating spectrum on backup path {backup_path}"
            )
            self._allocate_on_path(backup_path)

        # Track allocated request for failure handling
        self._track_allocated_request()

    def _track_allocated_request(self) -> None:
        """
        Track an allocated request for failure handling.

        Records request information including paths, protection status, and timing
        to enable proper handling when failures occur.
        """
        if self.sdn_props.request_id is None:
            return

        # Get backup path if this is a protected request
        backup_path = self.spectrum_obj.spectrum_props.backup_path

        # Store request information
        self.sdn_props.allocated_requests[self.sdn_props.request_id] = {
            "request_id": self.sdn_props.request_id,
            "primary_path": self.sdn_props.path_list.copy()
            if self.sdn_props.path_list
            else None,
            "backup_path": backup_path.copy() if backup_path else None,
            "is_protected": self.sdn_props.is_protected,
            "active_path": self.sdn_props.active_path,
            "arrive_time": self.sdn_props.arrive,
            "depart_time": self.sdn_props.depart,
            "bandwidth": self.sdn_props.bandwidth,
            "source": self.sdn_props.source,
            "destination": self.sdn_props.destination,
            "start_slot": self.spectrum_obj.spectrum_props.start_slot,
            "end_slot": self.spectrum_obj.spectrum_props.end_slot,
            "core_number": self.spectrum_obj.spectrum_props.core_number,
            "current_band": self.spectrum_obj.spectrum_props.current_band,
        }

        logger.debug(
            f"Tracked allocated request {self.sdn_props.request_id}: "
            f"protected={self.sdn_props.is_protected}, "
            f"primary={self.sdn_props.path_list}, backup={backup_path}"
        )

    def _update_request_statistics(self, bandwidth: float | None) -> None:
        """
        Update request statistics with allocation results.

        :param bandwidth: Allocated bandwidth for the request
        :type bandwidth: str
        """
        if bandwidth is not None:
            self.sdn_props.bandwidth_list.append(bandwidth)
        for stat_key in self.sdn_props.stat_key_list:
            # Skip remaining_bw - it's tracked separately for grooming
            if stat_key == "remaining_bw":
                continue

            # Map stat_key to corresponding spectrum_props attribute
            spectrum_key = stat_key.split("_", maxsplit=1)[0]
            if spectrum_key == "crosstalk":
                spectrum_key = "crosstalk_cost"
            elif spectrum_key == "snr":
                spectrum_key = "crosstalk_cost"  # SNR cost is same as crosstalk_cost
            elif spectrum_key == "xt":
                spectrum_key = "crosstalk_cost"  # XT cost is same as crosstalk_cost
            elif spectrum_key == "core":
                spectrum_key = "core_number"
            elif spectrum_key == "band":
                spectrum_key = "current_band"
            elif spectrum_key == "start":
                spectrum_key = "start_slot"
            elif spectrum_key == "end":
                spectrum_key = "end_slot"
            elif spectrum_key == "modulation":
                spectrum_key = "modulation"
            elif stat_key == "lightpath_id_list":
                spectrum_key = "lightpath_id"
            elif stat_key == "lightpath_bandwidth_list":
                spectrum_key = "lightpath_bandwidth"

            self.sdn_props.update_params(
                key=stat_key,
                spectrum_key=spectrum_key,
                spectrum_obj=self.spectrum_obj.spectrum_props,
            )

    # Backward compatibility alias for tests
    def _update_req_stats(self, bandwidth: float | None = None, remaining: str | None = None) -> None:
        """Legacy method name for _update_request_statistics."""
        # Note: remaining parameter is accepted for compatibility but not currently used
        self._update_request_statistics(bandwidth)

    def _update_grooming_stats(self) -> None:
        """Update grooming statistics after request processing."""
        if not hasattr(self, "stats") or self.stats is None:
            return

        if (
            not hasattr(self.stats, "grooming_stats")
            or self.stats.grooming_stats is None
        ):
            return

        # Determine grooming outcome
        was_groomed = getattr(self.sdn_props, "was_groomed", False) or False
        was_partially_groomed = getattr(self.sdn_props, "was_partially_groomed", False)

        # Get bandwidth
        bandwidth = float(self.sdn_props.bandwidth) if self.sdn_props.bandwidth else 0.0

        # Count new lightpaths established
        was_new_lps = getattr(self.sdn_props, "was_new_lp_established", [])
        new_lightpaths = len(was_new_lps) if isinstance(was_new_lps, list) else 0

        self.stats.grooming_stats.update_grooming_outcome(
            was_groomed=was_groomed,
            was_partially_groomed=was_partially_groomed,
            bandwidth=bandwidth,
            new_lightpaths=new_lightpaths,
        )

    def _handle_slicing_request(
        self,
        path_list: list[Any],
        path_index: int,
        forced_segments: int,
        force_slicing: bool,
    ) -> bool:
        """
        Handle slicing request using the dedicated slicing manager.

        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param path_index: Index of the current path being processed
        :type path_index: int
        :param forced_segments: Number of segments to force (-1 for auto)
        :type forced_segments: int
        :param force_slicing: Whether slicing is forced
        :type force_slicing: bool
        :return: True if slicing was successful
        :rtype: bool
        """
        if self.engine_props["dynamic_lps"]:
            return self.slicing_manager.handle_dynamic_slicing_direct(
                path_list=path_list,
                path_index=path_index,
                forced_segments=forced_segments,
                sdn_controller=self,
            )
        return self.slicing_manager.handle_static_slicing_direct(
            path_list=path_list, forced_segments=forced_segments, sdn_controller=self
        )

    def _check_snr_after_allocation(self, lightpath_id: int) -> bool:
        """
        Recheck SNR after spectrum allocation (for grooming).

        :param lightpath_id: ID of newly allocated lightpath
        :type lightpath_id: int
        :return: True if SNR is acceptable, False otherwise
        :rtype: bool
        """
        if not self.engine_props.get("snr_recheck", False):
            return True

        # Note: SNR rechecking functionality will be implemented when
        # SnrMeasurements supports the recheck_snr_after_allocation method
        # For now, return True (accept all allocations)
        return True

    def _handle_congestion_with_grooming(self, remaining_bw: int) -> None:
        """
        Handle allocation failure with grooming rollback.

        If partial grooming occurred but remaining bandwidth cannot be allocated,
        rollback the newly created lightpaths but keep the groomed portion.

        :param remaining_bw: Remaining bandwidth that could not be allocated
        :type remaining_bw: int
        """
        if self.sdn_props.bandwidth is not None and remaining_bw != int(
            self.sdn_props.bandwidth
        ):
            # Rollback newly established lightpaths
            was_new_lps = getattr(self.sdn_props, "was_new_lp_established", [])
            if isinstance(was_new_lps, list):
                for lpid in list(was_new_lps):
                    self.release(lightpath_id=lpid, slicing_flag=True)
                    was_new_lps.remove(lpid)

                    # Remove from tracking lists
                    lp_idx = self.sdn_props.lightpath_id_list.index(lpid)
                    self.sdn_props.lightpath_id_list.pop(lp_idx)
                    self.sdn_props.lightpath_bandwidth_list.pop(lp_idx)
                    self.sdn_props.start_slot_list.pop(lp_idx)
                    self.sdn_props.band_list.pop(lp_idx)
                    self.sdn_props.core_list.pop(lp_idx)
                    self.sdn_props.end_slot_list.pop(lp_idx)
                    self.sdn_props.crosstalk_list.pop(lp_idx)
                    self.sdn_props.bandwidth_list.pop(lp_idx)
                    self.sdn_props.modulation_list.pop(lp_idx)

        self.sdn_props.number_of_transponders = 1
        self.sdn_props.is_sliced = False
        self.sdn_props.was_partially_routed = False

        if getattr(self.sdn_props, "was_partially_groomed", False):
            self.sdn_props.remaining_bw = int(self.sdn_props.bandwidth) - sum(
                map(int, self.sdn_props.bandwidth_list)
            )
        else:
            if self.sdn_props.bandwidth is not None:
                self.sdn_props.remaining_bw = int(self.sdn_props.bandwidth)

        self.sdn_props.was_new_lp_established = []

    def _handle_congestion(
        self, remaining_bandwidth: int | None = None, remaining_bw: int | None = None
    ) -> None:
        """
        Handle allocation failure due to network congestion.

        :param remaining_bandwidth: Remaining bandwidth that could not be allocated
        :type remaining_bandwidth: int
        :param remaining_bw: Legacy parameter name for remaining_bandwidth
        :type remaining_bw: int
        """
        # Handle backward compatibility
        if remaining_bw is not None:
            remaining_bandwidth = remaining_bw

        if remaining_bandwidth is None:
            raise ValueError("Must provide remaining_bandwidth")
        self.sdn_props.was_routed = False
        self.sdn_props.block_reason = "congestion"
        self.sdn_props.number_of_transponders = 1

        if self.sdn_props.bandwidth is not None and remaining_bandwidth != int(
            self.sdn_props.bandwidth
        ):
            # Release all newly established lightpaths for this request
            was_new_lps = getattr(self.sdn_props, "was_new_lp_established", [])
            if isinstance(was_new_lps, list):
                for lpid in list(was_new_lps):
                    # Remove current request from lightpath's requests_dict before releasing
                    light_id = tuple(sorted([self.sdn_props.path_list[0], self.sdn_props.path_list[-1]]))
                    if (light_id in self.sdn_props.lightpath_status_dict and
                            lpid in self.sdn_props.lightpath_status_dict[light_id]):
                        lp_info = self.sdn_props.lightpath_status_dict[light_id][lpid]
                        if self.sdn_props.request_id in lp_info["requests_dict"]:
                            # Restore remaining bandwidth before removing request
                            allocated_bw = lp_info["requests_dict"][self.sdn_props.request_id]
                            lp_info["remaining_bandwidth"] += allocated_bw
                            lp_info["requests_dict"].pop(self.sdn_props.request_id)

                    self.release(lightpath_id=lpid, slicing_flag=True)
                    was_new_lps.remove(lpid)

                    # Remove from tracking lists
                    if lpid in self.sdn_props.lightpath_id_list:
                        lp_idx = self.sdn_props.lightpath_id_list.index(lpid)
                        self.sdn_props.lightpath_id_list.pop(lp_idx)
                        self.sdn_props.lightpath_bandwidth_list.pop(lp_idx)
                        self.sdn_props.start_slot_list.pop(lp_idx)
                        self.sdn_props.band_list.pop(lp_idx)
                        self.sdn_props.core_list.pop(lp_idx)
                        self.sdn_props.end_slot_list.pop(lp_idx)
                        self.sdn_props.crosstalk_list.pop(lp_idx)
                        self.sdn_props.bandwidth_list.pop(lp_idx)
                        self.sdn_props.modulation_list.pop(lp_idx)

        self.sdn_props.is_sliced = False

    def _initialize_request_statistics(self) -> None:
        """Initialize request statistics for a new request."""
        self.sdn_props.bandwidth_list = []
        self.sdn_props.reset_params()

    def _setup_routing(
        self, force_route_matrix: list[Any] | None, force_mod_format: str | None
    ) -> tuple[list[Any], float]:
        """
        Setup routing for the request.

        :param force_route_matrix: Optional forced routing matrix
        :type force_route_matrix: list[Any] | None
        :param force_mod_format: Optional forced modulation format
        :type force_mod_format: str | None
        :return: Tuple of (route_matrix, route_time)
        :rtype: tuple[list[Any], float]
        """
        start_time = time.time()
        if force_route_matrix is None:
            self.route_obj.get_route()
            route_matrix = self.route_obj.route_props.paths_matrix
        else:
            route_matrix = force_route_matrix
            if force_mod_format:
                formats_matrix = [[force_mod_format]]
            else:
                formats_matrix = [[]]
            self.route_obj.route_props.modulation_formats_matrix = formats_matrix
            self.route_obj.route_props.weights_list = [0]
        route_time = time.time() - start_time
        return route_matrix, route_time

    def _get_ml_prediction(
        self, ml_model: Any | None, request_dict: dict[str, Any]
    ) -> float:
        """
        Get ML model prediction for forced segments.

        :param ml_model: Optional machine learning model
        :type ml_model: Any | None
        :param request_dict: Request dictionary
        :type request_dict: dict[str, Any]
        :return: Forced segments prediction (-1 for auto)
        :rtype: float
        """
        if ml_model is not None:
            input_df = get_ml_obs(
                request_dict=request_dict,
                engine_properties=self.engine_props,
                sdn_properties=self.sdn_props,
            )
            prediction = ml_model.predict(input_df)[0]
            return float(prediction)
        return -1.0

    def _process_single_path(
        self,
        path_list: list[Any],
        path_index: int,
        mod_format_list: list[str],
        forced_segments: float,
        force_slicing: bool,
        segment_slicing: bool,
        forced_index: int | None,
        force_core: int | None,
        forced_band: str | None,
        backup_mod_format_list: list[str] | None = None,
    ) -> bool:
        """
        Process allocation for a single path.

        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param path_index: Index of the current path
        :type path_index: int
        :param mod_format_list: List of modulation formats for primary path
        :type mod_format_list: list[str]
        :param forced_segments: Number of forced segments
        :type forced_segments: float
        :param force_slicing: Whether to force slicing
        :type force_slicing: bool
        :param segment_slicing: Whether segment slicing is enabled
        :type segment_slicing: bool
        :param forced_index: Optional forced spectrum index
        :type forced_index: int | None
        :param force_core: Optional forced core number
        :type force_core: int | None
        :param forced_band: Optional forced band
        :type forced_band: str | None
        :param backup_mod_format_list: Optional modulation formats for backup path (1+1)
        :type backup_mod_format_list: list[str] | None
        :return: True if path processing was successful
        :rtype: bool
        """
        self.sdn_props.path_list = path_list
        self.sdn_props.path_index = path_index

        # Handle slicing scenarios
        if segment_slicing or force_slicing or forced_segments > 1:
            success = self._handle_slicing_request(
                path_list, path_index, int(forced_segments), force_slicing
            )
            if not success:
                self.sdn_props.number_of_transponders = 1
                return False
        else:
            # Handle standard allocation
            self.spectrum_obj.spectrum_props.forced_index = forced_index
            self.spectrum_obj.spectrum_props.forced_core = force_core
            self.spectrum_obj.spectrum_props.path_list = path_list
            self.spectrum_obj.spectrum_props.forced_band = forced_band
            self.spectrum_obj.get_spectrum(
                mod_format_list=mod_format_list,
                backup_mod_format_list=backup_mod_format_list,
            )

            if self.spectrum_obj.spectrum_props.is_free is not True:
                self.sdn_props.block_reason = "congestion"
                return False

            # Generate and assign unique lightpath ID for this allocation
            lp_id = self.sdn_props.get_lightpath_id()
            self.spectrum_obj.spectrum_props.lightpath_id = lp_id
            self.sdn_props.was_new_lp_established.append(lp_id)

            # Determine bandwidth for statistics (handle partial grooming)
            if self.sdn_props.was_partially_groomed:
                allocate_bandwidth = str(self.sdn_props.remaining_bw)
            else:
                allocate_bandwidth = self.sdn_props.bandwidth

            # Set lightpath bandwidth only if not already set by slicing code
            if not hasattr(self.spectrum_obj.spectrum_props, 'lightpath_bandwidth') or \
               self.spectrum_obj.spectrum_props.lightpath_bandwidth is None:
                self.spectrum_obj.spectrum_props.lightpath_bandwidth = allocate_bandwidth

            self._update_request_statistics(bandwidth=allocate_bandwidth)

            # Create lightpath entry in status dict for grooming/tracking
            self.spectrum_obj._update_lightpath_status()

        return True

    def _finalize_successful_allocation(
        self,
        path_index: int,
        route_time: float,
        force_slicing: bool,
        segment_slicing: bool,
    ) -> None:
        """
        Finalize a successful allocation.

        :param path_index: Index of the successful path
        :type path_index: int
        :param route_time: Time taken for routing
        :type route_time: float
        :param force_slicing: Whether slicing was forced
        :type force_slicing: bool
        :param segment_slicing: Whether segment slicing was used
        :type segment_slicing: bool
        """
        self.sdn_props.was_routed = True
        self.sdn_props.route_time = route_time
        self.sdn_props.path_weight = self.route_obj.route_props.weights_list[path_index]
        self.sdn_props.spectrum_object = self.spectrum_obj.spectrum_props

        if not segment_slicing and not force_slicing:
            self.sdn_props.is_sliced = False
            self.allocate()

        # Update grooming statistics
        if self.engine_props.get("is_grooming_enabled", False):
            self._update_grooming_stats()

    def handle_event(
        self,
        request_dict: dict[str, Any],
        request_type: str,
        force_slicing: bool = False,
        force_route_matrix: list[Any] | None = None,
        forced_index: int | None = None,
        force_core: int | None = None,
        ml_model: Any | None = None,
        force_mod_format: str | None = None,
        forced_band: str | None = None,
    ) -> None:
        """
        Handle any event that occurs in the simulation.

        Controls the main flow of request processing including routing, spectrum
        allocation, and various slicing strategies.

        :param request_dict: Request dictionary containing request parameters
        :type request_dict: dict[str, Any]
        :param request_type: Type of request ('arrival' or 'release')
        :type request_type: str
        :param force_slicing: Whether to force light path segment slicing
        :type force_slicing: bool
        :param force_route_matrix: Optional forced routing matrix
        :type force_route_matrix: list[Any] | None
        :param forced_index: Optional forced start index for spectrum allocation
        :type forced_index: int | None
        :param force_mod_format: Optional forced modulation format
        :type force_mod_format: str | None
        :param force_core: Optional forced core number
        :type force_core: int | None
        :param ml_model: Optional machine learning model for predictions
        :type ml_model: Any | None
        :param forced_band: Optional forced spectral band
        :type forced_band: str | None
        """
        # Handle release requests
        if request_type == "release":
            lightpath_id_list: list[int | None] = []
            if self.engine_props.get("is_grooming_enabled", False):
                groom_result = self.grooming_obj.handle_grooming(request_type)
                if isinstance(groom_result, list):
                    # Convert list[int] to list[int | None]
                    lightpath_id_list = list(groom_result)
            else:
                # Convert list[int] to list[int | None]
                lightpath_id_list = list(self.sdn_props.lightpath_id_list)

            # Update throughput once per request, then release each lightpath
            self._update_throughput()
            for lightpath_id in lightpath_id_list:
                self.release(lightpath_id=lightpath_id)
            return

        self._initialize_request_statistics()
        self.sdn_props.number_of_transponders = 1

        # Try grooming first if enabled
        if self.engine_props.get("is_grooming_enabled", False):
            # Set lightpath status dict for grooming object
            if hasattr(self.grooming_obj, "lightpath_status_dict"):
                self.grooming_obj.lightpath_status_dict = (
                    self.sdn_props.lightpath_status_dict
                )
            groom_result = self.grooming_obj.handle_grooming(request_type)

            if groom_result:
                # Fully groomed - done!
                self._update_grooming_stats()
                return

            # Not groomed or partially groomed
            self.sdn_props.was_new_lp_established = []

            if getattr(self.sdn_props, "was_partially_groomed", False):
                # Force route on same path as groomed portion
                force_route_matrix = [self.sdn_props.path_list]

                # Get modulation formats for remaining bandwidth
                from fusion.utils.data import sort_nested_dict_values

                if self.sdn_props.modulation_formats_dict is not None:
                    mod_formats_dict = sort_nested_dict_values(
                        original_dict=self.sdn_props.modulation_formats_dict,
                        nested_key="max_length",
                    )
                    force_mod_format = (
                        list(mod_formats_dict.keys())[0] if mod_formats_dict else None
                    )

        # Setup routing
        route_matrix, route_time = self._setup_routing(
            force_route_matrix, force_mod_format
        )

        # Get ML prediction if available
        forced_segments = self._get_ml_prediction(ml_model, request_dict)

        # Try allocation with different strategies
        segment_slicing = False
        while True:
            for path_index, path_list in enumerate(route_matrix):
                if path_list is not False:
                    # Check path feasibility if failures are active
                    if (
                        self.failure_manager
                        and not self.failure_manager.is_path_feasible(path_list)
                    ):
                        logger.debug(
                            f"Path {path_list} (index {path_index}) is "
                            f"infeasible due to active failures"
                        )
                        continue  # Skip this path and try next one

                    # Check backup path feasibility for protected requests
                    # Get corresponding backup path from backup_paths_matrix if available
                    backup_path = None
                    if (
                        hasattr(self.route_obj.route_props, "backup_paths_matrix")
                        and len(self.route_obj.route_props.backup_paths_matrix)
                        > path_index
                    ):
                        backup_path = self.route_obj.route_props.backup_paths_matrix[
                            path_index
                        ]
                    elif self.sdn_props.backup_path is not None:
                        # Fallback to sdn_props.backup_path for backward compatibility
                        backup_path = self.sdn_props.backup_path

                    if (
                        self.failure_manager
                        and backup_path is not None
                        and not self.failure_manager.is_path_feasible(backup_path)
                    ):
                        logger.debug(
                            f"Backup path {backup_path} "
                            f"(for primary {path_list}) is infeasible due to active failures"
                        )
                        continue  # Skip this path pair and try next one

                    # Set backup path in sdn_props for spectrum assignment
                    backup_mod_format_list = None
                    if backup_path is not None:
                        self.sdn_props.backup_path = backup_path
                        self.sdn_props.is_protected = True

                        # Get backup modulation formats for 1+1 protection validation
                        if (
                            hasattr(
                                self.route_obj.route_props,
                                "backup_modulation_formats_matrix",
                            )
                            and len(
                                self.route_obj.route_props.backup_modulation_formats_matrix
                            )
                            > path_index
                        ):
                            backup_mod_format_list = self.route_obj.route_props.backup_modulation_formats_matrix[
                                path_index
                            ]

                    mod_format_list = (
                        self.route_obj.route_props.modulation_formats_matrix[path_index]
                    )

                    # Process the path
                    success = self._process_single_path(
                        path_list,
                        path_index,
                        mod_format_list,
                        forced_segments,
                        force_slicing,
                        segment_slicing,
                        forced_index,
                        force_core,
                        forced_band,
                        backup_mod_format_list,
                    )

                    if success:
                        self._finalize_successful_allocation(
                            path_index, route_time, force_slicing, segment_slicing
                        )
                        return

            # Try segment slicing if not already tried
            if (
                self.engine_props["max_segments"] > 1
                and self.sdn_props.bandwidth != "25"
                and not segment_slicing
            ):
                segment_slicing = True
                continue

            # All paths exhausted
            self.sdn_props.block_reason = "distance"
            self.sdn_props.was_routed = False
            return

    # Backward compatibility methods for tests
    def _allocate_slicing(
        self, num_segments: int, mod_format: str, path_list: list[Any], bandwidth: str
    ) -> None:
        """
        Backward compatibility wrapper for allocate_slicing method.

        :param num_segments: Number of segments to allocate
        :type num_segments: int
        :param mod_format: Modulation format to use
        :type mod_format: str
        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param bandwidth: Bandwidth requirement for each segment
        :type bandwidth: str
        """
        self.slicing_manager.allocate_slicing_direct(
            num_segments=num_segments,
            mod_format=mod_format,
            path_list=path_list,
            bandwidth=bandwidth,
            sdn_controller=self,
        )

    def _handle_dynamic_slicing(
        self, path_list: list[Any], path_index: int, forced_segments: int
    ) -> None:
        """
        Backward compatibility wrapper for handle_dynamic_slicing method.

        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param path_index: Index of the current path being processed
        :type path_index: int
        :param forced_segments: Number of forced segments (unused)
        :type forced_segments: int
        """
        self.slicing_manager.handle_dynamic_slicing_direct(
            path_list=path_list,
            path_index=path_index,
            forced_segments=forced_segments,
            sdn_controller=self,
        )
