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
        if self.sdn_props.path_list is None:
            return

        # Use provided lightpath_id or fall back to request_id
        release_id = (
            lightpath_id if lightpath_id is not None else self.sdn_props.request_id
        )

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

                    # Release using lightpath_id instead of request_id
                    if release_id is None:
                        continue
                    request_id_indices = np.where(core_array == release_id)
                    guard_band_indices = np.where(core_array == (release_id * -1))

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

        # Throughput calculation (existing code)
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

        # Handle grooming-specific cleanup
        if not slicing_flag and lightpath_id is not None:
            self._release_lightpath_resources(lightpath_id)

    def _release_lightpath_resources(self, lightpath_id: int) -> None:
        """
        Release transponders and update lightpath status dict.

        :param lightpath_id: ID of lightpath to release
        :type lightpath_id: int
        """
        if (
            self.sdn_props.transponder_usage_dict is None
            or self.sdn_props.path_list is None
            or self.sdn_props.lightpath_status_dict is None
        ):
            return

        # Always update transponders
        for node in [self.sdn_props.source, self.sdn_props.destination]:
            if node not in self.sdn_props.transponder_usage_dict:
                logger.warning("Node %s not in transponder usage dict", node)
                continue
            self.sdn_props.transponder_usage_dict[node]["available_transponder"] += 1

        light_id = tuple(
            sorted([self.sdn_props.path_list[0], self.sdn_props.path_list[-1]])
        )

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
                average_bw_usage = 0.0
                # Note: average_bandwidth_usage may not exist yet
                # Skip if not available
                try:
                    from fusion.utils.network import (  # type: ignore[attr-defined]
                        average_bandwidth_usage,
                    )

                    average_bw_usage = average_bandwidth_usage(
                        bw_dict=lp_status["time_bw_usage"],
                        departure_time=self.sdn_props.depart,
                    )
                except (ImportError, AttributeError):
                    pass

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
            logger.debug("Released lightpath %d", lightpath_id)

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

    def allocate(self) -> None:
        """
        Allocate spectrum resources for a network request.

        Assigns spectrum slots to the current request across all links in the path,
        including guard bands if configured.

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

        start_slot = self.spectrum_obj.spectrum_props.start_slot
        end_slot = self.spectrum_obj.spectrum_props.end_slot
        core_num = self.spectrum_obj.spectrum_props.core_number
        band = self.spectrum_obj.spectrum_props.current_band
        lightpath_id = self.spectrum_obj.spectrum_props.lightpath_id

        # Use lightpath_id if available, otherwise fall back to request_id
        if lightpath_id is None:
            lightpath_id = self.sdn_props.request_id
            if lightpath_id is None:
                raise ValueError("Neither lightpath_id nor request_id is available")

        if self.engine_props["guard_slots"] != 0:
            end_slot = end_slot - 1
        else:
            end_slot = end_slot + 1

        for link_tuple in zip(
            self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False
        ):
            link_dict = self.sdn_props.network_spectrum_dict[
                (link_tuple[0], link_tuple[1])
            ]
            reverse_link_dict = self.sdn_props.network_spectrum_dict[
                (link_tuple[1], link_tuple[0])
            ]

            spectrum_slots_set = set(
                link_dict["cores_matrix"][band][core_num][start_slot:end_slot]
            )
            reverse_spectrum_slots_set = set(
                reverse_link_dict["cores_matrix"][band][core_num][start_slot:end_slot]
            )

            if spectrum_slots_set == {} or reverse_spectrum_slots_set == {}:
                raise ValueError("Nothing detected on the spectrum when allocating.")

            if spectrum_slots_set != {0.0} or reverse_spectrum_slots_set != {0.0}:
                raise BufferError("Attempted to allocate a taken spectrum.")

            if link_tuple in self.sdn_props.network_spectrum_dict:
                self.sdn_props.network_spectrum_dict[link_tuple]["usage_count"] += 1
            self.sdn_props.network_spectrum_dict[(link_tuple[1], link_tuple[0])][
                "usage_count"
            ] += 1

            core_matrix = link_dict["cores_matrix"]
            reverse_core_matrix = reverse_link_dict["cores_matrix"]

            # Use lightpath_id instead of request_id
            core_matrix[band][core_num][start_slot:end_slot] = lightpath_id
            reverse_core_matrix[band][core_num][start_slot:end_slot] = lightpath_id

            if self.engine_props["guard_slots"]:
                self._allocate_guard_band(
                    band=band,
                    core_matrix=core_matrix,
                    reverse_core_matrix=reverse_core_matrix,
                    core_num=core_num,
                    end_slot=end_slot,
                    lightpath_id=lightpath_id,
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
            # Skip grooming-specific keys that are tracked directly in SDNProps
            # (not retrieved from SpectrumProps)
            if stat_key in (
                "lightpath_bandwidth_list",
                "lightpath_id_list",
                "remaining_bw",
            ):
                continue

            spectrum_key = stat_key.split("_", maxsplit=1)[0]
            if spectrum_key == "crosstalk":
                spectrum_key = "crosstalk_cost"
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

            self.sdn_props.update_params(
                key=stat_key,
                spectrum_key=spectrum_key,
                spectrum_obj=self.spectrum_obj.spectrum_props,
            )

    # Backward compatibility alias for tests
    def _update_req_stats(self, bandwidth: float | None = None) -> None:
        """Legacy method name for _update_request_statistics."""
        self._update_request_statistics(bandwidth)

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
            # Type ignore: remaining_bw can be list or int depending on context
            self.sdn_props.remaining_bw = (
                self.sdn_props.lightpath_bandwidth_list  # type: ignore[assignment]
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
            self.release()

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
    ) -> bool:
        """
        Process allocation for a single path.

        :param path_list: List of nodes in the routing path
        :type path_list: list[Any]
        :param path_index: Index of the current path
        :type path_index: int
        :param mod_format_list: List of modulation formats
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
            self.spectrum_obj.get_spectrum(mod_format_list=mod_format_list)

            if self.spectrum_obj.spectrum_props.is_free is not True:
                self.sdn_props.block_reason = "congestion"
                return False
            self._update_request_statistics(bandwidth=self.sdn_props.bandwidth)

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
                if hasattr(self.sdn_props, "lightpath_id_list") and isinstance(
                    self.sdn_props.lightpath_id_list, list
                ):
                    # Convert list[int] to list[int | None]
                    lightpath_id_list = list(self.sdn_props.lightpath_id_list)

            # If no lightpath IDs, release with None (uses request_id)
            if not lightpath_id_list:
                self.release(lightpath_id=None)
            else:
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
