import time
from typing import Any

import numpy as np

from fusion.core.properties import SDNProps
from fusion.core.routing import Routing
from fusion.core.spectrum_assignment import SpectrumAssignment
# Removed unused imports: sort_dict_keys, get_path_mod, find_path_len
from fusion.modules.ml import get_ml_obs
from fusion.modules.spectrum.light_path_slicing import LightPathSlicingManager
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


class SDNController:
    """Software-defined network controller for managing network requests.

    This class provides functionality for routing, spectrum allocation, and resource
    management in software-defined optical networks. It handles request allocation,
    release, and various slicing strategies.

    :param engine_props: Engine configuration properties
    :type engine_props: Dict[str, Any]
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

    def release(self) -> None:
        """Remove a previously allocated request from the network.

        Deallocates spectrum resources and updates throughput statistics for
        the current request across all links in the path.
        """
        for source, dest in zip(
            self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False
        ):
            for band in self.engine_props["band_list"]:
                for core_num in range(self.engine_props["cores_per_link"]):
                    core_arr = self.sdn_props.network_spectrum_dict[(source, dest)][
                        "cores_matrix"
                    ][band][core_num]
                    request_id_indices = np.where(core_arr == self.sdn_props.request_id)
                    guard_band_indices = np.where(
                        core_arr == (self.sdn_props.request_id * -1)
                    )

                    for req_index in request_id_indices:
                        self.sdn_props.network_spectrum_dict[(source, dest)][
                            "cores_matrix"
                        ][band][core_num][req_index] = 0
                        self.sdn_props.network_spectrum_dict[(dest, source)][
                            "cores_matrix"
                        ][band][core_num][req_index] = 0
                    for gb_index in guard_band_indices:
                        self.sdn_props.network_spectrum_dict[(source, dest)][
                            "cores_matrix"
                        ][band][core_num][gb_index] = 0
                        self.sdn_props.network_spectrum_dict[(dest, source)][
                            "cores_matrix"
                        ][band][core_num][gb_index] = 0

        try:
            duration = self.sdn_props.depart - self.sdn_props.arrive  # seconds
            bandwidth = int(self.sdn_props.bandwidth)  # Gbps
            data_transferred = bandwidth * duration  # Gbps·s

            for source, dest in zip(
                self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False
            ):
                self.sdn_props.network_spectrum_dict[(source, dest)][
                    "throughput"
                ] += data_transferred
                self.sdn_props.network_spectrum_dict[(dest, source)][
                    "throughput"
                ] += data_transferred
        except (TypeError, ValueError) as e:
            logger.warning(
                "Throughput update skipped due to missing or invalid timing/bandwidth: %s",
                e,
            )

    def _allocate_gb(
        self,
        band: str,
        core_matrix: list[Any],
        rev_core_matrix: list[Any],
        core_num: int,
        end_slot: int,
    ) -> None:
        """Allocate guard band slots for spectrum isolation.

        :param band: Spectral band identifier
        :type band: str
        :param core_matrix: Core matrix for forward direction
        :type core_matrix: List[Any]
        :param rev_core_matrix: Core matrix for reverse direction
        :type rev_core_matrix: List[Any]
        :param core_num: Core number to allocate on
        :type core_num: int
        :param end_slot: End slot position for guard band
        :type end_slot: int
        :raises BufferError: If attempting to allocate already taken spectrum
        """
        if (
            core_matrix[band][core_num][end_slot] != 0.0
            or rev_core_matrix[band][core_num][end_slot] != 0.0
        ):
            raise BufferError("Attempted to allocate a taken spectrum.")

        core_matrix[band][core_num][end_slot] = self.sdn_props.request_id * -1
        rev_core_matrix[band][core_num][end_slot] = self.sdn_props.request_id * -1

    def allocate(self) -> None:
        """Allocate spectrum resources for a network request.

        Assigns spectrum slots to the current request across all links in the path,
        including guard bands if configured.

        :raises BufferError: If attempting to allocate already taken spectrum
        :raises ValueError: If no spectrum is detected during allocation
        """
        start_slot = self.spectrum_obj.spectrum_props.start_slot
        end_slot = self.spectrum_obj.spectrum_props.end_slot
        core_num = self.spectrum_obj.spectrum_props.core_number
        band = self.spectrum_obj.spectrum_props.current_band

        if self.engine_props["guard_slots"] != 0:
            end_slot = end_slot - 1
        else:
            end_slot += 1

        for link_tuple in zip(
            self.sdn_props.path_list, self.sdn_props.path_list[1:], strict=False
        ):
            # Remember, Python list indexing is up to and NOT including!
            link_dict = self.sdn_props.network_spectrum_dict[
                (link_tuple[0], link_tuple[1])
            ]
            rev_link_dict = self.sdn_props.network_spectrum_dict[
                (link_tuple[1], link_tuple[0])
            ]

            spectrum_slots_set = set(
                link_dict["cores_matrix"][band][core_num][start_slot:end_slot]
            )
            reverse_spectrum_slots_set = set(
                rev_link_dict["cores_matrix"][band][core_num][start_slot:end_slot]
            )

            if spectrum_slots_set == {} or reverse_spectrum_slots_set == {}:
                raise ValueError("Nothing detected on the spectrum when allocating.")

            if spectrum_slots_set != {0.0} or reverse_spectrum_slots_set != {0.0}:
                raise BufferError("Attempted to allocate a taken spectrum.")

            self.sdn_props.network_spectrum_dict[link_tuple]["usage_count"] += 1
            self.sdn_props.network_spectrum_dict[(link_tuple[1], link_tuple[0])][
                "usage_count"
            ] += 1

            core_matrix = link_dict["cores_matrix"]
            rev_core_matrix = rev_link_dict["cores_matrix"]
            core_matrix[band][core_num][start_slot:end_slot] = self.sdn_props.request_id
            rev_core_matrix[band][core_num][
                start_slot:end_slot
            ] = self.sdn_props.request_id

            if self.engine_props["guard_slots"]:
                self._allocate_gb(
                    core_matrix=core_matrix,
                    rev_core_matrix=rev_core_matrix,
                    end_slot=end_slot,
                    core_num=core_num,
                    band=band,
                )

    def _update_req_stats(self, bandwidth: str) -> None:
        """Update request statistics with allocation results.

        :param bandwidth: Allocated bandwidth for the request
        :type bandwidth: str
        """
        self.sdn_props.bandwidth_list.append(bandwidth)
        for stat_key in self.sdn_props.stat_key_list:
            spectrum_key = stat_key.split("_")[0]  # pylint: disable=use-maxsplit-arg
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

            self.sdn_props.update_params(
                key=stat_key, spectrum_key=spectrum_key, spectrum_obj=self.spectrum_obj
            )

    def _handle_slicing_request(
        self,
        path_list: list[Any],
        path_index: int,
        forced_segments: int,
        force_slicing: bool,
    ) -> bool:  # pylint: disable=unused-argument
        """Handle slicing request using the dedicated slicing manager.

        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
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

    def _handle_congestion(self, remaining_bw: int) -> None:
        """Handle allocation failure due to network congestion.

        :param remaining_bw: Remaining bandwidth that could not be allocated
        :type remaining_bw: int
        """
        self.sdn_props.was_routed = False
        self.sdn_props.block_reason = "congestion"
        self.sdn_props.number_of_transponders = 1

        if remaining_bw != int(self.sdn_props.bandwidth):
            self.release()

        self.sdn_props.is_sliced = False

    def _init_req_stats(self) -> None:
        """Initialize request statistics for a new request."""
        self.sdn_props.bandwidth_list = []
        self.sdn_props.reset_params()

    def _setup_routing(
        self, force_route_matrix: list[Any] | None, force_mod_format: str | None
    ) -> tuple[list[Any], float]:
        """Setup routing for the request.

        :param force_route_matrix: Optional forced routing matrix
        :type force_route_matrix: Optional[List[Any]]
        :param force_mod_format: Optional forced modulation format
        :type force_mod_format: Optional[str]
        :return: Tuple of (route_matrix, route_time)
        :rtype: tuple[List[Any], float]
        """
        start_time = time.time()
        if force_route_matrix is None:
            self.route_obj.get_route()
            route_matrix = self.route_obj.route_props.paths_matrix
        else:
            route_matrix = force_route_matrix
            self.route_obj.route_props.modulation_formats_matrix = [force_mod_format]
            self.route_obj.route_props.weights_list = [0]
        route_time = time.time() - start_time
        return route_matrix, route_time

    def _get_ml_prediction(
        self, ml_model: Any | None, request_dict: dict[str, Any]
    ) -> float:
        """Get ML model prediction for forced segments.

        :param ml_model: Optional machine learning model
        :type ml_model: Optional[Any]
        :param request_dict: Request dictionary
        :type request_dict: Dict[str, Any]
        :return: Forced segments prediction (-1 for auto)
        :rtype: float
        """
        if ml_model is not None:
            input_df = get_ml_obs(
                request_dict=request_dict,
                engine_properties=self.engine_props,
                sdn_properties=self.sdn_props,
            )
            return ml_model.predict(input_df)[0]
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
        """Process allocation for a single path.

        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
        :param path_index: Index of the current path
        :type path_index: int
        :param mod_format_list: List of modulation formats
        :type mod_format_list: List[str]
        :param forced_segments: Number of forced segments
        :type forced_segments: float
        :param force_slicing: Whether to force slicing
        :type force_slicing: bool
        :param segment_slicing: Whether segment slicing is enabled
        :type segment_slicing: bool
        :param forced_index: Optional forced spectrum index
        :type forced_index: Optional[int]
        :param force_core: Optional forced core number
        :type force_core: Optional[int]
        :param forced_band: Optional forced band
        :type forced_band: Optional[str]
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
            self._update_req_stats(bandwidth=self.sdn_props.bandwidth)

        return True

    def _finalize_successful_allocation(
        self,
        path_index: int,
        route_time: float,
        force_slicing: bool,
        segment_slicing: bool,
    ) -> None:
        """Finalize a successful allocation.

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
        """Handle any event that occurs in the simulation.

        Controls the main flow of request processing including routing, spectrum allocation,
        and various slicing strategies.

        :param request_dict: Request dictionary containing request parameters
        :type request_dict: Dict[str, Any]
        :param request_type: Type of request ('arrival' or 'release')
        :type request_type: str
        :param force_slicing: Whether to force light path segment slicing
        :type force_slicing: bool
        :param force_route_matrix: Optional forced routing matrix
        :type force_route_matrix: Optional[List[Any]]
        :param forced_index: Optional forced start index for spectrum allocation
        :type forced_index: Optional[int]
        :param force_mod_format: Optional forced modulation format
        :type force_mod_format: Optional[str]
        :param force_core: Optional forced core number
        :type force_core: Optional[int]
        :param ml_model: Optional machine learning model for predictions
        :type ml_model: Optional[Any]
        :param forced_band: Optional forced spectral band
        :type forced_band: Optional[str]
        """
        self._init_req_stats()
        self.sdn_props.number_of_transponders = 1

        if request_type == "release":
            self.release()
            return

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
        """Backward compatibility wrapper for allocate_slicing method.

        :param num_segments: Number of segments to allocate
        :type num_segments: int
        :param mod_format: Modulation format to use
        :type mod_format: str
        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
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
        """Backward compatibility wrapper for handle_dynamic_slicing method.

        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
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
