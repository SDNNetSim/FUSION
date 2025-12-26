from collections.abc import Generator
from typing import Any

from fusion.utils.data import sort_dict_keys
from fusion.utils.logging_config import get_logger
from fusion.utils.network import find_path_length, get_path_modulation

# Need to access SDN controller's protected methods
# Some arguments/variables are kept for interface compatibility or future use

logger = get_logger(__name__)

# Backward compatibility aliases for tests
find_path_len = find_path_length
get_path_mod = get_path_modulation


class LightPathSlicingManager:
    """
    Manages light path segment slicing for optical network requests.

    This class handles the allocation of network requests using segment slicing
    strategies including static and dynamic slicing approaches.

    :param engine_props: Engine configuration properties
    :type engine_props: Dict[str, Any]
    :param sdn_props: SDN controller properties
    :type sdn_props: Any
    :param spectrum_obj: Spectrum assignment object
    :type spectrum_obj: Any
    """

    def __init__(
        self, engine_props: dict[str, Any], sdn_props: Any, spectrum_obj: Any
    ) -> None:
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.spectrum_obj = spectrum_obj

    def allocate_slicing(
        self, num_segments: int, mod_format: str, path_list: list[Any], bandwidth: str
    ) -> Generator[tuple[str, str | None], None, None]:
        """
        Allocate network request using segment slicing.

        :param num_segments: Number of segments to allocate
        :type num_segments: int
        :param mod_format: Modulation format to use
        :type mod_format: str
        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
        :param bandwidth: Bandwidth requirement for each segment
        :type bandwidth: str
        """
        self.sdn_props.number_of_transponders = num_segments
        self.spectrum_obj.spectrum_props.path_list = path_list
        mod_format_list = [mod_format]

        for _ in range(num_segments):
            self.spectrum_obj.get_spectrum(
                mod_format_list=mod_format_list, slice_bandwidth=bandwidth
            )
            if self.spectrum_obj.spectrum_props.is_free:
                # Delegate allocation back to SDN controller
                yield "allocate", bandwidth
            else:
                self.sdn_props.was_routed = False
                self.sdn_props.block_reason = "congestion"
                yield "release", None
                break

    def handle_static_slicing(
        self, path_list: list[Any], forced_segments: int
    ) -> Generator[tuple[str, str | None], None, bool]:
        """
        Handle static segment slicing for a network request.

        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
        :param forced_segments: Number of segments to force (-1 for auto)
        :type forced_segments: int
        :return: True if slicing was successful, False otherwise
        :rtype: bool
        """

        bandwidth_modulation_dict = sort_dict_keys(
            dictionary=self.engine_props["mod_per_bw"]
        )

        # Always use original request bandwidth for tier selection (matches v5)
        effective_bandwidth = self.sdn_props.bandwidth

        for bandwidth, mods_dict in bandwidth_modulation_dict.items():
            # We can't slice to a larger or equal bandwidth
            if int(bandwidth) >= int(effective_bandwidth):
                continue

            path_len = find_path_length(
                path_list=path_list, topology=self.engine_props["topology"]
            )
            mod_format = get_path_modulation(
                modulation_formats=mods_dict, path_length=path_len
            )
            if not mod_format or not isinstance(mod_format, str):
                continue

            self.sdn_props.was_routed = True
            num_segments = int(int(effective_bandwidth) / int(bandwidth))

            if num_segments > self.engine_props["max_segments"]:
                self.sdn_props.was_routed = False
                self.sdn_props.block_reason = "max_segments"
                break

            if forced_segments not in (-1, num_segments):
                self.sdn_props.was_routed = False
                continue

            # Process allocation through generator
            success = True
            for action, allocation_bandwidth in self.allocate_slicing(
                num_segments=num_segments,
                mod_format=mod_format,
                path_list=path_list,
                bandwidth=bandwidth,
            ):
                if action == "allocate":
                    yield "allocate", allocation_bandwidth
                elif action == "release":
                    yield "release", None
                    success = False
                    break

            if success and self.sdn_props.was_routed:
                self.sdn_props.is_sliced = True
                return True

            self.sdn_props.is_sliced = False

        return False

    def handle_static_slicing_direct(
        self, path_list: list[Any], forced_segments: int, sdn_controller: Any
    ) -> bool:
        """
        Handle static slicing using original logic with direct method calls.

        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
        :param forced_segments: Number of segments to force (-1 for auto)
        :type forced_segments: int
        :param sdn_controller: Reference to the SDN controller
        :type sdn_controller: Any
        :return: True if slicing was successful
        :rtype: bool
        """

        bandwidth_modulation_dict = sort_dict_keys(
            dictionary=self.engine_props["mod_per_bw"]
        )

        # Always use original request bandwidth for tier selection (matches v5)
        effective_bandwidth = self.sdn_props.bandwidth

        for bandwidth, mods_dict in bandwidth_modulation_dict.items():
            # We can't slice to a larger or equal bandwidth
            if int(bandwidth) >= int(effective_bandwidth):
                continue

            path_len = find_path_length(
                path_list=path_list, topology=self.engine_props["topology"]
            )
            mod_format = get_path_modulation(
                modulation_formats=mods_dict, path_length=path_len
            )
            if not mod_format or not isinstance(mod_format, str):
                continue

            self.sdn_props.was_routed = True
            num_segments = int(int(effective_bandwidth) / int(bandwidth))

            if num_segments > self.engine_props["max_segments"]:
                self.sdn_props.was_routed = False
                self.sdn_props.block_reason = "max_segments"
                break

            if forced_segments not in (-1, num_segments):
                self.sdn_props.was_routed = False
                continue

            # Use the original allocate_slicing logic directly
            self.sdn_props.number_of_transponders = num_segments
            self.spectrum_obj.spectrum_props.path_list = path_list
            mod_format_list = [mod_format]

            # Check if this is 1+1 protected
            backup_path_val = getattr(self.sdn_props, "backup_path", None)
            is_protected = backup_path_val is not None

            # TEMP: Force log to appear
            logger.debug(
                f"[DEBUG] Slicing: is_protected={is_protected}, "
                f"num_segments={num_segments}, bandwidth={bandwidth}"
            )

            for segment_idx in range(num_segments):
                logger.debug(
                    f"Slicing segment {segment_idx + 1}/{num_segments}: "
                    f"bandwidth={bandwidth}, mod_format={mod_format}"
                )
                self.spectrum_obj.get_spectrum(
                    mod_format_list=mod_format_list, slice_bandwidth=bandwidth
                )
                if self.spectrum_obj.spectrum_props.is_free:
                    # Generate unique lightpath ID for this segment
                    lp_id = self.sdn_props.get_lightpath_id()
                    self.spectrum_obj.spectrum_props.lightpath_id = lp_id
                    self.spectrum_obj.spectrum_props.lightpath_bandwidth = bandwidth
                    self.sdn_props.was_new_lp_established.append(lp_id)

                    sdn_controller.allocate()
                    sdn_controller._update_req_stats(bandwidth=bandwidth)
                else:
                    # Rollback previously allocated segments
                    remaining_bw = int(effective_bandwidth) - (segment_idx * int(bandwidth))

                    # FEATURE: Support partial serving (v5 behavior)
                    # If can_partially_serve is enabled and SOME segments were allocated, accept partial service
                    if self.engine_props.get("can_partially_serve", False):
                        # Check if any segments were allocated (segment_idx > 0)
                        if segment_idx > 0:
                            # Some segments were allocated
                            if (
                                self.sdn_props.was_partially_groomed or
                                self.sdn_props.path_index >= self.engine_props.get("k_paths", 1) - 1
                            ):
                                # Accept partial service
                                self.sdn_props.is_sliced = True
                                self.sdn_props.was_partially_routed = True
                                self.sdn_props.was_routed = True
                                self.sdn_props.remaining_bw = remaining_bw
                                return True

                    sdn_controller._handle_congestion(remaining_bw=remaining_bw)
                    break

            if self.sdn_props.was_routed:
                self.sdn_props.is_sliced = True
                return True

            self.sdn_props.is_sliced = False

        return False

    def handle_dynamic_slicing_direct(
        self,
        path_list: list[Any],
        path_index: int,
        forced_segments: int,
        sdn_controller: Any,
    ) -> bool:
        """
        Handle dynamic slicing using original logic with direct method calls.

        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
        :param path_index: Index of the current path being processed
        :type path_index: int
        :param forced_segments: Number of forced segments (unused)
        :type forced_segments: int
        :param sdn_controller: Reference to the SDN controller
        :type sdn_controller: Any
        :return: True if slicing was successful
        :rtype: bool
        """
        # Use remaining_bw if grooming occurred, otherwise use full bandwidth (matches v5)
        remaining_bw = (
            self.sdn_props.remaining_bw
            if self.sdn_props.was_partially_groomed
            else int(self.sdn_props.bandwidth)
        )

        _ = find_path_len(path_list=path_list, topology=self.engine_props["topology"])
        bw_mod_dict = sort_dict_keys(self.engine_props["mod_per_bw"])

        self.spectrum_obj.spectrum_props.path_list = path_list
        self.sdn_props.number_of_transponders = 0

        if self.engine_props["fixed_grid"]:
            # Fixed-grid dynamic slicing
            return self._handle_fixed_grid_dynamic_slicing(
                remaining_bw, path_index, sdn_controller
            )
        else:
            # Flex-grid dynamic slicing
            return self._handle_flex_grid_dynamic_slicing(
                remaining_bw, path_index, bw_mod_dict, sdn_controller
            )

    def _handle_fixed_grid_dynamic_slicing(
        self,
        remaining_bw: int,
        path_index: int,
        sdn_controller: Any,
    ) -> bool:
        """Handle fixed-grid dynamic slicing."""
        # DEBUG: Dump spectrum state for request 5 before slicing
        if self.sdn_props.request_id == 5:
            path = self.spectrum_obj.spectrum_props.path_list
            print(f"\n[LEGACY_SPECTRUM_STATE] req=5 BEFORE SLICING - path={path[:3] if path else 'None'}...")
            if path and len(path) >= 2 and self.sdn_props.network_spectrum_dict:
                link = (path[0], path[1])
                if link in self.sdn_props.network_spectrum_dict:
                    cores_matrix = self.sdn_props.network_spectrum_dict[link]["cores_matrix"]
                    for band, band_cores in cores_matrix.items():
                        for core_num, core_arr in enumerate(band_cores):
                            # Show first 10 slots
                            slots_preview = list(core_arr[:10])
                            occupied = [i for i, v in enumerate(slots_preview) if v != 0]
                            if occupied:
                                print(f"[LEGACY_SPECTRUM_STATE] link={link} band={band} core={core_num} slots[0:10]={slots_preview} occupied_indices={occupied}")

        initial_remaining = remaining_bw
        iteration = 0
        while remaining_bw > 0:
            iteration += 1
            self.sdn_props.was_routed = True
            _, bandwidth = self.spectrum_obj.get_spectrum_dynamic_slicing(
                _mod_format_list=[], path_index=path_index
            )

            # DEBUG: Show bandwidth values for req 40
            if self.sdn_props.request_id == 40:
                print(f"[LEGACY_DYN_SLICE] req=40 iter={iteration} remaining_bw={remaining_bw} bandwidth={bandwidth} is_free={self.spectrum_obj.spectrum_props.is_free}")

            if self.spectrum_obj.spectrum_props.is_free:
                lp_id = self.sdn_props.get_lightpath_id()
                self.spectrum_obj.spectrum_props.lightpath_id = lp_id
                dedicated_bw = min(bandwidth, remaining_bw)

                if self.sdn_props.was_partially_groomed:
                    lightpath_bw = bandwidth
                    stats_bw = str(dedicated_bw)
                    remaining_bw -= bandwidth
                else:
                    lightpath_bw = str(bandwidth)
                    stats_bw = str(dedicated_bw)
                    remaining_bw -= bandwidth

                self.spectrum_obj.spectrum_props.lightpath_bandwidth = lightpath_bw
                self.sdn_props.was_new_lp_established.append(lp_id)

                sdn_controller.allocate()
                sdn_controller._update_req_stats(bandwidth=stats_bw)
                self.spectrum_obj._update_lightpath_status()
                self.sdn_props.number_of_transponders += 1
                self.sdn_props.is_sliced = True
                self.sdn_props.remaining_bw = max(0, remaining_bw)

                # SNR recheck after allocation (v5 behavior)
                # This ensures lightpaths are validated immediately and rolled back
                # if SNR requirements are not met, preventing orphaned allocations
                if not sdn_controller._check_snr_after_allocation(lp_id):
                    # Rollback this lightpath and stop (matches v5 behavior)
                    self.sdn_props.was_routed = False
                    self.sdn_props.block_reason = "snr_recheck_failed"
                    remaining_bw += bandwidth
                    sdn_controller._handle_congestion(remaining_bw)
                    break
            else:
                if self.engine_props.get("can_partially_serve", False):
                    initial_bw = int(self.sdn_props.bandwidth)
                    if remaining_bw != initial_bw:
                        if (
                            self.sdn_props.was_partially_groomed
                            or self.sdn_props.path_index
                            >= self.engine_props.get("k_paths", 1) - 1
                        ):
                            self.sdn_props.is_sliced = True
                            self.sdn_props.was_partially_routed = True
                            self.sdn_props.remaining_bw = max(0, remaining_bw)
                            return True

                sdn_controller._handle_congestion(remaining_bw=remaining_bw)
                break

        total_allocated = initial_remaining - remaining_bw
        return bool(self.sdn_props.was_routed)

    def _handle_flex_grid_dynamic_slicing(
        self,
        remaining_bw: int,
        path_index: int,
        bw_mod_dict: dict[str, Any],
        sdn_controller: Any,
    ) -> bool:
        """Handle flex-grid dynamic slicing."""
        initial_bw = int(self.sdn_props.bandwidth)
        print(f"[LEGACY] req={self.sdn_props.request_id} FLEX_GRID_SLICING initial_bw={initial_bw} remaining_bw={remaining_bw}")

        for bandwidth_str, mods_dict in bw_mod_dict.items():
            # Skip bandwidth tiers >= request bandwidth
            if int(bandwidth_str) >= initial_bw:
                continue

            print(f"[LEGACY] req={self.sdn_props.request_id} TRYING_BW_TIER bw={bandwidth_str} remaining={remaining_bw}")
            iteration = 0
            while remaining_bw > 0:
                if remaining_bw < int(bandwidth_str):
                    break

                iteration += 1
                self.sdn_props.was_routed = True
                mod_format, _ = self.spectrum_obj.get_spectrum_dynamic_slicing(
                    _mod_format_list=[],
                    path_index=path_index,
                    mod_format_dict=mods_dict,
                )
                # In flex-grid slicing, bandwidth is pre-calculated from the tier
                bw = int(bandwidth_str)

                sp = self.spectrum_obj.spectrum_props
                print(f"[LEGACY] req={self.sdn_props.request_id} SLICE_ATTEMPT iter={iteration} bw_tier={bandwidth_str} is_free={sp.is_free} start={sp.start_slot} end={sp.end_slot} mod={sp.modulation}")

                if self.spectrum_obj.spectrum_props.is_free:
                    lp_id = self.sdn_props.get_lightpath_id()
                    self.spectrum_obj.spectrum_props.lightpath_id = lp_id
                    self.spectrum_obj.spectrum_props.lightpath_bandwidth = bw

                    sdn_controller.allocate()

                    dedicated_bw = bw if remaining_bw > bw else remaining_bw
                    sdn_controller._update_req_stats(bandwidth=str(dedicated_bw))
                    self.sdn_props.was_new_lp_established.append(lp_id)
                    self.spectrum_obj._update_lightpath_status()

                    remaining_bw -= bw
                    self.sdn_props.number_of_transponders += 1
                    self.sdn_props.is_sliced = True
                    self.sdn_props.was_partially_routed = False
                    self.sdn_props.remaining_bw = max(0, remaining_bw)

                    print(f"[LEGACY] req={self.sdn_props.request_id} SLICE_CREATED lp_id={lp_id} bw={dedicated_bw} remaining={remaining_bw}")

                    # SNR recheck after allocation (v5 behavior)
                    # This ensures lightpaths are validated immediately and rolled back
                    # if SNR requirements are not met, preventing orphaned allocations
                    if not sdn_controller._check_snr_after_allocation(lp_id):
                        # Rollback this lightpath and stop (matches v5 behavior)
                        self.sdn_props.was_routed = False
                        self.sdn_props.block_reason = "snr_recheck_failed"
                        remaining_bw += bw
                        print(f"[LEGACY] req={self.sdn_props.request_id} SLICE_SNR_FAIL lp_id={lp_id} rollback")
                        sdn_controller._handle_congestion(remaining_bw)
                        break
                else:
                    print(f"[LEGACY] req={self.sdn_props.request_id} SLICE_FAIL no_spectrum remaining={remaining_bw}")
                    break

            if remaining_bw <= 0:
                break

        if remaining_bw <= 0:
            if self.sdn_props.was_routed:
                self.sdn_props.is_sliced = True
                return True

        # Handle partial serving if enabled
        if self.engine_props.get("can_partially_serve", False):
            if remaining_bw != initial_bw:
                if (
                    self.sdn_props.was_partially_groomed
                    or self.sdn_props.path_index
                    >= self.engine_props.get("k_paths", 1) - 1
                ):
                    self.sdn_props.is_sliced = True
                    self.sdn_props.was_partially_routed = True
                    self.sdn_props.remaining_bw = max(0, remaining_bw)
                    return True

        if remaining_bw > 0:
            self.sdn_props.was_routed = False
            self.sdn_props.block_reason = "congestion"
            sdn_controller._handle_congestion(remaining_bw=remaining_bw)

        return bool(self.sdn_props.was_routed)

    def allocate_slicing_direct(
        self,
        num_segments: int,
        mod_format: str,
        path_list: list[Any],
        bandwidth: str,
        sdn_controller: Any,
    ) -> None:
        """
        Direct implementation of allocate slicing logic.

        :param num_segments: Number of segments to allocate
        :type num_segments: int
        :param mod_format: Modulation format to use
        :type mod_format: str
        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
        :param bandwidth: Bandwidth requirement for each segment
        :type bandwidth: str
        :param sdn_controller: Reference to the SDN controller
        :type sdn_controller: Any
        """
        self.sdn_props.number_of_transponders = num_segments
        self.spectrum_obj.spectrum_props.path_list = path_list
        remaining_bw = int(self.sdn_props.bandwidth)
        mod_format_list = [mod_format]
        for _ in range(num_segments):
            self.spectrum_obj.get_spectrum(
                mod_format_list=mod_format_list, slice_bandwidth=bandwidth
            )
            if self.spectrum_obj.spectrum_props.is_free:
                remaining_bw -= int(bandwidth)

                # Generate unique lightpath ID for this segment
                lp_id = self.sdn_props.get_lightpath_id()
                self.spectrum_obj.spectrum_props.lightpath_id = lp_id
                self.spectrum_obj.spectrum_props.lightpath_bandwidth = bandwidth
                self.sdn_props.was_new_lp_established.append(lp_id)

                sdn_controller.allocate()
                sdn_controller._update_req_stats(bandwidth=bandwidth, remaining=str(remaining_bw))
            else:
                # Rollback previously allocated segments
                remaining_bw_calc = int(self.sdn_props.bandwidth) - (int(self.sdn_props.bandwidth) - remaining_bw)
                sdn_controller._handle_congestion(remaining_bw=remaining_bw_calc)
                break

    def handle_dynamic_slicing(
        self, path_list: list[Any], path_index: int, forced_segments: int
    ) -> Generator[tuple[str, str | int | None], None, None]:
        """
        Handle dynamic slicing for a network request.

        Attempts to allocate bandwidth using dynamic slicing when traditional
        allocation fails due to fragmentation.

        :param path_list: List of nodes in the routing path
        :type path_list: List[Any]
        :param path_index: Index of the current path being processed
        :type path_index: int
        :param forced_segments: Number of segments to force (unused)
        :type forced_segments: int
        """
        remaining_bw = int(self.sdn_props.bandwidth)
        _ = find_path_len(path_list=path_list, topology=self.engine_props["topology"])
        _ = sort_dict_keys(self.engine_props["mod_per_bw"])

        self.spectrum_obj.spectrum_props.path_list = path_list
        self.sdn_props.number_of_transponders = 0

        while remaining_bw > 0:
            if not self.engine_props["fixed_grid"]:
                raise NotImplementedError(
                    "Dynamic slicing for non-fixed grid is not implemented."
                )

            self.sdn_props.was_routed = True
            _, bandwidth = self.spectrum_obj.get_spectrum_dynamic_slicing(
                _mod_format_list=[], path_index=path_index
            )

            if self.spectrum_obj.spectrum_props.is_free:
                yield "allocate", None
                dedicated_bw = min(bandwidth, remaining_bw)
                yield "update_stats", str(dedicated_bw)
                remaining_bw -= bandwidth
                self.sdn_props.number_of_transponders += 1
                self.sdn_props.is_sliced = True
            else:
                yield "handle_congestion", remaining_bw
                break
