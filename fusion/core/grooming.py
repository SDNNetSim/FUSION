"""
Traffic grooming module for optical network simulations.

This module provides traffic grooming functionality to efficiently pack
multiple requests onto existing lightpaths, improving resource utilization.
"""

from typing import Any

from fusion.core.properties import GroomingProps
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


class Grooming:
    """
    Traffic grooming handler for optical networks.

    This class manages the grooming of network requests to existing lightpaths,
    including partial grooming and service release operations.

    :param engine_props: Engine configuration properties
    :type engine_props: dict[str, Any]
    :param sdn_props: SDN controller properties
    :type sdn_props: Any
    """

    def __init__(self, engine_props: dict[str, Any], sdn_props: Any) -> None:
        """
        Initialize grooming handler.

        :param engine_props: Engine configuration dictionary
        :type engine_props: dict[str, Any]
        :param sdn_props: SDN properties object
        :type sdn_props: Any
        """
        self.grooming_props = GroomingProps()
        self.engine_props = engine_props
        self.sdn_props = sdn_props

        logger.debug("Initialized grooming handler")

    def _find_path_max_bw(self, light_id: tuple) -> dict[str, Any] | None:
        """
        Find the path group with the maximum total remaining bandwidth.

        Groups lightpaths by their physical path and returns the group
        with the highest total remaining bandwidth. Skips degraded lightpaths.

        :param light_id: Tuple of (source, destination) representing the path ID
        :type light_id: tuple
        :return: Dictionary of the path group with maximum remaining bandwidth or None
        :rtype: dict[str, Any] | None
        """
        path_groups: dict[tuple, dict[str, Any]] = {}

        for lp_id, lp_info in self.sdn_props.lightpath_status_dict[light_id].items():
            # Skip lightpaths with degraded SNRs
            if lp_info.get("is_degraded", False):
                continue

            if lp_info["remaining_bandwidth"] > 0:
                path_key = tuple(lp_info["path"])
                reverse_path_key = tuple(reversed(lp_info["path"]))

                # Normalize path key (treat path and reverse as same group)
                normalized_path_key = min(path_key, reverse_path_key)

                if normalized_path_key not in path_groups:
                    path_groups[normalized_path_key] = {
                        "total_remaining_bandwidth": 0,
                        "lightpaths": [],
                        "lp_id_list": [],
                    }

                path_groups[normalized_path_key]["total_remaining_bandwidth"] += (
                    lp_info["remaining_bandwidth"]
                )
                path_groups[normalized_path_key]["lightpaths"].append((lp_id, lp_info))
                path_groups[normalized_path_key]["lp_id_list"].append(lp_id)

        # Find the path group with the maximum total remaining bandwidth
        if not path_groups:
            return None

        return max(
            path_groups.values(),
            key=lambda group: group["total_remaining_bandwidth"],
            default=None,
        )

    def _end_to_end_grooming(self) -> bool:
        """
        Groom arrival requests to already established lightpaths.

        Attempts to allocate the requested bandwidth using existing lightpaths
        between the source and destination. Supports partial grooming where
        part of the request is groomed and the remainder requires new lightpaths.

        :return: True if the request was fully groomed, False otherwise
        :rtype: bool
        """
        light_id = tuple(sorted([self.sdn_props.source, self.sdn_props.destination]))

        if light_id not in self.sdn_props.lightpath_status_dict:
            return False

        max_path_group = self._find_path_max_bw(light_id)
        if not max_path_group or max_path_group["total_remaining_bandwidth"] == 0:
            return False

        remaining_bw = int(self.sdn_props.bandwidth)

        for lp_id in max_path_group["lp_id_list"]:
            lp_info = self.sdn_props.lightpath_status_dict[light_id][lp_id]

            if lp_info["remaining_bandwidth"] == 0:
                continue

            # Determine how much bandwidth to allocate from this lightpath
            if lp_info["remaining_bandwidth"] > remaining_bw:
                tmp_remaining_bw = remaining_bw
                remaining_bw = 0
            else:
                tmp_remaining_bw = lp_info["remaining_bandwidth"]
                remaining_bw -= lp_info["remaining_bandwidth"]
                self.sdn_props.is_sliced = True

            # Update lightpath status
            lp_info["requests_dict"].update(
                {self.sdn_props.request_id: tmp_remaining_bw}
            )
            lp_info["remaining_bandwidth"] -= tmp_remaining_bw

            # Calculate utilization percentage
            lp_usage = 1 - (
                lp_info["remaining_bandwidth"] / lp_info["lightpath_bandwidth"]
            )
            lp_info["time_bw_usage"].update({self.sdn_props.arrive: lp_usage * 100})

            # Update SDN properties with this lightpath's allocation
            self.sdn_props.bandwidth_list.append(str(tmp_remaining_bw))
            self.sdn_props.core_list.append(lp_info["core"])
            self.sdn_props.band_list.append(lp_info["band"])
            self.sdn_props.start_slot_list.append(lp_info["start_slot"])
            self.sdn_props.end_slot_list.append(lp_info["end_slot"])
            self.sdn_props.modulation_list.append(lp_info["mod_format"])
            self.sdn_props.path_list = lp_info["path"]
            self.sdn_props.snr_list.append(lp_info["snr_cost"])
            self.sdn_props.xt_list.append(lp_info["xt_cost"])
            self.sdn_props.lightpath_bandwidth_list.append(
                lp_info["lightpath_bandwidth"]
            )
            self.sdn_props.lightpath_id_list.append(lp_id)
            self.sdn_props.path_weight = lp_info["path_weight"]

            if remaining_bw == 0:
                # Fully groomed - no new lightpath needed
                self.sdn_props.was_routed = True
                self.sdn_props.was_groomed = True
                self.sdn_props.was_partially_groomed = False
                self.sdn_props.number_of_transponders = 0
                self.sdn_props.was_new_lp_established = []
                self.sdn_props.remaining_bw = "0"

                logger.debug(
                    "Request %s fully groomed using %d lightpaths",
                    self.sdn_props.request_id,
                    len(max_path_group["lp_id_list"]),
                )
                return True

        # Partially groomed - some bandwidth still needs allocation
        self.sdn_props.was_partially_groomed = True
        self.sdn_props.was_groomed = False
        self.sdn_props.remaining_bw = remaining_bw

        logger.debug(
            "Request %s partially groomed, %d bandwidth remaining",
            self.sdn_props.request_id,
            remaining_bw,
        )
        return False

    def _release_service(self) -> list[int]:
        """
        Remove a previously allocated request from the lightpaths.

        Frees up bandwidth on lightpaths that were used by this request
        and identifies lightpaths that are now completely unused.

        :return: List of lightpath IDs that are no longer carrying any requests
        :rtype: list[int]
        """
        release_lp = []
        light_id = tuple(sorted([self.sdn_props.source, self.sdn_props.destination]))

        # Initialize remaining_bw for release tracking (not restored from arrival)
        if self.sdn_props.remaining_bw is None:
            self.sdn_props.remaining_bw = int(self.sdn_props.bandwidth)


        for lp_id in self.sdn_props.lightpath_id_list[:]:
            index = self.sdn_props.lightpath_id_list.index(lp_id)
            lp_info = self.sdn_props.lightpath_status_dict[light_id][lp_id]

            # Get allocated bandwidth for this request
            req_bw = lp_info["requests_dict"][self.sdn_props.request_id]

            # Remove request from lightpath
            lp_info["requests_dict"].pop(self.sdn_props.request_id)
            lp_info["remaining_bandwidth"] += req_bw
            self.sdn_props.remaining_bw = int(self.sdn_props.remaining_bw) - req_bw

            # Clean up tracking lists
            self.sdn_props.lightpath_id_list.pop(index)
            self.sdn_props.lightpath_bandwidth_list.pop(index)

            # Check if lightpath is now completely unused
            if lp_info["remaining_bandwidth"] == float(lp_info["lightpath_bandwidth"]):
                release_lp.append(lp_id)
            else:
                # Update utilization for partially used lightpath
                lp_usage = 1 - (
                    lp_info["remaining_bandwidth"] / float(lp_info["lightpath_bandwidth"])
                )
                lp_info["time_bw_usage"].update({self.sdn_props.depart: lp_usage * 100})

        logger.debug(
            "Released request %s from %d lightpaths, %d lightpaths now empty",
            self.sdn_props.request_id,
            len(self.sdn_props.lightpath_id_list),
            len(release_lp),
        )
        return release_lp

    def handle_grooming(self, request_type: str) -> bool | list[int]:
        """
        Control grooming operations based on request type.

        Entry point for grooming operations. Routes to appropriate
        method based on whether this is an arrival or release request.

        :param request_type: Type of request ("arrival" or "release")
        :type request_type: str
        :return: For arrivals: bool indicating if fully groomed.
                 For releases: list of lightpath IDs to release.
        :rtype: bool | list[int]
        """
        if request_type == "release":
            return self._release_service()
        return self._end_to_end_grooming()
