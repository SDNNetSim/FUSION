import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from fusion.sim.utils import find_free_channels, find_free_slots, get_channel_overlaps

if TYPE_CHECKING:
    from fusion.core.properties import SpectrumProps


class SpectrumHelpers:
    """
    Helper utilities for spectrum assignment algorithms.

    This class provides utility methods for spectrum assignment operations
    including spectrum availability checking, band management, and slot allocation.

    :param engine_props: Dictionary containing engine configuration
    :param sdn_props: Object containing SDN controller properties
    :param spectrum_props: Object containing spectrum assignment properties
    """

    def __init__(
        self, engine_props: dict, sdn_props: Any, spectrum_props: "SpectrumProps"
    ) -> None:
        """
        Initialize spectrum helper utilities.

        :param engine_props: Dictionary containing engine configuration
        :param sdn_props: Object containing SDN controller properties
        :param spectrum_props: Object containing spectrum assignment properties
        """
        self.engine_props = engine_props
        self.spectrum_props = spectrum_props
        self.sdn_props = sdn_props

        self.start_index: int | None = None
        self.end_index: int | None = None
        self.core_number: int | None = None
        self.current_band: str | None = None

    def _check_free_spectrum(self, link_tuple: tuple, rev_link_tuple: tuple) -> bool:
        """
        Check if spectrum slots are free on both forward and reverse links.

        :param link_tuple: Tuple representing the forward link (source, destination)
        :param rev_link_tuple: Tuple representing the reverse link (destination, source)
        :return: True if spectrum slots are free on both links, False otherwise
        :raises ValueError: If spectrum set is empty
        """
        network_dict = self.sdn_props.network_spectrum_dict
        core_arr = network_dict[link_tuple]["cores_matrix"][self.current_band][
            self.core_number
        ]
        rev_core_arr = network_dict[rev_link_tuple]["cores_matrix"][self.current_band][
            self.core_number
        ]
        if (
            self.spectrum_props.slots_needed == 1
            and self.engine_props["guard_slots"] == 0
        ):
            return bool(
                core_arr[self.start_index] == 0.0
                and rev_core_arr[self.start_index] == 0.0
            )

        if self.engine_props["guard_slots"] == 0:
            tmp_end_index = (self.end_index or 0) + 1
        else:
            tmp_end_index = self.end_index or 0

        spectrum_set = core_arr[
            self.start_index : tmp_end_index + self.engine_props["guard_slots"]
        ]
        rev_spectrum_set = rev_core_arr[
            self.start_index : tmp_end_index + self.engine_props["guard_slots"]
        ]

        if len(spectrum_set) == 0 or len(rev_spectrum_set) == 0:
            raise ValueError("Spectrum set cannot be empty.")

        if set(spectrum_set) == {0.0} and set(rev_spectrum_set) == {0.0}:
            return True

        return False

    def check_other_links(self) -> None:
        """
        Check spectrum availability on remaining links in the path.

        This method validates that the spectrum slots identified as free on the
        first link are also available on all subsequent links in the path.

        Updates:
            spectrum_props.is_free: Set to False if any link has conflicting
                                    spectrum
        """
        self.spectrum_props.is_free = True
        path_list = self.spectrum_props.path_list
        if path_list is None:
            raise ValueError("path_list must not be None")
        for node in range(len(path_list) - 1):
            link_tuple = (
                path_list[node],
                path_list[node + 1],
            )
            rev_link_tuple = (
                path_list[node + 1],
                path_list[node],
            )

            if not self._check_free_spectrum(
                link_tuple=link_tuple, rev_link_tuple=rev_link_tuple
            ):
                self.spectrum_props.is_free = False
                return

    def _update_spec_props(self) -> Any:
        if getattr(self.spectrum_props, "forced_core", None) is not None:
            self.core_number = self.spectrum_props.forced_core

        if getattr(self.spectrum_props, "forced_band", None) is not None:
            self.current_band = self.spectrum_props.forced_band

        if self.engine_props["allocation_method"] == "last_fit":
            self.spectrum_props.start_slot = self.end_index
            self.spectrum_props.end_slot = (self.start_index or 0) + self.engine_props[
                "guard_slots"
            ]
        else:
            self.spectrum_props.start_slot = self.start_index
            self.spectrum_props.end_slot = (self.end_index or 0) + self.engine_props[
                "guard_slots"
            ]

        self.spectrum_props.core_number = self.core_number
        self.spectrum_props.current_band = self.current_band
        return self.spectrum_props

    def check_super_channels(self, open_slots_matrix: list, flag: str) -> bool:
        """
        Find available super-channel for current request allocation.

        :param open_slots_matrix: Matrix of available super-channel indexes
        :param flag: Allocation flag for forced index checking
        :return: True if request can be allocated, False otherwise
        :rtype: bool
        """
        if self.spectrum_props.slots_needed is None:
            raise ValueError("slots_needed must not be None")
        for super_channel in open_slots_matrix:
            if len(super_channel) >= (
                self.spectrum_props.slots_needed + self.engine_props["guard_slots"]
            ):
                for start_index in super_channel:
                    if flag == "forced_index" and start_index != getattr(
                        self.spectrum_props, "forced_index", None
                    ):
                        continue
                    self.start_index = start_index
                    if self.engine_props["allocation_method"] == "last_fit":
                        self.end_index = (
                            (self.start_index or 0)
                            - self.spectrum_props.slots_needed
                            - self.engine_props["guard_slots"]
                        ) + 1
                    else:
                        self.end_index = (
                            (self.start_index or 0)
                            + self.spectrum_props.slots_needed
                            + self.engine_props["guard_slots"]
                        ) - 1
                    if self.end_index not in super_channel:
                        break
                    self.spectrum_props.is_free = True

                    if self.spectrum_props.path_list is None:
                        raise ValueError("path_list must not be None")
                    if len(self.spectrum_props.path_list) > 2:
                        self.check_other_links()

                    if (
                        self.spectrum_props.is_free is not False
                        or len(self.spectrum_props.path_list) <= 2
                    ):
                        self._update_spec_props()
                        return True

        return False

    @staticmethod
    def _find_link_inters(info_dict: dict, source_dest: tuple) -> None:
        for core_num in info_dict["free_slots_dict"][source_dest]:
            if core_num not in info_dict["slots_inters_dict"]:
                tmp_dict = {
                    core_num: set(info_dict["free_slots_dict"][source_dest][core_num])
                }
                info_dict["slots_inters_dict"].update(tmp_dict)

                tmp_dict = {
                    core_num: set(
                        info_dict["free_channels_dict"][source_dest][core_num]
                    )
                }
                info_dict["channel_inters_dict"].update(tmp_dict)
            else:
                slot_inters_dict = info_dict["slots_inters_dict"][core_num]
                free_slots_set = set(
                    info_dict["free_slots_dict"][source_dest][core_num]
                )
                slot_inters = slot_inters_dict & free_slots_set
                info_dict["slots_inters_dict"][core_num] = slot_inters

                tmp_list = []
                for item in info_dict["channel_inters_dict"][core_num]:
                    if item in info_dict["free_channels_dict"][source_dest][core_num]:
                        tmp_list.append(item)

                info_dict["channel_inters_dict"][core_num] = tmp_list

    def find_link_inters(self) -> dict:
        """
        Find slots and channels with potential intersections.

        :return: Dictionary with free slots, channels and intersections
        :rtype: dict
        """
        info_dict: dict[str, dict] = {
            "free_slots_dict": {},
            "free_channels_dict": {},
            "slots_inters_dict": {},
            "channel_inters_dict": {},
        }

        path_list = self.spectrum_props.path_list
        if path_list is None:
            raise ValueError("path_list must not be None")
        for source_dest in zip(
            path_list,
            path_list[1:],
            strict=False,
        ):
            free_slots = find_free_slots(
                network_spectrum_dict=self.sdn_props.network_spectrum_dict,
                link_tuple=source_dest,
            )
            free_channels = find_free_channels(
                network_spectrum_dict=self.sdn_props.network_spectrum_dict,
                slots_needed=self.spectrum_props.slots_needed,
                link_tuple=source_dest,
            )

            info_dict["free_slots_dict"].update({source_dest: free_slots})
            info_dict["free_channels_dict"].update({source_dest: free_channels})

        return info_dict

    def find_best_core(self) -> int:
        """
        Finds the core with the least amount of overlapping super channels.

        :return: The core with the least amount of overlapping channels.
        :rtype: int
        """
        path_info = self.find_link_inters()
        all_channels = get_channel_overlaps(
            path_info["free_channels_dict"], path_info["free_slots_dict"]
        )
        overlapping_results = copy.deepcopy(all_channels[list(all_channels.keys())[0]])
        for _, channels in all_channels.items():
            for ch_type, channels_type in channels.items():
                for band, band_channels in channels_type.items():
                    for core_num, channel in band_channels.items():
                        tmp_dict = overlapping_results[ch_type][band][core_num]
                        overlapping_results[ch_type][band][core_num] = np.intersect1d(
                            tmp_dict, channel
                        )
        c_band_dict = overlapping_results["non_over_dict"]["c"]
        sorted_cores = sorted(c_band_dict, key=lambda k: len(c_band_dict[k]))

        # TODO: Comment why
        if len(sorted_cores) > 1:
            if 6 in sorted_cores:
                sorted_cores.remove(6)
        return int(sorted_cores[0])
