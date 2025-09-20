"""
Best Fit spectrum assignment algorithm implementation.
"""

# pylint: disable=duplicate-code

import itertools
from operator import itemgetter
from typing import Any

import numpy as np

from fusion.core.properties import SpectrumProps
from fusion.interfaces.spectrum import AbstractSpectrumAssigner
from fusion.modules.spectrum.utils import SpectrumHelpers


class BestFitSpectrum(AbstractSpectrumAssigner):
    """Best Fit spectrum assignment algorithm.

    This algorithm assigns spectrum by finding the smallest available contiguous
    set of slots that can accommodate the request, minimizing fragmentation.
    """

    def __init__(self, engine_props: dict, sdn_props: object, route_props: object):
        """Initialize Best Fit spectrum assignment algorithm.

        Args:
            engine_props: Dictionary containing engine configuration
            sdn_props: Object containing SDN controller properties
            route_props: Object containing routing properties
        """
        super().__init__(engine_props, sdn_props, route_props)
        self.spectrum_props = SpectrumProps()
        self.spec_help_obj = SpectrumHelpers(
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
        )
        self._assignments_made = 0
        self._total_slots_assigned = 0

    @property
    def algorithm_name(self) -> str:
        """Return the name of the spectrum assignment algorithm."""
        return "best_fit"

    @property
    def supports_multiband(self) -> bool:
        """Indicate whether this algorithm supports multi-band assignment."""
        return True

    def assign(self, path: list[Any], request: Any) -> dict[str, Any] | None:
        """Assign spectrum resources along the given path for the request.

        Args:
            path: List of nodes representing the path
            request: Request object containing traffic demand and spectrum requirements

        Returns:
            Dictionary containing spectrum assignment details or None if assignment fails
        """
        # Store path and request info
        self.spectrum_props.path_list = path

        # Get slots needed for the request
        if hasattr(request, "slots_needed"):
            self.spectrum_props.slots_needed = request.slots_needed
        elif hasattr(request, "bandwidth"):
            self.spectrum_props.slots_needed = self._calculate_slots_needed(
                request.bandwidth
            )
        else:
            self.spectrum_props.slots_needed = 1

        # Reset assignment state
        self.spectrum_props.is_free = False
        self.spectrum_props.start_slot = None
        self.spectrum_props.end_slot = None
        self.spectrum_props.core_number = None
        self.spectrum_props.current_band = None

        # Try best fit allocation
        success = self._find_best_fit()

        if success:
            self._assignments_made += 1
            self._total_slots_assigned += self.spectrum_props.slots_needed

            return {
                "start_slot": self.spectrum_props.start_slot,
                "end_slot": self.spectrum_props.end_slot,
                "core_number": self.spectrum_props.core_number,
                "band": self.spectrum_props.current_band,
                "is_free": self.spectrum_props.is_free,
                "slots_needed": self.spectrum_props.slots_needed,
            }

        return None

    def _calculate_slots_needed(self, bandwidth: float) -> int:
        """Calculate number of slots needed for given bandwidth."""
        slots_per_gbps = self.engine_props.get("slots_per_gbps", 1)
        return int(np.ceil(bandwidth * slots_per_gbps))

    def _find_best_fit(self) -> bool:
        """Find best fit spectrum assignment using the original algorithm logic."""
        channels_list = []

        # Get all potential super channels
        for i in range(len(self.spectrum_props.path_list) - 1):
            src = self.spectrum_props.path_list[i]
            dest = self.spectrum_props.path_list[i + 1]

            for core_num in range(self.engine_props.get("cores_per_link", 1)):
                if (
                    hasattr(self.spectrum_props, "forced_core")
                    and self.spectrum_props.forced_core is not None
                    and self.spectrum_props.forced_core != core_num
                ):
                    continue

                for band in self.engine_props.get("band_list", ["c"]):
                    if (
                        hasattr(self.spectrum_props, "forced_band")
                        and self.spectrum_props.forced_band is not None
                        and self.spectrum_props.forced_band != band
                    ):
                        continue

                    link_key = (src, dest)
                    if link_key not in self.sdn_props.network_spectrum_dict:
                        continue

                    link_dict = self.sdn_props.network_spectrum_dict[link_key]
                    if band not in link_dict["cores_matrix"]:
                        continue

                    core_arr = link_dict["cores_matrix"][band][core_num]
                    open_slots_arr = np.where(np.array(core_arr) == 0)[0]

                    # Group consecutive slots into channels
                    tmp_matrix = [
                        list(map(itemgetter(1), g))
                        for k, g in itertools.groupby(
                            enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1]
                        )
                    ]

                    for channel_list in tmp_matrix:
                        if len(channel_list) >= self.spectrum_props.slots_needed:
                            channels_list.append(
                                {
                                    "link": (src, dest),
                                    "core": core_num,
                                    "channel": channel_list,
                                    "band": band,
                                }
                            )

        # Sort the list of candidate super channels by size (best fit = smallest fit)
        channels_list = sorted(channels_list, key=lambda d: len(d["channel"]))

        return self._allocate_best_fit(channels_list)

    def _allocate_best_fit(self, channels_list: list) -> bool:
        """Allocate spectrum using best fit from sorted channel list."""
        for channel_dict in channels_list:
            for start_index in channel_dict["channel"]:
                end_index = (
                    start_index
                    + self.spectrum_props.slots_needed
                    + self.engine_props.get("guard_slots", 0)
                ) - 1

                if end_index not in channel_dict["channel"]:
                    break

                # Check if this assignment works for multi-hop paths
                if len(self.spectrum_props.path_list) > 2:
                    self.spec_help_obj.start_index = start_index
                    self.spec_help_obj.end_index = end_index
                    self.spec_help_obj.core_number = channel_dict["core"]
                    self.spec_help_obj.current_band = channel_dict["band"]
                    self.spec_help_obj.check_other_links()

                if (
                    self.spectrum_props.is_free
                    or len(self.spectrum_props.path_list) <= 2
                ):
                    self.spectrum_props.is_free = True
                    self.spectrum_props.start_slot = start_index
                    self.spectrum_props.end_slot = end_index
                    self.spectrum_props.core_number = channel_dict["core"]
                    self.spectrum_props.current_band = channel_dict["band"]
                    return True

        return False

    def check_spectrum_availability(
        self, path: list[Any], start_slot: int, end_slot: int, core_num: int, band: str
    ) -> bool:
        """Check if spectrum slots are available along the entire path."""
        for i in range(len(path) - 1):
            source, dest = path[i], path[i + 1]
            link_key = (source, dest)

            if link_key not in self.sdn_props.network_spectrum_dict:
                return False

            link_dict = self.sdn_props.network_spectrum_dict[link_key]

            if band not in link_dict["cores_matrix"] or core_num >= len(
                link_dict["cores_matrix"][band]
            ):
                return False

            core_array = link_dict["cores_matrix"][band][core_num]

            # Check if all required slots are free
            for slot in range(start_slot, end_slot + 1):
                if slot >= len(core_array) or core_array[slot] != 0:
                    return False

        return True

    def allocate_spectrum(
        self,
        path: list[Any],
        start_slot: int,
        end_slot: int,
        core_num: int,
        band: str,
        request_id: Any,
    ) -> bool:
        """Allocate spectrum resources along the path."""
        for i in range(len(path) - 1):
            source, dest = path[i], path[i + 1]
            link_key = (source, dest)

            if link_key not in self.sdn_props.network_spectrum_dict:
                return False

            link_dict = self.sdn_props.network_spectrum_dict[link_key]
            core_array = link_dict["cores_matrix"][band][core_num]

            # Allocate slots with request ID
            for slot in range(start_slot, end_slot + 1):
                core_array[slot] = request_id

        return True

    def deallocate_spectrum(
        self, path: list[Any], start_slot: int, end_slot: int, core_num: int, band: str
    ) -> bool:
        """Deallocate spectrum resources along the path."""
        for i in range(len(path) - 1):
            source, dest = path[i], path[i + 1]
            link_key = (source, dest)

            if link_key not in self.sdn_props.network_spectrum_dict:
                return False

            link_dict = self.sdn_props.network_spectrum_dict[link_key]
            core_array = link_dict["cores_matrix"][band][core_num]

            # Free slots by setting to 0
            for slot in range(start_slot, end_slot + 1):
                core_array[slot] = 0

        return True

    def get_fragmentation_metric(self, path: list[Any]) -> float:
        """Calculate fragmentation metric for the given path."""
        total_fragmentation = 0.0
        link_count = 0

        for i in range(len(path) - 1):
            source, dest = path[i], path[i + 1]
            link_key = (source, dest)

            if link_key in self.sdn_props.network_spectrum_dict:
                link_dict = self.sdn_props.network_spectrum_dict[link_key]
                link_fragmentation = self._calculate_link_fragmentation(link_dict)
                total_fragmentation += link_fragmentation
                link_count += 1

        return total_fragmentation / link_count if link_count > 0 else 0.0

    def _calculate_link_fragmentation(self, link_dict: dict) -> float:
        """Calculate fragmentation for a single link."""
        total_segments = 0
        total_free_slots = 0

        for band in link_dict["cores_matrix"]:
            for core_array in link_dict["cores_matrix"][band]:
                core_arr = np.array(core_array)
                free_slots = np.where(core_arr == 0)[0]

                if len(free_slots) > 0:
                    total_free_slots += len(free_slots)

                    # Count number of contiguous segments
                    segments = 1
                    for i in range(1, len(free_slots)):
                        if free_slots[i] - free_slots[i - 1] > 1:
                            segments += 1

                    total_segments += segments

        # Higher fragmentation = more segments for same number of free slots
        if total_free_slots == 0:
            return 0.0

        return total_segments / total_free_slots

    def get_metrics(self) -> dict[str, Any]:
        """Get spectrum assignment algorithm performance metrics."""
        avg_slots = (
            self._total_slots_assigned / self._assignments_made
            if self._assignments_made > 0
            else 0
        )

        return {
            "algorithm": self.algorithm_name,
            "assignments_made": self._assignments_made,
            "total_slots_assigned": self._total_slots_assigned,
            "average_slots_per_assignment": avg_slots,
            "supports_multiband": self.supports_multiband,
            "fragmentation_optimized": True,
        }

    def reset(self) -> None:
        """Reset the spectrum assignment algorithm state."""
        self._assignments_made = 0
        self._total_slots_assigned = 0
        self.spectrum_props = SpectrumProps()
