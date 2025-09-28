"""
Last Fit spectrum assignment algorithm implementation.
"""

from typing import Any

import numpy as np

from fusion.core.properties import SpectrumProps
from fusion.interfaces.spectrum import AbstractSpectrumAssigner
from fusion.modules.spectrum.utils import SpectrumHelpers


class LastFitSpectrum(AbstractSpectrumAssigner):
    """
    Last Fit spectrum assignment algorithm.

    This algorithm assigns spectrum by finding the last (highest index)
    available contiguous set of slots that can accommodate the request.
    """

    def __init__(self, engine_props: dict, sdn_props: object, route_props: object):
        """
        Initialize Last Fit spectrum assignment algorithm.

        :param engine_props: Dictionary containing engine configuration
        :param sdn_props: Object containing SDN controller properties
        :param route_props: Object containing routing properties
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
        return "last_fit"

    @property
    def supports_multiband(self) -> bool:
        """Indicate whether this algorithm supports multi-band assignment."""
        return True

    def assign(self, path: list[Any], request: Any) -> dict[str, Any] | None:
        """
        Assign spectrum resources along the given path for the request.

        :param path: List of nodes representing the path
        :param request: Request object containing traffic demand and spectrum
            requirements
        :return: Dictionary with spectrum assignment details or None if fails
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

        # Try last fit allocation
        success = self._find_last_fit()

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

    def _find_last_fit(self) -> bool:
        """Find last available spectrum slots using last fit strategy."""
        # Set up cores and bands to check (in reverse order for last fit)
        if self.spectrum_props.forced_core is not None:
            core_list = [self.spectrum_props.forced_core]
        else:
            core_list = list(
                reversed(range(0, self.engine_props.get("cores_per_link", 1)))
            )

        if self.spectrum_props.forced_band is not None:
            band_list = [self.spectrum_props.forced_band]
        else:
            band_list = list(reversed(self.engine_props.get("band_list", ["c"])))

        # If we have path information, try direct link search
        path_list = getattr(self.spectrum_props, "path_list", None)
        if path_list is not None and len(path_list) > 1:
            return self._find_last_fit_on_path(core_list, band_list)

        return False

    def _find_last_fit_on_path(
        self, core_list: list[int], band_list: list[str]
    ) -> bool:
        """Find last fit spectrum assignment along the entire path."""
        best_assignment = None

        # Check all links in the path for spectrum availability
        path_list = self.spectrum_props.path_list
        for i in range(len(path_list) - 1):
            source = path_list[i]
            dest = path_list[i + 1]
            link_key = (source, dest)

            network_dict = self.sdn_props.network_spectrum_dict
            if link_key not in network_dict:
                continue

            link_dict = network_dict[link_key]

            # Try to find best assignment for this link
            link_assignment = self._find_best_assignment_for_link(
                link_dict, core_list, band_list
            )
            if link_assignment and (
                best_assignment is None
                or link_assignment["start_slot"] > best_assignment["start_slot"]
            ):
                best_assignment = link_assignment

        if best_assignment:
            self.spectrum_props.start_slot = best_assignment["start_slot"]
            self.spectrum_props.end_slot = best_assignment["end_slot"]
            self.spectrum_props.core_number = best_assignment["core_number"]
            self.spectrum_props.current_band = best_assignment["band"]
            self.spectrum_props.is_free = True
            return True

        return False

    def _find_best_assignment_for_link(
        self, link_dict: dict, core_list: list[int], band_list: list[str]
    ) -> dict | None:
        """Find the best assignment for a single link."""
        best_assignment = None

        # Try each core and band combination (in reverse order)
        for core_num in core_list:
            for band in band_list:
                assignment = self._try_core_band_assignment(link_dict, core_num, band)
                if assignment and (
                    best_assignment is None
                    or assignment["start_slot"] > best_assignment["start_slot"]
                ):
                    best_assignment = assignment

        return best_assignment

    def _try_core_band_assignment(
        self, link_dict: dict, core_num: int, band: str
    ) -> dict | None:
        """Try to find an assignment for a specific core and band."""
        if band not in link_dict["cores_matrix"]:
            return None

        core_array = link_dict["cores_matrix"][band][core_num]

        # Find last contiguous block that fits
        assignment = self._find_last_contiguous_block(core_array, core_num, band)
        if not assignment:
            return None

        # Verify this assignment works for entire path
        if self._verify_path_assignment(
            assignment["start_slot"], assignment["end_slot"], core_num, band
        ):
            return assignment

        return None

    def _find_last_contiguous_block(
        self, core_array: Any, core_num: int, band: str
    ) -> dict | None:
        """Find the last (highest index) contiguous block that fits the request."""
        if not hasattr(core_array, "__len__"):
            return None

        # Find all free slots (value 0)
        free_slots = np.where(np.array(core_array) == 0)[0]

        slots_needed = self.spectrum_props.slots_needed
        if len(free_slots) < slots_needed:
            return None

        # Search from the end (highest indices) backwards
        for i in range(len(free_slots) - slots_needed, -1, -1):
            start_slot = free_slots[i]
            required_slots = list(range(start_slot, start_slot + slots_needed))

            # Check if all required slots are available and contiguous
            if all(slot in free_slots for slot in required_slots):
                # Check continuity constraint
                is_contiguous = True
                for j in range(len(required_slots) - 1):
                    if required_slots[j + 1] - required_slots[j] != 1:
                        is_contiguous = False
                        break

                if is_contiguous:
                    return {
                        "start_slot": start_slot,
                        "end_slot": start_slot + slots_needed - 1,
                        "core_number": core_num,
                        "band": band,
                    }

        return None

    def _verify_path_assignment(
        self, start_slot: int, end_slot: int, core_num: int, band: str
    ) -> bool:
        """Verify spectrum assignment is available along entire path."""
        path_list = self.spectrum_props.path_list
        for i in range(len(path_list) - 1):
            source = path_list[i]
            dest = path_list[i + 1]

            if not self.check_spectrum_availability(
                [source, dest], start_slot, end_slot, core_num, band
            ):
                return False

        return True

    def check_spectrum_availability(
        self, path: list[Any], start_slot: int, end_slot: int, core_num: int, band: str
    ) -> bool:
        """Check if spectrum slots are available along the entire path."""
        for i in range(len(path) - 1):
            source, dest = path[i], path[i + 1]
            link_key = (source, dest)

            network_dict = self.sdn_props.network_spectrum_dict
            if link_key not in network_dict:
                return False

            link_dict = network_dict[link_key]

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

            network_dict = self.sdn_props.network_spectrum_dict
            if link_key not in network_dict:
                return False

            link_dict = network_dict[link_key]
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

            network_dict = self.sdn_props.network_spectrum_dict
            if link_key not in network_dict:
                return False

            link_dict = network_dict[link_key]
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

            network_dict = self.sdn_props.network_spectrum_dict
            if link_key in network_dict:
                link_dict = network_dict[link_key]
                link_fragmentation = self._calculate_link_fragmentation(link_dict)
                total_fragmentation += link_fragmentation
                link_count += 1

        return total_fragmentation / link_count if link_count > 0 else 0.0

    def _calculate_link_fragmentation(self, link_dict: dict) -> float:
        """Calculate fragmentation for a single link."""
        total_free_blocks = 0
        total_free_slots = 0

        for band in link_dict["cores_matrix"]:
            for core_array in link_dict["cores_matrix"][band]:
                blocks, free_slots = self._analyze_core_array(core_array)
                total_free_blocks += blocks
                total_free_slots += free_slots

        if total_free_slots == 0:
            return 0.0

        return 1.0 - (1.0 / total_free_blocks) if total_free_blocks > 0 else 0.0

    def _analyze_core_array(self, core_array: list) -> tuple[int, int]:
        """Analyze core array and return free blocks and total free slots."""
        free_slots = np.where(np.array(core_array) == 0)[0]

        if len(free_slots) == 0:
            return 0, 0

        # Count contiguous blocks
        blocks = self._count_contiguous_blocks(core_array)
        return blocks, len(free_slots)

    def _count_contiguous_blocks(self, core_array: list) -> int:
        """Count the number of contiguous free blocks in a core array."""
        blocks = 0
        in_block = False

        for slot in core_array:
            if slot == 0:  # Free slot
                if not in_block:
                    blocks += 1
                    in_block = True
            else:  # Occupied slot
                in_block = False

        return blocks

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
            "assignment_strategy": "last_available",
        }

    def reset(self) -> None:
        """Reset the spectrum assignment algorithm state."""
        self._assignments_made = 0
        self._total_slots_assigned = 0
        self.spectrum_props = SpectrumProps()
