"""
Spectrum assignment module for optical network requests.

This module provides functionality for finding and allocating available spectrum
for optical network requests in software-defined optical networks.
"""

import itertools
from operator import itemgetter
from typing import Any

import numpy as np

from fusion.core.properties import RoutingProps, SDNProps, SpectrumProps
from fusion.core.snr_measurements import SnrMeasurements
from fusion.modules.spectrum.utils import SpectrumHelpers
from fusion.utils.data import sort_nested_dict_values
from fusion.utils.logging_config import get_logger
from fusion.utils.spectrum import find_common_channels_on_paths

logger = get_logger(__name__)


class SpectrumAssignment:
    """
    Find and allocate available spectrum for optical network requests.

    This class provides methods for spectrum assignment using various allocation
    strategies including best-fit, first-fit, last-fit, and priority-based allocation.

    :param engine_props: Engine configuration properties
    :type engine_props: dict[str, Any]
    :param sdn_props: SDN properties object
    :type sdn_props: SDNProps
    :param route_props: Routing properties object
    :type route_props: RoutingProps
    """

    def __init__(
        self,
        engine_props: dict[str, Any],
        sdn_props: SDNProps,
        route_props: RoutingProps,
    ) -> None:
        self.spectrum_props = SpectrumProps()
        self.engine_props_dict = engine_props
        self.sdn_props = sdn_props
        self.route_props = route_props

        self.snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props_dict,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props,
        )
        self.spectrum_helpers = SpectrumHelpers(
            engine_props=self.engine_props_dict,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
        )

    def _allocate_best_fit_spectrum(
        self, candidate_channels_list: list[dict[str, Any]]
    ) -> None:
        """
        Allocate spectrum using best-fit strategy.

        :param candidate_channels_list: List of candidate channel dictionaries
        :type candidate_channels_list: list[dict[str, Any]]
        """
        for channel_dict in candidate_channels_list:
            for start_index in channel_dict["channel"]:
                end_index = (
                    start_index
                    + self.spectrum_props.slots_needed
                    + self.engine_props_dict["guard_slots"]
                ) - 1
                if end_index not in channel_dict["channel"]:
                    break

                if (
                    self.spectrum_props.path_list is not None
                    and len(self.spectrum_props.path_list) > 2
                ):
                    self.spectrum_helpers.start_index = start_index
                    self.spectrum_helpers.end_index = end_index
                    self.spectrum_helpers.core_number = channel_dict["core"]
                    self.spectrum_helpers.current_band = channel_dict["band"]
                    self.spectrum_helpers.check_other_links()

                if self.spectrum_props.is_free or (
                    self.spectrum_props.path_list is not None
                    and len(self.spectrum_props.path_list) <= 2
                ):
                    self.spectrum_props.is_free = True
                    self.spectrum_props.start_slot = start_index
                    self.spectrum_props.end_slot = (
                        end_index + self.engine_props_dict["guard_slots"]
                    )
                    self.spectrum_props.end_slot = end_index
                    self.spectrum_props.core_number = channel_dict["core"]
                    self.spectrum_props.current_band = channel_dict["band"]
                    return

    def find_best_fit(self) -> None:
        """Search for and allocate best-fit super channel on each link along path."""
        candidate_channels_list = []

        # Get all potential super channels
        if self.spectrum_props.path_list is None:
            raise ValueError("Path list must be initialized")

        for source_node, destination_node in zip(
            self.spectrum_props.path_list[:-1],
            self.spectrum_props.path_list[1:],
            strict=False,
        ):
            for core_number in range(self.engine_props_dict["cores_per_link"]):
                if (
                    self.spectrum_props.forced_core is not None
                    and self.spectrum_props.forced_core != core_number
                ):
                    continue

                for band in self.engine_props_dict["band_list"]:
                    if (
                        self.spectrum_props.forced_band is not None
                        and self.spectrum_props.forced_band != band
                    ):
                        continue

                    if self.sdn_props.network_spectrum_dict is None:
                        raise ValueError("Network spectrum dict must be initialized")

                    core_spectrum_array = self.sdn_props.network_spectrum_dict[
                        (source_node, destination_node)
                    ]["cores_matrix"][band][core_number]
                    available_slots_array = np.where(core_spectrum_array == 0)[0]

                    contiguous_blocks_matrix = [
                        list(map(itemgetter(1), group))
                        for key, group in itertools.groupby(
                            enumerate(available_slots_array),
                            lambda index_slot: index_slot[0] - index_slot[1],
                        )
                    ]
                    for contiguous_channel_list in contiguous_blocks_matrix:
                        slots_needed = self.spectrum_props.slots_needed
                        if (
                            slots_needed is not None
                            and len(contiguous_channel_list) >= slots_needed
                        ):
                            candidate_channels_list.append(
                                {
                                    "link": (source_node, destination_node),
                                    "core": core_number,
                                    "channel": contiguous_channel_list,
                                    "band": band,
                                }
                            )

        # Sort the list of candidate super channels
        candidate_channels_list = sorted(
            candidate_channels_list,
            key=lambda channel_dict: len(channel_dict["channel"]),
        )
        self._allocate_best_fit_spectrum(
            candidate_channels_list=candidate_channels_list
        )

    def _get_cores_and_bands_lists(self) -> tuple[list[int], list[str]]:
        """
        Get core numbers and band lists based on forced values and allocation method.

        This is shared logic used by both normal and protected spectrum allocation.

        :return: Tuple containing core numbers list and available bands list
        :rtype: tuple[list[int], list[str]]
        """
        if self.spectrum_props.forced_core is not None:
            core_numbers_list = [self.spectrum_props.forced_core]
        elif self.engine_props_dict["allocation_method"] in (
            "priority_first",
            "priority_last",
        ):
            core_numbers_list = [0, 2, 4, 1, 3, 5, 6]
        else:
            core_numbers_list = list(range(0, self.engine_props_dict["cores_per_link"]))

        if self.spectrum_props.forced_band is not None:
            available_bands_list = [self.spectrum_props.forced_band]
        else:
            available_bands_list = self.engine_props_dict["band_list"]

        return core_numbers_list, available_bands_list

    def _setup_first_last_allocation(
        self,
    ) -> tuple[list[list[np.ndarray]], list[int], list[str]]:
        """
        Setup matrices for first/last allocation strategies.

        :return: Tuple containing cores spectrum matrix, core numbers list, band list
        :rtype: tuple[list[list[np.ndarray]], list[int], list[str]]
        """
        cores_spectrum_matrix = []

        core_numbers_list, available_bands_list = self._get_cores_and_bands_lists()

        if self.spectrum_props.cores_matrix is None:
            raise ValueError("Cores matrix must be initialized")

        for current_core_number in core_numbers_list:
            cores_spectrum_matrix.append(
                [
                    self.spectrum_props.cores_matrix[band][current_core_number]  # type: ignore
                    for band in available_bands_list
                ]
            )

        return (
            cores_spectrum_matrix,
            core_numbers_list,
            self.engine_props_dict["band_list"],
        )

    def _get_available_slots_matrix(
        self, available_slots_array: np.ndarray, allocation_flag: str
    ) -> list[list[int]]:
        """
        Convert array of available slots into matrix of contiguous blocks.

        Based on allocation flag.

        :param available_slots_array: Array of available slot indices.
        :type available_slots_array: np.ndarray
        :param allocation_flag: Allocation method flag (e.g., 'first_fit', 'last_fit').
        :type allocation_flag: str
        :return: A matrix of contiguous available slot blocks.
        :rtype: list
        """
        if allocation_flag in ("last_fit", "priority_last"):
            return [
                list(map(itemgetter(1), group))[::-1]
                for key, group in itertools.groupby(
                    enumerate(available_slots_array),
                    lambda index_slot: index_slot[0] - index_slot[1],
                )
            ]
        if allocation_flag in ("first_fit", "priority_first", "forced_index"):
            return [
                list(map(itemgetter(1), group))
                for key, group in itertools.groupby(
                    enumerate(available_slots_array),
                    lambda index_slot: index_slot[0] - index_slot[1],
                )
            ]

        raise NotImplementedError(
            f"Invalid allocation flag, got: {allocation_flag} and expected "
            f"'last_fit' or 'first_fit'."
        )

    def handle_first_last_allocation(self, allocation_flag: str) -> None:
        """
        Handle first-fit or last-fit allocation without priority or SNR."

        :param allocation_flag: A flag to determine which allocation method to be used
        :type allocation_flag: str
        """
        (cores_spectrum_matrix, core_numbers_list, band_list) = (
            self._setup_first_last_allocation()
        )

        for core_spectrum_array, core_number in zip(
            cores_spectrum_matrix, core_numbers_list, strict=False
        ):
            for band_index, band in enumerate(band_list):
                available_slots_array = np.where(core_spectrum_array[band_index] == 0)[
                    0
                ]
                available_slots_matrix = self._get_available_slots_matrix(
                    available_slots_array, allocation_flag
                )

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(
                    open_slots_matrix=available_slots_matrix, flag=allocation_flag
                )
                if was_allocated:
                    return

    def handle_first_last_priority_bsc(self, allocation_flag: str) -> None:
        """
        Handle first-fit or last-fit allocation with multi-band priority (BSC).

        :param allocation_flag: A flag to determine which allocation method to be used
        :type allocation_flag: str
        """
        (cores_spectrum_matrix, core_numbers_list, band_list) = (
            self._setup_first_last_allocation()
        )

        for band_index, band in enumerate(band_list):
            for core_spectrum_array, core_number in zip(
                cores_spectrum_matrix, core_numbers_list, strict=False
            ):
                available_slots_array = np.where(core_spectrum_array[band_index] == 0)[
                    0
                ]
                available_slots_matrix = self._get_available_slots_matrix(
                    available_slots_array, allocation_flag
                )

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(
                    open_slots_matrix=available_slots_matrix, flag=allocation_flag
                )
                if was_allocated:
                    if (
                        self.engine_props_dict["cores_per_link"] in [13, 19]
                        and self.engine_props_dict["snr_type"]
                        == "snr_e2e_external_resources"
                    ):
                        if self._handle_snr_external_resources(
                            allocation_flag, available_slots_matrix
                        ):
                            return

                        self.spectrum_props.is_free = False
                        continue

                    return

    def handle_first_last_priority_band(self, allocation_flag: str) -> None:
        """
        Handle first-fit or last-fit allocation with band priority (non-BSC).

        :param allocation_flag: A flag to determine which allocation method to be used
        :type allocation_flag: str
        """
        (cores_spectrum_matrix, core_numbers_list, band_list) = (
            self._setup_first_last_allocation()
        )

        for core_spectrum_array, core_number in zip(
            cores_spectrum_matrix, core_numbers_list, strict=False
        ):
            for band_index, band in enumerate(band_list):
                available_slots_array = np.where(core_spectrum_array[band_index] == 0)[
                    0
                ]

                available_slots_matrix = self._get_available_slots_matrix(
                    available_slots_array, allocation_flag
                )

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(
                    open_slots_matrix=available_slots_matrix, flag=allocation_flag
                )
                if was_allocated:
                    if (
                        self.engine_props_dict["cores_per_link"] in [13, 19]
                        and self.engine_props_dict["snr_type"]
                        == "snr_e2e_external_resources"
                    ):
                        if self._handle_snr_external_resources(
                            allocation_flag, available_slots_matrix
                        ):
                            return

                        self.spectrum_props.is_free = False
                        continue
                    return

    def _handle_snr_external_resources(
        self, allocation_flag: str, available_slots_matrix: list[list[int]]
    ) -> bool:
        """
        Handle SNR external resource checks during allocation.

        :param allocation_flag: Allocation flag (e.g., 'first_fit', 'last_fit')
        :type allocation_flag: str
        :param available_slots_matrix: Matrix of available slot blocks
        :type available_slots_matrix: list[list[int]]
        :return: Whether the allocation was successful
        :rtype: bool
        """

        for slots_row in available_slots_matrix:
            while slots_row:
                if self.sdn_props.path_index is None:
                    raise ValueError(
                        "Path index must be initialized for external SNR checks"
                    )
                slots_row = self.snr_measurements.check_snr_ext_open_slots(
                    self.sdn_props.path_index, slots_row
                )
                if slots_row:
                    was_allocated = self.spectrum_helpers.check_super_channels(
                        open_slots_matrix=[slots_row], flag=allocation_flag
                    )
                    if was_allocated:
                        return True

                    break

                break

        return False

    def handle_crosstalk_aware_allocation(self) -> None:
        """
        Allocate request with minimum cross-talk interference on neighboring cores.

        NOTE: Current implementation only supports 7-core configurations.
        """
        best_core_number = self.spectrum_helpers.find_best_core()
        if best_core_number in [0, 2, 4, 6]:
            self.spectrum_props.forced_core = best_core_number
            return self.handle_first_last_allocation(allocation_flag="first_fit")

        return self.handle_first_last_allocation(allocation_flag="last_fit")

    def _determine_spectrum_allocation(self) -> None:
        """Determine spectrum allocation method based on engine properties."""
        if self.spectrum_props.forced_index is not None:
            self.handle_first_last_allocation(allocation_flag="forced_index")
        elif self.engine_props_dict["allocation_method"] == "best_fit":
            self.find_best_fit()
        elif self.engine_props_dict["allocation_method"] in (
            "first_fit",
            "last_fit",
            "priority_first",
            "priority_last",
        ):
            if self.engine_props_dict["spectrum_priority"] == "BSC":
                self.handle_first_last_priority_bsc(
                    allocation_flag=self.engine_props_dict["allocation_method"]
                )
            else:
                self.handle_first_last_priority_band(
                    allocation_flag=self.engine_props_dict["allocation_method"]
                )
        elif self.engine_props_dict["allocation_method"] == "xt_aware":
            self.handle_crosstalk_aware_allocation()
        else:
            raise NotImplementedError(
                f"Expected first_fit or best_fit, got: "
                f"{self.engine_props_dict['allocation_method']}"
            )

    def _initialize_spectrum_information(self) -> None:
        """Initialize spectrum information for the request."""
        # Reset properties to prevent carryover from previous requests
        self.spectrum_props.lightpath_bandwidth = None
        self.spectrum_props.crosstalk_cost = None
        self.spectrum_props.lightpath_id = None
        self.spectrum_props.modulation = None
        self.spectrum_props.start_slot = None
        self.spectrum_props.end_slot = None
        self.spectrum_props.slots_needed = None
        self.spectrum_props.current_band = None
        self.spectrum_props.core_number = None
        self.spectrum_props.is_free = False
        self.spectrum_props.slicing_flag = False  # Reset slicing flag for each request
        path_list = self.spectrum_props.path_list
        if path_list is None or len(path_list) < 2:
            raise ValueError("Path list must be initialized with at least 2 nodes")

        forward_link_tuple = (
            path_list[0],
            path_list[1],
        )
        reverse_link_tuple = (
            path_list[1],
            path_list[0],
        )
        if self.sdn_props.network_spectrum_dict is None:
            raise ValueError("Network spectrum dict must be initialized")

        self.spectrum_props.cores_matrix = self.sdn_props.network_spectrum_dict[
            forward_link_tuple
        ]["cores_matrix"]
        self.spectrum_props.reverse_cores_matrix = self.sdn_props.network_spectrum_dict[
            reverse_link_tuple
        ]["cores_matrix"]
        self.spectrum_props.is_free = False

    def _find_protected_spectrum(
        self, primary_path: list[int], backup_path: list[int]
    ) -> None:
        """
        Find spectrum available on both primary and backup paths for 1+1 protection.

        Dispatches to appropriate method based on spectrum_priority configuration.

        :param primary_path: Primary path as list of node IDs
        :type primary_path: list[int]
        :param backup_path: Backup path as list of node IDs
        :type backup_path: list[int]
        """
        if self.sdn_props.network_spectrum_dict is None:
            raise ValueError("Network spectrum dict must be initialized")

        if self.spectrum_props.slots_needed is None:
            raise ValueError(
                "Slots needed must be set before finding protected spectrum"
            )

        # Dispatch based on spectrum priority (same pattern as normal allocation)
        if self.engine_props_dict.get("spectrum_priority") == "BSC":
            self._find_protected_spectrum_bsc(primary_path, backup_path)
        else:
            self._find_protected_spectrum_band(primary_path, backup_path)

    def _find_protected_spectrum_bsc(
        self, primary_path: list[int], backup_path: list[int]
    ) -> None:
        """
        Find protected spectrum with band-first priority (BSC).

        Iterates bands in outer loop, cores in inner loop. Matches the pattern
        of handle_first_last_priority_bsc for consistency.

        :param primary_path: Primary path as list of node IDs
        :type primary_path: list[int]
        :param backup_path: Backup path as list of node IDs
        :type backup_path: list[int]
        """
        core_numbers_list, available_bands_list = self._get_cores_and_bands_lists()

        # Band-first priority: iterate bands in outer loop
        for band in available_bands_list:
            for core in core_numbers_list:
                if self._try_protected_allocation(
                    primary_path, backup_path, band, core
                ):
                    return

        # No spectrum found on any core/band combination
        self.spectrum_props.is_free = False
        self.sdn_props.block_reason = "no_common_spectrum"

    def _find_protected_spectrum_band(
        self, primary_path: list[int], backup_path: list[int]
    ) -> None:
        """
        Find protected spectrum with core-first priority (non-BSC).

        Iterates cores in outer loop, bands in inner loop. Matches the pattern
        of handle_first_last_priority_band for consistency.

        :param primary_path: Primary path as list of node IDs
        :type primary_path: list[int]
        :param backup_path: Backup path as list of node IDs
        :type backup_path: list[int]
        """
        core_numbers_list, available_bands_list = self._get_cores_and_bands_lists()

        # Core-first priority: iterate cores in outer loop
        for core in core_numbers_list:
            for band in available_bands_list:
                if self._try_protected_allocation(
                    primary_path, backup_path, band, core
                ):
                    return

        # No spectrum found on any core/band combination
        self.spectrum_props.is_free = False
        self.sdn_props.block_reason = "no_common_spectrum"

    def _try_protected_allocation(
        self, primary_path: list[int], backup_path: list[int], band: str, core: int
    ) -> bool:
        """
        Try to allocate protected spectrum on specific band and core.

        :param primary_path: Primary path as list of node IDs
        :type primary_path: list[int]
        :param backup_path: Backup path as list of node IDs
        :type backup_path: list[int]
        :param band: Spectrum band identifier
        :type band: str
        :param core: Core number
        :type core: int
        :return: True if allocation successful, False otherwise
        :rtype: bool
        """
        if (
            self.sdn_props.network_spectrum_dict is None
            or self.spectrum_props.slots_needed is None
        ):
            return False

        # Find common available slot starting indices on both paths
        common_starts = find_common_channels_on_paths(
            network_spectrum_dict=self.sdn_props.network_spectrum_dict,
            paths=[primary_path, backup_path],
            slots_needed=self.spectrum_props.slots_needed,
            band=band,
            core=core,
        )

        if not common_starts:
            return False

        # Use first available common slot range
        start_slot = common_starts[0]
        end_slot = start_slot + self.spectrum_props.slots_needed

        # Set spectrum_props (same pattern as existing allocation methods)
        self.spectrum_props.start_slot = start_slot
        self.spectrum_props.end_slot = end_slot
        self.spectrum_props.core_number = core
        self.spectrum_props.current_band = band
        self.spectrum_props.is_free = True

        # Store backup path for allocation phase
        self.spectrum_props.backup_path = backup_path

        logger.debug(
            f"1+1 protection: Found common spectrum slots {start_slot}-{end_slot} "
            f"on band {band}, core {core}"
        )

        return True

    def _calculate_slots_needed(
        self, modulation: str, slice_bandwidth: str | None = None
    ) -> int | None:
        """
        Calculate slots needed for modulation and bandwidth.

        Handles special case for partial grooming where remaining bandwidth
        needs to be rounded up to the next available bandwidth tier.

        :param modulation: Modulation format
        :type modulation: str
        :param slice_bandwidth: Bandwidth for slicing
        :type slice_bandwidth: str | None
        :return: Number of slots needed, or None if no tier available
        :rtype: int | None
        """
        if self.engine_props_dict["fixed_grid"]:
            return 1

        # Handle partial grooming
        if self.sdn_props.was_partially_groomed:
            if self.sdn_props.remaining_bw is None:
                raise ValueError("Remaining bandwidth must be set for partial grooming")

            remaining_bw = int(self.sdn_props.remaining_bw)

            # Find next higher bandwidth tier
            available_bw_tiers = [
                int(k)
                for k in self.engine_props_dict["mod_per_bw"].keys()
                if int(k) >= remaining_bw
            ]

            if not available_bw_tiers:
                return None

            bw_tmp = min(available_bw_tiers)
            slots_needed: int = self.engine_props_dict["mod_per_bw"][str(bw_tmp)][
                modulation
            ]["slots_needed"]

            return slots_needed

        # Standard case
        if slice_bandwidth:
            result: int = self.engine_props_dict["mod_per_bw"][slice_bandwidth][
                modulation
            ]["slots_needed"]
            return result

        if self.sdn_props.modulation_formats_dict is None:
            raise ValueError("Modulation formats dict must be initialized")

        return int(self.sdn_props.modulation_formats_dict[modulation]["slots_needed"])

    def _update_lightpath_status(self) -> None:
        """
        Update lightpath status dictionary after allocation.

        Called after spectrum is allocated to track the new lightpath
        for future grooming operations and dynamic slicing bandwidth tracking.
        """
        # Only skip if both grooming and dynamic_lps are disabled
        if (not self.engine_props_dict.get("is_grooming_enabled", False) and
            not self.engine_props_dict.get("dynamic_lps", False)):
            return

        if self.sdn_props.source is None or self.sdn_props.destination is None:
            raise ValueError("Source and destination must be initialized")

        light_id = tuple(sorted([self.sdn_props.source, self.sdn_props.destination]))
        lp_id = self.spectrum_props.lightpath_id

        if lp_id is None:
            raise ValueError("Lightpath ID must be initialized")

        # Only track NEW lightpaths (matching v5 behavior)
        # Groomed requests reuse existing lightpaths and shouldn't create new entries
        was_new_lps = getattr(self.sdn_props, "was_new_lp_established", [])
        if lp_id not in was_new_lps:
            return

        # Initialize light_id entry if needed
        if self.sdn_props.lightpath_status_dict is None:
            self.sdn_props.lightpath_status_dict = {}

        if light_id not in self.sdn_props.lightpath_status_dict:
            self.sdn_props.lightpath_status_dict[light_id] = {}

        # Get lightpath bandwidth
        lp_bandwidth = self.spectrum_props.lightpath_bandwidth
        if lp_bandwidth is None:
            # Calculate from modulation and slots if not set by SNR
            mod = self.spectrum_props.modulation
            if mod is not None and self.sdn_props.modulation_formats_dict is not None:
                lp_bandwidth = self.sdn_props.modulation_formats_dict[mod].get(
                    "bandwidth", 0
                )
            else:
                lp_bandwidth = 0

        if self.sdn_props.path_list is None:
            raise ValueError("Path list must be initialized")

        # Calculate initial utilization based on dedicated vs total bandwidth
        # For dynamic slicing: dedicated_bw might be < lightpath_bandwidth
        # Convert lp_bandwidth to float for calculations
        lp_bandwidth_float = float(lp_bandwidth) if lp_bandwidth else 0.0
        dedicated_bw = lp_bandwidth_float  # Default: full capacity used
        if self.sdn_props.bandwidth_list and len(self.sdn_props.bandwidth_list) > 0:
            # Get the most recently allocated bandwidth (the dedicated amount)
            dedicated_bw = float(self.sdn_props.bandwidth_list[-1])

        initial_utilization = (
            (dedicated_bw / lp_bandwidth_float) * 100.0 if lp_bandwidth_float > 0 else 0.0
        )

        # Skip creating lightpath entry if bandwidth is 0 or None
        # This prevents phantom 0-bandwidth lightpaths from being tracked
        if lp_bandwidth_float == 0:
            return

        remaining_bw_calc = lp_bandwidth_float - dedicated_bw

        # For NEW lightpaths, populate requests_dict with current request
        # For groomed lightpaths, this will be updated by the grooming module
        requests_dict_initial = {self.sdn_props.request_id: int(dedicated_bw)}

        self.sdn_props.lightpath_status_dict[light_id][lp_id] = {
            "path": self.sdn_props.path_list,
            "path_weight": self.sdn_props.path_weight,
            "core": self.spectrum_props.core_number,
            "band": self.spectrum_props.current_band,
            "start_slot": self.spectrum_props.start_slot,
            "end_slot": self.spectrum_props.end_slot,
            "mod_format": self.spectrum_props.modulation,
            "lightpath_bandwidth": lp_bandwidth_float,  # Use float version for calculations
            "remaining_bandwidth": remaining_bw_calc,  # Account for dedicated bandwidth
            "snr_cost": self.spectrum_props.crosstalk_cost,
            "xt_cost": self.spectrum_props.crosstalk_cost,
            "is_degraded": False,
            "requests_dict": requests_dict_initial,  # Populated with current request
            "time_bw_usage": {
                self.sdn_props.arrive: initial_utilization  # Correct initial utilization
            },  # Track utilization over time
        }


    def get_spectrum(
        self,
        mod_format_list: list[str],
        slice_bandwidth: str | None = None,
        backup_mod_format_list: list[str] | None = None,
    ) -> None:
        """
        Find available spectrum for the current request.

        For 1+1 protected requests, validates BOTH primary and backup paths
        have feasible modulation formats before attempting spectrum search.

        :param mod_format_list: List of modulation formats for primary path
        :type mod_format_list: list[str]
        :param slice_bandwidth: Bandwidth used for light-segment slicing
        :type slice_bandwidth: str | None
        :param backup_mod_format_list: List of modulation formats for backup path (1+1)
        :type backup_mod_format_list: list[str] | None
        """
        self._initialize_spectrum_information()

        # For 1+1 protection: validate backup path has feasible modulation formats
        backup_path = getattr(self.sdn_props, "backup_path", None)
        if backup_path is not None and backup_mod_format_list is not None:
            # Check if any modulation format in backup list is feasible
            backup_has_feasible = False
            for backup_mod in backup_mod_format_list:
                if backup_mod and backup_mod is not False:
                    backup_has_feasible = True
                    break

            if not backup_has_feasible:
                # Backup path has no feasible modulation formats
                self.sdn_props.block_reason = "backup_path_distance"
                logger.debug(
                    f"1+1 protection: Backup path has no feasible modulation formats. "
                    f"Backup mods: {backup_mod_format_list}"
                )
                return

        for modulation_format in mod_format_list:
            # Handle case where modulation_format might be a nested list
            if isinstance(modulation_format, list):
                if len(modulation_format) > 0:
                    modulation_format = modulation_format[0]
                else:
                    continue

            # Skip invalid modulation formats (False, None, empty string, etc.)
            if not modulation_format or modulation_format is False:
                self.sdn_props.block_reason = "distance"
                continue

            # Validate modulation format exists in the appropriate dictionary
            if slice_bandwidth:
                modulation_bandwidth_dict = self.engine_props_dict["mod_per_bw"][
                    slice_bandwidth
                ]
                if modulation_format not in modulation_bandwidth_dict:
                    self.sdn_props.block_reason = "distance"
                    continue
            elif not self.engine_props_dict["fixed_grid"]:
                if self.sdn_props.modulation_formats_dict is None:
                    raise ValueError("Modulation formats dict must be initialized")
                if modulation_format not in self.sdn_props.modulation_formats_dict:
                    self.sdn_props.block_reason = "distance"
                    continue

            # Calculate slots needed using the new method
            self.spectrum_props.slots_needed = self._calculate_slots_needed(
                modulation_format, slice_bandwidth
            )

            if self.spectrum_props.slots_needed is None:
                continue

            # Check if this is a protected (1+1) request
            backup_path = getattr(self.sdn_props, "backup_path", None)

            if backup_path is not None and self.spectrum_props.path_list is not None:
                # Protected request - find spectrum on both paths
                logger.debug(
                    f"1+1 protection: Finding common spectrum on primary "
                    f"{self.spectrum_props.path_list} and backup {backup_path}"
                )
                self._find_protected_spectrum(
                    self.spectrum_props.path_list, backup_path
                )
            else:
                # Regular request - use existing logic
                self._determine_spectrum_allocation()

            if self.spectrum_props.is_free:
                self.spectrum_props.modulation = modulation_format

                # Handle SNR checks
                if (
                    self.engine_props_dict["snr_type"] != "None"
                    and self.engine_props_dict["snr_type"] is not None
                ):
                    if self.sdn_props.path_index is None:
                        raise ValueError(
                            "Path index must be initialized for SNR calculations"
                        )
                    snr_is_acceptable, crosstalk_cost, lp_bw = (
                        self.snr_measurements.handle_snr(self.sdn_props.path_index)
                    )
                    self.spectrum_props.crosstalk_cost = crosstalk_cost
                    # Don't set lightpath_bandwidth here - LEGACY calculates LP capacity
                    # from modulation_formats_dict in _create_lightpath_info, not from SNR lp_bw

                    if not snr_is_acceptable:
                        self.spectrum_props.is_free = False
                        self.sdn_props.block_reason = "xt_threshold"
                        continue

                    self.spectrum_props.is_free = True
                    self.sdn_props.block_reason = None

                # Lightpath ID will be generated by sdn_controller after get_spectrum() returns
                # This matches v5 behavior where get_spectrum() only finds spectrum,
                # and the caller (sdn_controller) generates the lightpath ID
                return

            self.sdn_props.block_reason = "congestion"
            continue

    def get_spectrum_dynamic_slicing(
        self,
        _mod_format_list: list[str],
        _slice_bandwidth: str | None = None,
        path_index: int | None = None,
        mod_format_dict: dict[str, Any] | None = None,
    ) -> tuple[str | bool, int | bool]:
        """
        Find available spectrum for dynamic slicing.

        :param _mod_format_list: List of modulation formats to attempt allocation
        :type _mod_format_list: list[str]
        :param _slice_bandwidth: Bandwidth used for light-segment slicing
        :type _slice_bandwidth: str | None
        :param path_index: Index of the path for dynamic slicing
        :type path_index: int | None
        :param mod_format_dict: Modulation format dictionary for flex-grid slicing
        :type mod_format_dict: dict[str, Any] | None
        :return: Tuple of modulation format and bandwidth
        :rtype: tuple[str | bool, int | bool]
        """
        self._initialize_spectrum_information()

        # Set slicing flag for dynamic slicing mode
        self.spectrum_props.slicing_flag = True

        if self.engine_props_dict["fixed_grid"]:
            self.spectrum_props.slots_needed = 1
            self._determine_spectrum_allocation()
            if self.spectrum_props.is_free:
                if path_index is None:
                    raise ValueError(
                        "Path index must be initialized for dynamic slicing "
                        "SNR calculations"
                    )
                modulation_format, bandwidth, snr_value = (
                    self.snr_measurements.handle_snr_dynamic_slicing(path_index)
                )
                if bandwidth == 0:
                    self.spectrum_props.is_free = False
                    self.sdn_props.block_reason = "xt_threshold"
                else:
                    self.spectrum_props.modulation = modulation_format
                    self.spectrum_props.crosstalk_cost = snr_value
                    self.spectrum_props.is_free = True
                    self.sdn_props.block_reason = None
                return modulation_format or False, int(bandwidth)

            failed_modulation_format, failed_bandwidth = (False, False)
            return failed_modulation_format, failed_bandwidth

        # Flex-grid dynamic slicing
        if mod_format_dict is None:
            logger.warning("mod_format_dict is required for flex-grid dynamic slicing")
            return False, False

        # Sort modulation formats by max_length (highest first for shortest reach)
        sorted_mod_formats = sort_nested_dict_values(
            original_dict=mod_format_dict, nested_key="max_length"
        )
        mod_format_list = list(sorted_mod_formats.keys())

        for mod in mod_format_list:
            self.spectrum_props.slots_needed = mod_format_dict[mod]["slots_needed"]
            self.spectrum_props.modulation = mod
            self._determine_spectrum_allocation()

            if self.spectrum_props.is_free:
                if path_index is None:
                    raise ValueError(
                        "Path index must be initialized for dynamic slicing "
                        "SNR calculations"
                    )
                resp, bandwidth, snr_value = (
                    self.snr_measurements.handle_snr_dynamic_slicing(path_index)
                )
                if not resp:
                    continue
                self.spectrum_props.crosstalk_cost = snr_value
                self.spectrum_props.is_free = True
                self.sdn_props.block_reason = None
                return mod, int(bandwidth)

        self.spectrum_props.is_free = False
        return False, False
