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
        route_props: RoutingProps
    ) -> None:
        self.spectrum_props = SpectrumProps()
        self.engine_props_dict = engine_props
        self.sdn_props = sdn_props
        self.route_props = route_props

        self.snr_measurements = SnrMeasurements(
            engine_props_dict=self.engine_props_dict,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props,
            route_props=self.route_props
        )
        self.spectrum_helpers = SpectrumHelpers(
            engine_props=self.engine_props_dict,
            sdn_props=self.sdn_props,
            spectrum_props=self.spectrum_props
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
            for start_index in channel_dict['channel']:
                end_index = (
                    start_index + self.spectrum_props.slots_needed +
                    self.engine_props_dict['guard_slots']
                ) - 1
                if end_index not in channel_dict['channel']:
                    break

                if (self.spectrum_props.path_list is not None and
                    len(self.spectrum_props.path_list) > 2):
                    self.spectrum_helpers.start_index = start_index
                    self.spectrum_helpers.end_index = end_index
                    self.spectrum_helpers.core_number = channel_dict['core']
                    self.spectrum_helpers.current_band = channel_dict['band']
                    self.spectrum_helpers.check_other_links()

                if (
                    self.spectrum_props.is_free or
                    (self.spectrum_props.path_list is not None and
                     len(self.spectrum_props.path_list) <= 2)
                ):
                    self.spectrum_props.is_free = True
                    self.spectrum_props.start_slot = start_index
                    self.spectrum_props.end_slot = (
                        end_index + self.engine_props_dict['guard_slots']
                    )
                    self.spectrum_props.end_slot = end_index
                    self.spectrum_props.core_number = channel_dict['core']
                    self.spectrum_props.current_band = channel_dict['band']
                    return

    def find_best_fit(self) -> None:
        """Search for and allocate best-fit super channel on each link along path."""
        candidate_channels_list = []

        # Get all potential super channels
        if self.spectrum_props.path_list is None:
            raise ValueError("Path list must be initialized")

        for (source_node, destination_node) in zip(
            self.spectrum_props.path_list[:-1],
            self.spectrum_props.path_list[1:],
            strict=False
        ):
            for core_number in range(self.engine_props_dict['cores_per_link']):
                if (
                    self.spectrum_props.forced_core is not None and
                    self.spectrum_props.forced_core != core_number
                ):
                    continue

                for band in self.engine_props_dict['band_list']:
                    if (
                        self.spectrum_props.forced_band is not None and
                        self.spectrum_props.forced_band != band
                    ):
                        continue

                    if self.sdn_props.network_spectrum_dict is None:
                        raise ValueError("Network spectrum dict must be initialized")

                    core_spectrum_array = (
                        self.sdn_props.network_spectrum_dict[
                            (source_node, destination_node)
                        ]['cores_matrix'][band][core_number]
                    )
                    available_slots_array = np.where(core_spectrum_array == 0)[0]

                    contiguous_blocks_matrix = [
                        list(map(itemgetter(1), group))
                        for key, group in itertools.groupby(
                            enumerate(available_slots_array),
                            lambda index_slot: index_slot[0] - index_slot[1]
                        )
                    ]
                    for contiguous_channel_list in contiguous_blocks_matrix:
                        slots_needed = self.spectrum_props.slots_needed
                        if (
                            slots_needed is not None and
                            len(contiguous_channel_list) >= slots_needed
                        ):
                            candidate_channels_list.append({
                                'link': (source_node, destination_node),
                                'core': core_number,
                                'channel': contiguous_channel_list,
                                'band': band
                            })

        # Sort the list of candidate super channels
        candidate_channels_list = sorted(
            candidate_channels_list,
            key=lambda channel_dict: len(channel_dict['channel'])
        )
        self._allocate_best_fit_spectrum(
            candidate_channels_list=candidate_channels_list
        )

    def _setup_first_last_allocation(
        self,
    ) -> tuple[list[list[np.ndarray]], list[int], list[str]]:
        """
        Setup matrices for first/last allocation strategies.

        :return: Tuple containing cores spectrum matrix, core numbers list, band list
        :rtype: tuple[list[list[np.ndarray]], list[int], list[str]]
        """
        cores_spectrum_matrix = []

        if self.spectrum_props.forced_core is not None:
            core_numbers_list = [self.spectrum_props.forced_core]
        elif self.engine_props_dict['allocation_method'] in (
            'priority_first', 'priority_last'
        ):
            core_numbers_list = [0, 2, 4, 1, 3, 5, 6]
        else:
            core_numbers_list = list(
                range(0, self.engine_props_dict['cores_per_link'])
            )

        if self.spectrum_props.forced_band is not None:
            available_bands_list = [self.spectrum_props.forced_band]
        else:
            available_bands_list = self.engine_props_dict['band_list']

        if self.spectrum_props.cores_matrix is None:
            raise ValueError("Cores matrix must be initialized")

        for current_core_number in core_numbers_list:
            cores_spectrum_matrix.append([
                self.spectrum_props.cores_matrix[band][current_core_number]  # type: ignore
                for band in available_bands_list
            ])

        return (
            cores_spectrum_matrix,
            core_numbers_list,
            self.engine_props_dict['band_list']
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
        if allocation_flag in ('last_fit', 'priority_last'):
            return [
                list(map(itemgetter(1), group))[::-1]
                for key, group in itertools.groupby(
                    enumerate(available_slots_array),
                    lambda index_slot: index_slot[0] - index_slot[1]
                )
            ]
        if allocation_flag in ('first_fit', 'priority_first', 'forced_index'):
            return [
                list(map(itemgetter(1), group))
                for key, group in itertools.groupby(
                    enumerate(available_slots_array),
                    lambda index_slot: index_slot[0] - index_slot[1]
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
        (
            cores_spectrum_matrix,
            core_numbers_list,
            band_list
        ) = self._setup_first_last_allocation()

        for core_spectrum_array, core_number in zip(
            cores_spectrum_matrix, core_numbers_list, strict=False
        ):
            for band_index, band in enumerate(band_list):
                available_slots_array = np.where(
                    core_spectrum_array[band_index] == 0
                )[0]
                available_slots_matrix = self._get_available_slots_matrix(
                    available_slots_array, allocation_flag
                )

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(
                    open_slots_matrix=available_slots_matrix,
                    flag=allocation_flag
                )
                if was_allocated:
                    return

    def handle_first_last_priority_bsc(self, allocation_flag: str) -> None:
        """
        Handle first-fit or last-fit allocation with multi-band priority (BSC).

        :param allocation_flag: A flag to determine which allocation method to be used
        :type allocation_flag: str
        """
        (
            cores_spectrum_matrix,
            core_numbers_list,
            band_list
        ) = self._setup_first_last_allocation()

        for band_index, band in enumerate(band_list):
            for core_spectrum_array, core_number in zip(
                cores_spectrum_matrix, core_numbers_list, strict=False
            ):
                available_slots_array = np.where(
                    core_spectrum_array[band_index] == 0
                )[0]
                available_slots_matrix = self._get_available_slots_matrix(
                    available_slots_array, allocation_flag
                )

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(
                    open_slots_matrix=available_slots_matrix,
                    flag=allocation_flag
                )
                if was_allocated:
                    if (
                        self.engine_props_dict['cores_per_link'] in [13, 19] and
                        self.engine_props_dict['snr_type'] ==
                        'snr_e2e_external_resources'
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
        (
            cores_spectrum_matrix,
            core_numbers_list,
            band_list
        ) = self._setup_first_last_allocation()

        for core_spectrum_array, core_number in zip(
            cores_spectrum_matrix, core_numbers_list, strict=False
        ):
            for band_index, band in enumerate(band_list):
                available_slots_array = np.where(
                    core_spectrum_array[band_index] == 0
                )[0]
                available_slots_matrix = self._get_available_slots_matrix(
                    available_slots_array, allocation_flag
                )

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(
                    open_slots_matrix=available_slots_matrix,
                    flag=allocation_flag
                )
                if was_allocated:
                    if (
                        self.engine_props_dict['cores_per_link'] in [13, 19] and
                        self.engine_props_dict['snr_type'] ==
                        'snr_e2e_external_resources'
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
                        open_slots_matrix=[slots_row],
                        flag=allocation_flag
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
            return self.handle_first_last_allocation(allocation_flag='first_fit')

        return self.handle_first_last_allocation(allocation_flag='last_fit')

    def _determine_spectrum_allocation(self) -> None:
        """Determine spectrum allocation method based on engine properties."""
        if self.spectrum_props.forced_index is not None:
            self.handle_first_last_allocation(allocation_flag='forced_index')
        elif self.engine_props_dict['allocation_method'] == 'best_fit':
            self.find_best_fit()
        elif self.engine_props_dict['allocation_method'] in (
            'first_fit', 'last_fit', 'priority_first', 'priority_last'
        ):
            if self.engine_props_dict['spectrum_priority'] == 'BSC':
                self.handle_first_last_priority_bsc(
                    allocation_flag=self.engine_props_dict['allocation_method']
                )
            else:
                self.handle_first_last_priority_band(
                    allocation_flag=self.engine_props_dict['allocation_method']
                )
        elif self.engine_props_dict['allocation_method'] == 'xt_aware':
            self.handle_crosstalk_aware_allocation()
        else:
            raise NotImplementedError(
                f"Expected first_fit or best_fit, got: "
                f"{self.engine_props_dict['allocation_method']}"
            )

    def _initialize_spectrum_information(self) -> None:
        """Initialize spectrum information for the request."""
        path_list = self.spectrum_props.path_list
        if path_list is None or len(path_list) < 2:
            raise ValueError("Path list must be initialized with at least 2 nodes")

        forward_link_tuple = (
            self.spectrum_props.path_list[0],
            self.spectrum_props.path_list[1]
        )
        reverse_link_tuple = (
            self.spectrum_props.path_list[1],
            self.spectrum_props.path_list[0]
        )
        if self.sdn_props.network_spectrum_dict is None:
            raise ValueError("Network spectrum dict must be initialized")

        self.spectrum_props.cores_matrix = (
            self.sdn_props.network_spectrum_dict[forward_link_tuple]['cores_matrix']
        )
        self.spectrum_props.reverse_cores_matrix = (
            self.sdn_props.network_spectrum_dict[reverse_link_tuple]['cores_matrix']
        )
        self.spectrum_props.is_free = False

    def get_spectrum(
        self, mod_format_list: list[str], slice_bandwidth: str | None = None
    ) -> None:
        """
        Find available spectrum for the current request.

        :param mod_format_list: List of modulation formats to attempt allocation
        :type mod_format_list: list[str]
        :param slice_bandwidth: Bandwidth used for light-segment slicing
        :type slice_bandwidth: str | None
        """
        self._initialize_spectrum_information()
        for modulation_format in mod_format_list:
            if modulation_format is False:
                self.sdn_props.block_reason = 'distance'
                continue

            if slice_bandwidth:
                modulation_bandwidth_dict = (
                    self.engine_props_dict['mod_per_bw'][slice_bandwidth]
                )
                self.spectrum_props.slots_needed = (
                    modulation_bandwidth_dict[modulation_format]['slots_needed']
                )
            else:
                if self.engine_props_dict['fixed_grid']:
                    self.spectrum_props.slots_needed = 1
                else:
                    if self.sdn_props.modulation_formats_dict is None:
                        raise ValueError("Modulation formats dict must be initialized")
                    self.spectrum_props.slots_needed = (
                        self.sdn_props.modulation_formats_dict[modulation_format][
                            'slots_needed'
                        ]
                    )

            if self.spectrum_props.slots_needed is None:
                raise ValueError('Slots needed cannot be none.')

            self._determine_spectrum_allocation()

            if self.spectrum_props.is_free:
                self.spectrum_props.modulation = modulation_format
                if (
                    self.engine_props_dict['snr_type'] != 'None' and
                    self.engine_props_dict['snr_type'] is not None
                ):
                    if self.sdn_props.path_index is None:
                        raise ValueError(
                            "Path index must be initialized for SNR calculations"
                        )
                    snr_is_acceptable, crosstalk_cost = (
                        self.snr_measurements.handle_snr(self.sdn_props.path_index)
                    )
                    self.spectrum_props.crosstalk_cost = crosstalk_cost
                    if not snr_is_acceptable:
                        self.spectrum_props.is_free = False
                        self.sdn_props.block_reason = 'xt_threshold'
                        continue

                    self.spectrum_props.is_free = True
                    self.sdn_props.block_reason = None

                return

            self.sdn_props.block_reason = 'congestion'
            continue

    def get_spectrum_dynamic_slicing(
        self,
        _mod_format_list: list[str],
        _slice_bandwidth: str | None = None,
        path_index: int | None = None
    ) -> tuple[str | bool, int | bool]:
        """
        Find available spectrum for dynamic slicing.

        :param _mod_format_list: List of modulation formats to attempt allocation
        :type _mod_format_list: list[str]
        :param _slice_bandwidth: Bandwidth used for light-segment slicing
        :type _slice_bandwidth: str | None
        :param path_index: Index of the path for dynamic slicing
        :type path_index: int | None
        :return: Tuple of modulation format and bandwidth
        :rtype: tuple[str | bool, int | bool]
        """
        self._initialize_spectrum_information()

        if self.engine_props_dict['fixed_grid']:
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

        no_allocation_modulation_format, no_allocation_bandwidth = False, False
        return no_allocation_modulation_format, no_allocation_bandwidth
