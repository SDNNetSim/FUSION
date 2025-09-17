# pylint: disable=duplicate-code
import itertools
from operator import itemgetter

import numpy as np

from fusion.core.properties import SpectrumProps, SDNProps, RoutingProps
from fusion.core.snr_measurements import SnrMeasurements
from fusion.modules.spectrum.utils import SpectrumHelpers


class SpectrumAssignment:
    """
    Attempt to find the available spectrum for a given request.
    """

    def __init__(self, engine_props: dict, sdn_props: SDNProps, route_props: RoutingProps):
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

    def _allocate_best_fit_spectrum(self, candidate_channels_list: list):
        for channel_dict in candidate_channels_list:
            for start_index in channel_dict['channel']:
                end_index = (start_index + self.spectrum_props.slots_needed + self.engine_props_dict['guard_slots']) - 1
                if end_index not in channel_dict['channel']:
                    break

                if len(self.spectrum_props.path_list) > 2:
                    self.spectrum_helpers.start_index = start_index
                    self.spectrum_helpers.end_index = end_index
                    self.spectrum_helpers.core_number = channel_dict['core']
                    self.spectrum_helpers.current_band = channel_dict['band']
                    self.spectrum_helpers.check_other_links()

                if self.spectrum_props.is_free or len(self.spectrum_props.path_list) <= 2:
                    self.spectrum_props.is_free = True
                    self.spectrum_props.start_slot = start_index
                    self.spectrum_props.end_slot = end_index + self.engine_props_dict['guard_slots']
                    self.spectrum_props.end_slot = end_index
                    self.spectrum_props.core_number = channel_dict['core']
                    self.spectrum_props.current_band = channel_dict['band']
                    return

    def find_best_fit(self):
        """
        Searches for and allocates the best-fit super channel on each link along the path.
        """
        candidate_channels_list = []

        # Get all potential super channels
        for (source_node, destination_node) in zip(self.spectrum_props.path_list[:-1],
                                                   self.spectrum_props.path_list[1:]):
            for core_number in range(self.engine_props_dict['cores_per_link']):
                if self.spectrum_props.forced_core is not None and self.spectrum_props.forced_core != core_number:
                    continue

                for band in self.engine_props_dict['band_list']:
                    if self.spectrum_props.forced_band is not None and self.spectrum_props.forced_band != band:
                        continue

                    core_spectrum_array = \
                        self.sdn_props.network_spectrum_dict[(source_node, destination_node)]['cores_matrix'][band][
                            core_number]
                    available_slots_array = np.where(core_spectrum_array == 0)[0]

                    contiguous_blocks_matrix = [list(map(itemgetter(1), group)) for key, group in
                                                itertools.groupby(enumerate(available_slots_array),
                                                                  lambda index_slot: index_slot[0] - index_slot[1])]
                    for contiguous_channel_list in contiguous_blocks_matrix:
                        if len(contiguous_channel_list) >= self.spectrum_props.slots_needed:
                            candidate_channels_list.append({
                                'link': (source_node, destination_node),
                                'core': core_number,
                                'channel': contiguous_channel_list,
                                'band': band
                            })

        # Sort the list of candidate super channels
        candidate_channels_list = sorted(candidate_channels_list, key=lambda channel_dict: len(channel_dict['channel']))
        self._allocate_best_fit_spectrum(candidate_channels_list=candidate_channels_list)

    def _setup_first_last_allocation(self):
        cores_spectrum_matrix = []

        if self.spectrum_props.forced_core is not None:
            core_numbers_list = [self.spectrum_props.forced_core]
        elif self.engine_props_dict['allocation_method'] in ('priority_first', 'priority_last'):
            core_numbers_list = [0, 2, 4, 1, 3, 5, 6]
        else:
            core_numbers_list = list(range(0, self.engine_props_dict['cores_per_link']))

        if self.spectrum_props.forced_band is not None:
            available_bands_list = [self.spectrum_props.forced_band]
        else:
            available_bands_list = self.engine_props_dict['band_list']

        for current_core_number in core_numbers_list:
            cores_spectrum_matrix.append(
                [self.spectrum_props.cores_matrix[band][current_core_number] for band in available_bands_list])

        return cores_spectrum_matrix, core_numbers_list, self.engine_props_dict['band_list']

    def _get_available_slots_matrix(self, available_slots_array, allocation_flag):
        """
        Converts an array of available slots into a matrix of contiguous blocks based on allocation flag.

        :param available_slots_array: Array of available slot indices.
        :type available_slots_array: np.ndarray
        :param allocation_flag: Allocation method flag (e.g., 'first_fit', 'last_fit').
        :type allocation_flag: str
        :return: A matrix of contiguous available slot blocks.
        :rtype: list
        """
        if allocation_flag in ('last_fit', 'priority_last'):
            return [list(map(itemgetter(1), group))[::-1] for key, group in
                    itertools.groupby(enumerate(available_slots_array),
                                      lambda index_slot: index_slot[0] - index_slot[1])]
        if allocation_flag in ('first_fit', 'priority_first', 'forced_index'):
            return [list(map(itemgetter(1), group)) for key, group in
                    itertools.groupby(enumerate(available_slots_array),
                                      lambda index_slot: index_slot[0] - index_slot[1])]

        raise NotImplementedError(
            f"Invalid allocation flag, got: {allocation_flag} and expected 'last_fit' or 'first_fit'.")

    def handle_first_last_allocation(self, allocation_flag: str):
        """
        Handles either first-fit or last-fit spectrum allocation without any priority or SNR considerations.

        :param allocation_flag: A flag to determine which allocation method to be used.
        :type allocation_flag: str
        """
        cores_spectrum_matrix, core_numbers_list, band_list = self._setup_first_last_allocation()

        for core_spectrum_array, core_number in zip(cores_spectrum_matrix, core_numbers_list):
            for band_index, band in enumerate(band_list):
                available_slots_array = np.where(core_spectrum_array[band_index] == 0)[0]
                available_slots_matrix = self._get_available_slots_matrix(available_slots_array, allocation_flag)

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(open_slots_matrix=available_slots_matrix,
                                                                           flag=allocation_flag)
                if was_allocated:
                    return

    def handle_first_last_priority_bsc(self, allocation_flag: str):
        """
        Handles first-fit or last-fit spectrum allocation with multi-band priority (BSC).

        :param allocation_flag: A flag to determine which allocation method to be used.
        :type allocation_flag: str
        """
        cores_spectrum_matrix, core_numbers_list, band_list = self._setup_first_last_allocation()

        for band_index, band in enumerate(band_list):
            for core_spectrum_array, core_number in zip(cores_spectrum_matrix, core_numbers_list):
                available_slots_array = np.where(core_spectrum_array[band_index] == 0)[0]
                available_slots_matrix = self._get_available_slots_matrix(available_slots_array, allocation_flag)

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(open_slots_matrix=available_slots_matrix,
                                                                           flag=allocation_flag)
                if was_allocated:
                    if (self.engine_props_dict['cores_per_link'] in [13, 19] and
                            self.engine_props_dict['snr_type'] == 'snr_e2e_external_resources'):
                        if self._handle_snr_external_resources(allocation_flag, available_slots_matrix):
                            return

                        self.spectrum_props.is_free = False
                        continue

                    return

    def handle_first_last_priority_band(self, allocation_flag: str):
        """
        Handles first-fit or last-fit spectrum allocation with band priority (non-BSC).

        :param allocation_flag: A flag to determine which allocation method to be used.
        :type allocation_flag: str
        """
        cores_spectrum_matrix, core_numbers_list, band_list = self._setup_first_last_allocation()

        for core_spectrum_array, core_number in zip(cores_spectrum_matrix, core_numbers_list):
            for band_index, band in enumerate(band_list):
                available_slots_array = np.where(core_spectrum_array[band_index] == 0)[0]
                available_slots_matrix = self._get_available_slots_matrix(available_slots_array, allocation_flag)

                self.spectrum_helpers.core_number = core_number
                self.spectrum_helpers.current_band = band
                was_allocated = self.spectrum_helpers.check_super_channels(open_slots_matrix=available_slots_matrix,
                                                                           flag=allocation_flag)
                if was_allocated:
                    if (self.engine_props_dict['cores_per_link'] in [13, 19] and
                            self.engine_props_dict['snr_type'] == 'snr_e2e_external_resources'):
                        if self._handle_snr_external_resources(allocation_flag, available_slots_matrix):
                            return

                        self.spectrum_props.is_free = False
                        continue
                    return

    def _handle_snr_external_resources(self, allocation_flag, available_slots_matrix):
        """
        Handles SNR external resource checks during allocation.

        :param allocation_flag: Allocation flag (e.g., 'first_fit', 'last_fit').
        :type allocation_flag: str
        :param available_slots_matrix: Matrix of available slot blocks.
        :type available_slots_matrix: list
        :return: Whether the allocation was successful.
        :rtype: bool
        """

        for slots_row in available_slots_matrix:
            while slots_row:
                slots_row = self.snr_measurements.check_snr_ext_open_slots(self.sdn_props.path_index, slots_row)
                if slots_row:
                    was_allocated = self.spectrum_helpers.check_super_channels(open_slots_matrix=[slots_row],
                                                                               flag=allocation_flag)
                    if was_allocated:
                        return True

                    break

                break

        return False

    def handle_crosstalk_aware_allocation(self):
        """
        Attempts to allocate a request with the least amount of cross-talk interference on neighboring cores.
        NOTE: Current implementation only supports 7-core configurations.

        :return: The information of the request if allocated or False if not possible.
        :rtype: dict or bool
        """
        best_core_number = self.spectrum_helpers.find_best_core()
        if best_core_number in [0, 2, 4, 6]:
            self.spectrum_props.forced_core = best_core_number
            return self.handle_first_last_allocation(allocation_flag='first_fit')

        return self.handle_first_last_allocation(allocation_flag='last_fit')

    def _determine_spectrum_allocation(self):
        """
        Determines the spectrum allocation method based on engine properties and spectrum requirements.
        """
        if self.spectrum_props.forced_index is not None:
            self.handle_first_last_allocation(allocation_flag='forced_index')
        elif self.engine_props_dict['allocation_method'] == 'best_fit':
            self.find_best_fit()
        elif self.engine_props_dict['allocation_method'] in ('first_fit', 'last_fit', 'priority_first',
                                                             'priority_last'):
            if self.engine_props_dict['spectrum_priority'] == 'BSC':
                self.handle_first_last_priority_bsc(allocation_flag=self.engine_props_dict['allocation_method'])
            else:
                self.handle_first_last_priority_band(allocation_flag=self.engine_props_dict['allocation_method'])
        elif self.engine_props_dict['allocation_method'] == 'xt_aware':
            self.handle_crosstalk_aware_allocation()
        else:
            raise NotImplementedError(
                f"Expected first_fit or best_fit, got: {self.engine_props_dict['allocation_method']}")

    def _initialize_spectrum_information(self):
        forward_link_tuple = (self.spectrum_props.path_list[0], self.spectrum_props.path_list[1])
        reverse_link_tuple = (self.spectrum_props.path_list[1], self.spectrum_props.path_list[0])
        self.spectrum_props.cores_matrix = self.sdn_props.network_spectrum_dict[forward_link_tuple]['cores_matrix']
        self.spectrum_props.reverse_cores_matrix = self.sdn_props.network_spectrum_dict[reverse_link_tuple][
            'cores_matrix']
        self.spectrum_props.is_free = False

    def get_spectrum(self, mod_format_list: list, slice_bandwidth: str = None):
        """
        Controls the class, attempts to find an available spectrum.

        :param mod_format_list: A list of modulation formats to attempt allocation.
        :type mod_format_list: list
        :param slice_bandwidth: A bandwidth used for light-segment slicing.
        :type slice_bandwidth: str or None
        """
        self._initialize_spectrum_information()
        for modulation_format in mod_format_list:
            if modulation_format is False:
                self.sdn_props.block_reason = 'distance'
                continue

            if slice_bandwidth:
                modulation_bandwidth_dict = self.engine_props_dict['mod_per_bw'][slice_bandwidth]
                self.spectrum_props.slots_needed = modulation_bandwidth_dict[modulation_format]['slots_needed']
            else:
                if self.engine_props_dict['fixed_grid']:
                    self.spectrum_props.slots_needed = 1
                else:
                    self.spectrum_props.slots_needed = self.sdn_props.modulation_formats_dict[modulation_format][
                        'slots_needed']

            if self.spectrum_props.slots_needed is None:
                raise ValueError('Slots needed cannot be none.')

            self._determine_spectrum_allocation()

            if self.spectrum_props.is_free:
                self.spectrum_props.modulation = modulation_format
                if self.engine_props_dict['snr_type'] != 'None' and self.engine_props_dict['snr_type'] is not None:
                    snr_is_acceptable, crosstalk_cost = self.snr_measurements.handle_snr(self.sdn_props.path_index)
                    self.spectrum_props.crosstalk_cost = crosstalk_cost
                    if not snr_is_acceptable:
                        self.spectrum_props.is_free = False
                        self.sdn_props.block_reason = 'xt_threshold'
                        continue

                    self.spectrum_props.is_free = True
                    self.sdn_props.block_reason = None

                return

            self.spectrum_props.block_reason = 'congestion'
            continue

    def get_spectrum_dynamic_slicing(self, _mod_format_list: list, _slice_bandwidth: str = None,
                                     path_index: int = None):
        """
        Controls the class, attempts to find an available spectrum.

        :param _mod_format_list: A list of modulation formats to attempt allocation.
        :type _mod_format_list: list
        :param _slice_bandwidth: A bandwidth used for light-segment slicing.
        :type _slice_bandwidth: str or None
        :param path_index: Index of the path for dynamic slicing.
        :type path_index: int or None
        """
        self._initialize_spectrum_information()

        if self.engine_props_dict['fixed_grid']:
            self.spectrum_props.slots_needed = 1
            self._determine_spectrum_allocation()
            if self.spectrum_props.is_free:
                modulation_format, bandwidth, snr_value = self.snr_measurements.handle_snr_dynamic_slicing(path_index)
                if bandwidth == 0:
                    self.spectrum_props.is_free = False
                    self.sdn_props.block_reason = "xt_threshold"
                else:
                    self.spectrum_props.modulation = modulation_format
                    self.spectrum_props.crosstalk_cost = snr_value
                    self.spectrum_props.is_free = True
                    self.sdn_props.block_reason = None
                return modulation_format, bandwidth

            failed_modulation_format, failed_bandwidth = (False, False)
            return failed_modulation_format, failed_bandwidth

        no_allocation_modulation_format, no_allocation_bandwidth = False, False
        return no_allocation_modulation_format, no_allocation_bandwidth
