import itertools
from operator import itemgetter

import numpy as np

from helper_scripts.sim_helpers import sort_nested_dict_vals
from arg_scripts.spectrum_args import SpectrumProps
from helper_scripts.spectrum_helpers import SpectrumHelpers
from src.snr_measurements import SnrMeasurements


class SpectrumAssignment:
    """
    Attempt to find the available spectrum for a given request.
    """

    def __init__(self, engine_props: dict, sdn_props: object, route_props: object):
        self.spectrum_props = SpectrumProps()
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.route_props = route_props

        self.snr_obj = SnrMeasurements(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                       spectrum_props=self.spectrum_props, route_props=self.route_props)
        self.spec_help_obj = SpectrumHelpers(engine_props=self.engine_props, sdn_props=self.sdn_props,
                                             spectrum_props=self.spectrum_props)

    def _allocate_best_fit(self, channels_list: list):
        for channel_dict in channels_list:
            for start_index in channel_dict['channel']:
                end_index = (start_index + self.spectrum_props.slots_needed + self.engine_props['guard_slots']) - 1
                if end_index not in channel_dict['channel']:
                    break

                if len(self.spectrum_props.path_list) > 2:
                    self.spec_help_obj.start_index = start_index
                    self.spec_help_obj.end_index = end_index
                    self.spec_help_obj.core_num = channel_dict['core']
                    # TODO: This needs to be double checked
                    self.spec_help_obj.curr_band = channel_dict['band']
                    self.spec_help_obj.check_other_links()

                if self.spectrum_props.is_free or len(self.spectrum_props.path_list) <= 2:
                    self.spectrum_props.is_free = True
                    self.spectrum_props.start_slot = start_index
                    self.spectrum_props.end_slot = end_index + self.engine_props['guard_slots']
                    self.spectrum_props.end_slot = end_index
                    self.spectrum_props.core_num = channel_dict['core']
                    # TODO: This needs to be checked
                    self.spectrum_props.curr_band = channel_dict['band']
                    return

    # TODO: No support for multi-band
    def find_best_fit(self):
        """
        Searches for and allocates the best-fit super channel on each link along the path.
        """
        channels_list = list()

        # Get all potential super channels
        for (src, dest) in zip(self.spectrum_props.path_list[:-1], self.spectrum_props.path_list[1:]):
            for core_num in range(self.engine_props['cores_per_link']):
                if self.spectrum_props.forced_core is not None and self.spectrum_props.forced_core != core_num:
                    continue

                for band in self.engine_props['band_list']:
                    if self.spectrum_props.forced_band is not None and self.spectrum_props.forced_band != band:
                        continue

                    core_arr = self.sdn_props.net_spec_dict[(src, dest)]['cores_matrix'][band][core_num]
                    open_slots_arr = np.where(core_arr == 0)[0]

                    tmp_matrix = [list(map(itemgetter(1), g)) for k, g in
                                  itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                    for channel_list in tmp_matrix:
                        if len(channel_list) >= self.spectrum_props.slots_needed:
                            channels_list.append(
                                {'link': (src, dest), 'core': core_num, 'channel': channel_list, 'band': band})

        # Sort the list of candidate super channels
        channels_list = sorted(channels_list, key=lambda d: len(d['channel']))
        self._allocate_best_fit(channels_list=channels_list)

    def _setup_first_last(self):
        core_matrix = list()

        if self.spectrum_props.forced_core is not None:
            core_list = [self.spectrum_props.forced_core]
        elif self.engine_props['allocation_method'] in ('priority_first', 'priority_last'):
            core_list = [0, 2, 4, 1, 3, 5, 6]
        else:
            core_list = list(range(0, self.engine_props['cores_per_link']))

        if self.spectrum_props.forced_band is not None:
            band_list = [self.spectrum_props.forced_band]
        else:
            band_list = self.engine_props['band_list']

        for curr_core in core_list:
            core_matrix.append([self.spectrum_props.cores_matrix[band][curr_core] for band in band_list])

        return core_matrix, core_list, self.engine_props['band_list']

    def handle_first_last(self, flag: str):
        """
        Handles either first-fit or last-fit spectrum allocation.

        :param flag: A flag to determine which allocation method to be used.
        """
        # TODO: Cores matrix is now a dictionary, change name
        core_matrix, core_list, band_list = self._setup_first_last()

        if self.engine_props['spectrum_allocation_priority'] == 'BSC':
            
            for band_index in range(len(band_list)):  # pylint: disable=consider-using-enumerate
                for core_arr, core_num in zip(core_matrix, core_list):
                    open_slots_arr = np.where(core_arr[band_index] == 0)[0]

                    # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
                    if flag in ('last_fit', 'priority_last'):
                        open_slots_matrix = [list(map(itemgetter(1), g))[::-1] for k, g in
                                            itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                    elif flag in ('first_fit', 'priority_first', 'forced_index'):
                        open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                            itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                    else:
                        raise NotImplementedError(f'Invalid flag, got: {flag} and expected last_fit or first_fit.')

                    self.spec_help_obj.core_num = core_num
                    self.spec_help_obj.curr_band = band_list[band_index]
                    was_allocated = self.spec_help_obj.check_super_channels(open_slots_matrix=open_slots_matrix, flag=flag)
                    if was_allocated:
                        return
        else:
            for core_arr, core_num in zip(core_matrix, core_list):
                for band_index in range(len(band_list)):  # pylint: disable=consider-using-enumerate
                    open_slots_arr = np.where(core_arr[band_index] == 0)[0]

                    # Source: https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
                    if flag in ('last_fit', 'priority_last'):
                        open_slots_matrix = [list(map(itemgetter(1), g))[::-1] for k, g in
                                            itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                    elif flag in ('first_fit', 'priority_first', 'forced_index'):
                        open_slots_matrix = [list(map(itemgetter(1), g)) for k, g in
                                            itertools.groupby(enumerate(open_slots_arr), lambda i_x: i_x[0] - i_x[1])]
                    else:
                        raise NotImplementedError(f'Invalid flag, got: {flag} and expected last_fit or first_fit.')

                    self.spec_help_obj.core_num = core_num
                    self.spec_help_obj.curr_band = band_list[band_index]
                    was_allocated = self.spec_help_obj.check_super_channels(open_slots_matrix=open_slots_matrix, flag=flag)
                    if was_allocated:
                        return

    # fixme: Only works for 7 cores
    def xt_aware(self):
        """
        Attempts to allocate a request with the least amount of cross-talk interference on neighboring cores.

        :return: The information of the request if allocated or False if not possible.
        :rtype: dict
        """
        core = self.spec_help_obj.find_best_core()
        if core in [0, 2, 4, 6]:
            self.spectrum_props.forced_core = core
            return self.handle_first_last(flag='first_fit')

        return self.handle_first_last(flag='last_fit')

    def _get_spectrum(self):
        if self.spectrum_props.forced_index is not None:
            self.handle_first_last(flag='forced_index')
        elif self.engine_props['allocation_method'] == 'best_fit':
            self.find_best_fit()
        elif self.engine_props['allocation_method'] in ('first_fit', 'last_fit', 'priority_first', 'priority_last'):
            self.handle_first_last(flag=self.engine_props['allocation_method'])
        elif self.engine_props['allocation_method'] == 'xt_aware':
            self.xt_aware()
        else:
            raise NotImplementedError(f"Expected first_fit or best_fit, got: {self.engine_props['allocation_method']}")

    def _init_spectrum_info(self):
        link_tuple = (self.spectrum_props.path_list[0], self.spectrum_props.path_list[1])
        rev_link_tuple = (self.spectrum_props.path_list[1], self.spectrum_props.path_list[0])
        self.spectrum_props.cores_matrix = self.sdn_props.net_spec_dict[link_tuple]['cores_matrix']
        self.spectrum_props.rev_cores_matrix = self.sdn_props.net_spec_dict[rev_link_tuple]['cores_matrix']
        self.spectrum_props.is_free = False

    def get_spectrum(self, mod_format_list: list, slice_bandwidth: str = None, path_index: int = None):
        """
        Controls the class, attempts to find an available spectrum.

        :param mod_format_list: A list of modulation formats to attempt allocation.
        :param slice_bandwidth: A bandwidth used for light-segment slicing.
        """
        self._init_spectrum_info()
        for modulation in mod_format_list:
            if modulation is False:
                self.sdn_props.block_reason = 'distance'
                continue

            if slice_bandwidth:
                bandwidth_dict = self.engine_props['mod_per_bw'][slice_bandwidth]
                self.spectrum_props.slots_needed = bandwidth_dict[modulation]['slots_needed']
            else:
                if self.engine_props['fixed_grid']:
                    self.spectrum_props.slots_needed = 1
                else:
                    if self.sdn_props.was_partially_groomed:
                        bw_tmp = min((int(k) for k in self.engine_props['mod_per_bw'] if int(k) > int(self.sdn_props.remaining_bw)), default=None)
                        self.spectrum_props.slots_needed = self.engine_props['mod_per_bw'][str(bw_tmp)][modulation]['slots_needed']
                    self.spectrum_props.slots_needed = self.sdn_props.mod_formats_dict[modulation]['slots_needed']

            if self.spectrum_props.slots_needed is None:
                raise ValueError('Slots needed cannot be none.')

            self._get_spectrum()

            if self.spectrum_props.is_free:
                self.spectrum_props.modulation = modulation
                if self.engine_props['snr_type'] != 'None' and self.engine_props['snr_type'] is not None:
                    snr_check, xt_cost, lp_bw = self.snr_obj.handle_snr(path_index)
                    self.spectrum_props.xt_cost = xt_cost
                    if not snr_check:
                        self.spectrum_props.is_free = False
                        self.sdn_props.block_reason = 'xt_threshold'
                        continue
                    self.spectrum_props.lightpath_bandwidth = lp_bw
                    self.spectrum_props.is_free = True
                    self.sdn_props.block_reason = None

                return

            self.spectrum_props.block_reason = 'congestion'
            continue


    def get_spectrum_dynamic_slicing(self, mod_format_dict: dict, slice_bandwidth: str = None, path_index: int = None):
        """
        Controls the class, attempts to find an available spectrum.

        :param mod_format_list: A list of modulation formats to attempt allocation.
        :param slice_bandwidth: A bandwidth used for light-segment slicing.
        """
        self._init_spectrum_info()

        if self.engine_props['fixed_grid']:
            self.spectrum_props.slots_needed = 1
            self.spectrum_props.slicing_flag = True
            self._get_spectrum()
            if self.spectrum_props.is_free:
                mod_format, bw, snr_val = self.snr_obj.handle_snr_dynamic_slicing(path_index)
                self.spectrum_props.modulation = mod_format
                self.spectrum_props.xt_cost = snr_val
                self.spectrum_props.is_free = True
                self.sdn_props.block_reason = None
                return mod_format, bw
            else:
                return 0, 0
        else:
            mod_format = sort_nested_dict_vals(original_dict=mod_format_dict,
                                                        nested_key='max_length')
            mod_format_list = list(mod_format.keys())
            for mod in mod_format_list:
                self.spectrum_props.slots_needed = mod_format_dict[mod]['slots_needed']
                self.spectrum_props.modulation = mod
                self.spectrum_props.slicing_flag = True
                self._get_spectrum()
                if self.spectrum_props.is_free:
                    resp, bw, snr_val = self.snr_obj.handle_snr_dynamic_slicing(path_index)
                    if not resp:
                        continue
                    self.spectrum_props.xt_cost = snr_val
                    self.spectrum_props.is_free = True
                    self.sdn_props.block_reason = None
                    return mod, bw
            self.spectrum_props.is_free = False
            return False, 0


