import copy

import numpy as np

from helper_scripts.sim_helpers import find_free_channels, find_free_slots, get_channel_overlaps


class SpectrumHelpers:
    """
    Contains methods that assist with the spectrum assignment class.
    """

    def __init__(self, engine_props: dict, sdn_props: object, spectrum_props: object):
        self.engine_props = engine_props
        self.spectrum_props = spectrum_props
        self.sdn_props = sdn_props

        self.start_index = None
        self.end_index = None
        self.core_num = None
        self.curr_band = None

    def _check_free_spectrum(self, link_tuple: tuple, rev_link_tuple: tuple):
        core_arr = self.sdn_props.net_spec_dict[link_tuple]['cores_matrix'][self.curr_band][self.core_num]
        rev_core_arr = self.sdn_props.net_spec_dict[rev_link_tuple]['cores_matrix'][self.curr_band][self.core_num]
        if self.spectrum_props.slots_needed == 1 and self.engine_props['guard_slots'] == 0:
            return bool(core_arr[self.start_index] == 0.0 and rev_core_arr[self.start_index] == 0.0)

        if self.engine_props['guard_slots'] == 0:
            tmp_end_index = self.end_index + 1
        else:
            tmp_end_index = self.end_index

        spectrum_set = core_arr[self.start_index:tmp_end_index + self.engine_props['guard_slots']]
        rev_spectrum_set = rev_core_arr[self.start_index:tmp_end_index + self.engine_props['guard_slots']]

        if set(spectrum_set) == {} or set(rev_spectrum_set) == {}:  # pylint: disable=use-implicit-booleaness-not-comparison
            raise ValueError('Spectrum set cannot be empty.')

        if set(spectrum_set) == {0.0} and set(rev_spectrum_set) == {0.0}: # pylint: disable=use-implicit-booleaness-not-comparison
            return True

        return False

    def check_other_links(self):
        """
        Checks other links in the path since the first link was free.
        """
        self.spectrum_props.is_free = True
        for node in range(len(self.spectrum_props.path_list) - 1):
            link_tuple = (self.spectrum_props.path_list[node], self.spectrum_props.path_list[node + 1])
            rev_link_tuple = (self.spectrum_props.path_list[node + 1], self.spectrum_props.path_list[node])

            if not self._check_free_spectrum(link_tuple=link_tuple, rev_link_tuple=rev_link_tuple):
                self.spectrum_props.is_free = False
                return

    def _update_spec_props(self):
        if self.spectrum_props.forced_core is not None:
            self.core_num = self.spectrum_props.forced_core

        if self.spectrum_props.forced_band is not None:
            self.curr_band = self.spectrum_props.forced_band

        if self.engine_props['allocation_method'] == 'last_fit':
            self.spectrum_props.start_slot = self.end_index
            self.spectrum_props.end_slot = self.start_index + self.engine_props['guard_slots']
        else:
            self.spectrum_props.start_slot = self.start_index
            self.spectrum_props.end_slot = self.end_index + self.engine_props['guard_slots']

        self.spectrum_props.core_num = self.core_num
        self.spectrum_props.curr_band = self.curr_band
        return self.spectrum_props

    def check_super_channels(self, open_slots_matrix: list, flag: str):
        """
        Given a matrix of available super-channels, find one that can allocate the current request.

        :param open_slots_matrix: A matrix where each entry is an available super-channel's indexes.
        :return: If the request can be successfully allocated.
        :rtype: bool
        """
        for super_channel in open_slots_matrix:
            if len(super_channel) >= (self.spectrum_props.slots_needed + self.engine_props['guard_slots']):
                for start_index in super_channel:
                    if flag == 'forced_index' and start_index != self.spectrum_props.forced_index:
                        continue
                    self.start_index = start_index
                    if self.engine_props['allocation_method'] == 'last_fit':
                        self.end_index = (self.start_index - self.spectrum_props.slots_needed - self.engine_props[
                            'guard_slots']) + 1
                    else:
                        self.end_index = (self.start_index + self.spectrum_props.slots_needed + self.engine_props[
                            'guard_slots']) - 1
                    if self.end_index not in super_channel:
                        break
                    self.spectrum_props.is_free = True

                    if len(self.spectrum_props.path_list) > 2:
                        self.check_other_links()

                    if self.spectrum_props.is_free is not False or len(self.spectrum_props.path_list) <= 2:
                        self._update_spec_props()
                        return True

        return False

    @staticmethod
    def _find_link_inters(info_dict: dict, source_dest: tuple):
        for core_num in info_dict['free_slots_dict'][source_dest]:
            if core_num not in info_dict['slots_inters_dict']:
                tmp_dict = {core_num: set(info_dict['free_slots_dict'][source_dest][core_num])}
                info_dict['slots_inters_dict'].update(tmp_dict)

                tmp_dict = {core_num: set(info_dict['free_channels_dict'][source_dest][core_num])}
                info_dict['channel_inters_dict'].update(tmp_dict)
            else:
                slot_inters_dict = info_dict['slots_inters_dict'][core_num]
                free_slots_set = set(info_dict['free_slots_dict'][source_dest][core_num])
                slot_inters = slot_inters_dict & free_slots_set
                info_dict['slots_inters_dict'][core_num] = slot_inters

                tmp_list = list()
                for item in info_dict['channel_inters_dict'][core_num]:
                    if item in info_dict['free_channels_dict'][source_dest][core_num]:
                        tmp_list.append(item)

                info_dict['channel_inters_dict'][core_num] = tmp_list

    def find_link_inters(self):
        """
        Finds all slots and super-channels and spectral slots that have potential intersections.

        :return: The free slots and super-channels along with intersecting slots and super-channels.
        :rtype: dict
        """
        info_dict = {'free_slots_dict': {}, 'free_channels_dict': {}, 'slots_inters_dict': {},
                     'channel_inters_dict': {}}

        for source_dest in zip(self.spectrum_props.path_list, self.spectrum_props.path_list[1:]):
            free_slots = find_free_slots(net_spec_dict=self.sdn_props.net_spec_dict, link_tuple=source_dest)
            free_channels = find_free_channels(net_spec_dict=self.sdn_props.net_spec_dict,
                                               slots_needed=self.spectrum_props.slots_needed,
                                               link_tuple=source_dest)

            info_dict['free_slots_dict'].update({source_dest: free_slots})
            info_dict['free_channels_dict'].update({source_dest: free_channels})

        return info_dict

    def find_best_core(self):
        """
        Finds the core with the least amount of overlapping super channels.

        :return: The core with the least amount of overlapping channels.
        :rtype: int
        """
        path_info = self.find_link_inters()
        all_channels = get_channel_overlaps(path_info['free_channels_dict'],
                                            path_info['free_slots_dict'])
        overlapping_results = copy.deepcopy(all_channels[list(all_channels.keys())[0]])
        for _, channels in all_channels.items():
            for ch_type, channels_type in channels.items():
                for band, band_channels in channels_type.items():
                    for core_num, channel in band_channels.items():
                        tmp_dict = overlapping_results[ch_type][band][core_num]
                        overlapping_results[ch_type][band][core_num] = np.intersect1d(tmp_dict, channel)
        c_band_dict = overlapping_results['non_over_dict']['c']
        sorted_cores = sorted(c_band_dict, key=lambda k: len(c_band_dict[k]))

        # TODO: Comment why
        if len(sorted_cores) > 1:
            if 6 in sorted_cores:
                sorted_cores.remove(6)
        return sorted_cores[0]
