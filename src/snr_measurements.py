import math

import numpy as np
import networkx as nx

from arg_scripts.snr_args import SNRProps
from helper_scripts.snr_helpers import get_slot_index, get_loaded_files, compute_response


# fixme: Only works for seven cores
class SnrMeasurements:
    """
    Handles signal-to-noise ratio calculations for a given request.
    """

    def __init__(self, engine_props: dict, sdn_props: object, spectrum_props: object, route_props: object):
        self.snr_props = SNRProps()
        self.engine_props = engine_props
        self.sdn_props = sdn_props
        self.spectrum_props = spectrum_props
        self.route_props = route_props

        self.channels_list = None
        self.link_id = None
        self.num_slots = None

    def _calculate_sci_psd(self):
        """
        Calculates the self-phase power spectral density.

        :return: The self-phase power spectral density.
        :rtype: float
        """
        rho_param = (math.pi ** 2) * np.abs(self.snr_props.link_dict['dispersion'])
        rho_param /= (2 * self.snr_props.link_dict['attenuation'])

        sci_psd = self.snr_props.center_psd ** 2
        sci_psd *= math.asinh(rho_param * (self.snr_props.bandwidth ** 2))
        return sci_psd

    def _update_link_xci(self, req_id: float, curr_link: np.ndarray, slot_index: int, curr_xci: float):
        """
        Given the spectrum contents, updates the link's cross-phase modulation noise.

        :return: The updated cross-phase modulation noise.
        :rtype: float
        """
        channel_bw = len(np.where(req_id == curr_link[self.spectrum_props.core_num])[0])
        channel_bw *= self.engine_props['bw_per_slot']
        channel_freq = ((slot_index * self.engine_props['bw_per_slot']) + (channel_bw / 2)) * 10 ** 9
        channel_bw *= 10 ** 9
        channel_psd = self.engine_props['input_power'] / channel_bw

        if self.snr_props.center_freq != channel_freq:
            log_term = abs(self.snr_props.center_freq - channel_freq) + (channel_bw / 2)
            log_term /= (abs(self.snr_props.center_freq - channel_freq) - (channel_bw / 2))
            calculated_xci = (channel_psd ** 2) * math.log(abs(log_term))
            new_xci = curr_xci + calculated_xci
        else:
            new_xci = curr_xci

        return new_xci

    def _calculate_xci(self, link_num: int):
        """
        Calculates the cross-phase modulation noise on a link for a single request.

        :return: The total cross-phase modulation noise on the link
        :rtype: float
        """
        self.channels_list = []
        # Cross-phase modulation noise
        xci_noise = 0
        # TODO: Only works for c-band
        for slot_index in range(self.engine_props['c_band']):
            source = self.spectrum_props.path_list[link_num]
            dest = self.spectrum_props.path_list[link_num + 1]
            curr_link = self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix']['c']
            req_id = curr_link[self.spectrum_props.core_num][slot_index]

            # Spectrum is occupied
            if req_id > 0 and req_id not in self.channels_list:
                self.channels_list.append(req_id)
                xci_noise = self._update_link_xci(req_id=req_id, curr_link=curr_link,
                                                  slot_index=slot_index, curr_xci=xci_noise)

        return xci_noise

    def _calculate_pxt(self, num_adjacent: int):
        """
        Calculates the cross-talk noise power.

        :return: The cross-talk noise power normalized by the number of adjacent cores.
        :rtype: float
        """
        # A statistical mean of the cross-talk
        mean_xt = 2 * self.snr_props.link_dict['bending_radius']
        mean_xt *= self.snr_props.link_dict['mode_coupling_co'] ** 2
        mean_xt /= (self.snr_props.link_dict['propagation_const'] * self.snr_props.link_dict['core_pitch'])
        # The cross-talk noise power
        power_xt = num_adjacent * mean_xt * self.snr_props.length * 1e3 * self.engine_props['input_power']

        return power_xt

    @staticmethod
    def calculate_xt(num_adjacent: int, link_length: int):
        """
        Calculates the cross-talk interference based on the number of adjacent cores.

        :return: The cross-talk normalized by the number of adjacent cores.
        :rtype: float
        """
        mean_xt = 3.78e-9
        resp_xt = 1 - math.exp(-2 * mean_xt * link_length * 1e3)
        resp_xt /= (1 + math.exp(-2 * mean_xt * link_length * 1e3))

        return resp_xt * num_adjacent

    def _handle_egn_model(self):
        """
        Calculates the power spectral density correction based on the EGN model.

        :return: The total power spectral density correction
        """
        # The harmonic number series
        hn_series = 0
        for i in range(1, math.ceil((len(self.channels_list) - 1) / 2) + 1):
            hn_series = hn_series + 1 / i

        # The effective span length
        power = -2 * self.snr_props.link_dict['attenuation'] * self.snr_props.length * 10 ** 3
        eff_span_len = 1 - math.e ** power
        eff_span_len /= (2 * self.snr_props.link_dict['attenuation'])
        baud_rate = int(self.snr_props.req_bit_rate) * 10 ** 9 / 2

        temp_coef = self.engine_props['topology_info']['links'][self.link_id]['fiber']['non_linearity'] ** 2
        temp_coef *= eff_span_len ** 2
        temp_coef *= (self.snr_props.center_psd ** 3 * self.snr_props.bandwidth ** 2)
        temp_coef /= ((baud_rate ** 2) * math.pi * self.snr_props.link_dict['dispersion'] *
                      (self.snr_props.length * 10 ** 3))

        # The PSD correction term
        psd_correction = (80 / 81) * self.engine_props['phi'][self.spectrum_props.modulation] * temp_coef * hn_series

        return psd_correction

    def _calculate_psd_nli(self):
        """
        Calculates the power spectral density non-linear interference for a link.

        :return: The total power spectral density non-linear interference
        :rtype float
        """
        psd_nli = self.snr_props.sci_psd + self.snr_props.xci_psd
        psd_nli *= (self.snr_props.mu_param * self.snr_props.center_psd)
        if self.engine_props['egn_model']:
            psd_correction = self._handle_egn_model()
            psd_nli -= psd_correction

        return psd_nli

    def _update_link_params(self, link_num: int):
        """
        Updates needed parameters for each link used for calculating SNR or XT.
        """
        non_linearity = self.engine_props['topology_info']['links'][self.link_id]['fiber']['non_linearity'] ** 2
        self.snr_props.mu_param = 3 * non_linearity
        mu_denominator = 2 * math.pi * self.snr_props.link_dict['attenuation']
        mu_denominator *= np.abs(self.snr_props.link_dict['dispersion'])
        self.snr_props.mu_param /= mu_denominator

        self.snr_props.sci_psd = self._calculate_sci_psd()
        self.snr_props.xci_psd = self._calculate_xci(link_num=link_num)

        self.snr_props.length = self.engine_props['topology_info']['links'][self.link_id]['span_length']
        link_length = self.engine_props['topology_info']['links'][self.link_id]['length']
        self.snr_props.num_span = link_length / self.snr_props.length

    def _init_center_vars(self):
        """
        Updates variables for the center frequency, bandwidth, and PSD for the current request.
        """
        self.snr_props.center_freq = self.spectrum_props.start_slot * self.engine_props['bw_per_slot']
        self.snr_props.center_freq += ((self.num_slots * self.engine_props['bw_per_slot']) / 2)
        self.snr_props.center_freq *= 10 ** 9

        self.snr_props.bandwidth = self.num_slots * self.engine_props['bw_per_slot'] * 10 ** 9
        self.snr_props.center_psd = self.engine_props['input_power'] / self.snr_props.bandwidth

    def check_snr(self):
        """
        Determines whether the SNR threshold can be met for a single request.

        :return: Whether the SNR threshold can be met.
        :rtype: bool
        """
        total_snr = 0
        self._init_center_vars()
        for link_num in range(0, len(self.spectrum_props.path_list) - 1):
            source = self.spectrum_props.path_list[link_num]
            dest = self.spectrum_props.path_list[link_num + 1]
            self.link_id = self.sdn_props.net_spec_dict[(source, dest)]['link_num']

            self.snr_props.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']
            self._update_link_params(link_num=link_num)

            psd_nli = self._calculate_psd_nli()
            psd_ase = self.snr_props.plank * self.snr_props.light_frequency * self.snr_props.nsp
            psd_ase *= (math.exp(self.snr_props.link_dict['attenuation'] * self.snr_props.length * 10 ** 3) - 1)

            if self.engine_props['xt_noise']:
                # fixme number of adjacent set to a constant negative 100
                p_xt = self._calculate_pxt(num_adjacent=-100)
            else:
                p_xt = 0

            curr_snr = self.snr_props.center_psd * self.snr_props.bandwidth
            curr_snr /= (((psd_ase + psd_nli) * self.snr_props.bandwidth + p_xt) * self.snr_props.num_span)

            total_snr += (1 / curr_snr)

        total_snr = 10 * math.log10(1 / total_snr)

        resp = total_snr > self.snr_props.req_snr
        return resp, p_xt

    def check_adjacent_cores(self, link_tuple: tuple):
        """
        Given a link, finds the number of cores which have overlapping channels on a fiber.

        :return: The number of adjacent cores that have overlapping channels.
        """
        resp = 0
        if self.spectrum_props.core_num != 6:
            # The neighboring core directly before the currently selected core
            before = 5 if self.spectrum_props.core_num == 0 else self.spectrum_props.core_num - 1
            # The neighboring core directly after the currently selected core
            after = 0 if self.spectrum_props.core_num == 5 else self.spectrum_props.core_num + 1
            adj_cores_list = [before, after, 6]
        else:
            adj_cores_list = list(range(6))

        for curr_slot in range(self.spectrum_props.start_slot, self.spectrum_props.end_slot):
            overlapped = 0
            for core_num in adj_cores_list:
                band = self.spectrum_props.curr_band
                core_contents = self.sdn_props.net_spec_dict[link_tuple]['cores_matrix'][band][core_num][curr_slot]
                if core_contents > 0.0:
                    overlapped += 1

            # Determine which slot has the maximum number of overlapping channels
            if overlapped > resp:
                resp = overlapped

        return resp

    def find_worst_xt(self, flag: str):
        """
        Finds the worst possible cross-talk.

        :param flag: Determines which type of cross-talk is being considered.
        :type flag: str

        :return: The maximum length of the link found and the cross-talk calculated.
        :rtype: tuple
        """
        if flag == 'intra_core':
            edge_lengths = nx.get_edge_attributes(self.engine_props['topology'], 'length')
            max_link = max(edge_lengths, key=edge_lengths.get, default=None)

            self.link_id = self.sdn_props.net_spec_dict[max_link]['link_num']
            max_length = edge_lengths.get(max_link, 0.0)
            self.snr_props.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']

            resp = self.calculate_xt(num_adjacent=6, link_length=max_length)
            resp = 10 * math.log10(resp)
        else:
            raise NotImplementedError

        return resp, max_length

    def check_xt(self):
        """
        Checks the amount of cross-talk interference on a single request.

        :return: Whether the cross-talk interference threshold can be met
        :rtype: bool
        """
        cross_talk = 0

        self._init_center_vars()
        for link_num in range(0, len(self.spectrum_props.path_list) - 1):
            link_tuple = (self.spectrum_props.path_list[link_num], self.spectrum_props.path_list[link_num + 1])

            self.link_id = self.sdn_props.net_spec_dict[link_tuple]['link_num']
            link_length = self.engine_props['topology_info']['links'][self.link_id]['length']
            self.snr_props.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']
            self._update_link_params(link_num=link_num)

            num_adjacent = self.check_adjacent_cores(link_tuple=link_tuple)
            cross_talk += self.calculate_xt(num_adjacent=num_adjacent, link_length=link_length)

        if cross_talk == 0:
            resp = True
        else:
            cross_talk = 10 * math.log10(cross_talk)
            resp = cross_talk < self.engine_props['requested_xt'][self.spectrum_props.modulation]

        return resp, cross_talk

    def find_num_adjacent_cores(self):
        """
        Finds the number of adjacent cores for selected core.

        :return: The number of adjacent cores.
        """
        resp = 0
        if self.engine_props['cores_per_link'] == 4:
            resp = 2
        elif self.engine_props['cores_per_link'] == 7:
            resp = 6 if self.spectrum_props.core_num == 6 else 3
        elif self.engine_props['cores_per_link'] == 13:
            if self.spectrum_props.core_num < 6:
                resp = 2
            elif 6 <= self.spectrum_props.core_num < 12:
                resp = 5
            elif self.spectrum_props.core_num == 12:
                resp = 6
        elif self.engine_props['cores_per_link'] == 19:
            if self.spectrum_props.core_num >= 12:
                resp = 6
            elif self.spectrum_props.core_num % 2 == 0:
                resp = 3
            else:
                resp = 4
        return resp

    def check_snr_ext(self, path_index: int):
        """
        Checks the SNR on a single request using the external resources.

        :return: Whether the SNR threshold can be met and SNR value.
        :rtype: tuple
        """
        # Fetch loaded files
        if self.engine_props['multi_fiber']:
            num_adjacent = 0
        else:
            num_adjacent = self.find_num_adjacent_cores()
        loaded_data, loaded_data_gsnr = get_loaded_files(
            num_adjacent, self.engine_props['cores_per_link'],
            self.snr_props.file_mapping_dict,
            network=self.engine_props['network'],
        )

        # Compute slot index
        slot_index = get_slot_index(
            self.spectrum_props.curr_band, self.spectrum_props.start_slot, self.engine_props
        )

        # Fetch modulation format and SNR value
        mod_format = loaded_data[self.route_props.connection_index][slot_index][path_index]
        snr_val = loaded_data_gsnr[self.route_props.connection_index][slot_index][path_index]

        # Determine response
        resp = compute_response(mod_format, self.snr_props, self.spectrum_props, self.sdn_props)

        return resp, snr_val

    def check_snr_ext_slicing(self, path_index):
        """
        Checks the SNR on a single request using the external resources for slicing.

        :return: Modulation format, supported bandwidth, and SNR value.
        :rtype: tuple
        """
        # Fetch loaded files
        if self.engine_props['multi_fiber']:
            num_adjacent = 0
        else:
            num_adjacent = self.find_num_adjacent_cores()
        loaded_data, loaded_data_gsnr = get_loaded_files(
            num_adjacent, self.engine_props['cores_per_link'],
            self.snr_props.file_mapping_dict,
            network=self.engine_props['network']
        )

        # Compute slot index
        slot_index = get_slot_index(
            self.spectrum_props.curr_band, self.spectrum_props.start_slot, self.engine_props
        )

        # Retrieve modulation format and supported bandwidth
        mod_format_key = loaded_data[self.route_props.connection_index][slot_index][path_index]
        if mod_format_key == 0:
            mod_format = None
            supported_bw = 0
        else:
            mod_format = self.snr_props.mod_format_mapping_dict[mod_format_key]
            supported_bw = self.snr_props.bw_mapping_dict[mod_format]

        # Retrieve SNR value
        snr_val = loaded_data_gsnr[self.route_props.connection_index][slot_index][path_index]

        return mod_format, supported_bw, snr_val

    def check_snr_ext_open_slots(self, path_index, open_slots_list):
        """
        Checks the SNR on a single request using the external resources for slicing.

        :return: Modulation format, supported bandwidth, and SNR value.
        :rtype: tuple
        """
        # Fetch loaded files
        if self.engine_props['multi_fiber']:
            num_adjacent = 0
        else:
            num_adjacent = self.find_num_adjacent_cores()
        loaded_data, _ = get_loaded_files(
            num_adjacent, self.engine_props['cores_per_link'],
            self.snr_props.file_mapping_dict,
            self.engine_props['network']
        )

        # Retrieve modulation format and supported bandwidth
        for open_slot in open_slots_list[:]:
            slot_index = get_slot_index(
                self.spectrum_props.curr_band, open_slot, self.engine_props
            )
            mod_format_key = loaded_data[self.route_props.connection_index][slot_index][path_index]
            if mod_format_key == 0:
                open_slots_list.remove(open_slot)

        return open_slots_list

    def handle_snr(self, path_index):
        """
        Controls the methods of this class.

        :return: Whether snr is acceptable for allocation or not for a given request and its cost
        :rtype: tuple
        """
        self.num_slots = self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1
        if self.engine_props['snr_type'] == "snr_calc_nli":
            snr_check, xt_cost = self.check_snr()
        elif self.engine_props['snr_type'] == "xt_calculation":
            snr_check, xt_cost = self.check_xt()
        elif self.engine_props['snr_type'] == "snr_e2e_external_resources":
            snr_check, xt_cost = self.check_snr_ext(path_index)
        else:
            raise NotImplementedError(f"Unexpected snr_type flag got: {self.engine_props['snr_type']}")

        return snr_check, xt_cost

    def handle_snr_dynamic_slicing(self, path_index):
        """
        Controls the methods of this class.

        :return: Whether snr is acceptable for allocation or not for a given request and its cost
        :rtype: tuple
        """
        self.num_slots = self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1
        if self.engine_props['snr_type'] == "snr_e2e_external_resources":
            mod_format, bandwidth, snr_val = self.check_snr_ext_slicing(path_index)
        else:
            raise NotImplementedError(f"Unexpected snr_type flag got: {self.engine_props['snr_type']}")

        return mod_format, bandwidth, snr_val
