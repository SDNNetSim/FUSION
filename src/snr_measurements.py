import math

import numpy as np
import networkx as nx

from arg_scripts.snr_args import SNRProps
from helper_scripts.snr_helpers import get_slot_index, get_loaded_files, compute_response
from helper_scripts.sim_helpers import sort_nested_dict_vals


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
        # TODO: update XCI and SCI and ASE based on actual center frequency 
        # self.snr_props.center_freq += self.snr_props.link_dict['frequency_start_c'] 

        self.snr_props.bandwidth = self.num_slots * self.engine_props['bw_per_slot'] * 10 ** 9
        self.snr_props.center_psd = self.engine_props['input_power'] / self.snr_props.bandwidth

    def _calculate_psd_ase(self):
        """
        Calculates the power spectral density ASE noise for a link.

        :return: The total power spectral density ASE noise
        :rtype float
        """
        psd_ase = self.snr_props.plank * self.snr_props.light_frequency * self.snr_props.nsp[self.spectrum_props.curr_band]
        psd_ase *= (math.exp(self.snr_props.link_dict['attenuation'] * self.snr_props.length * 10 ** 3) - 1)
        return psd_ase


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
            psd_ase = self._calculate_psd_ase()

            if self.engine_props['xt_noise']:
                # fixme number of adjacent set to a constant negative 100
                p_xt = self._calculate_pxt(num_adjacent=-100)
            else:
                p_xt = 0

            curr_snr = self.snr_props.center_psd * self.snr_props.bandwidth
            curr_snr /= (((psd_ase + psd_nli) * self.snr_props.bandwidth + p_xt) * self.snr_props.num_span)

            total_snr += (1 / curr_snr)

        total_snr = 10 * math.log10(1 / total_snr)

        resp = total_snr > self.snr_props.req_snr['QPSK']
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


    def check_gsnr(self):
        """
        Calculate and check the SNR of Lightpath.

        :return: Whether the SNR threshold can be met, SNR value, Suppoted bitrate.
        :rtyp
        """

        bw_mapping = {
        "64-QAM": 800,
        "32-QAM": 700,
        "16-QAM": 600,
        "8-QAM": 500,
        "QPSK": 400,
        "BPSK": 200
        }

        gsnr_path_ase_nli = 0
        gsnr_link_ase_nli_db = []
        gsnr_span_ase_nli_db = []
        for link_num in range(0, len(self.spectrum_props.path_list) - 1):
            source = self.spectrum_props.path_list[link_num]
            dest = self.spectrum_props.path_list[link_num + 1]
            self.link_id = self.sdn_props.net_spec_dict[(source, dest)]['link_num']
            self.snr_props.length = self.engine_props['topology_info']['links'][self.link_id]['span_length']
            self.snr_props.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']
            center_freq = self.snr_props.link_dict['frequency_start_c'] + self.spectrum_props.start_slot * self.engine_props['bw_per_slot'] * 10 ** 9
            center_freq += ((self.num_slots * self.engine_props['bw_per_slot']) / 2) * 10 ** 9
            self.snr_props.bandwidth = self.num_slots * self.engine_props['bw_per_slot'] * 10 ** 9
            gsnr_link_ase_nli = 0

            self.snr_props.num_span =  math.ceil(self.engine_props['topology_info']['links'][self.link_id]['length'] / self.snr_props.length)
            p_ase_span = self.snr_props.bandwidth * self.snr_props.plank * center_freq * 2 * self.snr_props.nsp[self.spectrum_props.curr_band]
            p_ase_span *= (math.exp(self.snr_props.link_dict['attenuation'] * self.snr_props.length * 10 ** 3) - 1)
            l_eff = (1 - np.exp(-2 * self.snr_props.link_dict['attenuation'] * self.snr_props.length * 10 ** 3)) / (2 * self.snr_props.link_dict['attenuation'])
            l_eff_a = 1 / (2 * self.snr_props.link_dict['attenuation'])


            for _ in range(self.snr_props.num_span):
                self.channels_list = []
                sum_phi = 0
                for slot_index in range(self.engine_props['c_band']):
                    curr_link = self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix']['c'][self.spectrum_props.core_num]
                    req_id = curr_link[self.spectrum_props.core_num][slot_index]

                    # Spectrum is occupied
                    if (req_id > 0 and req_id not in self.channels_list) or slot_index == self.spectrum_props.start_slot:
                        channel_mod = None
                        if req_id > 0:
                            for _, lp_dict in self.sdn_props.lightpath_status_dict.items():
                                if req_id in lp_dict:
                                    channel_mod = lp_dict[req_id]['mod_format']

                            channel_bw = len(np.where(req_id == curr_link[self.spectrum_props.core_num])[0])
                            self.channels_list.append(req_id)
                        else:
                            channel_bw = self.num_slots
                        if  channel_mod is None and req_id >0:
                            if req_id in self.sdn_props.lightpath_id_list:
                                idx = self.sdn_props.lightpath_id_list.index(req_id)
                                channel_mod = self.sdn_props.modulation_list[idx]
                            else:
                                raise NotImplementedError(f"Unexpected lightpath id: {req_id}")
                        channel_bw *= self.engine_props['bw_per_slot']
                        channel_freq = self.snr_props.link_dict['frequency_start_c'] + ((slot_index * self.engine_props['bw_per_slot']) + (channel_bw / 2)) * 10 ** 9
                        channel_bw *= 10 ** 9
                        if center_freq == channel_freq:
                            phi = np.arcsinh(np.pi**2 * abs(self.snr_props.link_dict['dispersion']) * self.snr_props.bandwidth **2 / (4 * self.snr_props.link_dict['attenuation']))
                        else:
                            phi = (
                                np.arcsinh((np.pi**2) * abs(self.snr_props.link_dict['dispersion']) * l_eff_a * self.snr_props.bandwidth * (channel_freq - center_freq + (channel_bw * 0.5))) -
                                np.arcsinh((np.pi**2) * abs(self.snr_props.link_dict['dispersion']) * l_eff_a * self.snr_props.bandwidth * (channel_freq - center_freq - (channel_bw * 0.5))) -
                                (self.engine_props['phi'][channel_mod] * (channel_bw / abs(channel_freq - center_freq)) * (5 / 3) * (l_eff / (self.snr_props.length * 10 ** 3)))
                            )
                        sum_phi += phi
                p_nli_span = (self.engine_props['input_power'] / self.snr_props.bandwidth)**3 * (8 / (27 * np.pi * abs(self.snr_props.link_dict['dispersion']))) * self.snr_props.link_dict['non_linearity']**2 * l_eff * sum_phi * self.snr_props.bandwidth
                gsnr_span_ase_nli_db.append(10 * np.log10(self.engine_props['input_power'] / (p_ase_span + p_nli_span)))
                gsnr_link_ase_nli += (self.engine_props['input_power'] / (p_ase_span + p_nli_span))**-1
            gsnr_link_ase_nli_db.append(10 * np.log10(gsnr_link_ase_nli**-1))
            gsnr_path_ase_nli += gsnr_link_ase_nli
        gsnr_db = 10 * np.log10(1 / gsnr_path_ase_nli)
        resp = gsnr_db >= self.snr_props.req_snr[self.spectrum_props.modulation]
        bw_resp = 0
        if resp:
            if self.engine_props['fixed_grid']:
                if not self.spectrum_props.slicing_flag:
                    if bw_mapping[self.spectrum_props.modulation] >= int(self.sdn_props.bandwidth):
                        bw_resp = bw_mapping[self.spectrum_props.modulation]
                    else:
                        resp = False
                else:
                    bw_resp = bw_mapping[self.spectrum_props.modulation]
            else:
                if self.spectrum_props.slicing_flag:
                    bw_resp = None
                else:
                    bw_resp = int(self.sdn_props.bandwidth)

        if self.spectrum_props.slicing_flag and self.engine_props['fixed_grid'] and self.engine_props['dynamic_lps']:
            mod_formats_dict = sort_nested_dict_vals(original_dict=self.sdn_props.mod_formats_dict,
                                            nested_key='max_length')
            force_mod_format = list(mod_formats_dict.keys())
            for mod in force_mod_format:
                if gsnr_db >= self.snr_props.req_snr[mod]:
                    resp = mod
                    bw_resp = bw_mapping[mod]
                    break
        return resp, gsnr_db, bw_resp


    def _compute_nli_mb(self, source, dest, p_total):
        """
            Computes the nonlinear interference GSNR (SNR_NLI) of single link using Eq. (3) in https://doi.org/10.1049/cp.2019.0892 .

            :param source: Determines link start node.
            :type source: str
            :param dest: Determines link end node.
            :type dest: str
            :param p_total: Link's power profile.
            :type p_total: float


            :return: Inverse nonlinear SNR value of given link.
            :rtype: float

        """
        f_c = (self.snr_props.link_dict['frequency_end_c'] + self.snr_props.link_dict['frequency_start_l'])/2 # (self.snr_props.link_dict['frequency_start_c'] + self.snr_props.link_dict['frequency_start_l'])/2
        gamma = self.snr_props.link_dict['non_linearity']
        alpha_i = self.snr_props.link_dict['attenuation']
        alpha_bar_i = self.snr_props.link_dict['attenuation']
        c_ri = self.snr_props.link_dict['raman_gain_slope']
        beta_2 = self.snr_props.link_dict['gvd']
        beta_3 = self.snr_props.link_dict['gvd_slope']
        p_i = self.engine_props['input_power']
        pi = np.pi
        bi = ((self.num_slots * self.engine_props['bw_per_slot'])) * 10 ** 9
        f_i = self.snr_props.link_dict['frequency_start_' + self.spectrum_props.curr_band] + self.spectrum_props.start_slot * self.engine_props['bw_per_slot'] * 10 ** 9
        f_i += (bi / 2)
        ti = (alpha_i + alpha_bar_i - p_total * c_ri * (f_i-f_c))**2
        ai = alpha_i + alpha_bar_i
        phi_i = (1.5* pi ** 2) * (beta_2 + 2 * beta_3 * (f_i-f_c))
        self.channels_list = []
        spm = 0
        xpm = 0
        epsilon = 0.0001
        for band in self.engine_props['band_list']:
            for slot_index in range(self.engine_props[band + '_band']):
                curr_link = self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band]
                req_id = curr_link[self.spectrum_props.core_num][slot_index]
                self.link_id = self.sdn_props.net_spec_dict[(source, dest)]['link_num']
                self.snr_props.length = self.engine_props['topology_info']['links'][self.link_id]['span_length']
                l = self.snr_props.length * 1000
                if (req_id > 0 and req_id not in self.channels_list) or (slot_index == self.spectrum_props.start_slot and band == self.spectrum_props.curr_band ):
                    ch_k_mod = None
                    if req_id > 0:
                        for _, lp_dict in self.sdn_props.lightpath_status_dict.items():
                            if req_id in lp_dict:
                                ch_k_mod = lp_dict[req_id]['mod_format']

                        b_k = len(np.where(req_id == curr_link[self.spectrum_props.core_num])[0])
                        p_k = self.engine_props['input_power']
                        self.channels_list.append(req_id)
                    else:
                        b_k = self.num_slots
                    if  ch_k_mod is None and req_id >0:
                        if req_id in self.sdn_props.lightpath_id_list:
                            idx = self.sdn_props.lightpath_id_list.index(req_id)
                            ch_k_mod = self.sdn_props.modulation_list[idx]
                        else:
                            raise NotImplementedError(f"Unexpected lightpath id: {req_id}")
                    alpha_k = self.snr_props.link_dict['attenuation']
                    alpha_bar_k = self.snr_props.link_dict['attenuation']
                    b_k *= self.engine_props['bw_per_slot']
                    f_k = self.snr_props.link_dict['frequency_start_' + band] + ((slot_index * self.engine_props['bw_per_slot']) + (b_k / 2)) * 10 ** 9
                    b_k *= 10 ** 9
                    if f_i == f_k:
                        spm += (4 / 9) * ((pi * (gamma ** 2) * (self.snr_props.num_span ** (1+epsilon)) * (p_i ** 2)) / ((bi ** 2) * phi_i * alpha_bar_i * (2 * alpha_i + alpha_bar_i) ) ) * (
                            ((ti- (alpha_i ** 2)) / alpha_i) * np.arcsinh((phi_i * (bi ** 2)) / (pi * alpha_i)) +
                            ((ai ** 2 - ti) / ai) * np.arcsinh((phi_i * (bi ** 2)) / (pi * ai))
                        )
                    else:
                        fk_fi = abs(f_k - f_i)
                        tk = (alpha_k + alpha_bar_k - p_total * c_ri * (f_k-f_c))**2
                        ak = alpha_k + alpha_bar_k
                        n_tilde =  0 if self.snr_props.num_span == 1 else self.snr_props.num_span
                        phi_i_k = (-4 * pi**2 * (f_k - f_i)) * (beta_2 + pi * beta_3 *(f_i + f_k - 2 * f_c))
                        phi = (-4 * pi**2 * (beta_2 + pi * beta_3 * (f_i + f_k )) ) * l
                        xpm += (32 / 27) * (((gamma ** 2) * (p_k ** 2)) / b_k) * (
                            ((self.snr_props.num_span + (5 / 6) * ( -1 * self.engine_props['phi'][ch_k_mod])) / (phi_i_k * alpha_bar_k * (2 * alpha_k + alpha_bar_k))) * (
                                ((tk - (alpha_k ** 2)) / alpha_k) * np.arctan((phi_i_k * bi) / alpha_k) +
                                (((ak ** 2) - tk) / ak) * np.arctan((phi_i_k * bi) / ak)
                            ) +
                            (5 / 3) * ((-1 * self.engine_props['phi'][ch_k_mod] * pi * n_tilde * tk) / (abs(phi) * (b_k ** 2) * (alpha_k ** 2) * (ak ** 2))) * (
                                (2 * fk_fi - b_k) * math.log((2 * fk_fi - b_k) / (2 * fk_fi + b_k)) + 2 * b_k
                            )
                        )
        snr_nli_inv = spm + xpm
        return snr_nli_inv


    def _compute_ase_mb(self, source, dest, p_total):
        """
            Computes the ASE SNR (P_ASE) for MB-EONs using Eq. (5) in https://doi.org/10.1364/JOCN.515081.

            :param source: Determines link start node.
            :type source: str
            :param dest: Determines link end node.
            :type dest: str
            :param p_total: Link's power profile.
            :type p_total: float


            :return: Inverse ASE SNR value of given link.
            :rtype: float
        """
        p_ase_link = 0.0
        c_ri = self.snr_props.link_dict['raman_gain_slope']
        b_i = ((self.num_slots * self.engine_props['bw_per_slot'])) * 10 ** 9
        f_i = self.snr_props.link_dict['frequency_start_' + self.spectrum_props.curr_band] + self.spectrum_props.start_slot * self.engine_props['bw_per_slot'] * 10 ** 9
        f_i += (b_i / 2)
        alpha = self.snr_props.link_dict['attenuation']
        self.channels_list = []
        p_profile = 0.0
        l_eff = (1 - np.exp(-alpha * self.snr_props.length * 10 ** 3)) / (alpha)
        p_profile += (2/b_i) * np.exp(-p_total * c_ri * l_eff * f_i) * np.sinh(p_total * c_ri * l_eff * (b_i/2))
        for band in self.engine_props['band_list']:
            for slot_index in range(self.engine_props[band + '_band']):
                curr_link = self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band]
                req_id = curr_link[self.spectrum_props.core_num][slot_index]
                if (req_id > 0 and req_id not in self.channels_list):
                    b_k = len(np.where(req_id == curr_link[self.spectrum_props.core_num])[0]) * self.engine_props['bw_per_slot']
                    f_k = self.snr_props.link_dict['frequency_start_' + band] + ((slot_index * self.engine_props['bw_per_slot']) + (b_k / 2)) * 10 ** 9
                    b_k *= 10 ** 9
                    p_profile += (2/b_k) * np.exp(-p_total * c_ri * l_eff * f_k) * np.sinh(p_total * c_ri * l_eff * (b_k/2))
                    self.channels_list.append(req_id)


        p_profile = (p_total**2 * c_ri * l_eff * np.exp((-alpha * self.snr_props.length * 10 ** 3) - (p_total * c_ri * l_eff * f_i))) / p_profile
        p_ase_span = 2 * self.snr_props.nsp[self.spectrum_props.curr_band] * self.snr_props.plank * f_i * b_i * ( (self.engine_props['input_power']/p_profile)- 1)
        p_ase_link = p_ase_span * self.snr_props.num_span

        snr_ase_inv = p_ase_link / self.engine_props['input_power']
        return snr_ase_inv


    def _gsnr_calc_mb(self):
        """
            Calculate GSNR for multi band EON (C+L band) for single request.


            :return: GSNR value of given resources.
            :rtype: float
        """
        snr_nli_inv = 0.0
        snr_ase_inv = 0.0
        snr_lp_inv = 0.0
        for link_num in range(0, len(self.spectrum_props.path_list) - 1):
            source = self.spectrum_props.path_list[link_num]
            dest = self.spectrum_props.path_list[link_num + 1]
            self.link_id = self.sdn_props.net_spec_dict[(source, dest)]['link_num']
            self.snr_props.link_dict = self.engine_props['topology_info']['links'][self.link_id]['fiber']
            self.snr_props.length = self.engine_props['topology_info']['links'][self.link_id]['span_length']
            self.snr_props.num_span =  math.ceil(self.engine_props['topology_info']['links'][self.link_id]['length'] / self.snr_props.length)
            p_total = 1
            number_lp = set()
            for band in self.engine_props['band_list']:
                curr_link = self.sdn_props.net_spec_dict[(source, dest)]['cores_matrix'][band][self.spectrum_props.core_num]
                number_lp.update(np.unique(curr_link[curr_link > 0]))
            p_total += len(number_lp)
            p_total *= self.engine_props['input_power']
            snr_nli_inv_l = self._compute_nli_mb(source, dest, p_total)
            snr_ase_inv_l = self._compute_ase_mb(source, dest, p_total)
            snr_nli_inv += snr_nli_inv_l
            snr_ase_inv += snr_ase_inv_l
        snr_lp_inv += (snr_nli_inv + snr_ase_inv)
        gsnr = 1 / snr_lp_inv
        return gsnr

    def check_gsnr_mb(self):
        """
        check GSNR of multiband and create reposnse based on request and modulation type

        :return: Whether GSNR is acceptable for allocation or not for a given request, GSNR value, supported bandwidth,.
        :rtype: tuple
        """

        bw_mapping = {
        "64-QAM": 800,
        "32-QAM": 700,
        "16-QAM": 600,
        "8-QAM": 500,
        "QPSK": 400,
        "BPSK": 200
        }

        gsnr = self._gsnr_calc_mb()
        gsnr_db = 10 * np.log10(gsnr)
        resp = gsnr_db >= self.snr_props.req_snr[self.spectrum_props.modulation] + self.snr_props.snr_margin
        bw_resp = 0
        if resp:
            if self.engine_props['fixed_grid']:
                if not self.spectrum_props.slicing_flag:
                    if bw_mapping[self.spectrum_props.modulation] >= int(self.sdn_props.bandwidth):
                        bw_resp = bw_mapping[self.spectrum_props.modulation]
                    else:
                        resp = False
                else:
                    bw_resp = bw_mapping[self.spectrum_props.modulation]
            else:
                if self.spectrum_props.slicing_flag:
                    bw_resp = None
                else:
                    bw_resp = int(self.sdn_props.bandwidth)

        if self.spectrum_props.slicing_flag and self.engine_props['fixed_grid'] and self.engine_props['dynamic_lps']:
            mod_formats_dict = sort_nested_dict_vals(original_dict=self.sdn_props.mod_formats_dict,
                                            nested_key='max_length')
            force_mod_format = list(mod_formats_dict.keys())
            for mod in force_mod_format:
                if gsnr_db >= self.snr_props.req_snr[mod]:
                    resp = mod
                    bw_resp = bw_mapping[mod]
                    break
        return resp, gsnr_db, bw_resp


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
        elif self.engine_props['snr_type'] == "gsnr":
            if self.engine_props['band_list'] == ['c']:
                snr_check, xt_cost, bw_resp = self.check_gsnr()
            elif self.engine_props['band_list'] == ['c', 'l']:
                snr_check, xt_cost, bw_resp = self.check_gsnr_mb()
        elif self.engine_props['snr_type'] == "snr_e2e_external_resources":
            snr_check, xt_cost, bw_resp = self.check_snr_ext(path_index)
        else:
            raise NotImplementedError(f"Unexpected snr_type flag got: {self.engine_props['snr_type']}")

        return snr_check, xt_cost, bw_resp

    def handle_snr_dynamic_slicing(self, path_index):
        """
        Controls the methods of this class.

        :return: Whether snr is acceptable for allocation or not for a given request and its cost
        :rtype: tuple
        """
        self.num_slots = self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1
        if self.engine_props['snr_type'] == "snr_e2e_external_resources":
            mod_format, bandwidth, snr_val = self.check_snr_ext_slicing(path_index)
        elif self.engine_props['snr_type'] == "gsnr":
            if self.engine_props['band_list'] == ['c']:
                mod_format, snr_val, bandwidth = self.check_gsnr()
            elif self.engine_props['band_list'] == ['c', 'l']:
                mod_format, snr_val, bandwidth  = self.check_gsnr_mb()
        else:
            raise NotImplementedError(f"Unexpected snr_type flag got: {self.engine_props['snr_type']}")

        return mod_format, bandwidth, snr_val
