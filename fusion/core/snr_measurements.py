"""
SNR measurements module for signal-to-noise ratio calculations.

This module provides functionality for calculating SNR, cross-talk interference,
and other signal quality metrics for optical network requests.
"""

import math
from typing import Any

import networkx as nx
import numpy as np

from fusion.core.properties import RoutingProps, SDNProps, SNRProps, SpectrumProps
from fusion.modules.snr.utils import compute_response, get_loaded_files, get_slot_index

# Constants for calculations
POWER_CONVERSION_FACTOR = 10**9  # Convert to GHz
LENGTH_CONVERSION_FACTOR = 1e3  # Convert km to m
DB_CONVERSION_FACTOR = 10  # For dB calculations
MEAN_XT_CONSTANT = 3.78e-9  # Mean cross-talk constant
EGN_COEFFICIENT = 80 / 81  # EGN model coefficient
ADJACENT_CORES_PLACEHOLDER = -100  # Placeholder for adjacent cores calculation


class SnrMeasurements:
    """
    Handle signal-to-noise ratio calculations for a given request.

    This class provides methods for calculating SNR, cross-talk interference,
    and other signal quality metrics for optical network requests.
    Note: Currently optimized for seven-core fibers.

    :param engine_props_dict: Engine configuration properties
    :type engine_props_dict: dict[str, Any]
    :param sdn_props: SDN properties object
    :type sdn_props: SDNProps
    :param spectrum_props: Spectrum properties object
    :type spectrum_props: SpectrumProps
    :param route_props: Routing properties object
    :type route_props: RoutingProps
    """

    def __init__(
        self,
        engine_props_dict: dict[str, Any],
        sdn_props: SDNProps,
        spectrum_props: SpectrumProps,
        route_props: RoutingProps,
    ) -> None:
        self.snr_props = SNRProps()
        self.engine_props_dict = engine_props_dict
        self.sdn_props = sdn_props
        self.spectrum_props = spectrum_props
        self.route_props = route_props

        self.channels_list: list[float] | None = None
        self.link_id: int | None = None
        self.number_of_slots: int | None = None

    def _calculate_sci_psd(self) -> float:
        """
        Calculate the self-phase power spectral density.

        :return: The self-phase power spectral density
        :rtype: float
        """
        if (self.snr_props.link_dictionary is None or
            self.snr_props.center_psd is None or
            self.snr_props.bandwidth is None):
            raise ValueError("Required SNR properties are not initialized")

        rho_parameter = (math.pi**2) * np.abs(
            self.snr_props.link_dictionary["dispersion"]
        )
        rho_parameter /= 2 * self.snr_props.link_dictionary["attenuation"]

        sci_psd = self.snr_props.center_psd**2
        sci_psd *= math.asinh(rho_parameter * (self.snr_props.bandwidth**2))
        return sci_psd

    def _update_link_xci(
        self,
        request_id: float,
        current_link: np.ndarray,
        slot_index: int,
        current_xci: float,
    ) -> float:
        """
        Update the link's cross-phase modulation noise.

        Given the spectrum contents, updates the link's cross-phase modulation noise.

        :param request_id: Request identifier
        :type request_id: float
        :param current_link: Current link spectrum array
        :type current_link: np.ndarray
        :param slot_index: Spectrum slot index
        :type slot_index: int
        :param current_xci: Current cross-phase modulation noise
        :type current_xci: float
        :return: The updated cross-phase modulation noise
        :rtype: float
        """
        channel_bandwidth = len(
            np.where(request_id == current_link[self.spectrum_props.core_number])[0]
        )
        channel_bandwidth *= self.engine_props_dict["bw_per_slot"]
        channel_frequency = (
            (slot_index * self.engine_props_dict["bw_per_slot"])
            + (channel_bandwidth / 2)
        ) * POWER_CONVERSION_FACTOR
        channel_bandwidth *= POWER_CONVERSION_FACTOR
        channel_psd = self.engine_props_dict["input_power"] / channel_bandwidth

        if (self.snr_props.center_frequency is not None and
            self.snr_props.center_frequency != channel_frequency):
            log_term = abs(self.snr_props.center_frequency - channel_frequency) + (
                channel_bandwidth / 2
            )
            log_term /= abs(self.snr_props.center_frequency - channel_frequency) - (
                channel_bandwidth / 2
            )
            calculated_xci = (channel_psd**2) * math.log(abs(log_term))
            new_xci = current_xci + calculated_xci
        else:
            new_xci = current_xci

        return float(new_xci)

    def _calculate_xci(self, link_number: int) -> float:
        """
        Calculate the cross-phase modulation noise on a link.

        :param link_number: Link number in the path
        :type link_number: int
        :return: The total cross-phase modulation noise on the link
        :rtype: float
        """
        if (self.spectrum_props.path_list is None or
            self.sdn_props.network_spectrum_dict is None):
            raise ValueError("Required spectrum properties are not initialized")

        self.channels_list = []
        # Cross-phase modulation noise
        xci_noise = 0.0
        for slot_index in range(self.engine_props_dict["c_band"]):
            source = self.spectrum_props.path_list[link_number]
            dest = self.spectrum_props.path_list[link_number + 1]
            current_link = self.sdn_props.network_spectrum_dict[(source, dest)][
                "cores_matrix"
            ]["c"]
            request_id = current_link[self.spectrum_props.core_number][slot_index]

            # Spectrum is occupied
            if request_id > 0 and request_id not in self.channels_list:
                self.channels_list.append(request_id)
                xci_noise = self._update_link_xci(
                    request_id=request_id,
                    current_link=current_link,
                    slot_index=slot_index,
                    current_xci=xci_noise,
                )

        return xci_noise

    def _calculate_pxt(self, number_of_adjacent_cores: int) -> float:
        """
        Calculate the cross-talk noise power.

        :param number_of_adjacent_cores: Number of adjacent cores
        :type number_of_adjacent_cores: int
        :return: Cross-talk noise power normalized by adjacent cores
        :rtype: float
        """
        if self.snr_props.link_dictionary is None:
            raise ValueError("Link dictionary is not initialized")

        # A statistical mean of the cross-talk
        mean_xt = 2 * self.snr_props.link_dictionary["bending_radius"]
        mean_xt *= self.snr_props.link_dictionary["mode_coupling_co"] ** 2
        mean_xt /= (
            self.snr_props.link_dictionary["propagation_const"]
            * self.snr_props.link_dictionary["core_pitch"]
        )
        # The cross-talk noise power
        power_xt = (
            number_of_adjacent_cores
            * mean_xt
            * self.snr_props.length
            * LENGTH_CONVERSION_FACTOR
            * self.engine_props_dict["input_power"]
        )

        return float(power_xt)

    @staticmethod
    def calculate_xt(number_of_adjacent_cores: int, link_length: int) -> float:
        """
        Calculate cross-talk interference based on adjacent cores.

        :param number_of_adjacent_cores: Number of adjacent cores
        :type number_of_adjacent_cores: int
        :param link_length: Length of the link
        :type link_length: int
        :return: Cross-talk normalized by the number of adjacent cores
        :rtype: float
        """
        mean_xt = MEAN_XT_CONSTANT
        resp_xt = 1 - math.exp(-2 * mean_xt * link_length * LENGTH_CONVERSION_FACTOR)
        resp_xt /= 1 + math.exp(-2 * mean_xt * link_length * LENGTH_CONVERSION_FACTOR)

        return resp_xt * number_of_adjacent_cores

    def _handle_egn_model(self) -> float:
        """
        Calculates the power spectral density correction based on the EGN model.

        :return: The total power spectral density correction
        """
        if (self.channels_list is None or
            self.snr_props.link_dictionary is None or
            self.snr_props.length is None):
            raise ValueError("Required properties are not initialized")

        # The harmonic number series
        hn_series = 0.0
        for i in range(1, math.ceil((len(self.channels_list) - 1) / 2) + 1):
            hn_series = hn_series + 1 / i

        # The effective span length
        power = (
            -2
            * self.snr_props.link_dictionary["attenuation"]
            * self.snr_props.length
            * LENGTH_CONVERSION_FACTOR
        )
        effective_span_length = 1 - math.e**power
        effective_span_length /= 2 * self.snr_props.link_dictionary["attenuation"]
        baud_rate = int(self.snr_props.request_bit_rate) * 10**9 / 2

        temp_coefficient = (
            self.engine_props_dict["topology_info"]["links"][self.link_id]["fiber"][
                "non_linearity"
            ]
            ** 2
        )
        temp_coefficient *= effective_span_length**2
        if (self.snr_props.center_psd is None or self.snr_props.bandwidth is None):
            raise ValueError("Center PSD and bandwidth must be initialized")
        temp_coefficient *= self.snr_props.center_psd**3 * self.snr_props.bandwidth**2
        temp_coefficient /= (
            (baud_rate**2)
            * math.pi
            * self.snr_props.link_dictionary["dispersion"]
            * (self.snr_props.length * LENGTH_CONVERSION_FACTOR)
        )

        # The PSD correction term
        if self.spectrum_props.modulation is None:
            raise ValueError("Modulation format must be initialized")
        psd_correction = (
            EGN_COEFFICIENT
            * self.engine_props_dict["phi"][self.spectrum_props.modulation]
            * temp_coefficient
            * hn_series
        )

        return float(psd_correction)

    def _calculate_psd_nli(self) -> float:
        """
        Calculates the power spectral density non-linear interference for a link.

        :return: The total power spectral density non-linear interference
        :rtype float
        """
        if (self.snr_props.self_channel_interference_psd is None or
            self.snr_props.cross_channel_interference_psd is None or
            self.snr_props.mu_parameter is None or
            self.snr_props.center_psd is None):
            raise ValueError("Required interference PSD properties are not initialized")

        psd_nli = (
            self.snr_props.self_channel_interference_psd
            + self.snr_props.cross_channel_interference_psd
        )
        psd_nli *= self.snr_props.mu_parameter * self.snr_props.center_psd
        if self.engine_props_dict["egn_model"]:
            psd_correction = self._handle_egn_model()
            psd_nli -= psd_correction

        return psd_nli

    def _update_link_parameters(self, link_number: int) -> None:
        """
        Updates needed parameters for each link used for calculating SNR or XT.
        """
        if (self.link_id is None or self.snr_props.link_dictionary is None):
            raise ValueError("Link ID and link dictionary must be initialized")

        non_linearity = (
            self.engine_props_dict["topology_info"]["links"][self.link_id]["fiber"][
                "non_linearity"
            ]
            ** 2
        )
        self.snr_props.mu_parameter = 3 * non_linearity
        mu_denominator = 2 * math.pi * self.snr_props.link_dictionary["attenuation"]
        mu_denominator *= np.abs(self.snr_props.link_dictionary["dispersion"])
        self.snr_props.mu_parameter /= mu_denominator

        self.snr_props.self_channel_interference_psd = self._calculate_sci_psd()
        self.snr_props.cross_channel_interference_psd = self._calculate_xci(
            link_number=link_number
        )

        self.snr_props.length = self.engine_props_dict["topology_info"]["links"][
            self.link_id
        ]["span_length"]
        link_length = self.engine_props_dict["topology_info"]["links"][self.link_id][
            "length"
        ]
        self.snr_props.number_of_spans = link_length / self.snr_props.length

    def _init_center_frequency_and_bandwidth(self) -> None:
        """
        Updates variables for the center frequency, bandwidth, and PSD for the current
        request.
        """
        if self.number_of_slots is None:
            raise ValueError("Number of slots must be initialized")

        self.snr_props.center_frequency = (
            self.spectrum_props.start_slot * self.engine_props_dict["bw_per_slot"]
        )
        self.snr_props.center_frequency += (
            self.number_of_slots * self.engine_props_dict["bw_per_slot"]
        ) / 2
        self.snr_props.center_frequency *= POWER_CONVERSION_FACTOR

        self.snr_props.bandwidth = (
            self.number_of_slots
            * self.engine_props_dict["bw_per_slot"]
            * POWER_CONVERSION_FACTOR
        )
        self.snr_props.center_psd = (
            self.engine_props_dict["input_power"] / self.snr_props.bandwidth
        )

    def check_snr(self) -> tuple[bool, float]:
        """
        Determines whether the SNR threshold can be met for a single request.

        :return: Whether the SNR threshold can be met.
        :rtype: bool
        """
        if (self.spectrum_props.path_list is None or
            self.sdn_props.network_spectrum_dict is None):
            raise ValueError("Path list and network spectrum dict must be initialized")

        total_snr = 0.0
        self._init_center_frequency_and_bandwidth()
        for link_num in range(0, len(self.spectrum_props.path_list) - 1):
            source = self.spectrum_props.path_list[link_num]
            dest = self.spectrum_props.path_list[link_num + 1]
            self.link_id = self.sdn_props.network_spectrum_dict[(source, dest)][
                "link_num"
            ]

            self.snr_props.link_dictionary = self.engine_props_dict["topology_info"][
                "links"
            ][self.link_id]["fiber"]
            self._update_link_parameters(link_number=link_num)

            psd_nli = self._calculate_psd_nli()
            psd_ase = (
                self.snr_props.planck_constant
                * self.snr_props.light_frequency
                * self.snr_props.noise_spectral_density
            )
            psd_ase *= (
                math.exp(
                    self.snr_props.link_dictionary["attenuation"]
                    * self.snr_props.length
                    * LENGTH_CONVERSION_FACTOR
                )
                - 1
            )

            if self.engine_props_dict["xt_noise"]:
                # fixme number of adjacent set to a constant negative 100
                power_xt = self._calculate_pxt(
                    number_of_adjacent_cores=ADJACENT_CORES_PLACEHOLDER
                )
            else:
                power_xt = 0.0

            if (self.snr_props.center_psd is None or
                self.snr_props.bandwidth is None or
                self.snr_props.number_of_spans is None):
                raise ValueError("Required SNR properties are not initialized")

            current_snr = self.snr_props.center_psd * self.snr_props.bandwidth
            current_snr /= (
                (psd_ase + psd_nli) * self.snr_props.bandwidth + power_xt
            ) * self.snr_props.number_of_spans

            total_snr += 1 / current_snr

        total_snr = DB_CONVERSION_FACTOR * math.log10(1 / total_snr)

        response = total_snr > self.snr_props.request_snr
        return response, power_xt

    def check_adjacent_cores(self, link_tuple: tuple[str, str]) -> int:
        """
        Given a link, finds the number of cores which have overlapping channels on a
        fiber.

        :return: The number of adjacent cores that have overlapping channels.
        """
        if (self.spectrum_props.core_number is None or
            self.sdn_props.network_spectrum_dict is None or
            self.spectrum_props.start_slot is None or
            self.spectrum_props.end_slot is None):
            raise ValueError("Required spectrum properties must be initialized")

        overlapping_adjacent_cores = 0
        if self.spectrum_props.core_number != 6:
            # The neighboring core directly before the currently selected core
            before = (
                5
                if self.spectrum_props.core_number == 0
                else self.spectrum_props.core_number - 1
            )
            # The neighboring core directly after the currently selected core
            after = (
                0
                if self.spectrum_props.core_number == 5
                else self.spectrum_props.core_number + 1
            )
            adjacent_cores_list = [
                before,
                after,
                6,
            ]  # 6 is the center core for 7-core fiber
        else:
            adjacent_cores_list = list(range(6))

        for current_slot in range(
            self.spectrum_props.start_slot, self.spectrum_props.end_slot
        ):
            overlapped = 0
            for core_num in adjacent_cores_list:
                band = self.spectrum_props.current_band
                core_contents = self.sdn_props.network_spectrum_dict[link_tuple][
                    "cores_matrix"
                ][band][core_num][current_slot]
                if core_contents > 0.0:
                    overlapped += 1

            # Determine which slot has the maximum number of overlapping channels
            if overlapped > overlapping_adjacent_cores:
                overlapping_adjacent_cores = overlapped

        return overlapping_adjacent_cores

    def find_worst_xt(self, flag: str) -> tuple[float, float]:
        """
        Finds the worst possible cross-talk.

        :param flag: Determines which type of cross-talk is being considered.
        :type flag: str

        :return: The maximum length of the link found and the cross-talk calculated.
        :rtype: tuple
        """
        if flag == "intra_core":
            edge_lengths = nx.get_edge_attributes(
                self.engine_props_dict["topology"], "length"
            )
            max_link = max(edge_lengths, key=edge_lengths.get, default=None)

            if (max_link is None or self.sdn_props.network_spectrum_dict is None):
                raise ValueError(
                    "No valid links found or network spectrum dict not initialized"
                )

            self.link_id = self.sdn_props.network_spectrum_dict[max_link]["link_num"]
            max_length = edge_lengths.get(max_link, 0.0)
            self.snr_props.link_dictionary = self.engine_props_dict["topology_info"][
                "links"
            ][self.link_id]["fiber"]

            response = self.calculate_xt(
                number_of_adjacent_cores=6, link_length=max_length
            )
            response = DB_CONVERSION_FACTOR * math.log10(response)
        else:
            raise NotImplementedError

        return response, max_length

    def check_xt(self) -> tuple[bool, float]:
        """
        Checks the amount of cross-talk interference on a single request.

        :return: Whether the cross-talk interference threshold can be met
        :rtype: bool
        """
        if (self.spectrum_props.path_list is None or
            self.sdn_props.network_spectrum_dict is None):
            raise ValueError("Path list and network spectrum dict must be initialized")

        cross_talk = 0.0

        self._init_center_frequency_and_bandwidth()
        for link_num in range(0, len(self.spectrum_props.path_list) - 1):
            link_tuple = (
                self.spectrum_props.path_list[link_num],
                self.spectrum_props.path_list[link_num + 1],
            )

            self.link_id = self.sdn_props.network_spectrum_dict[link_tuple]["link_num"]
            link_length = self.engine_props_dict["topology_info"]["links"][
                self.link_id
            ]["length"]
            self.snr_props.link_dictionary = self.engine_props_dict["topology_info"][
                "links"
            ][self.link_id]["fiber"]
            self._update_link_parameters(link_number=link_num)

            number_of_adjacent_cores = self.check_adjacent_cores(
                link_tuple=(str(link_tuple[0]), str(link_tuple[1]))
            )
            cross_talk += self.calculate_xt(
                number_of_adjacent_cores=number_of_adjacent_cores,
                link_length=link_length,
            )

        if cross_talk == 0:
            response = True
        else:
            cross_talk = DB_CONVERSION_FACTOR * math.log10(cross_talk)
            response = (
                cross_talk
                < self.engine_props_dict["requested_xt"][self.spectrum_props.modulation]
            )

        return response, cross_talk

    def find_number_of_adjacent_cores(self) -> int:
        """
        Finds the number of adjacent cores for selected core.

        :return: The number of adjacent cores.
        """
        if self.spectrum_props.core_number is None:
            raise ValueError("Core number must be initialized")

        number_of_adjacent_cores = 0
        if self.engine_props_dict["cores_per_link"] == 4:
            number_of_adjacent_cores = 2
        elif self.engine_props_dict["cores_per_link"] == 7:
            number_of_adjacent_cores = 6 if self.spectrum_props.core_number == 6 else 3
        elif self.engine_props_dict["cores_per_link"] == 13:
            if self.spectrum_props.core_number < 6:
                number_of_adjacent_cores = 2
            elif 6 <= self.spectrum_props.core_number < 12:
                number_of_adjacent_cores = 5
            elif self.spectrum_props.core_number == 12:
                number_of_adjacent_cores = 6
        elif self.engine_props_dict["cores_per_link"] == 19:
            if self.spectrum_props.core_number >= 12:
                number_of_adjacent_cores = 6
            elif self.spectrum_props.core_number % 2 == 0:
                number_of_adjacent_cores = 3
            else:
                number_of_adjacent_cores = 4
        return number_of_adjacent_cores

    def check_snr_ext(self, path_index: int) -> tuple[bool, float]:
        """
        Checks the SNR on a single request using the external resources.

        :return: Whether the SNR threshold can be met and SNR value.
        :rtype: tuple
        """
        # Fetch loaded files
        if self.engine_props_dict["multi_fiber"]:
            number_of_adjacent_cores = 0
        else:
            number_of_adjacent_cores = self.find_number_of_adjacent_cores()
        loaded_data, loaded_data_gsnr = get_loaded_files(
            number_of_adjacent_cores,
            self.engine_props_dict["cores_per_link"],
            self.snr_props.file_mapping_dict,
            network=self.engine_props_dict["network"],
        )

        # Compute slot index
        if (
            self.spectrum_props.current_band is None
            or self.spectrum_props.start_slot is None
        ):
            raise ValueError("current_band and start_slot must not be None")
        slot_index = get_slot_index(
            self.spectrum_props.current_band,
            self.spectrum_props.start_slot,
            self.engine_props_dict,
        )

        # Fetch modulation format and SNR value
        modulation_format = loaded_data[self.route_props.connection_index][slot_index][
            path_index
        ]
        snr_value = loaded_data_gsnr[self.route_props.connection_index][slot_index][
            path_index
        ]

        # Determine response
        response = compute_response(
            modulation_format, self.snr_props, self.spectrum_props, self.sdn_props
        )

        return response, snr_value

    def check_snr_ext_slicing(self, path_index: int) -> tuple[str | None, float, float]:
        """
        Checks the SNR on a single request using the external resources for slicing.

        :return: Modulation format, supported bandwidth, and SNR value.
        :rtype: tuple
        """
        # Fetch loaded files
        if self.engine_props_dict["multi_fiber"]:
            number_of_adjacent_cores = 0
        else:
            number_of_adjacent_cores = self.find_number_of_adjacent_cores()
        loaded_data, loaded_data_gsnr = get_loaded_files(
            number_of_adjacent_cores,
            self.engine_props_dict["cores_per_link"],
            self.snr_props.file_mapping_dict,
            network=self.engine_props_dict["network"],
        )

        # Compute slot index
        if (
            self.spectrum_props.current_band is None
            or self.spectrum_props.start_slot is None
        ):
            raise ValueError("current_band and start_slot must not be None")
        slot_index = get_slot_index(
            self.spectrum_props.current_band,
            self.spectrum_props.start_slot,
            self.engine_props_dict,
        )

        # Retrieve modulation format and supported bandwidth
        modulation_format_key = loaded_data[self.route_props.connection_index][
            slot_index
        ][path_index]
        if modulation_format_key == 0:
            modulation_format = None
            supported_bandwidth = 0
        else:
            modulation_format = self.snr_props.modulation_format_mapping_dict[
                modulation_format_key
            ]
            supported_bandwidth = self.snr_props.bandwidth_mapping_dict[
                modulation_format
            ]

        # Retrieve SNR value
        snr_value = loaded_data_gsnr[self.route_props.connection_index][slot_index][
            path_index
        ]

        return modulation_format, supported_bandwidth, snr_value

    def check_snr_ext_open_slots(
        self, path_index: int, open_slots_list: list[int]
    ) -> list[int]:
        """
        Checks the SNR on a single request using the external resources for slicing.

        :return: Modulation format, supported bandwidth, and SNR value.
        :rtype: tuple
        """
        # Fetch loaded files
        if self.engine_props_dict["multi_fiber"]:
            number_of_adjacent_cores = 0
        else:
            number_of_adjacent_cores = self.find_number_of_adjacent_cores()
        loaded_data, _ = get_loaded_files(
            number_of_adjacent_cores,
            self.engine_props_dict["cores_per_link"],
            self.snr_props.file_mapping_dict,
            self.engine_props_dict["network"],
        )

        # Retrieve modulation format and supported bandwidth
        for open_slot in open_slots_list[:]:
            if self.spectrum_props.current_band is None:
                raise ValueError("current_band must not be None")
            slot_index = get_slot_index(
                self.spectrum_props.current_band, open_slot, self.engine_props_dict
            )
            modulation_format_key = loaded_data[self.route_props.connection_index][
                slot_index
            ][path_index]
            if modulation_format_key == 0:
                open_slots_list.remove(open_slot)

        return open_slots_list

    def handle_snr(self, path_index: int) -> tuple[bool, float]:
        """
        Controls the methods of this class.

        :return: Whether snr is acceptable for allocation or not for a given
                 request and its cost
        :rtype: tuple
        """
        if (self.spectrum_props.end_slot is None or
            self.spectrum_props.start_slot is None):
            raise ValueError("End slot and start slot must be initialized")

        self.number_of_slots = (
            self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1
        )
        if self.engine_props_dict["snr_type"] == "snr_calc_nli":
            snr_check, xt_cost = self.check_snr()
        elif self.engine_props_dict["snr_type"] == "xt_calculation":
            snr_check, xt_cost = self.check_xt()
        elif self.engine_props_dict["snr_type"] == "snr_e2e_external_resources":
            snr_check, xt_cost = self.check_snr_ext(path_index)
        else:
            raise NotImplementedError(
                f"Unexpected snr_type flag got: {self.engine_props_dict['snr_type']}"
            )

        return snr_check, xt_cost

    def handle_snr_dynamic_slicing(
        self, path_index: int
    ) -> tuple[str | None, float, float]:
        """
        Controls the methods of this class.

        :return: Whether snr is acceptable for allocation or not for a given
                 request and its cost
        :rtype: tuple
        """
        if (self.spectrum_props.end_slot is None or
            self.spectrum_props.start_slot is None):
            raise ValueError("End slot and start slot must be initialized")

        self.number_of_slots = (
            self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1
        )
        if self.engine_props_dict["snr_type"] == "snr_e2e_external_resources":
            modulation_format, bandwidth, snr_value = self.check_snr_ext_slicing(
                path_index
            )
        else:
            raise NotImplementedError(
                f"Unexpected snr_type flag got: {self.engine_props_dict['snr_type']}"
            )

        return modulation_format, bandwidth, snr_value
