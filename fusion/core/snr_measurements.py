"""
SNR measurements module for signal-to-noise ratio calculations.

This module provides functionality for calculating SNR, cross-talk interference,
and other signal quality metrics for optical network requests.
"""

import logging
import math
from typing import Any

import networkx as nx
import numpy as np

from fusion.core.properties import RoutingProps, SDNProps, SNRProps, SpectrumProps
from fusion.modules.snr.utils import compute_response, get_loaded_files, get_slot_index

logger = logging.getLogger(__name__)

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
        if (
            self.snr_props.link_dictionary is None
            or self.snr_props.center_psd is None
            or self.snr_props.bandwidth is None
        ):
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

        # If there's no channel bandwidth (empty link), return current xci
        if channel_bandwidth == 0:
            return current_xci

        channel_frequency = (
            (slot_index * self.engine_props_dict["bw_per_slot"])
            + (channel_bandwidth / 2)
        ) * POWER_CONVERSION_FACTOR
        channel_bandwidth *= POWER_CONVERSION_FACTOR
        channel_psd = self.engine_props_dict["input_power"] / channel_bandwidth

        if (
            self.snr_props.center_frequency is not None
            and self.snr_props.center_frequency != channel_frequency
        ):
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
        if (
            self.spectrum_props.path_list is None
            or self.sdn_props.network_spectrum_dict is None
        ):
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
        if (
            self.channels_list is None
            or self.snr_props.link_dictionary is None
            or self.snr_props.length is None
        ):
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
        if self.snr_props.center_psd is None or self.snr_props.bandwidth is None:
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
        if (
            self.snr_props.self_channel_interference_psd is None
            or self.snr_props.cross_channel_interference_psd is None
            or self.snr_props.mu_parameter is None
            or self.snr_props.center_psd is None
        ):
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
        if self.link_id is None or self.snr_props.link_dictionary is None:
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
        if (
            self.spectrum_props.path_list is None
            or self.sdn_props.network_spectrum_dict is None
        ):
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

            if (
                self.snr_props.center_psd is None
                or self.snr_props.bandwidth is None
                or self.snr_props.number_of_spans is None
            ):
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
        if (
            self.spectrum_props.core_number is None
            or self.sdn_props.network_spectrum_dict is None
            or self.spectrum_props.start_slot is None
            or self.spectrum_props.end_slot is None
        ):
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

            if max_link is None or self.sdn_props.network_spectrum_dict is None:
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
        if (
            self.spectrum_props.path_list is None
            or self.sdn_props.network_spectrum_dict is None
        ):
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
    def check_gsnr(
        self
    ) -> tuple[bool | str, float, int]:
        """
        Calculate and check the GSNR (Generalized SNR) of Lightpath.

        Ported from v5 for proper modulation selection based on path quality.

        :return: Whether the SNR threshold can be met, GSNR value, Supported bitrate.
        :rtype: tuple[bool | str, float, int]
        """
        from fusion.utils.data import sort_nested_dict_values

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
            self.link_id = self.sdn_props.network_spectrum_dict[(source, dest)]['link_num']
            self.snr_props.length = self.engine_props_dict['topology_info']['links'][self.link_id]['span_length']
            self.snr_props.link_dict = self.engine_props_dict['topology_info']['links'][self.link_id]['fiber']
            center_freq = self.snr_props.link_dict['frequency_start_c'] + self.spectrum_props.start_slot * self.engine_props_dict['bw_per_slot'] * 10 ** 9
            center_freq += ((self.number_of_slots * self.engine_props_dict['bw_per_slot']) / 2) * 10 ** 9
            self.snr_props.bandwidth = self.number_of_slots * self.engine_props_dict['bw_per_slot'] * 10 ** 9
            gsnr_link_ase_nli = 0

            self.snr_props.num_span = math.ceil(self.engine_props_dict['topology_info']['links'][self.link_id]['length'] / self.snr_props.length)
            p_ase_span = self.snr_props.bandwidth * self.snr_props.planck_constant * center_freq * 2 * self.snr_props.nsp[self.spectrum_props.current_band]
            p_ase_span *= (math.exp(self.snr_props.link_dict['attenuation'] * self.snr_props.length * 10 ** 3) - 1)
            l_eff = (1 - np.exp(-2 * self.snr_props.link_dict['attenuation'] * self.snr_props.length * 10 ** 3)) / (2 * self.snr_props.link_dict['attenuation'])
            l_eff_a = 1 / (2 * self.snr_props.link_dict['attenuation'])

            for _ in range(self.snr_props.num_span):
                self.channels_list = []
                sum_phi = 0
                for slot_index in range(self.engine_props_dict['c_band']):
                    curr_link = self.sdn_props.network_spectrum_dict[(source, dest)]['cores_matrix'][self.spectrum_props.current_band]
                    req_id = curr_link[self.spectrum_props.core_number][slot_index]

                    # Spectrum is occupied
                    if (req_id > 0 and req_id not in self.channels_list) or slot_index == self.spectrum_props.start_slot:
                        channel_mod = None
                        if req_id > 0:
                            for _, lp_dict in self.sdn_props.lightpath_status_dict.items():
                                if req_id in lp_dict:
                                    channel_mod = lp_dict[req_id]['mod_format']

                            channel_bw = len(np.where(req_id == curr_link[self.spectrum_props.core_number])[0])
                            self.channels_list.append(req_id)
                        else:
                            channel_bw = self.number_of_slots
                        
                        if channel_mod is None and req_id > 0:
                            if req_id in self.sdn_props.lightpath_id_list:
                                idx = self.sdn_props.lightpath_id_list.index(req_id)
                                channel_mod = self.sdn_props.modulation_list[idx]
                            else:
                                raise NotImplementedError(f"Unexpected lightpath id: {req_id}")
                        
                        channel_bw *= self.engine_props_dict['bw_per_slot']
                        channel_freq = self.snr_props.link_dict['frequency_start_c'] + ((slot_index * self.engine_props_dict['bw_per_slot']) + (channel_bw / 2)) * 10 ** 9
                        channel_bw *= 10 ** 9
                        
                        if center_freq == channel_freq:
                            phi = np.arcsinh(np.pi**2 * abs(self.snr_props.link_dict['dispersion']) * self.snr_props.bandwidth **2 / (4 * self.snr_props.link_dict['attenuation']))
                        else:
                            phi = (
                                np.arcsinh((np.pi**2) * abs(self.snr_props.link_dict['dispersion']) * l_eff_a * self.snr_props.bandwidth * (channel_freq - center_freq + (channel_bw * 0.5))) -
                                np.arcsinh((np.pi**2) * abs(self.snr_props.link_dict['dispersion']) * l_eff_a * self.snr_props.bandwidth * (channel_freq - center_freq - (channel_bw * 0.5))) -
                                (self.engine_props_dict['phi'][channel_mod] * (channel_bw / abs(channel_freq - center_freq)) * (5 / 3) * (l_eff / (self.snr_props.length * 10 ** 3)))
                            )
                        sum_phi += phi
                
                p_nli_span = (self.engine_props_dict['input_power'] / self.snr_props.bandwidth)**3 * (8 / (27 * np.pi * abs(self.snr_props.link_dict['dispersion']))) * self.snr_props.link_dict['non_linearity']**2 * l_eff * sum_phi * self.snr_props.bandwidth
                gsnr_span_ase_nli_db.append(10 * np.log10(self.engine_props_dict['input_power'] / (p_ase_span + p_nli_span)))
                gsnr_link_ase_nli += (self.engine_props_dict['input_power'] / (p_ase_span + p_nli_span))**-1
            
            gsnr_link_ase_nli_db.append(10 * np.log10(gsnr_link_ase_nli**-1))
            gsnr_path_ase_nli += gsnr_link_ase_nli
        
        gsnr_db = 10 * np.log10(1 / gsnr_path_ase_nli)

        # Dynamic modulation selection for slicing with dynamic lightpaths
        if self.spectrum_props.slicing_flag and self.engine_props_dict['fixed_grid'] and self.engine_props_dict.get('dynamic_lps', False):
            mod_formats_dict = sort_nested_dict_values(
                original_dict=self.sdn_props.modulation_formats_dict,
                nested_key='max_length'
            )
            force_mod_format = list(mod_formats_dict.keys())
            resp = False
            bw_resp = 0
            for mod in force_mod_format:
                req_snr_val = self.snr_props.req_snr[mod]
                meets_req = gsnr_db >= req_snr_val
                if meets_req:
                    resp = mod
                    bw_resp = bw_mapping[mod]
                    break
        else:
            # Standard modulation check (modulation must be set)
            if self.spectrum_props.modulation is None:
                raise ValueError("Modulation format must be set for non-dynamic slicing")

            req_snr_threshold = self.snr_props.req_snr[self.spectrum_props.modulation]
            resp = gsnr_db >= req_snr_threshold
            # INSTRUMENTATION: GSNR threshold comparison
            bw_resp = 0

            if resp:
                if self.engine_props_dict['fixed_grid']:
                    if not self.spectrum_props.slicing_flag:
                        if bw_mapping[self.spectrum_props.modulation] >= int(self.sdn_props.bandwidth):
                            bw_resp = bw_mapping[self.spectrum_props.modulation]
                        else:
                            resp = False
                    else:
                        bw_resp = bw_mapping[self.spectrum_props.modulation]
                else:
                    if self.spectrum_props.slicing_flag:
                        bw_resp = 0
                    else:
                        bw_resp = int(self.sdn_props.bandwidth)
        
        return resp, gsnr_db, bw_resp

    def _compute_nli_mb(
        self, source: str, dest: str, p_total: float
    ) -> float:
        """
        Compute nonlinear interference GSNR (SNR_NLI) for multi-band EON.

        Uses Eq. (3) in https://doi.org/10.1049/cp.2019.0892

        :param source: Link start node
        :type source: str
        :param dest: Link end node
        :type dest: str
        :param p_total: Link's power profile
        :type p_total: float
        :return: Inverse nonlinear SNR value of given link
        :rtype: float
        """
        if self.snr_props.link_dictionary is None:
            raise ValueError("link_dictionary must be initialized")
        if self.spectrum_props.current_band is None:
            raise ValueError("current_band must be initialized")
        if self.spectrum_props.start_slot is None:
            raise ValueError("start_slot must be initialized")
        if self.number_of_slots is None:
            raise ValueError("number_of_slots must be initialized")

        link_dict = self.snr_props.link_dictionary
        f_c = (link_dict['frequency_end_c'] + link_dict['frequency_start_l']) / 2
        gamma = link_dict['non_linearity']
        alpha_i = link_dict['attenuation']
        alpha_bar_i = link_dict['attenuation']
        c_ri = link_dict['raman_gain_slope']
        beta_2 = link_dict['gvd']
        beta_3 = link_dict['gvd_slope']
        p_i = self.engine_props_dict['input_power']
        pi = np.pi
        bi = (self.number_of_slots * self.engine_props_dict['bw_per_slot']) * 10 ** 9
        f_i = link_dict['frequency_start_' + self.spectrum_props.current_band] + self.spectrum_props.start_slot * self.engine_props_dict['bw_per_slot'] * 10 ** 9
        f_i += (bi / 2)
        ti = (alpha_i + alpha_bar_i - p_total * c_ri * (f_i - f_c)) ** 2
        ai = alpha_i + alpha_bar_i
        phi_i = (1.5 * pi ** 2) * (beta_2 + 2 * beta_3 * (f_i - f_c))
        self.channels_list = []
        spm = 0.0
        xpm = 0.0
        epsilon = 0.0001

        for band in self.engine_props_dict['band_list']:
            for slot_index in range(self.engine_props_dict[band + '_band']):
                if self.sdn_props.network_spectrum_dict is None:
                    raise ValueError("network_spectrum_dict must be initialized")
                curr_link = self.sdn_props.network_spectrum_dict[(source, dest)]['cores_matrix'][band]
                req_id = curr_link[self.spectrum_props.core_number][slot_index]
                self.link_id = self.sdn_props.network_spectrum_dict[(source, dest)]['link_num']
                self.snr_props.length = self.engine_props_dict['topology_info']['links'][self.link_id]['span_length']
                length_m = self.snr_props.length * 1000

                if (req_id > 0 and req_id not in self.channels_list) or (slot_index == self.spectrum_props.start_slot and band == self.spectrum_props.current_band):
                    ch_k_mod = None
                    if req_id > 0:
                        for _, lp_dict in self.sdn_props.lightpath_status_dict.items():
                            if req_id in lp_dict:
                                ch_k_mod = lp_dict[req_id]['mod_format']

                        b_k = len(np.where(req_id == curr_link[self.spectrum_props.core_number])[0])
                        p_k = self.engine_props_dict['input_power']
                        self.channels_list.append(req_id)
                    else:
                        b_k = self.number_of_slots

                    if ch_k_mod is None and req_id > 0:
                        if req_id in self.sdn_props.lightpath_id_list:
                            idx = self.sdn_props.lightpath_id_list.index(req_id)
                            ch_k_mod = self.sdn_props.modulation_list[idx]
                        else:
                            raise NotImplementedError(f"Unexpected lightpath id: {req_id}")

                    alpha_k = link_dict['attenuation']
                    alpha_bar_k = link_dict['attenuation']
                    b_k *= self.engine_props_dict['bw_per_slot']
                    f_k = link_dict['frequency_start_' + band] + ((slot_index * self.engine_props_dict['bw_per_slot']) + (b_k / 2)) * 10 ** 9
                    b_k *= 10 ** 9

                    if f_i == f_k:
                        spm += (4 / 9) * ((pi * (gamma ** 2) * (self.snr_props.number_of_spans ** (1 + epsilon)) * (p_i ** 2)) / ((bi ** 2) * phi_i * alpha_bar_i * (2 * alpha_i + alpha_bar_i))) * (
                            ((ti - (alpha_i ** 2)) / alpha_i) * np.arcsinh((phi_i * (bi ** 2)) / (pi * alpha_i)) +
                            ((ai ** 2 - ti) / ai) * np.arcsinh((phi_i * (bi ** 2)) / (pi * ai))
                        )
                    else:
                        fk_fi = abs(f_k - f_i)
                        tk = (alpha_k + alpha_bar_k - p_total * c_ri * (f_k - f_c)) ** 2
                        ak = alpha_k + alpha_bar_k
                        n_tilde = 0 if self.snr_props.number_of_spans == 1 else self.snr_props.number_of_spans
                        phi_i_k = (-4 * pi ** 2 * (f_k - f_i)) * (beta_2 + pi * beta_3 * (f_i + f_k - 2 * f_c))
                        phi = (-4 * pi ** 2 * (beta_2 + pi * beta_3 * (f_i + f_k))) * length_m
                        xpm += (32 / 27) * (((gamma ** 2) * (p_k ** 2)) / b_k) * (
                            ((self.snr_props.number_of_spans + (5 / 6) * (-1 * self.engine_props_dict['phi'][ch_k_mod])) / (phi_i_k * alpha_bar_k * (2 * alpha_k + alpha_bar_k))) * (
                                ((tk - (alpha_k ** 2)) / alpha_k) * np.arctan((phi_i_k * bi) / alpha_k) +
                                (((ak ** 2) - tk) / ak) * np.arctan((phi_i_k * bi) / ak)
                            ) +
                            (5 / 3) * ((-1 * self.engine_props_dict['phi'][ch_k_mod] * pi * n_tilde * tk) / (abs(phi) * (b_k ** 2) * (alpha_k ** 2) * (ak ** 2))) * (
                                (2 * fk_fi - b_k) * math.log((2 * fk_fi - b_k) / (2 * fk_fi + b_k)) + 2 * b_k
                            )
                        )

        snr_nli_inv = spm + xpm
        return snr_nli_inv

    def _compute_ase_mb(
        self, source: str, dest: str, p_total: float
    ) -> float:
        """
        Compute ASE SNR (P_ASE) for multi-band EONs.

        Uses Eq. (5) in https://doi.org/10.1364/JOCN.515081

        :param source: Link start node
        :type source: str
        :param dest: Link end node
        :type dest: str
        :param p_total: Link's power profile
        :type p_total: float
        :return: Inverse ASE SNR value of given link
        :rtype: float
        """
        if self.snr_props.link_dictionary is None:
            raise ValueError("link_dictionary must be initialized")
        if self.spectrum_props.current_band is None:
            raise ValueError("current_band must be initialized")
        if self.spectrum_props.start_slot is None:
            raise ValueError("start_slot must be initialized")
        if self.number_of_slots is None:
            raise ValueError("number_of_slots must be initialized")

        link_dict = self.snr_props.link_dictionary
        p_ase_link = 0.0
        c_ri = link_dict['raman_gain_slope']
        b_i = (self.number_of_slots * self.engine_props_dict['bw_per_slot']) * 10 ** 9
        f_i = link_dict['frequency_start_' + self.spectrum_props.current_band] + self.spectrum_props.start_slot * self.engine_props_dict['bw_per_slot'] * 10 ** 9
        f_i += (b_i / 2)
        alpha = link_dict['attenuation']
        self.channels_list = []
        p_profile = 0.0
        l_eff = (1 - np.exp(-alpha * self.snr_props.length * 10 ** 3)) / alpha
        p_profile += (2 / b_i) * np.exp(-p_total * c_ri * l_eff * f_i) * np.sinh(p_total * c_ri * l_eff * (b_i / 2))

        for band in self.engine_props_dict['band_list']:
            for slot_index in range(self.engine_props_dict[band + '_band']):
                if self.sdn_props.network_spectrum_dict is None:
                    raise ValueError("network_spectrum_dict must be initialized")
                curr_link = self.sdn_props.network_spectrum_dict[(source, dest)]['cores_matrix'][band]
                req_id = curr_link[self.spectrum_props.core_number][slot_index]
                if req_id > 0 and req_id not in self.channels_list:
                    b_k = len(np.where(req_id == curr_link[self.spectrum_props.core_number])[0]) * self.engine_props_dict['bw_per_slot']
                    f_k = link_dict['frequency_start_' + band] + ((slot_index * self.engine_props_dict['bw_per_slot']) + (b_k / 2)) * 10 ** 9
                    b_k *= 10 ** 9
                    p_profile += (2 / b_k) * np.exp(-p_total * c_ri * l_eff * f_k) * np.sinh(p_total * c_ri * l_eff * (b_k / 2))
                    self.channels_list.append(req_id)

        p_profile = (p_total ** 2 * c_ri * l_eff * np.exp((-alpha * self.snr_props.length * 10 ** 3) - (p_total * c_ri * l_eff * f_i))) / p_profile
        p_ase_span = 2 * self.snr_props.nsp[self.spectrum_props.current_band] * self.snr_props.planck_constant * f_i * b_i * ((self.engine_props_dict['input_power'] / p_profile) - 1)
        p_ase_link = p_ase_span * self.snr_props.number_of_spans

        snr_ase_inv = p_ase_link / self.engine_props_dict['input_power']
        return snr_ase_inv

    def _gsnr_calc_mb(self) -> float:
        """
        Calculate GSNR for multi-band EON (C+L band) for single request.

        :return: GSNR value of given resources
        :rtype: float
        """
        if self.spectrum_props.path_list is None:
            raise ValueError("path_list must be initialized")

        snr_nli_inv = 0.0
        snr_ase_inv = 0.0
        snr_lp_inv = 0.0

        for link_num in range(len(self.spectrum_props.path_list) - 1):
            source = self.spectrum_props.path_list[link_num]
            dest = self.spectrum_props.path_list[link_num + 1]
            if self.sdn_props.network_spectrum_dict is None:
                raise ValueError("network_spectrum_dict must be initialized")
            self.link_id = self.sdn_props.network_spectrum_dict[(source, dest)]['link_num']
            self.snr_props.link_dictionary = self.engine_props_dict['topology_info']['links'][self.link_id]['fiber']
            self.snr_props.length = self.engine_props_dict['topology_info']['links'][self.link_id]['span_length']
            self.snr_props.number_of_spans = math.ceil(self.engine_props_dict['topology_info']['links'][self.link_id]['length'] / self.snr_props.length)

            # Calculate total power
            p_total = 1
            number_lp: set = set()
            for band in self.engine_props_dict['band_list']:
                curr_link = self.sdn_props.network_spectrum_dict[(source, dest)]['cores_matrix'][band][self.spectrum_props.core_number]
                number_lp.update(np.unique(curr_link[curr_link > 0]))
            p_total += len(number_lp)
            p_total *= self.engine_props_dict['input_power']

            snr_nli_inv_l = self._compute_nli_mb(source, dest, p_total)
            snr_ase_inv_l = self._compute_ase_mb(source, dest, p_total)
            snr_nli_inv += snr_nli_inv_l
            snr_ase_inv += snr_ase_inv_l

        snr_lp_inv += (snr_nli_inv + snr_ase_inv)
        gsnr = 1 / snr_lp_inv
        return gsnr

    def check_gsnr_mb(self) -> tuple[bool | str, float, int]:
        """
        Check GSNR of multiband and create response based on request and modulation type.

        Ported from v5 for multi-band (C+L) GSNR calculations.

        :return: Whether GSNR is acceptable for allocation or not for a given
                 request, GSNR value, and supported bandwidth
        :rtype: tuple[bool | str, float, int]
        """
        from fusion.utils.data import sort_nested_dict_values

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

        # v5 used snr_margin=0, so we just check against req_snr
        resp: bool | str = gsnr_db >= self.snr_props.req_snr[self.spectrum_props.modulation]
        bw_resp = 0

        if resp:
            if self.engine_props_dict['fixed_grid']:
                if not self.spectrum_props.slicing_flag:
                    if bw_mapping[self.spectrum_props.modulation] >= int(self.sdn_props.bandwidth):
                        bw_resp = bw_mapping[self.spectrum_props.modulation]
                    else:
                        resp = False
                else:
                    bw_resp = bw_mapping[self.spectrum_props.modulation]
            else:
                if self.spectrum_props.slicing_flag:
                    bw_resp = 0
                else:
                    bw_resp = int(self.sdn_props.bandwidth)

        # Dynamic modulation selection for slicing with dynamic lightpaths
        if self.spectrum_props.slicing_flag and self.engine_props_dict['fixed_grid'] and self.engine_props_dict.get('dynamic_lps', False):
            mod_formats_dict = sort_nested_dict_values(
                original_dict=self.sdn_props.modulation_formats_dict,
                nested_key='max_length'
            )
            force_mod_format = list(mod_formats_dict.keys())
            for mod in force_mod_format:
                if gsnr_db >= self.snr_props.req_snr[mod]:
                    resp = mod
                    bw_resp = bw_mapping[mod]
                    break

        return resp, gsnr_db, bw_resp

    def handle_snr(self, path_index: int) -> tuple[bool, float, float]:
        """
        Controls the methods of this class.

        :return: Whether snr is acceptable for allocation or not for a given
                 request, its cost, and lightpath bandwidth
        :rtype: tuple
        """
        if (
            self.spectrum_props.end_slot is None
            or self.spectrum_props.start_slot is None
        ):
            raise ValueError("End slot and start slot must be initialized")

        self.number_of_slots = (
            self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1
        )
        if self.engine_props_dict["snr_type"] == "snr_calc_nli":
            snr_check, xt_cost = self.check_snr()
        elif self.engine_props_dict["snr_type"] == "xt_calculation":
            snr_check, xt_cost = self.check_xt()
        elif self.engine_props_dict["snr_type"] == "gsnr":
            # GSNR (Generalized SNR) - proper implementation ported from v5
            if self.engine_props_dict["band_list"] == ["c"]:
                snr_check, xt_cost, lp_bw = self.check_gsnr()
            elif self.engine_props_dict["band_list"] == ["c", "l"]:
                snr_check, xt_cost, lp_bw = self.check_gsnr_mb()
            else:
                raise ValueError(
                    f"Unsupported band_list: {self.engine_props_dict['band_list']}"
                )

            # Note: check_gsnr/check_gsnr_mb can return either a boolean (standard mode) or
            # a modulation string (dynamic mode with slicing_flag=True).
            # In handle_snr, we should NOT override the pre-selected modulation.
            # Dynamic modulation selection is handled by handle_snr_dynamic_slicing.
            # Here we just need to treat the return value as truthy/falsy.
            # A modulation string means GSNR check passed (truthy).
            # False means GSNR check failed.
        elif self.engine_props_dict["snr_type"] == "snr_e2e_external_resources":
            snr_check, xt_cost = self.check_snr_ext(path_index)
        else:
            raise NotImplementedError(
                f"Unexpected snr_type flag got: {self.engine_props_dict['snr_type']}"
            )

        # Calculate lightpath bandwidth from modulation format
        lp_bandwidth = 0.0
        if self.spectrum_props.modulation is not None:
            if self.spectrum_props.modulation in self.snr_props.bandwidth_mapping_dict:
                lp_bandwidth = float(
                    self.snr_props.bandwidth_mapping_dict[
                        self.spectrum_props.modulation
                    ]
                )
            elif (
                self.sdn_props.modulation_formats_dict is not None
                and self.spectrum_props.modulation
                in self.sdn_props.modulation_formats_dict
            ):
                lp_bandwidth = float(
                    self.sdn_props.modulation_formats_dict[
                        self.spectrum_props.modulation
                    ].get("bandwidth", 0.0)
                )

        return snr_check, xt_cost, lp_bandwidth

    def recheck_snr_after_allocation(self, lightpath_id: int) -> tuple[bool, float]:
        """
        Recheck SNR after spectrum allocation.

        When grooming is enabled, validates that newly allocated spectrum
        still meets SNR requirements after considering all interference.

        :param lightpath_id: ID of newly allocated lightpath
        :type lightpath_id: int
        :return: Tuple of (snr_acceptable, crosstalk_cost)
        :rtype: tuple[bool, float]
        """
        if not self.engine_props_dict.get("snr_recheck", False):
            # Rechecking disabled, assume OK
            return True, 0.0

        logger.debug("Rechecking SNR for lightpath %d", lightpath_id)

        # Get allocated spectrum details
        start_slot = self.spectrum_props.start_slot
        end_slot = self.spectrum_props.end_slot
        core_num = self.spectrum_props.core_number
        band = self.spectrum_props.current_band
        path_list = self.spectrum_props.path_list

        if (
            start_slot is None
            or end_slot is None
            or core_num is None
            or band is None
            or path_list is None
        ):
            raise ValueError("Required spectrum properties must be initialized")

        # Calculate interference from adjacent cores
        adjacent_core_interference = 0.0
        if self.engine_props_dict.get("recheck_adjacent_cores", False):
            adjacent_core_interference = self._calculate_adjacent_core_interference(
                path_list, band, core_num, start_slot, end_slot
            )

        # Calculate cross-band interference
        crossband_interference = 0.0
        if self.engine_props_dict.get("recheck_crossband", False):
            crossband_interference = self._calculate_crossband_interference(
                path_list, core_num, start_slot, end_slot
            )

        # Total interference
        total_interference = adjacent_core_interference + crossband_interference

        # Re-calculate SNR with interference
        snr_margin = self._calculate_snr_with_interference(total_interference)

        # Check if SNR is still acceptable
        required_snr = self.snr_props.request_snr
        snr_acceptable = snr_margin >= required_snr

        if not snr_acceptable:
            logger.warning(
                "SNR recheck failed for lightpath %d: margin=%.2f < required=%.2f",
                lightpath_id,
                snr_margin,
                required_snr,
            )

        return snr_acceptable, total_interference

    def _calculate_adjacent_core_interference(
        self, path_list: list, band: str, core_num: int, start_slot: int, end_slot: int
    ) -> float:
        """
        Calculate interference from adjacent cores.

        :param path_list: Path nodes
        :type path_list: list
        :param band: Spectral band
        :type band: str
        :param core_num: Core number
        :type core_num: int
        :param start_slot: Start slot index
        :type start_slot: int
        :param end_slot: End slot index
        :type end_slot: int
        :return: Adjacent core interference value
        :rtype: float
        """
        interference = 0.0

        # Get adjacent cores (depends on core layout)
        adjacent_cores = self._get_adjacent_cores(core_num)

        for source, dest in zip(path_list, path_list[1:], strict=False):
            if self.sdn_props.network_spectrum_dict is None:
                raise ValueError("network_spectrum_dict must be initialized")

            link_dict = self.sdn_props.network_spectrum_dict[(source, dest)]

            for adj_core in adjacent_cores:
                if adj_core >= self.engine_props_dict["cores_per_link"]:
                    continue

                core_array = link_dict["cores_matrix"][band][adj_core]

                # Check for occupied slots in adjacent core
                occupied_slots = core_array[start_slot:end_slot]
                if np.any(occupied_slots != 0):
                    # Add interference (simplified - use actual crosstalk model)
                    interference += self.engine_props_dict.get(
                        "adjacent_core_xt_coefficient", 0.01
                    )

        return interference

    def _calculate_crossband_interference(
        self, path_list: list, core_num: int, start_slot: int, end_slot: int
    ) -> float:
        """
        Calculate interference from other spectral bands.

        :param path_list: Path nodes
        :type path_list: list
        :param core_num: Core number
        :type core_num: int
        :param start_slot: Start slot index
        :type start_slot: int
        :param end_slot: End slot index
        :type end_slot: int
        :return: Cross-band interference value
        :rtype: float
        """
        interference = 0.0

        current_band = self.spectrum_props.current_band
        if current_band is None:
            raise ValueError("current_band must be initialized")

        other_bands = [
            b for b in self.engine_props_dict["band_list"] if b != current_band
        ]

        for source, dest in zip(path_list, path_list[1:], strict=False):
            if self.sdn_props.network_spectrum_dict is None:
                raise ValueError("network_spectrum_dict must be initialized")

            link_dict = self.sdn_props.network_spectrum_dict[(source, dest)]

            for band in other_bands:
                core_array = link_dict["cores_matrix"][band][core_num]

                # Check for occupied slots in other band
                occupied_slots = core_array[start_slot:end_slot]
                if np.any(occupied_slots != 0):
                    # Add interference (simplified - use actual crosstalk model)
                    interference += self.engine_props_dict.get(
                        "crossband_xt_coefficient", 0.005
                    )

        return interference

    def _get_adjacent_cores(self, core_num: int) -> list[int]:
        """
        Get list of cores adjacent to the given core.

        :param core_num: Core number
        :type core_num: int
        :return: List of adjacent core numbers
        :rtype: list[int]
        """
        # Simplified adjacency - actual adjacency depends on fiber geometry
        total_cores = self.engine_props_dict["cores_per_link"]

        if total_cores == 1:
            return []

        # Simple linear adjacency model
        adjacent = []
        if core_num > 0:
            adjacent.append(core_num - 1)
        if core_num < total_cores - 1:
            adjacent.append(core_num + 1)

        return adjacent

    def _calculate_snr_with_interference(self, interference: float) -> float:
        """
        Calculate SNR including additional interference.

        :param interference: Additional interference value
        :type interference: float
        :return: SNR margin in dB
        :rtype: float
        """
        # Get base SNR calculation
        # Use existing check_snr logic but adjust for interference
        if (
            self.spectrum_props.path_list is None
            or self.sdn_props.network_spectrum_dict is None
        ):
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

            # Add additional interference from grooming
            power_xt = interference

            if (
                self.snr_props.center_psd is None
                or self.snr_props.bandwidth is None
                or self.snr_props.number_of_spans is None
            ):
                raise ValueError("Required SNR properties are not initialized")

            current_snr = self.snr_props.center_psd * self.snr_props.bandwidth
            current_snr /= (
                (psd_ase + psd_nli) * self.snr_props.bandwidth + power_xt
            ) * self.snr_props.number_of_spans

            total_snr += 1 / current_snr

        total_snr = DB_CONVERSION_FACTOR * math.log10(1 / total_snr)

        return total_snr

    def handle_snr_dynamic_slicing(
        self, path_index: int
    ) -> tuple[str | None, float, float]:
        """
        Controls the methods of this class.

        :return: Whether snr is acceptable for allocation or not for a given
                 request and its cost
        :rtype: tuple
        """
        if (
            self.spectrum_props.end_slot is None
            or self.spectrum_props.start_slot is None
        ):
            raise ValueError("End slot and start slot must be initialized")

        self.number_of_slots = (
            self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1
        )
        if self.engine_props_dict["snr_type"] == "snr_e2e_external_resources":
            modulation_format, bandwidth, snr_value = self.check_snr_ext_slicing(
                path_index
            )
        elif self.engine_props_dict["snr_type"] == "gsnr":
            # GSNR (Generalized SNR) for dynamic slicing
            if self.engine_props_dict["band_list"] == ["c"]:
                modulation_format, snr_value, bandwidth = self.check_gsnr()
            elif self.engine_props_dict["band_list"] == ["c", "l"]:
                modulation_format, snr_value, bandwidth = self.check_gsnr_mb()
            else:
                raise ValueError(
                    f"Unexpected band_list: {self.engine_props_dict['band_list']}"
                )
        else:
            raise NotImplementedError(
                f"Unexpected snr_type flag got: {self.engine_props_dict['snr_type']}"
            )

        return modulation_format, bandwidth, snr_value

    def _build_lightpath_list_from_net_spec(self) -> list[dict[str, Any]]:
        """
        Build lightpath list using net_spec_dict as primary source.

        Constructs a list of all active lightpaths by combining information from
        lightpath_status_dict and current lightpath_id_list.

        :return: List of lightpath dictionaries
        :rtype: list[dict[str, Any]]
        """
        lp_list = [
            {**self.sdn_props.lightpath_status_dict[light_id][lp_id], "id": lp_id}
            for light_id in self.sdn_props.lightpath_status_dict
            for lp_id in self.sdn_props.lightpath_status_dict[light_id]
        ]

        # Add currently being allocated lightpaths (not yet in status dict)
        if len(self.sdn_props.lightpath_id_list) > 1:
            for lp_cnt in range(len(self.sdn_props.lightpath_id_list)):
                # Safely access crosstalk_list with default value if not available
                snr_cost = (
                    self.sdn_props.crosstalk_list[lp_cnt]
                    if lp_cnt < len(self.sdn_props.crosstalk_list)
                    else 0.0
                )
                lp_list.append({
                    "id": self.sdn_props.lightpath_id_list[lp_cnt],
                    "path": self.sdn_props.path_list,
                    "core": self.sdn_props.core_list[lp_cnt],
                    "start_slot": self.sdn_props.start_slot_list[lp_cnt],
                    "end_slot": self.sdn_props.end_slot_list[lp_cnt],
                    "band": self.sdn_props.band_list[lp_cnt],
                    "mod_format": self.sdn_props.modulation_list[lp_cnt],
                    "snr_cost": snr_cost,
                })

        return lp_list

    def load_from_lp_info(self, lp_info: dict[str, Any]) -> None:
        """
        Load a specific lightpath's state into this SnrMeasurements object.

        :param lp_info: Lightpath information dictionary
        :type lp_info: dict[str, Any]
        """
        self.spectrum_props.path_list = lp_info["path"]
        self.spectrum_props.start_slot = lp_info["start_slot"]
        self.spectrum_props.end_slot = lp_info["end_slot"]
        self.spectrum_props.core_number = lp_info["core"]
        self.spectrum_props.current_band = lp_info["band"]
        self.spectrum_props.modulation = lp_info.get("mod_format", "QPSK")
        self.number_of_slots = self.spectrum_props.end_slot - self.spectrum_props.start_slot + 1

    def evaluate_lp(
        self, lp_info: dict[str, Any]
    ) -> tuple[bool, float]:
        """
        Compute SNR for a given lightpath and return whether it meets requirements.

        :param lp_info: Lightpath information dictionary
        :type lp_info: dict[str, Any]
        :return: Tuple of (meets_snr_requirement, snr_value_in_dB)
        :rtype: tuple[bool, float]
        """
        self.load_from_lp_info(lp_info)

        if self.engine_props_dict["snr_type"] == "gsnr":
            if self.engine_props_dict["band_list"] == ["c"]:
                resp, snr_val, _ = self.check_gsnr()
            elif self.engine_props_dict["band_list"] == ["c", "l"]:
                resp, snr_val, _ = self.check_gsnr_mb()
            else:
                raise ValueError(
                    f"Unsupported band_list: {self.engine_props_dict['band_list']}"
                )
        else:
            raise NotImplementedError(
                f"Unsupported snr_type: {self.engine_props_dict['snr_type']}"
            )

        return resp, snr_val

    def snr_recheck_after_allocation(
        self, new_lp_info: dict[str, Any]
    ) -> tuple[bool, list[tuple[int, float, float]]]:
        """
        Re-evaluate all overlapping lightpaths after new allocation.

        After a new lightpath is allocated, this method re-evaluates the SNR
        of all existing lightpaths that overlap with it (share links and cores).
        If any existing lightpath no longer meets its SNR requirements, the
        allocation is rejected.

        :param new_lp_info: New lightpath information dict
        :type new_lp_info: dict[str, Any]
        :return: Tuple of (all_pass, violations_list)
        :rtype: tuple[bool, list[tuple[int, float, float]]]
        """
        # Check feature toggle
        if not self.engine_props_dict.get("snr_recheck", False):
            return True, []

        # Import here to avoid circular dependency
        from fusion.utils.spectrum import get_overlapping_lightpaths

        # Build list of all active lightpaths (include new LP for interference calculation)
        all_active_lps = self._build_lightpath_list_from_net_spec()
        new_lp_id = new_lp_info.get("id")

        # Find lightpaths that overlap with the new one
        overlapping_lps = get_overlapping_lightpaths(
            new_lp=new_lp_info,
            lp_list=all_active_lps,
            cores_per_link=self.engine_props_dict["cores_per_link"],
            include_adjacent_cores=self.engine_props_dict.get("recheck_adjacent_cores", True),
            include_all_bands=self.engine_props_dict.get("recheck_crossband", True),
            bidirectional_links=self.engine_props_dict.get("bi_directional", False),
        )

        # Re-evaluate each overlapping lightpath (include new LP's interference)
        violations = []
        for lp in overlapping_lps:
            resp, observed_snr = self.evaluate_lp(lp)
            required_snr = self.snr_props.req_snr[lp["mod_format"]]
            if not resp:
                violations.append((lp["id"], observed_snr, required_snr))

        return not violations, violations
