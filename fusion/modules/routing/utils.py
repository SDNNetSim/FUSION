import copy
import math

import networkx as nx
import numpy as np

from fusion.sim.utils import find_free_channels, find_taken_channels

# Constants for routing calculations
FULLY_CONGESTED_LINK_COST = 1000.0
DEFAULT_BAND = "c"
CENTER_CORE_INDEX = 6
MAX_OUTER_CORES = 6


# Note: Consider integrating K-Shortest Path functionality directly into this utility module
# for better cohesion with routing helper functions


class RoutingHelpers:
    """
    Class containing methods to assist with finding routes in the routing class.
    """

    def __init__(self, route_props: object, engine_props: dict, sdn_props: object):
        self.route_props = route_props
        self.engine_props = engine_props
        self.sdn_props = sdn_props

    def _get_indexes(self, center_index: int):
        if self.sdn_props.slots_needed % 2 == 0:
            start_index = center_index - self.sdn_props.slots_needed // 2
            end_index = center_index + self.sdn_props.slots_needed // 2
        else:
            start_index = center_index - self.sdn_props.slots_needed // 2
            end_index = center_index + self.sdn_props.slots_needed // 2 + 1

        return start_index, end_index

    def _get_simulated_link(self):
        sim_link_list = np.zeros(self.engine_props["spectral_slots"])
        # Add to the step to account for the guard band
        total_slots = self.sdn_props.slots_needed + self.engine_props["guard_slots"]
        for i in range(0, len(sim_link_list), total_slots):
            value_to_set = i // self.sdn_props.slots_needed + 1
            sim_link_list[i : i + self.sdn_props.slots_needed + 2] = value_to_set

        # Add guard-bands
        sim_link_list[self.sdn_props.slots_needed :: total_slots] *= -1
        # Free the middle-most channel with respect to the number of slots needed
        center_index = len(sim_link_list) // 2
        start_index, end_index = self._get_indexes(center_index=center_index)

        sim_link_list[start_index:end_index] = 0
        return sim_link_list

    def _find_channel_mci(
        self, channels_list: list, center_freq: float, num_span: float
    ):
        total_mci = 0
        for channel in channels_list:
            # The current center frequency for the occupied channel
            curr_freq = channel[0] * self.route_props.frequency_spacing
            curr_freq += (len(channel) * self.route_props.frequency_spacing) / 2
            bandwidth = len(channel) * self.route_props.frequency_spacing
            # Power spectral density
            power_spec_dens = self.route_props.input_power / bandwidth

            curr_mci = abs(center_freq - curr_freq) + (bandwidth / 2.0)
            curr_mci = math.log(
                curr_mci / (abs(center_freq - curr_freq) - (bandwidth / 2.0))
            )
            curr_mci *= power_spec_dens**2

            total_mci += curr_mci

        total_mci = (total_mci / self.route_props.mci_worst) * num_span
        return total_mci

    def _find_link_cost(
        self, free_channels_dict: dict, taken_channels_dict: dict, num_span: float
    ):
        nli_cost = 0
        num_channels = 0
        for band, curr_channels_dict in free_channels_dict.items():
            for core_num, free_channels_list in curr_channels_dict.items():
                # Update MCI for available channel
                for channel in free_channels_list:
                    num_channels += 1
                    # Calculate the center frequency for the open channel
                    center_freq = channel[0] * self.route_props.frequency_spacing
                    center_freq += (
                        self.sdn_props.slots_needed * self.route_props.frequency_spacing
                    ) / 2

                    nli_cost += self._find_channel_mci(
                        channels_list=taken_channels_dict[band][core_num],
                        center_freq=center_freq,
                        num_span=num_span,
                    )

        # Return high cost if link is fully congested
        if num_channels == 0:
            return FULLY_CONGESTED_LINK_COST

        link_cost = nli_cost / num_channels
        return link_cost

    def find_worst_nli(self, span_count: float, band: str = DEFAULT_BAND):
        """Find the worst possible non-linear impairment cost.

        Args:
            span_count: The number of spans a link has
            band: Band to check NLI on (defaults to C-band)

        Returns:
            The worst possible NLI cost
        """
        links_list = list(self.sdn_props.network_spectrum_dict.keys())
        sim_link_list = self._get_simulated_link()

        orig_link_list = copy.copy(
            self.sdn_props.network_spectrum_dict[links_list[0]]["cores_matrix"][band]
        )
        self.sdn_props.network_spectrum_dict[links_list[0]]["cores_matrix"][band][
            0
        ] = sim_link_list

        free_channels_dict = find_free_channels(
            network_spectrum_dict=self.sdn_props.network_spectrum_dict,
            slots_needed=self.sdn_props.slots_needed,
            link_tuple=links_list[0],
        )
        taken_channels_dict = find_taken_channels(
            network_spectrum_dict=self.sdn_props.network_spectrum_dict,
            link_tuple=links_list[0],
        )
        nli_worst = self._find_link_cost(
            free_channels_dict=free_channels_dict,
            taken_channels_dict=taken_channels_dict,
            num_span=span_count,
        )

        self.sdn_props.network_spectrum_dict[links_list[0]]["cores_matrix"][
            band
        ] = orig_link_list
        return nli_worst

    @staticmethod
    def _find_adjacent_cores(core_num: int):
        """
        Identify the adjacent cores to the currently selected core.
        """
        # Every core will neighbor core 6
        adj_core_list = [6]
        if core_num == 0:
            adj_core_list.append(5)
        else:
            adj_core_list.append(core_num - 1)

        if core_num == 5:
            adj_core_list.append(0)
        else:
            adj_core_list.append(core_num + 1)

        return adj_core_list

    def _find_num_overlapped(
        self, channel: int, core_num: int, core_info_dict: dict, band: str
    ):
        num_overlapped = 0.0
        num_cores = len(core_info_dict[band])

        if core_num != 6 or num_cores <= 6:
            adj_cores_list = self._find_adjacent_cores(core_num=core_num)
            for curr_core in adj_cores_list:
                # Check bounds before accessing
                if (
                    curr_core < num_cores
                    and core_info_dict[band][curr_core][channel] > 0
                ):
                    num_overlapped += 1

            # Avoid division by zero
            if len([c for c in adj_cores_list if c < num_cores]) > 0:
                num_overlapped /= len([c for c in adj_cores_list if c < num_cores])
        # The number of overlapped cores for core six will be different since it's the center core
        else:
            for sub_core_num in range(min(6, num_cores)):
                if core_info_dict[band][sub_core_num][channel] > 0:
                    num_overlapped += 1

            num_overlapped /= 6

        return num_overlapped

    def find_xt_link_cost(self, free_slots_dict: dict, link_list: list):
        """Find the intra-core crosstalk cost for a single link.

        Note: Currently optimized for seven-core fiber systems.

        Args:
            free_slots_dict: Dictionary with all free slot indexes for each core
            link_list: The desired link to be checked

        Returns:
            The calculated XT cost for the link
        """
        xt_cost = 0
        free_slots = 0

        for band in free_slots_dict:
            for core_num in free_slots_dict[band]:
                free_slots += len(free_slots_dict[band][core_num])
                for channel in free_slots_dict[band][core_num]:
                    core_info_dict = self.sdn_props.network_spectrum_dict[link_list][
                        "cores_matrix"
                    ]
                    num_overlapped = self._find_num_overlapped(
                        channel=channel,
                        core_num=core_num,
                        core_info_dict=core_info_dict,
                        band=band,
                    )
                    xt_cost += num_overlapped

        # Return high cost if link is fully congested
        if free_slots == 0:
            return FULLY_CONGESTED_LINK_COST

        link_cost = xt_cost / free_slots
        return link_cost

    def get_nli_path(self, path_list: list):
        """
        Find the non-linear impairment for a single path.

        :param path_list: The given path.
        :return: The NLI calculation for the path.
        :rtype: float
        """
        nli_cost = 0
        for source, destination in zip(path_list, path_list[1:], strict=False):
            link_length = self.engine_props["topology"][source][destination]["length"]
            num_span = link_length / self.route_props.span_length
            link_tuple = (source, destination)
            nli_cost += self.get_nli_cost(link_tuple=link_tuple, num_span=num_span)

        return nli_cost

    def get_max_link_length(self):
        """
        Find the link with the maximum length in the entire network topology.
        """
        topology = self.engine_props["topology"]
        self.route_props.max_link_length = max(
            nx.get_edge_attributes(topology, "length").values(), default=0.0
        )

    def get_nli_cost(self, link_tuple: tuple, num_span: float):
        """
        Finds the non-linear impairment cost for a single link.

        :param link_tuple: The desired link.
        :param num_span: The number of span this link has.
        :return: The calculated NLI cost.
        :rtype: float
        """
        free_channels_dict = find_free_channels(
            network_spectrum_dict=self.sdn_props.network_spectrum_dict,
            slots_needed=self.sdn_props.slots_needed,
            link_tuple=link_tuple,
        )
        taken_channels_dict = find_taken_channels(
            network_spectrum_dict=self.sdn_props.network_spectrum_dict,
            link_tuple=link_tuple,
        )

        link_cost = self._find_link_cost(
            free_channels_dict=free_channels_dict,
            taken_channels_dict=taken_channels_dict,
            num_span=num_span,
        )

        source, dest = link_tuple[0], link_tuple[1]
        if self.route_props.max_link_length is None:
            self.get_max_link_length()

        nli_cost = (
            self.engine_props["topology"][source][dest]["length"]
            / self.route_props.max_link_length
        )
        nli_cost *= self.engine_props["beta"]
        nli_cost += (1 - self.engine_props["beta"]) * link_cost

        return nli_cost
