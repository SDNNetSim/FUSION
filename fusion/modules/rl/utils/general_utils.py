import os
from typing import Any

import numpy as np

from fusion.core.properties import RoutingProps, SDNProps
from fusion.core.spectrum_assignment import SpectrumAssignment
from fusion.modules.rl.utils.errors import ConfigurationError, RLUtilsError
from fusion.sim.utils.network import (
    classify_congestion,
    find_path_congestion,
    find_path_length,
    get_path_modulation,
)
from fusion.sim.utils.spectrum import get_shannon_entropy_fragmentation


class CoreUtilHelpers:
    """
    Contains methods to assist with reinforcement learning simulations.
    """

    def __init__(self, rl_props: Any, engine_props: Any, route_obj: Any):
        self.rl_props = rl_props

        self.engine_props = engine_props
        self.route_obj = route_obj

        self.topology = None

        self.core_number = None
        self.super_channel = None
        self.super_channel_indexes: list[list[int]] = []
        self.modulation_format = None
        self._last_processed_index = 0

    def update_snapshots(self) -> None:
        """
        Updates snapshot saves for the simulation.
        """
        arrival_count = self.rl_props.arrival_count
        snapshot_step = self.engine_props.engine_props["snapshot_step"]

        if (
            self.engine_props.engine_props["save_snapshots"]
            and (arrival_count + 1) % snapshot_step == 0
        ):
            self.engine_props.stats_obj.update_snapshot(
                network_spectrum_dict=self.engine_props.network_spectrum_dict,
                request_number=arrival_count + 1,
            )

    def get_super_channels(
        self, slots_needed: int, num_channels: int
    ) -> tuple[np.ndarray, bool]:
        """
        Gets the available 'J' super-channels for the agent to choose from
        along with a fragmentation score.

        :param slots_needed: Slots needed by the current request.
        :param num_channels: Number of channels needed by the current request.
        :return: A matrix of super-channels with their fragmentation score.
        :rtype: list
        """
        # NOTE: Currently hardcoded to use 'c' band - should be configurable
        # for different spectral bands
        path_list = self.rl_props.chosen_path_list[0]
        super_channel_index_matrix, horizontal_fragmentation_array = (
            get_shannon_entropy_fragmentation(
                path_list=path_list,
                network_spectrum=self.engine_props.network_spectrum_dict,
                spectral_slots=self.rl_props.spectral_slots,
                core_num=self.core_number if self.core_number is not None else 0,
                slots_needed=slots_needed,
                band="c",
            )
        )

        self.super_channel_indexes = super_channel_index_matrix[:num_channels].tolist()
        # There were not enough super-channels, do not penalize the agent
        no_penalty = len(self.super_channel_indexes) == 0

        response_fragmentation_matrix: list[float] = []
        for channel in self.super_channel_indexes:
            start_index = channel[0]
            response_fragmentation_matrix.append(
                horizontal_fragmentation_array[start_index]
            )

        response_fragmentation_matrix_np = np.array(response_fragmentation_matrix)
        response_fragmentation_matrix_np = np.where(
            np.isinf(response_fragmentation_matrix_np),
            100.0,
            response_fragmentation_matrix_np,
        )
        difference = self.rl_props.super_channel_space - len(
            response_fragmentation_matrix_np
        )

        if len(
            response_fragmentation_matrix_np
        ) < self.rl_props.super_channel_space or np.any(
            np.isinf(response_fragmentation_matrix_np)
        ):
            for _ in range(difference):
                response_fragmentation_matrix_np = np.append(
                    response_fragmentation_matrix_np, 100.0
                )

        return response_fragmentation_matrix_np, no_penalty

    def classify_paths(
        self, paths_list: np.ndarray
    ) -> list[tuple[int, list[Any], int]]:
        """
        Classify paths by their current congestion level.

        :param paths_list: A list of paths from source to destination.
        :return: The index of the path, the path itself, and its congestion
            index for every path.
        :rtype: list
        """
        information_list = []
        paths_list = (
            paths_list[:, 0] if isinstance(paths_list, np.ndarray) else paths_list
        )
        for path_index, current_path in enumerate(paths_list):
            current_congestion, _ = find_path_congestion(
                path_list=current_path,
                network_spectrum=self.engine_props.network_spectrum_dict,
            )
            congestion_index = classify_congestion(
                current_congestion=current_congestion,
                congestion_cutoff=self.engine_props.engine_props["cong_cutoff"],
            )

            information_list.append((path_index, current_path, congestion_index))

        return information_list

    def classify_cores(self, cores_list: list[Any]) -> list[tuple[int, Any, int]]:
        """
        Classify cores by their congestion level.

        :param cores_list: A list of cores.
        :return: The core index, the core itself, and the congestion level
            of that core for every core.
        :rtype: list
        :raises RLUtilsError: This functionality is not yet implemented
        """
        raise RLUtilsError(
            "Core classification functionality is not yet implemented. "
            "This feature requires additional development to analyze "
            "core-level congestion patterns."
        )

    def update_route_props(self, bandwidth: str, chosen_path: list[list[Any]]) -> None:
        """
        Updates the route properties.

        :param bandwidth: Bandwidth of the current request.
        :param chosen_path: Path of the current request.
        """
        self.route_obj.route_props.paths_matrix = chosen_path
        path_length = find_path_length(
            path_list=chosen_path[0],
            topology=self.engine_props.engine_props["topology"],
        )
        modulation_format = get_path_modulation(
            modulation_formats=self.engine_props.engine_props["mod_per_bw"][bandwidth],
            path_length=path_length,
        )
        self.route_obj.route_props.modulation_formats_matrix = [[modulation_format]]
        self.route_obj.route_props.weights_list.append(path_length)

    def handle_releases(self) -> None:
        """
        Checks if a request or multiple requests need to be released.
        """
        current_time = self.rl_props.arrival_list[
            min(self.rl_props.arrival_count, len(self.rl_props.arrival_list) - 1)
        ]["arrive"]

        departure_list = self.rl_props.depart_list
        while self._last_processed_index < len(departure_list):
            request_object = departure_list[self._last_processed_index]
            if request_object["depart"] > current_time:
                break

            # Build tuple key (req_id, time) to match v6 reqs_dict structure
            release_time = (request_object["req_id"], request_object["depart"])
            self.engine_props.handle_release(current_time=release_time)
            self._last_processed_index += 1

    def allocate(self) -> None:
        """
        Attempts to allocate a request.
        """
        current_request = self.rl_props.arrival_list[self.rl_props.arrival_count]
        # Build tuple key (req_id, time) to match v6 reqs_dict structure
        current_time = (current_request["req_id"], current_request["arrive"])

        if self.rl_props.forced_index is not None:
            try:
                forced_index = self.super_channel_indexes[self.rl_props.forced_index][0]
            # DRL agent picked a super-channel that is not available, block
            except IndexError:
                self.engine_props.stats_obj.blocked_reqs += 1
                self.engine_props.stats_obj.stats_props["block_reasons_dict"][
                    "congestion"
                ] += 1
                bandwidth = current_request["bandwidth"]
                self.engine_props.stats_obj.stats_props["block_bw_dict"][bandwidth] += 1
                return
        else:
            forced_index = None

        forced_modulation_format = self.route_obj.route_props.modulation_formats_matrix[
            0
        ]
        self.engine_props.handle_arrival(
            current_time=current_time,
            force_route_matrix=self.rl_props.chosen_path_list,
            force_core=self.rl_props.core_index,
            forced_index=forced_index,
            force_mod_format=forced_modulation_format,
        )

    @staticmethod
    def mock_handle_arrival(
        engine_props: dict[str, Any],
        sdn_props: dict[str, Any],
        path_list: list[Any],
        mod_format_list: list[str],
    ) -> bool:
        """
        Function to mock an arrival process or allocation in the network.

        :param engine_props: Properties of engine.
        :param sdn_props: Properties of the SDN controller.
        :param path_list: List of nodes, the current path.
        :param mod_format_list: Valid modulation formats.
        :return: If there are available spectral slots.
        :rtype: bool
        """
        # Create dummy route_props and sdn_props objects for mock allocation
        route_props = RoutingProps()
        sdn_props_obj = SDNProps()
        # Convert dict to SDNProps object
        for key, value in sdn_props.items():
            setattr(sdn_props_obj, key, value)
        spectrum_obj = SpectrumAssignment(
            engine_props=engine_props, sdn_props=sdn_props_obj, route_props=route_props
        )

        spectrum_obj.spectrum_props.forced_index = None
        spectrum_obj.spectrum_props.forced_core = None
        spectrum_obj.spectrum_props.path_list = path_list
        spectrum_obj.get_spectrum(mod_format_list=mod_format_list)
        # Request was blocked for this path
        if spectrum_obj.spectrum_props.is_free is not True:
            return False

        return True

    def update_mock_sdn(self, current_request: dict[str, Any]) -> SDNProps:
        """
        Updates the mock sdn dictionary to find select routes.

        :param current_request: The current request.
        :return: The mock return of the SDN controller.
        :rtype: dict
        """
        mock_sdn = SDNProps()
        params = {
            "req_id": current_request["req_id"],
            "source": current_request["source"],
            "destination": current_request["destination"],
            "bandwidth": current_request["bandwidth"],
            "network_spectrum_dict": self.engine_props.network_spectrum_dict,
            "topology": self.topology,
            "mod_formats_dict": current_request["mod_formats"],
            "num_trans": 1.0,
            "block_reason": None,
            "modulation_list": [],
            "crosstalk_list": [],
            "is_sliced": False,
            "core_list": [],
            "bandwidth_list": [],
            "path_weight": [],
        }

        for key, value in params.items():
            setattr(mock_sdn, key, value)

        return mock_sdn

    def reset_reqs_dict(self, seed: int) -> None:
        """
        Resets the request dictionary.

        :param seed: The random seed.
        """
        self._last_processed_index = 0
        self.engine_props.reqs_status_dict = {}
        self.engine_props.generate_requests(seed=seed)

        for request_time in self.engine_props.reqs_dict:
            if self.engine_props.reqs_dict[request_time]["request_type"] == "arrival":
                self.rl_props.arrival_list.append(
                    self.engine_props.reqs_dict[request_time]
                )
            else:
                self.rl_props.depart_list.append(
                    self.engine_props.reqs_dict[request_time]
                )


# NOTE: Current implementation is limited to 's1' simulation thread
# Future versions should support multi-threaded simulation environments
def determine_model_type(sim_dict: dict) -> str:
    """
    Determines the type of agent being used based on the provided simulation dictionary.

    :param sim_dict: A dictionary containing simulation configuration.
    :return: A string representing the model type ('path_algorithm',
        'core_algorithm', 'spectrum_algorithm').
    """
    if "s1" in sim_dict:
        sim_dict = sim_dict["s1"]
    if sim_dict.get("path_algorithm") is not None:
        return "path_algorithm"
    if sim_dict.get("core_algorithm") is not None:
        return "core_algorithm"
    if sim_dict.get("spectrum_algorithm") is not None:
        return "spectrum_algorithm"

    raise ConfigurationError(
        "No valid algorithm type found in simulation configuration. "
        f"Expected one of: 'path_algorithm', 'core_algorithm', or 'spectrum_algorithm' "
        f"in sim_dict keys: {list(sim_dict.keys())}. "
        "Please verify your configuration file contains a valid algorithm setting."
    )


def save_arr(arr: np.ndarray, sim_dict: dict[str, Any], file_name: str) -> None:
    """
    Save a numpy array to a specific file path constructed from simulation details.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm_type = sim_dict[model_type]

    network, date, time = sim_dict["network"], sim_dict["date"], sim_dict["sim_start"]
    file_path = os.path.join("logs", algorithm_type, network, date, time, file_name)
    np.save(file_path, arr)
