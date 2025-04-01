import os

import numpy as np

from src.spectrum_assignment import SpectrumAssignment
from src.routing import Routing

from helper_scripts.sim_helpers import find_path_len, get_path_mod, get_hfrag
from helper_scripts.sim_helpers import find_path_cong, classify_cong, find_core_cong

from reinforcement_learning.args.general_args import VALID_PATH_ALGORITHMS
from arg_scripts.sdn_args import SDNProps


class CoreUtilHelpers:
    """
    Contains methods to assist with reinforcement learning simulations.
    """

    def __init__(self, rl_props: object, engine_obj: object, route_obj: object):
        self.rl_props = rl_props

        self.engine_obj = engine_obj
        self.route_obj = route_obj

        self.topology = None

        self.core_num = None
        self.super_channel = None
        self.super_channel_indexes = list()
        self.mod_format = None
        self._last_processed_index = 0

    def update_snapshots(self):
        """
        Updates snapshot saves for the simulation.
        """
        arrival_count = self.rl_props.arrival_count
        snapshot_step = self.engine_obj.engine_props['snapshot_step']

        if self.engine_obj.engine_props['save_snapshots'] and (arrival_count + 1) % snapshot_step == 0:
            self.engine_obj.stats_obj.update_snapshot(net_spec_dict=self.engine_obj.net_spec_dict,
                                                      req_num=arrival_count + 1)

    def get_super_channels(self, slots_needed: int, num_channels: int):
        """
        Gets the available 'J' super-channels for the agent to choose from along with a fragmentation score.

        :param slots_needed: Slots needed by the current request.
        :param num_channels: Number of channels needed by the current request.
        :return: A matrix of super-channels with their fragmentation score.
        :rtype: list
        """
        # TODO: (drl_path_agents) The 'c' band used by default
        path_list = self.rl_props.chosen_path_list[0]
        sc_index_mat, hfrag_arr = get_hfrag(path_list=path_list, net_spec_dict=self.engine_obj.net_spec_dict,
                                            spectral_slots=self.rl_props.spectral_slots, core_num=self.core_num,
                                            slots_needed=slots_needed, band='c')

        self.super_channel_indexes = sc_index_mat[:num_channels]
        # There were not enough super-channels, do not penalize the agent
        no_penalty = len(self.super_channel_indexes) == 0

        resp_frag_mat = list()
        for channel in self.super_channel_indexes:
            start_index = channel[0]
            resp_frag_mat.append(hfrag_arr[start_index])

        resp_frag_mat = np.where(np.isinf(resp_frag_mat), 100.0, resp_frag_mat)
        difference = self.rl_props.super_channel_space - len(resp_frag_mat)

        if len(resp_frag_mat) < self.rl_props.super_channel_space or np.any(np.isinf(resp_frag_mat)):
            for _ in range(difference):
                resp_frag_mat = np.append(resp_frag_mat, 100.0)

        return resp_frag_mat, no_penalty

    def classify_paths(self, paths_list: list):
        """
        Classify paths by their current congestion level.

        :param paths_list: A list of paths from source to destination.
        :return: The index of the path, the path itself, and its congestion index for every path.
        :rtype: list
        """
        info_list = list()
        paths_list = paths_list[:, 0]
        for path_index, curr_path in enumerate(paths_list):
            curr_cong = find_path_cong(path_list=curr_path, net_spec_dict=self.engine_obj.net_spec_dict)
            cong_index = classify_cong(curr_cong=curr_cong)

            info_list.append((path_index, curr_path, cong_index))

        return info_list

    def classify_cores(self, cores_list: list):
        """
        Classify cores by their congestion level.

        :param cores_list: A list of cores.
        :return: The core index, the core itself, and the congestion level of that core for every core.
        :rtype: list
        """
        info_list = list()

        for core_index, curr_core in enumerate(cores_list):
            path_list = curr_core['path'][0]
            curr_cong = find_core_cong(core_index=core_index, net_spec_dict=self.engine_obj.net_spec_dict,
                                       path_list=path_list)
            cong_index = classify_cong(curr_cong=curr_cong)

            info_list.append((core_index, curr_core[cong_index], cong_index))

        return info_list

    def update_route_props(self, bandwidth: str, chosen_path: list):
        """
        Updates the route properties.

        :param bandwidth: Bandwidth of the current request.
        :param chosen_path: Path of the current request.
        """
        self.route_obj.route_props.paths_matrix = chosen_path
        path_len = find_path_len(path_list=chosen_path[0], topology=self.engine_obj.engine_props['topology'])
        mod_format = get_path_mod(mods_dict=self.engine_obj.engine_props['mod_per_bw'][bandwidth], path_len=path_len)
        self.route_obj.route_props.mod_formats_matrix = [[mod_format]]
        self.route_obj.route_props.weights_list.append(path_len)

    def handle_releases(self):
        """
        Checks if a request or multiple requests need to be released.
        """
        curr_time = self.rl_props.arrival_list[min(self.rl_props.arrival_count,
                                                   len(self.rl_props.arrival_list) - 1)]['arrive']

        depart_list = self.rl_props.depart_list
        while self._last_processed_index < len(depart_list):
            req_obj = depart_list[self._last_processed_index]
            if req_obj['depart'] > curr_time:
                break

            self.engine_obj.handle_release(curr_time=req_obj['depart'])
            self._last_processed_index += 1

    def allocate(self):
        """
        Attempts to allocate a request.
        """
        curr_time = self.rl_props.arrival_list[self.rl_props.arrival_count]['arrive']
        if self.rl_props.forced_index is not None:
            try:
                forced_index = self.super_channel_indexes[self.rl_props.forced_index][0]
            # DRL agent picked a super-channel that is not available, block
            except IndexError:
                self.engine_obj.stats_obj.blocked_reqs += 1
                self.engine_obj.stats_obj.stats_props['block_reasons_dict']['congestion'] += 1
                bandwidth = self.rl_props.arrival_list[self.rl_props.arrival_count]['bandwidth']
                self.engine_obj.stats_obj.stats_props['block_bw_dict'][bandwidth] += 1
                return
        else:
            forced_index = None

        force_mod_format = self.route_obj.route_props.mod_formats_matrix[0]
        self.engine_obj.handle_arrival(curr_time=curr_time, force_route_matrix=self.rl_props.chosen_path_list,
                                       force_core=self.rl_props.core_index,
                                       forced_index=forced_index, force_mod_format=force_mod_format)

    @staticmethod
    def mock_handle_arrival(engine_props: dict, sdn_props: dict, path_list: list, mod_format_list: list):
        """
        Function to mock an arrival process or allocation in the network.

        :param engine_props: Properties of engine.
        :param sdn_props: Properties of the SDN controller.
        :param path_list: List of nodes, the current path.
        :param mod_format_list: Valid modulation formats.
        :return: If there are available spectral slots.
        :rtype: bool
        """
        route_props = None
        spectrum_obj = SpectrumAssignment(engine_props=engine_props, sdn_props=sdn_props, route_props=route_props)

        spectrum_obj.spectrum_props.forced_index = None
        spectrum_obj.spectrum_props.forced_core = None
        spectrum_obj.spectrum_props.path_list = path_list
        spectrum_obj.get_spectrum(mod_format_list=mod_format_list)
        # Request was blocked for this path
        if spectrum_obj.spectrum_props.is_free is not True:
            return False

        return True

    def update_mock_sdn(self, curr_req: dict):
        """
        Updates the mock sdn dictionary to find select routes.

        :param curr_req: The current request.
        :return: The mock return of the SDN controller.
        :rtype: dict
        """
        mock_sdn = SDNProps()
        params = {
            'req_id': curr_req['req_id'],
            'source': curr_req['source'],
            'destination': curr_req['destination'],
            'bandwidth': curr_req['bandwidth'],
            'net_spec_dict': self.engine_obj.net_spec_dict,
            'topology': self.topology,
            'mod_formats_dict': curr_req['mod_formats'],
            'num_trans': 1.0,
            'block_reason': None,
            'modulation_list': list(),
            'xt_list': list(),
            'is_sliced': False,
            'core_list': list(),
            'bandwidth_list': list(),
            'path_weight': list(),
        }

        for key, value in params.items():
            setattr(mock_sdn, key, value)

        return mock_sdn

    def reset_reqs_dict(self, seed: int):
        """
        Resets the request dictionary.

        :param seed: The random seed.
        """
        self._last_processed_index = 0
        self.engine_obj.reqs_status_dict = dict()
        self.engine_obj.generate_requests(seed=seed)

        for req_time in self.engine_obj.reqs_dict:
            if self.engine_obj.reqs_dict[req_time]['request_type'] == 'arrival':
                self.rl_props.arrival_list.append(self.engine_obj.reqs_dict[req_time])
            else:
                self.rl_props.depart_list.append(self.engine_obj.reqs_dict[req_time])


class SimEnvHelpers:
    """
    Encapsulates high-level helper methods tailored for managing and enhancing the behavior of the `SimEnv` class during
    reinforcement learning simulations.
    """

    def __init__(self, sim_env: object):
        """
        Initializes the helper methods class with shared context.

        :param sim_env: The main simulation environment object.
        """
        self.sim_env = sim_env

        self.lowest_holding = None
        self.highest_holding = None

    def update_helper_obj(self, action: int, bandwidth: str):
        """
        Updates the helper object with new actions and configurations.
        """
        if self.sim_env.engine_obj.engine_props['is_drl_agent']:
            self.sim_env.rl_help_obj.path_index = action
        else:
            self.sim_env.rl_help_obj.path_index = self.sim_env.rl_props.path_index

        self.sim_env.rl_help_obj.core_num = self.sim_env.rl_props.core_index

        if self.sim_env.sim_dict['spectrum_algorithm'] in ('dqn', 'ppo', 'a2c'):
            self.sim_env.rl_help_obj.rl_props.forced_index = action
        else:
            self.sim_env.rl_help_obj.rl_props.forced_index = None

        self.sim_env.rl_help_obj.rl_props = self.sim_env.rl_props
        self.sim_env.rl_help_obj.engine_obj = self.sim_env.engine_obj
        self.sim_env.rl_help_obj.handle_releases()
        self.sim_env.rl_help_obj.update_route_props(chosen_path=self.sim_env.rl_props.chosen_path_list,
                                                    bandwidth=bandwidth)

    def determine_core_penalty(self):
        """
        Determines penalty for the core algorithm based on path availability.
        """
        # Default to first fit if all paths fail
        self.sim_env.rl_props.chosen_path = [self.sim_env.route_obj.route_props.paths_matrix[0]]
        self.sim_env.rl_props.chosen_path_index = 0
        for path_index, path_list in enumerate(self.sim_env.route_obj.route_props.paths_matrix):
            mod_format_list = self.sim_env.route_obj.route_props.mod_formats_matrix[path_index]

            was_allocated = self.sim_env.rl_help_obj.mock_handle_arrival(
                engine_props=self.sim_env.engine_obj.engine_props,
                sdn_props=self.sim_env.rl_props.mock_sdn_dict,
                mod_format_list=mod_format_list,
                path_list=path_list)

            if was_allocated:
                self.sim_env.rl_props.chosen_path_list = [path_list]
                self.sim_env.rl_props.chosen_path_index = path_index
                break

    def handle_test_train_obs(self, curr_req: dict):  # pylint: disable=unused-argument
        """
        Handles path and core selection during training/testing phases based on the current request.

        Returns:
            Path modulation format, if available.
        """
        if self.sim_env.sim_dict['is_training']:
            if self.sim_env.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self.sim_env.step_helper.handle_path_train_test()
            else:
                raise NotImplementedError
        else:
            self.sim_env.step_helpers.handle_path_train_test()

    def _scale_req_holding(self, holding_time):
        req_dict = self.sim_env.engine_obj.reqs_dict
        if self.lowest_holding is None or self.highest_holding is None:
            differences = [value['depart'] - arrival for arrival, value in req_dict.items()]

            self.lowest_holding = min(differences)
            self.highest_holding = max(differences)

        if self.lowest_holding == self.highest_holding:
            raise ValueError("x_max and x_min cannot be the same value.")

        scaled_holding = (holding_time - self.lowest_holding) / (self.highest_holding - self.lowest_holding)
        return scaled_holding

    def _get_paths_slots(self, bandwidth):
        # TODO: Can move this to the constructor...
        routing_obj = Routing(engine_props=self.sim_env.engine_obj.engine_props,
                              sdn_props=self.sim_env.engine_obj.sdn_obj.sdn_props)

        routing_obj.sdn_props.bandwidth = bandwidth
        routing_obj.sdn_props.source = str(self.sim_env.rl_props.source)
        routing_obj.sdn_props.destination = str(self.sim_env.rl_props.destination)
        routing_obj.get_route()
        route_props = routing_obj.route_props

        slots_needed_list = list()
        mod_bw_dict = self.sim_env.engine_obj.engine_props['mod_per_bw']
        for mod_format in route_props.mod_formats_matrix:
            if not mod_format[0]:
                slots_needed = -1
            else:
                slots_needed = mod_bw_dict[bandwidth][mod_format[0]]['slots_needed']
            slots_needed_list.append(slots_needed)

        return slots_needed_list, route_props.weights_list

    def get_drl_obs(self, bandwidth, holding_time):
        """
        Creates observation data for Deep Reinforcement Learning (DRL) in a graph-based
        environment.
        """
        source_obs = np.zeros(self.sim_env.rl_props.num_nodes)
        source_obs[self.sim_env.rl_props.source] = 1.0
        dest_obs = np.zeros(self.sim_env.rl_props.num_nodes)
        dest_obs[self.sim_env.rl_props.destination] = 1.0

        if not hasattr(self.sim_env, "bw_obs_list"):
            des_dict = self.sim_env.sim_dict['request_distribution']
            self.sim_env.bw_obs_list = sorted([k for k, v in des_dict.items() if v != 0], key=int)

        bw_index = self.sim_env.bw_obs_list.index(bandwidth)
        req_obs = np.zeros(len(self.sim_env.bw_obs_list))
        req_obs[bw_index] = 1.0
        req_holding_scaled = self._scale_req_holding(holding_time=holding_time)

        # TODO: Add and initialize bandwidth in self
        slots_needed, path_lengths = self._get_paths_slots(bandwidth=bandwidth)

        return {'source_obs': source_obs, 'dest_obs': dest_obs, 'req_obs': req_obs, 'req_holding': req_holding_scaled,
                'slots_needed': slots_needed, 'path_lengths': path_lengths}


# TODO: (drl_path_agents) Only works for s1
def determine_model_type(sim_dict: dict) -> str:
    """
    Determines the type of agent being used based on the provided simulation dictionary.

    :param sim_dict: A dictionary containing simulation configuration.
    :return: A string representing the model type ('path_algorithm', 'core_algorithm', 'spectrum_algorithm').
    """
    if 's1' in sim_dict:
        sim_dict = sim_dict['s1']
    if sim_dict.get('path_algorithm') is not None:
        return 'path_algorithm'
    if sim_dict.get('core_algorithm') is not None:
        return 'core_algorithm'
    if sim_dict.get('spectrum_algorithm') is not None:
        return 'spectrum_algorithm'

    raise ValueError("No valid algorithm type found in sim_dict. "
                     "Ensure 'path_algorithm', 'core_algorithm', or 'spectrum_algorithm' is set.")


def save_arr(arr: np.array, sim_dict: dict, file_name: str):
    """
    Save a numpy array to a specific file path constructed from simulation details.
    """
    model_type = determine_model_type(sim_dict=sim_dict)
    algorithm_type = sim_dict[model_type]

    network, date, time = sim_dict['network'], sim_dict['date'], sim_dict['sim_start']
    file_path = os.path.join('logs', algorithm_type, network, date, time, file_name)
    np.save(file_path, arr)




class FragmentationTracker:
    def __init__(self, num_nodes, core_count, spectral_slots):
        """
        Initializes the fragmentation tracker.

        :param num_nodes: Total number of nodes in the network topology.
        :param core_count: Number of cores per link.
        :param spectral_slots: Number of spectral slots per core.
        """
        self.num_nodes = num_nodes
        self.core_count = core_count
        self.spectral_slots = spectral_slots

        # Bitmask: [src, dst, core, slots]
        self.core_masks = np.zeros((num_nodes, num_nodes, core_count, spectral_slots), dtype=np.uint8)

        # Dirty mask to avoid unnecessary recomputation
        self.dirty_cores = np.zeros((num_nodes, num_nodes, core_count), dtype=bool)

        # Normalize by max possible transitions per core (not including zero-padding)
        self.norm_factor = (spectral_slots // 3) + 1

    def update(self, src: int, dst: int, core_index: int, start_slot: int, end_slot: int, is_allocate: bool = True):
        """
        Updates the core mask for a specific link (src→dst), core, and slot range.

        :param src: Source node index
        :param dst: Destination node index
        :param core_index: Core used for allocation
        :param start_slot: First slot index allocated
        :param end_slot: Last slot index allocated (inclusive)
        :param is_allocate: If True, mark as allocated (1); if False, free (0)
        """
        slot_indices = np.arange(start_slot, end_slot + 1)
        if is_allocate:
            self.core_masks[src, dst, core_index, slot_indices] = 1
        else:
            self.core_masks[src, dst, core_index, slot_indices] = 0

        self.dirty_cores[src, dst, core_index] = True

    def get_fragmentation(self, chosen_path: list[int], core_index: int):
        """
        Computes fragmentation for the specified core and chosen path.

        :param chosen_path: List of node indices representing the path.
        :param core_index: Core used during allocation.
        :return: Dictionary with formatted NumPy arrays for Gym observations.
        """
        # 1. Core-level fragmentation (last link in path)
        core_frag = 0.0
        if len(chosen_path) >= 2:
            last_src, last_dst = chosen_path[-2], chosen_path[-1]
            if self.dirty_cores[last_src, last_dst, core_index]:
                core_mask = self.core_masks[last_src, last_dst, core_index]
                padded = np.pad(core_mask, (1,), 'constant')
                transitions = np.abs(np.diff(padded))
                core_frag = np.sum(transitions) / (2 * self.norm_factor)

        # 2. Path-level fragmentation (avg across all links in path, all cores)
        path_transitions = 0
        total_links = len(chosen_path) - 1

        for i in range(total_links):
            src, dst = chosen_path[i], chosen_path[i + 1]
            for c in range(self.core_count):
                padded = np.pad(self.core_masks[src, dst, c], (1,), 'constant')
                transitions = np.abs(np.diff(padded))
                path_transitions += np.sum(transitions)

        if total_links > 0:
            path_frag = path_transitions / (2 * self.norm_factor * total_links * self.core_count)
        else:
            path_frag = 0.0

        self.dirty_cores.fill(False)

        return {
            'fragmentation': np.array([core_frag], dtype=np.float32),
            'path_frag': np.array([path_frag], dtype=np.float32)
        }
