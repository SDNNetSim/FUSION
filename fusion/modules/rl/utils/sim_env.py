import os

import numpy as np

from fusion.core.routing import Routing

from fusion.sim.sim_helpers import find_path_cong

from fusion.modules.rl.args.general_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS
from fusion.modules.rl.args.observation_args import OBS_DICT
from fusion.modules.rl.utils.topology import convert_networkx_topo


class SimEnvUtils:
    """
    Provides helper methods for managing steps, training/testing logic, and observations
    in the SimEnv reinforcement learning environment.
    """

    def __init__(self, sim_env):
        """
        Initializes the RL step helper with access to the SimEnv instance.

        :param sim_env: The main simulation environment object.
        """
        self.sim_env = sim_env

    def check_terminated(self):
        """
        Checks whether the simulation has reached termination conditions.

        :return: A boolean indicating if the simulation is terminated.
        """
        if self.sim_env.rl_props.arrival_count == (self.sim_env.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            if self.sim_env.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS and self.sim_env.sim_dict[
                'is_training']:
                self.sim_env.path_agent.end_iter()
            elif self.sim_env.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS and self.sim_env.sim_dict[
                'is_training']:
                self.sim_env.core_agent.end_iter()
            self.sim_env.engine_obj.end_iter(iteration=self.sim_env.iteration, print_flag=False, base_fp=base_fp)
            self.sim_env.iteration += 1
        else:
            terminated = False

        return terminated

    def handle_test_train_step(self, was_allocated: bool, path_length: int, trial: int):
        """
        Handles updates specific to training or testing during the current simulation step.

        :param was_allocated: Whether the resource allocation was successful.
        :param trial: The current trial number.
        :param path_length: The length of the chosen path.
        """
        if self.sim_env.sim_dict['is_training']:
            if self.sim_env.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self.sim_env.path_agent.update(was_allocated=was_allocated,
                                               net_spec_dict=self.sim_env.engine_obj.net_spec_dict,
                                               iteration=self.sim_env.iteration, path_length=path_length,
                                               trial=trial)
            elif self.sim_env.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            self.sim_env.path_agent.update(was_allocated=was_allocated,
                                           net_spec_dict=self.sim_env.engine_obj.net_spec_dict,
                                           iteration=self.sim_env.iteration, path_length=path_length)
            self.sim_env.core_agent.update(was_allocated=was_allocated,
                                           net_spec_dict=self.sim_env.engine_obj.net_spec_dict,
                                           iteration=self.sim_env.iteration)

    def handle_step(self, action: int, is_drl_agent: bool):
        """
        Handles path-related decisions during training and testing phases.
        """
        # Q-learning has access to its own paths, everything else needs the route object
        if 'bandit' in self.sim_env.sim_dict['path_algorithm'] or is_drl_agent:
            self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
            self.sim_env.route_obj.engine_props['route_method'] = 'k_shortest_path'
            self.sim_env.route_obj.get_route()

        self.sim_env.path_agent.get_route(route_obj=self.sim_env.route_obj, action=action)
        self.sim_env.rl_help_obj.rl_props.chosen_path_list = [self.sim_env.rl_props.chosen_path_list]
        self.sim_env.route_obj.route_props.paths_matrix = self.sim_env.rl_help_obj.rl_props.chosen_path_list
        self.sim_env.rl_props.core_index = None
        self.sim_env.rl_props.forced_index = None

    def handle_core_train(self):
        """
        Handles core-related logic during the training phase.
        """
        self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
        self.sim_env.route_obj.engine_props['route_method'] = 'k_shortest_path'
        self.sim_env.route_obj.get_route()
        self.sim_env.sim_env_helper.determine_core_penalty()

        self.sim_env.rl_props.forced_index = None

        self.sim_env.core_agent.get_core()

    def handle_spectrum_train(self):
        """
        Handles spectrum-related logic during the training phase.
        """
        self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
        self.sim_env.route_obj.engine_props['route_method'] = 'shortest_path'
        self.sim_env.route_obj.get_route()
        self.sim_env.rl_props.paths_list = self.sim_env.route_obj.route_props.paths_matrix
        self.sim_env.rl_props.chosen_path = self.sim_env.route_obj.route_props.paths_matrix
        self.sim_env.rl_props.path_index = 0
        self.sim_env.rl_props.core_index = None

    def get_obs(self, bandwidth, holding_time):
        """
        Generates the current observation for the agent based on the environment state.

        :return: A dictionary containing observation components.
        """
        if self.sim_env.rl_props.arrival_count == self.sim_env.engine_obj.engine_props['num_requests']:
            curr_req = self.sim_env.rl_props.arrival_list[self.sim_env.rl_props.arrival_count - 1]
        else:
            curr_req = self.sim_env.rl_props.arrival_list[self.sim_env.rl_props.arrival_count]

        self.sim_env.rl_help_obj.handle_releases()
        self.sim_env.rl_props.source = int(curr_req['source'])
        self.sim_env.rl_props.destination = int(curr_req['destination'])
        self.sim_env.rl_props.mock_sdn_dict = self.sim_env.rl_help_obj.update_mock_sdn(curr_req=curr_req)

        resp_dict = self.sim_env.sim_env_helper.get_drl_obs(bandwidth=bandwidth, holding_time=holding_time)
        return resp_dict


class SimEnvObs:
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
        self.routing_obj = None

        self.lowest_holding = None
        self.highest_holding = None
        self.edge_index = None
        self.edge_attr = None
        self.node_feats = None
        self.str2idx = None
        self.id2idx = None

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
        self.routing_obj = Routing(engine_props=self.sim_env.engine_obj.engine_props,
                                   sdn_props=self.sim_env.engine_obj.sdn_obj.sdn_props)

        self.routing_obj.sdn_props.bandwidth = bandwidth
        self.routing_obj.sdn_props.source = str(self.sim_env.rl_props.source)
        self.routing_obj.sdn_props.destination = str(self.sim_env.rl_props.destination)
        self.routing_obj.get_route()
        route_props = self.routing_obj.route_props

        slots_needed_list = list()
        mod_bw_dict = self.sim_env.engine_obj.engine_props['mod_per_bw']
        for mod_format in route_props.mod_formats_matrix:
            if not mod_format[0]:
                slots_needed = -1
            else:
                slots_needed = mod_bw_dict[bandwidth][mod_format[0]]['slots_needed']
            slots_needed_list.append(slots_needed)

        paths_cong = list()
        available_slots = list()
        for curr_path in route_props.paths_matrix:
            curr_cong, curr_slots = find_path_cong(path_list=curr_path,
                                                   net_spec_dict=self.sim_env.engine_obj.net_spec_dict)
            paths_cong.append(curr_cong)
            available_slots.append(curr_slots)

        norm_list = route_props.weights_list
        return slots_needed_list, norm_list, paths_cong, available_slots

    def get_path_masks(self, resp_dict: dict):
        """
        Encodes which paths are available via masks.
        """
        resp_dict['x'] = self.node_feats.numpy()
        resp_dict['edge_index'] = self.edge_index.numpy()
        resp_dict['edge_attr'] = self.edge_attr.numpy()

        edge_pairs = list(zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()))
        edge_map = {pair: idx for idx, pair in enumerate(edge_pairs)}
        edge_shape = self.edge_index.shape[1]

        paths_matrix = self.routing_obj.route_props.paths_matrix
        k_shape = self.sim_env.rl_props.k_paths
        masks = np.zeros((k_shape, edge_shape), dtype=np.float32)

        for i, path in enumerate(paths_matrix[:k_shape]):
            idx_path = [self.str2idx[label] for label in path]
            e_idxs = [edge_map[(u, v)] for u, v in zip(idx_path, idx_path[1:])]
            masks[i, e_idxs] = 1.0

        resp_dict['path_masks'] = masks

    def get_drl_obs(self, bandwidth, holding_time):
        """
        Creates observation data for Deep Reinforcement Learning (DRL) in a graph-based
        environment.
        """
        topo_graph = self.sim_env.engine_obj.engine_props['topology']
        include_graph = False
        resp_dict = dict()
        obs_space_key = self.sim_env.engine_obj.engine_props['obs_space']
        if 'graph' in obs_space_key:
            if None in (self.node_feats, self.edge_attr, self.edge_index):
                self.edge_index, self.edge_attr, self.node_feats, self.id2idx = convert_networkx_topo(
                    topo_graph, as_directed=True
                )
                self.str2idx = {str(n): idx for n, idx in self.id2idx.items()}
            include_graph = True
            obs_space_key = obs_space_key.replace('_graph', '')
        if 'source' in OBS_DICT[obs_space_key]:
            source_obs = np.zeros(self.sim_env.rl_props.num_nodes)
            source_obs[self.sim_env.rl_props.source] = 1.0
            resp_dict['source'] = source_obs
        if 'destination' in OBS_DICT[obs_space_key]:
            dest_obs = np.zeros(self.sim_env.rl_props.num_nodes)
            dest_obs[self.sim_env.rl_props.destination] = 1.0
            resp_dict['destination'] = dest_obs

        if not hasattr(self.sim_env, "bw_obs_list"):
            des_dict = self.sim_env.sim_dict['request_distribution']
            self.sim_env.bw_obs_list = sorted([k for k, v in des_dict.items() if v != 0], key=int)
        if 'request_bandwidth' in OBS_DICT[obs_space_key]:
            bw_index = self.sim_env.bw_obs_list.index(bandwidth)
            req_band = np.zeros(len(self.sim_env.bw_obs_list))
            req_band[bw_index] = 1.0
            resp_dict['request_bandwidth'] = req_band

        if 'holding_time' in OBS_DICT[obs_space_key]:
            req_holding_scaled = self._scale_req_holding(holding_time=holding_time)
            resp_dict['holding_time'] = req_holding_scaled

        # TODO: Add and initialize bandwidth in self
        # TODO: Filter these later, but they may not always be in the observation space
        slots_needed, path_lengths, paths_cong, available_slots = self._get_paths_slots(bandwidth=bandwidth)

        if 'slots_needed' in OBS_DICT[obs_space_key]:
            resp_dict['slots_needed'] = slots_needed
        if 'path_lengths' in OBS_DICT[obs_space_key]:
            resp_dict['path_lengths'] = path_lengths
        if 'paths_cong' in OBS_DICT[obs_space_key]:
            resp_dict['paths_cong'] = paths_cong
        if 'available_slots' in OBS_DICT[obs_space_key]:
            resp_dict['available_slots'] = available_slots
        if 'is_feasible' in OBS_DICT[obs_space_key]:
            raise NotImplementedError

        if include_graph:
            self.get_path_masks(resp_dict=resp_dict)

        return resp_dict
