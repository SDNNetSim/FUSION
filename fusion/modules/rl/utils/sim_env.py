import os
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from fusion.core.routing import Routing

if TYPE_CHECKING:
    from fusion.core.routing import Routing as RoutingType
else:
    RoutingType = Any
from fusion.modules.rl.args.general_args import (
    VALID_CORE_ALGORITHMS,
    VALID_PATH_ALGORITHMS,
)
from fusion.modules.rl.args.observation_args import OBS_DICT
from fusion.modules.rl.utils.errors import RLUtilsError
from fusion.modules.rl.utils.topology import convert_networkx_topo
from fusion.sim.utils.network import find_path_congestion


class SimEnvUtils:
    """
    Provides helper methods for managing steps, training/testing logic, and observations
    in the SimEnv reinforcement learning environment.
    """

    def __init__(self, sim_env: Any) -> None:
        """
        Initializes the RL step helper with access to the SimEnv instance.

        :param sim_env: The main simulation environment object
        :type sim_env: Any
        """
        self.sim_env = sim_env

    def check_terminated(self) -> bool:
        """
        Checks whether the simulation has reached termination conditions.

        :return: A boolean indicating if the simulation is terminated
        :rtype: bool
        """
        if (
            self.sim_env.rl_props.arrival_count
            == (self.sim_env.engine_obj.engine_props["num_requests"])
        ):
            terminated = True
            base_fp = os.path.join("data")
            if (
                self.sim_env.sim_dict["path_algorithm"] in VALID_PATH_ALGORITHMS
                and self.sim_env.sim_dict["is_training"]
            ):
                self.sim_env.path_agent.end_iter()
            elif (
                self.sim_env.sim_dict["core_algorithm"] in VALID_CORE_ALGORITHMS
                and self.sim_env.sim_dict["is_training"]
            ):
                self.sim_env.core_agent.end_iter()
            self.sim_env.engine_obj.end_iter(
                iteration=self.sim_env.iteration,
                print_flag=False,
                base_file_path=base_fp,
            )
            self.sim_env.iteration += 1
        else:
            terminated = False

        return terminated

    def handle_test_train_step(
        self, was_allocated: bool, path_length: int, trial: int
    ) -> None:
        """
        Handles updates specific to training or testing during the current
        simulation step.

        :param was_allocated: Whether the resource allocation was successful
        :type was_allocated: bool
        :param path_length: The length of the chosen path
        :type path_length: int
        :param trial: The current trial number
        :type trial: int
        """
        if self.sim_env.sim_dict["is_training"]:
            if self.sim_env.sim_dict["path_algorithm"] in VALID_PATH_ALGORITHMS:
                self.sim_env.path_agent.update(
                    was_allocated=was_allocated,
                    network_spectrum_dict=self.sim_env.engine_obj.network_spectrum_dict,
                    iteration=self.sim_env.iteration,
                    path_length=path_length,
                    trial=trial,
                )
            elif self.sim_env.sim_dict["core_algorithm"] in VALID_CORE_ALGORITHMS:
                raise RLUtilsError(
                    "Core algorithm handling is not yet implemented. "
                    "This feature requires additional development for "
                    "core-specific RL agents."
                )
            else:
                raise RLUtilsError(
                    "Unsupported algorithm configuration for observation handling. "
                    f"Expected valid path or core algorithm, got: "
                    f"path={self.sim_env.sim_dict.get('path_algorithm')}, "
                    f"core={self.sim_env.sim_dict.get('core_algorithm')}"
                )
        else:
            self.sim_env.path_agent.update(
                was_allocated=was_allocated,
                network_spectrum_dict=self.sim_env.engine_obj.network_spectrum_dict,
                iteration=self.sim_env.iteration,
                path_length=path_length,
            )
            self.sim_env.core_agent.update(
                was_allocated=was_allocated,
                network_spectrum_dict=self.sim_env.engine_obj.network_spectrum_dict,
                iteration=self.sim_env.iteration,
            )

    def handle_step(self, action: int, is_drl_agent: bool) -> None:
        """
        Handles path-related decisions during training and testing phases.

        :param action: The action to take
        :type action: int
        :param is_drl_agent: Whether the agent is a DRL agent
        :type is_drl_agent: bool
        """
        # Q-learning has access to its own paths, everything else needs the route object
        if "bandit" in self.sim_env.sim_dict["path_algorithm"] or is_drl_agent:
            self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
            self.sim_env.route_obj.engine_props["route_method"] = "k_shortest_path"
            self.sim_env.route_obj.get_route()

        self.sim_env.path_agent.get_route(
            route_obj=self.sim_env.route_obj, action=action
        )
        self.sim_env.rl_help_obj.rl_props.chosen_path_list = [
            self.sim_env.rl_props.chosen_path_list
        ]
        self.sim_env.route_obj.route_props.paths_matrix = (
            self.sim_env.rl_help_obj.rl_props.chosen_path_list
        )
        self.sim_env.rl_props.core_index = None
        self.sim_env.rl_props.forced_index = None

    def handle_core_train(self) -> None:
        """
        Handles core-related logic during the training phase.
        """
        self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
        self.sim_env.route_obj.engine_props["route_method"] = "k_shortest_path"
        self.sim_env.route_obj.get_route()
        self.sim_env.sim_env_helper.determine_core_penalty()

        self.sim_env.rl_props.forced_index = None

        self.sim_env.core_agent.get_core()

    def handle_spectrum_train(self) -> None:
        """
        Handles spectrum-related logic during the training phase.
        """
        self.sim_env.route_obj.sdn_props = self.sim_env.rl_props.mock_sdn_dict
        self.sim_env.route_obj.engine_props["route_method"] = "shortest_path"
        self.sim_env.route_obj.get_route()
        self.sim_env.rl_props.paths_list = (
            self.sim_env.route_obj.route_props.paths_matrix
        )
        self.sim_env.rl_props.chosen_path = (
            self.sim_env.route_obj.route_props.paths_matrix
        )
        self.sim_env.rl_props.path_index = 0
        self.sim_env.rl_props.core_index = None

    def get_obs(self, bandwidth: str, holding_time: float) -> dict[str, Any]:
        """
        Generates the current observation for the agent based on the environment state.

        :param bandwidth: The bandwidth requirement for the current request
        :type bandwidth: str
        :param holding_time: The holding time for the current request
        :type holding_time: float
        :return: A dictionary containing observation components
        :rtype: dict[str, Any]
        """
        if (
            self.sim_env.rl_props.arrival_count
            == self.sim_env.engine_obj.engine_props["num_requests"]
        ):
            curr_req = self.sim_env.rl_props.arrival_list[
                self.sim_env.rl_props.arrival_count - 1
            ]
        else:
            curr_req = self.sim_env.rl_props.arrival_list[
                self.sim_env.rl_props.arrival_count
            ]

        self.sim_env.rl_help_obj.handle_releases()
        self.sim_env.rl_props.source = int(curr_req["source"])
        self.sim_env.rl_props.destination = int(curr_req["destination"])
        self.sim_env.rl_props.mock_sdn_dict = self.sim_env.rl_help_obj.update_mock_sdn(
            current_request=curr_req
        )

        resp_dict = self.sim_env.sim_env_helper.get_drl_obs(
            bandwidth=bandwidth, holding_time=holding_time
        )
        return dict(resp_dict)


class SimEnvObs:
    """
    Encapsulates high-level helper methods tailored for managing and enhancing
    the behavior of the `SimEnv` class during
    reinforcement learning simulations.
    """

    def __init__(self, sim_env: Any) -> None:
        """
        Initializes the helper methods class with shared context.

        :param sim_env: The main simulation environment object
        :type sim_env: Any
        """
        self.sim_env = sim_env
        self.routing_obj: RoutingType | None = None

        self.lowest_holding = None
        self.highest_holding = None
        self.edge_index: torch.Tensor | None = None
        self.edge_attr: torch.Tensor | None = None
        self.node_feats: torch.Tensor | None = None
        self.str2idx: dict[str, int] | None = None
        self.id2idx: dict[Any, int] | None = None

    def update_helper_obj(self, action: int, bandwidth: str) -> None:
        """
        Updates the helper object with new actions and configurations.

        :param action: The action taken by the agent
        :type action: int
        :param bandwidth: The bandwidth requirement for the current request
        :type bandwidth: str
        """
        if self.sim_env.engine_obj.engine_props["is_drl_agent"]:
            self.sim_env.rl_help_obj.path_index = action
        else:
            self.sim_env.rl_help_obj.path_index = self.sim_env.rl_props.path_index

        self.sim_env.rl_help_obj.core_num = self.sim_env.rl_props.core_index

        if self.sim_env.sim_dict["spectrum_algorithm"] in ("dqn", "ppo", "a2c"):
            self.sim_env.rl_help_obj.rl_props.forced_index = action
        else:
            self.sim_env.rl_help_obj.rl_props.forced_index = None

        self.sim_env.rl_help_obj.rl_props = self.sim_env.rl_props
        self.sim_env.rl_help_obj.engine_props = self.sim_env.engine_obj
        self.sim_env.rl_help_obj.handle_releases()
        self.sim_env.rl_help_obj.update_route_props(
            chosen_path=self.sim_env.rl_props.chosen_path_list, bandwidth=bandwidth
        )

    def determine_core_penalty(self) -> None:
        """
        Determines penalty for the core algorithm based on path availability.
        """
        # Default to first fit if all paths fail
        self.sim_env.rl_props.chosen_path = [
            self.sim_env.route_obj.route_props.paths_matrix[0]
        ]
        self.sim_env.rl_props.chosen_path_index = 0
        for path_index, path_list in enumerate(
            self.sim_env.route_obj.route_props.paths_matrix
        ):
            mod_format_list = (
                self.sim_env.route_obj.route_props.modulation_formats_matrix[path_index]
            )

            was_allocated = self.sim_env.rl_help_obj.mock_handle_arrival(
                engine_props=self.sim_env.engine_obj.engine_props,
                sdn_props=self.sim_env.rl_props.mock_sdn_dict,
                mod_format_list=mod_format_list,
                path_list=path_list,
            )

            if was_allocated:
                self.sim_env.rl_props.chosen_path_list = [path_list]
                self.sim_env.rl_props.chosen_path_index = path_index
                break

    def handle_test_train_obs(self, curr_req: dict[str, Any]) -> None:  # pylint: disable=unused-argument
        """
        Handles path and core selection during training/testing phases based on
        the current request.

        :param curr_req: The current request being processed
        :type curr_req: dict[str, Any]
        :return: Path modulation format, if available
        :rtype: None
        """
        if self.sim_env.sim_dict["is_training"]:
            if self.sim_env.sim_dict["path_algorithm"] in VALID_PATH_ALGORITHMS:
                self.sim_env.step_helper.handle_path_train_test()
            else:
                raise NotImplementedError
        else:
            self.sim_env.step_helpers.handle_path_train_test()

    def _scale_req_holding(self, holding_time: float) -> float:
        """
        Scales the request holding time to a normalized value between 0 and 1.

        :param holding_time: The holding time to scale
        :type holding_time: float
        :return: The scaled holding time value
        :rtype: float
        """
        req_dict = self.sim_env.engine_obj.reqs_dict
        if self.lowest_holding is None or self.highest_holding is None:
            differences = [
                value["depart"] - arrival for arrival, value in req_dict.items()
            ]

            self.lowest_holding = min(differences)
            self.highest_holding = max(differences)

        if self.lowest_holding == self.highest_holding:
            raise ValueError("x_max and x_min cannot be the same value.")

        lowest_val = self.lowest_holding if self.lowest_holding is not None else 0.0
        highest_val = self.highest_holding if self.highest_holding is not None else 1.0

        scaled_holding = (holding_time - lowest_val) / (highest_val - lowest_val)
        return float(scaled_holding)

    def _get_paths_slots(
        self, bandwidth: str
    ) -> tuple[list[int], list[float], list[float], list[float]]:
        """
        Gets path-related information including slots needed and congestion metrics.

        :param bandwidth: The bandwidth requirement for the current request
        :type bandwidth: str
        :return: Tuple containing slots needed list, path lengths list, path congestion list, and available slots list
        :rtype: tuple[list[int], list[float], list[float], list[float]]
        """
        # NOTE: Consider moving routing object initialization to constructor
        # for better performance
        routing_obj = Routing(
            engine_props=self.sim_env.engine_obj.engine_props,
            sdn_props=self.sim_env.engine_obj.sdn_obj.sdn_props,
        )
        self.routing_obj = routing_obj

        routing_obj.sdn_props.bandwidth = bandwidth
        routing_obj.sdn_props.source = str(self.sim_env.rl_props.source)
        routing_obj.sdn_props.destination = str(self.sim_env.rl_props.destination)
        routing_obj.get_route()
        route_props = routing_obj.route_props

        slots_needed_list = []
        mod_bw_dict = self.sim_env.engine_obj.engine_props["mod_per_bw"]
        for mod_format in route_props.modulation_formats_matrix:
            if not mod_format[0]:
                slots_needed = -1
            else:
                slots_needed = mod_bw_dict[bandwidth][mod_format[0]]["slots_needed"]
            slots_needed_list.append(slots_needed)

        paths_cong = []
        available_slots = []
        for curr_path in route_props.paths_matrix:
            curr_cong, curr_slots = find_path_congestion(
                path_list=curr_path,
                network_spectrum=self.sim_env.engine_obj.network_spectrum_dict,
            )
            paths_cong.append(curr_cong)
            available_slots.append(curr_slots)

        norm_list = route_props.weights_list
        return slots_needed_list, norm_list, paths_cong, available_slots

    def get_path_masks(self, resp_dict: dict[str, Any]) -> None:
        """
        Encodes which paths are available via masks.

        :param resp_dict: The response dictionary to update with path mask information
        :type resp_dict: dict[str, Any]
        """
        if self.node_feats is not None:
            resp_dict["x"] = self.node_feats.numpy()
        if self.edge_index is not None:
            resp_dict["edge_index"] = self.edge_index.numpy()
        if self.edge_attr is not None:
            resp_dict["edge_attr"] = self.edge_attr.numpy()

        if self.edge_index is not None:
            edge_pairs = list(
                zip(
                    self.edge_index[0].tolist(),
                    self.edge_index[1].tolist(),
                    strict=False,
                )
            )
        else:
            edge_pairs = []
        edge_map = {pair: idx for idx, pair in enumerate(edge_pairs)}
        edge_shape = self.edge_index.shape[1] if self.edge_index is not None else 0

        paths_matrix: list[list[str]] = (
            self.routing_obj.route_props.paths_matrix
            if self.routing_obj is not None
            else []
        )
        k_shape = self.sim_env.rl_props.k_paths
        masks = np.zeros((k_shape, edge_shape), dtype=np.float32)

        if self.str2idx is not None:
            for i, path in enumerate(paths_matrix[:k_shape]):
                idx_path = [self.str2idx[label] for label in path]
                e_idxs = [
                    edge_map[(u, v)]
                    for u, v in zip(idx_path, idx_path[1:], strict=False)
                ]
                masks[i, e_idxs] = 1.0

        resp_dict["path_masks"] = masks

    def get_drl_obs(self, bandwidth: str, holding_time: float) -> dict[str, Any]:
        """
        Creates observation data for Deep Reinforcement Learning (DRL) in a graph-based
        environment.

        :param bandwidth: The bandwidth requirement for the current request
        :type bandwidth: str
        :param holding_time: The holding time for the current request
        :type holding_time: float
        :return: Dictionary containing observation data for DRL
        :rtype: dict[str, Any]
        """
        topo_graph = self.sim_env.engine_obj.engine_props["topology"]
        include_graph = False
        resp_dict = {}
        obs_space_key = self.sim_env.engine_obj.engine_props["obs_space"]
        if "graph" in obs_space_key:
            if None in (self.node_feats, self.edge_attr, self.edge_index):
                self.edge_index, self.edge_attr, self.node_feats, self.id2idx = (
                    convert_networkx_topo(topo_graph, as_directed=True)
                )
                if self.id2idx is not None:
                    self.str2idx = {str(n): idx for n, idx in self.id2idx.items()}
            include_graph = True
            obs_space_key = obs_space_key.replace("_graph", "")
        if "source" in OBS_DICT[obs_space_key]:
            source_obs = np.zeros(self.sim_env.rl_props.num_nodes)
            source_obs[self.sim_env.rl_props.source] = 1.0
            resp_dict["source"] = source_obs.tolist()
        if "destination" in OBS_DICT[obs_space_key]:
            dest_obs = np.zeros(self.sim_env.rl_props.num_nodes)
            dest_obs[self.sim_env.rl_props.destination] = 1.0
            resp_dict["destination"] = dest_obs.tolist()

        if not hasattr(self.sim_env, "bw_obs_list"):
            des_dict = self.sim_env.sim_dict["request_distribution"]
            self.sim_env.bw_obs_list = sorted(
                [k for k, v in des_dict.items() if v != 0], key=int
            )
        if "request_bandwidth" in OBS_DICT[obs_space_key]:
            bw_index = self.sim_env.bw_obs_list.index(bandwidth)
            req_band = np.zeros(len(self.sim_env.bw_obs_list))
            req_band[bw_index] = 1.0
            resp_dict["request_bandwidth"] = req_band.tolist()

        if "holding_time" in OBS_DICT[obs_space_key]:
            req_holding_scaled = self._scale_req_holding(holding_time=holding_time)
            resp_dict["holding_time"] = float(req_holding_scaled)

        # NOTE: Consider adding bandwidth as instance variable for better performance
        # NOTE: These features may not always be in the observation space
        slots_needed, path_lengths, paths_cong, available_slots = self._get_paths_slots(
            bandwidth=bandwidth
        )

        if "slots_needed" in OBS_DICT[obs_space_key]:
            resp_dict["slots_needed"] = list(slots_needed)
        if "path_lengths" in OBS_DICT[obs_space_key]:
            resp_dict["path_lengths"] = list(path_lengths)
        if "paths_cong" in OBS_DICT[obs_space_key]:
            resp_dict["paths_cong"] = list(paths_cong)
        if "available_slots" in OBS_DICT[obs_space_key]:
            resp_dict["available_slots"] = list(available_slots)
        if "is_feasible" in OBS_DICT[obs_space_key]:
            raise RLUtilsError(
                "Feasibility observation feature is not yet implemented. "
                "This feature requires additional development to determine feasibility."
            )

        if include_graph:
            self.get_path_masks(resp_dict=resp_dict)

        return resp_dict
