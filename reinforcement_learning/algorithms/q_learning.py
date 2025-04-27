import os
import json
import networkx as nx
import numpy as np

from reinforcement_learning.algorithms.algorithm_props import QProps
from helper_scripts.sim_helpers import (
    find_path_cong, classify_cong, calc_matrix_stats, find_core_cong
)
from helper_scripts.os_helpers import create_dir


class QLearning:
    """
    Q-learning agent responsible for handling routing and core selection.
    """

    def __init__(self, rl_props, engine_props):
        self.props = QProps()
        self.engine_props = engine_props
        self.rl_props = rl_props

        self.path_levels = engine_props['path_levels']
        self.iteration = 0
        self.learn_rate = None
        self.completed_sim = False
        self._initialize_matrices()

        self.rewards_stats_dict = None
        self.error_stats_dict = None

    def _initialize_matrices(self):
        """Initializes Q-tables for paths and cores."""
        self.props.epsilon = self.engine_props['epsilon_start']
        self.props.routes_matrix = self._create_routes_matrix()
        self.props.cores_matrix = self._create_cores_matrix()
        self._populate_q_tables()

    def _create_routes_matrix(self):
        """Creates an empty routes matrix."""
        return np.empty(
            (self.rl_props.num_nodes, self.rl_props.num_nodes, self.rl_props.k_paths, self.path_levels),
            dtype=[('path', 'O'), ('q_value', 'f8')]
        )

    def _create_cores_matrix(self):
        """Creates an empty cores matrix."""
        return np.empty(
            (self.rl_props.num_nodes, self.rl_props.num_nodes, self.rl_props.k_paths,
             self.engine_props['cores_per_link'], self.path_levels),
            dtype=[('path', 'O'), ('core_action', 'i8'), ('q_value', 'f8')]
        )

    def _populate_q_tables(self):
        """Populates the Q-tables with initial values."""
        topology = self.engine_props['topology']

        for src in range(self.rl_props.num_nodes):
            for dst in range(self.rl_props.num_nodes):
                if src == dst:
                    continue

                shortest_paths = list(
                    nx.shortest_simple_paths(topology, source=str(src), target=str(dst), weight='length'))
                for k, path in enumerate(shortest_paths[:self.rl_props.k_paths]):
                    for level in range(self.path_levels):
                        self.props.routes_matrix[src, dst, k, level] = (path, 0.0)
                        for core in range(self.engine_props['cores_per_link']):
                            self.props.cores_matrix[src, dst, k, core, level] = (path, core, 0.0)

    def get_max_future_q(self, path_list, net_spec_dict, matrix, flag, core_index=None):
        """Retrieves the maximum future Q-value based on congestion levels."""
        new_cong, _ = find_core_cong(core_index, net_spec_dict, path_list) if flag == 'core' else find_path_cong(
            path_list, net_spec_dict)
        new_cong_index = classify_cong(new_cong, cong_cutoff=self.engine_props['cong_cutoff'])

        max_future_q = matrix[core_index if flag == 'core' else new_cong_index]['q_value']
        return max_future_q

    def get_max_curr_q(self, cong_list, matrix_flag):
        """Gets the maximum current Q-value from the current state."""
        matrix = (
            self.props.routes_matrix[self.rl_props.source, self.rl_props.destination]
            if matrix_flag == 'routes_matrix'
            else self.props.cores_matrix[
                self.rl_props.source, self.rl_props.destination, self.rl_props.chosen_path_index]
        )

        q_values = [matrix[obj_index, level_index]['q_value'] for obj_index, _, level_index in cong_list]
        max_index = np.argmax(q_values)
        max_obj = (
            self.rl_props.paths_list[max_index]
            if matrix_flag == 'routes_matrix'
            else self.rl_props.cores_list[max_index]
        )

        return max_index, max_obj

    def update_q_matrix(self, reward, level_index, net_spec_dict, flag, trial: int, iteration, core_index=None):
        """Updates Q-values for either path or core selection."""
        matrix = self.props.cores_matrix if flag == 'core' else self.props.routes_matrix
        matrix = matrix[self.rl_props.source, self.rl_props.destination]
        # matrix = matrix[self.rl_props.chosen_path_index] if flag == 'core' else matrix
        matrix = matrix[self.rl_props.chosen_path_index] if flag == 'path' else matrix
        current_q = matrix[core_index if flag == 'core' else level_index]['q_value']

        max_future_q = self.get_max_future_q(matrix[core_index if flag == 'core' else level_index]['path'],
                                             net_spec_dict, matrix, flag, core_index)
        delta = reward + self.engine_props['gamma'] * max_future_q
        td_error = current_q - delta
        new_q = ((1 - self.learn_rate) * current_q) + (self.learn_rate * delta)

        self.iteration = iteration
        self.update_q_stats(reward, td_error, 'cores_dict' if flag == 'core' else 'routes_dict', trial=trial)
        matrix[core_index if flag == 'core' else level_index]['q_value'] = new_q

    def update_q_stats(self, reward, td_error, stats_flag, trial: int):
        """Updates statistics related to Q-learning performance."""
        episode = str(self.iteration)
        if episode not in self.props.rewards_dict[stats_flag]['rewards']:
            self.props.rewards_dict[stats_flag]['rewards'][episode] = [reward]
            self.props.errors_dict[stats_flag]['errors'][episode] = [td_error]
        else:
            self.props.rewards_dict[stats_flag]['rewards'][episode].append(reward)
            self.props.errors_dict[stats_flag]['errors'][episode].append(td_error)

        if (self.iteration % self.engine_props['save_step'] == 0 or self.iteration == self.engine_props[
            'max_iters'] - 1) and \
                len(self.props.rewards_dict[stats_flag]['rewards'][episode]) == self.engine_props['num_requests']:
            self._calc_q_averages(stats_flag, trial=trial)

    def _calc_q_averages(self, stats_flag, trial: int):
        """
        Calculates averages for rewards and errors at the end of an episode.
        Once the number of requests is reached, mark sim as complete, compute final stats,
        and trigger save_model with the current iteration.
        """
        self.rewards_stats_dict = calc_matrix_stats(self.props.rewards_dict[stats_flag]['rewards'])
        self.error_stats_dict = calc_matrix_stats(self.props.errors_dict[stats_flag]['errors'])
        self.save_model(trial)

    def _convert_q_tables_to_dict(self, which_table: str):
        """
        Converts either 'routes_matrix' or 'cores_matrix' into a dict suitable for JSON.
        which_table: 'routes' or 'cores'
        """
        q_dict = {}
        if which_table == 'routes':
            for src in range(self.rl_props.num_nodes):
                for dst in range(self.rl_props.num_nodes):
                    if src == dst:
                        continue
                    q_vals_list = list()
                    for k in range(self.rl_props.k_paths):
                        entry = self.props.routes_matrix[src, dst, k, 0]
                        current_q = entry[1]
                        q_vals_list.append(current_q)

                    q_dict[str((src, dst))] = q_vals_list
        elif which_table == 'cores':
            raise NotImplementedError

        return q_dict

    def save_model(self, trial: int):
        """
        Saves the Q-learning model similarly to bandits:
          - Uses iteration-based checks (if 'save_step' > 0).
          - Saves Q-tables to NumPy and also to JSON.
          - Names files consistently with 'e{erlang}_routes_c{...}' or 'e{erlang}_cores_c{...}'.
        """
        save_dir = os.path.join(
            'logs',
            'q_learning',
            self.engine_props['network'],
            self.engine_props['date'],
            self.engine_props['sim_start']
        )
        create_dir(save_dir)

        erlang = self.engine_props['erlang']
        cores_per_link = self.engine_props['cores_per_link']
        base_str = 'routes' if self.engine_props['path_algorithm'] == 'q_learning' else 'cores'
        filename_npy = f"rewards_e{erlang}_{base_str}_c{cores_per_link}_t{trial + 1}_iter_{self.iteration}.npy"
        save_path_npy = os.path.join(save_dir, filename_npy)

        if 'routes' in base_str:
            np.save(save_path_npy, self.rewards_stats_dict['average'])
            q_dict = self._convert_q_tables_to_dict('routes')
        else:
            raise NotImplementedError

        json_filename = f"state_vals_e{erlang}_{base_str}_c{cores_per_link}_t{trial + 1}.json"
        save_path_json = os.path.join(save_dir, json_filename)
        with open(save_path_json, 'w', encoding='utf-8') as file_obj:
            json.dump(q_dict, file_obj)
