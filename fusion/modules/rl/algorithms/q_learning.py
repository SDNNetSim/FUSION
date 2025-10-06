import json
from itertools import islice
from pathlib import Path
from typing import Any, cast

import networkx as nx
import numpy as np

from fusion.modules.rl.algorithms.algorithm_props import QProps
from fusion.modules.rl.errors import AlgorithmNotFoundError
from fusion.sim.utils.data import calculate_matrix_statistics
from fusion.sim.utils.network import (
    classify_congestion,
    find_core_congestion,
    find_path_congestion,
)
from fusion.utils.os import create_directory


class QLearning:
    """Q-learning agent responsible for handling routing and core selection."""

    def __init__(self, rl_props: object, engine_props: dict[str, Any]) -> None:
        self.props = QProps()
        self.engine_props = engine_props
        self.rl_props = rl_props

        self.path_levels = engine_props["path_levels"]
        self.iteration = 0
        self.learn_rate: float | None = None
        self.completed_sim = False
        self._initialize_matrices()

        self.rewards_stats_dict: dict[str, list[float]] | None = None
        self.error_stats_dict: dict[str, list[float]] | None = None

    def _ensure_rl_props_initialized(self) -> None:
        """Ensure rl_props has required attributes for type safety."""
        assert hasattr(self.rl_props, 'num_nodes'), "rl_props must have num_nodes"
        assert hasattr(self.rl_props, 'k_paths'), "rl_props must have k_paths"
        assert hasattr(self.rl_props, 'source'), "rl_props must have source"
        assert hasattr(self.rl_props, 'destination'), "rl_props must have destination"
        assert hasattr(self.rl_props, 'chosen_path_index'), \
            "rl_props must have chosen_path_index"
        assert hasattr(self.rl_props, 'paths_list'), "rl_props must have paths_list"
        assert hasattr(self.rl_props, 'cores_list'), "rl_props must have cores_list"

    @property
    def _rl_props(self) -> Any:
        """Get rl_props with type casting after validation."""
        self._ensure_rl_props_initialized()
        return cast(Any, self.rl_props)

    def _initialize_matrices(self) -> None:
        """Initialize Q-tables for paths and cores."""
        self._ensure_rl_props_initialized()
        self.props.epsilon = self.engine_props["epsilon_start"]
        self.props.routes_matrix = self._create_routes_matrix()
        self.props.cores_matrix = self._create_cores_matrix()
        self._populate_q_tables()

    def _create_routes_matrix(self) -> np.ndarray:
        """Create an empty routes matrix."""
        shape = (
            self._rl_props.num_nodes,
            self._rl_props.num_nodes,
            self._rl_props.k_paths,
            self.path_levels,
        )
        dtype = [("path", "O"), ("q_value", "f8")]
        return np.empty(shape, dtype=dtype)

    def _create_cores_matrix(self) -> np.ndarray:
        """Create an empty cores matrix."""
        shape = (
            self._rl_props.num_nodes,
            self._rl_props.num_nodes,
            self._rl_props.k_paths,
            self.engine_props["cores_per_link"],
            self.path_levels,
        )
        dtype = [("path", "O"), ("core_action", "i8"), ("q_value", "f8")]
        return np.empty(shape, dtype=dtype)

    def _populate_q_tables(self) -> None:
        """Populate Q-tables with initial values."""
        assert self.props.routes_matrix is not None
        assert self.props.cores_matrix is not None

        ssp = nx.shortest_simple_paths
        topology = self.engine_props["topology"]

        for src in range(self._rl_props.num_nodes):
            for dst in range(self._rl_props.num_nodes):
                if src == dst:
                    continue

                for k, path in enumerate(
                    islice(
                        ssp(
                            topology, source=str(src), target=str(dst), weight="length"
                        ),
                        self._rl_props.k_paths,
                    )
                ):
                    for level in range(self.path_levels):
                        self.props.routes_matrix[src, dst, k, level] = (path, 0.0)
                        for core in range(self.engine_props["cores_per_link"]):
                            self.props.cores_matrix[src, dst, k, core, level] = (
                                path,
                                core,
                                0.0,
                            )

    def get_max_future_q(
        self,
        path_list: Any,  # Can be list or ndarray
        network_spectrum_dict: dict[str, Any],
        matrix: np.ndarray,
        flag: str,
        core_index: int | None = None,
    ) -> float:
        """Retrieve the maximum future Q-value based on congestion levels."""
        # Convert path_list to list if it's an ndarray
        if hasattr(path_list, 'tolist'):
            path_list = path_list.tolist()

        if flag == "core":
            assert core_index is not None, "core_index must be provided for core flag"
            congestion_result = find_core_congestion(
                core_index, network_spectrum_dict, path_list
            )
            # Handle both tuple and single value returns for core
            if isinstance(congestion_result, tuple):
                new_cong = float(congestion_result[0])
            else:
                new_cong = float(congestion_result)
        else:
            # find_path_congestion returns a tuple
            congestion_tuple = find_path_congestion(path_list, network_spectrum_dict)
            new_cong = float(congestion_tuple[0])
        new_cong_index = classify_congestion(
            new_cong, congestion_cutoff=self.engine_props["cong_cutoff"]
        )

        max_future_q = matrix[core_index if flag == "core" else new_cong_index][
            "q_value"
        ]
        return float(max_future_q)

    def get_max_curr_q(self, cong_list: list, matrix_flag: str) -> tuple[int, Any]:
        """Get the maximum current Q-value from the current state."""
        self._ensure_rl_props_initialized()
        assert self.props.routes_matrix is not None
        assert self.props.cores_matrix is not None

        matrix = (
            self.props.routes_matrix[self._rl_props.source, self._rl_props.destination]
            if matrix_flag == "routes_matrix"
            else self.props.cores_matrix[
                self._rl_props.source,
                self._rl_props.destination,
                self._rl_props.chosen_path_index,
            ]
        )

        q_values = [
            matrix[obj_index, level_index]["q_value"]
            for obj_index, _, level_index in cong_list
        ]
        max_index = np.argmax(q_values)
        max_obj = (
            self._rl_props.paths_list[max_index]
            if matrix_flag == "routes_matrix"
            else self._rl_props.cores_list[max_index]
        )

        return int(max_index), max_obj

    def update_q_matrix(
        self,
        reward: float,
        level_index: int,
        network_spectrum_dict: dict[str, Any],
        flag: str,
        trial: int,
        iteration: int,
        core_index: int | None = None,
    ) -> None:
        """Update Q-values for either path or core selection."""
        self._ensure_rl_props_initialized()
        assert self.props.routes_matrix is not None
        assert self.props.cores_matrix is not None
        assert self.learn_rate is not None

        matrix = self.props.cores_matrix if flag == "core" else self.props.routes_matrix
        matrix = matrix[self._rl_props.source, self._rl_props.destination]
        matrix = matrix[self._rl_props.chosen_path_index] if flag == "core" else matrix
        current_q = matrix[core_index if flag == "core" else level_index]["q_value"]

        max_future_q = self.get_max_future_q(
            matrix[core_index if flag == "core" else level_index]["path"],
            network_spectrum_dict,
            matrix,
            flag,
            core_index,
        )
        delta = reward + self.engine_props["gamma"] * max_future_q
        td_error = current_q - delta
        new_q = ((1 - self.learn_rate) * current_q) + (self.learn_rate * delta)

        self.iteration = iteration
        self.update_q_stats(
            reward,
            td_error,
            "cores_dict" if flag == "core" else "routes_dict",
            trial=trial,
        )
        matrix[core_index if flag == "core" else level_index]["q_value"] = new_q

    def update_q_stats(
        self, reward: float, td_error: float, stats_flag: str, trial: int
    ) -> None:
        """Updates statistics related to Q-learning performance."""
        episode = str(self.iteration)
        if episode not in self.props.rewards_dict[stats_flag]["rewards"]:
            self.props.rewards_dict[stats_flag]["rewards"][episode] = [reward]
            self.props.errors_dict[stats_flag]["errors"][episode] = [td_error]
        else:
            self.props.rewards_dict[stats_flag]["rewards"][episode].append(reward)
            self.props.errors_dict[stats_flag]["errors"][episode].append(td_error)

        if (
            self.iteration % self.engine_props["save_step"] == 0
            or self.iteration == self.engine_props["max_iters"] - 1
        ) and len(
            self.props.rewards_dict[stats_flag]["rewards"][episode]
        ) == self.engine_props["num_requests"]:
            self._calc_q_averages(stats_flag, trial=trial)

    def _calc_q_averages(self, stats_flag: str, trial: int) -> None:
        """
        Calculate averages for rewards and errors at the end of an episode.

        Once the number of requests is reached, mark sim as complete, compute
        final stats, and trigger save_model with the current iteration.

        :param stats_flag: Flag indicating which statistics to calculate
        :type stats_flag: str
        :param trial: Current trial number
        :type trial: int
        """
        self.rewards_stats_dict = calculate_matrix_statistics(
            self.props.rewards_dict[stats_flag]["rewards"]
        )
        self.error_stats_dict = calculate_matrix_statistics(
            self.props.errors_dict[stats_flag]["errors"]
        )
        self.save_model(trial)

    def _convert_q_tables_to_dict(self, which_table: str) -> dict[str, list]:
        """
        Convert 'routes_matrix' or 'cores_matrix' into dict for JSON.

        :param which_table: 'routes' or 'cores'
        :type which_table: str
        :return: Dictionary suitable for JSON serialization
        :rtype: dict[str, list]
        """
        self._ensure_rl_props_initialized()
        assert self.props.routes_matrix is not None

        q_dict = {}
        if which_table == "routes":
            for src in range(self._rl_props.num_nodes):
                for dst in range(self._rl_props.num_nodes):
                    if src == dst:
                        continue
                    q_vals_list = []
                    for k in range(self._rl_props.k_paths):
                        entry = self.props.routes_matrix[src, dst, k, 0]
                        current_q = entry[1]
                        q_vals_list.append(current_q)

                    q_dict[str((src, dst))] = q_vals_list
        elif which_table == "cores":
            raise AlgorithmNotFoundError(
                "Core table conversion is not yet implemented. "
                "Only routes table conversion is currently supported."
            )

        return q_dict

    def save_model(self, trial: int) -> None:
        """
        Save the Q-learning model.

        Saves Q-tables to NumPy and also to JSON with consistent naming.

        :param trial: Trial number
        :type trial: int
        """
        assert self.rewards_stats_dict is not None

        save_dir_path = (
            Path("logs") /
            "q_learning" /
            self.engine_props["network"] /
            self.engine_props["date"] /
            self.engine_props["sim_start"]
        )
        create_directory(str(save_dir_path))

        erlang = self.engine_props["erlang"]
        cores_per_link = self.engine_props["cores_per_link"]
        base_str = (
            "routes" if self.engine_props["path_algorithm"] == "q_learning" else "cores"
        )
        filename_npy = (
            f"rewards_e{erlang}_{base_str}_c{cores_per_link}_t{trial + 1}"
            f"_iter_{self.iteration}.npy"
        )
        save_path_npy = save_dir_path / filename_npy

        if "routes" in base_str:
            np.save(save_path_npy, self.rewards_stats_dict["average"])
            q_dict = self._convert_q_tables_to_dict("routes")
        else:
            raise AlgorithmNotFoundError(
                "Core Q-learning model saving is not yet implemented. "
                "Only routes Q-learning models are currently supported."
            )

        json_filename = (
            f"state_vals_e{erlang}_{base_str}_c{cores_per_link}_t{trial + 1}.json"
        )
        save_path_json = save_dir_path / json_filename
        with open(save_path_json, "w", encoding="utf-8") as file_obj:
            json.dump(q_dict, file_obj)
