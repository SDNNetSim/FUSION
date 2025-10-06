"""Properties classes for RL algorithms."""

from typing import Any

import numpy as np


class RLProps:  # pylint: disable=too-few-public-methods
    """
    Main reinforcement learning properties used in run_rl_sim.py script.
    """

    def __init__(self) -> None:
        # Number of paths the agent has to choose from
        self.k_paths: int | None = None
        # Number of cores on every link
        self.cores_per_link: int | None = None
        # Spectral slots on every core
        self.spectral_slots: int | list[int] | None = None
        # Total nodes in the network topology
        self.num_nodes: int | None = None

        self.arrival_list: list[float] = []  # Inter-arrival times for every request
        self.depart_list: list[float] = []  # Departure times for every request

        self.mock_sdn_dict: dict[Any, Any] = {}  # A virtual SDN dictionary
        self.source = None  # Source node for a single request
        self.destination = None  # Destination node for a single request

        # Potential paths from source to destination
        self.paths_list: list[list[int]] = []
        self.path_index = None  # Index of the last path chosen in a RL simulation
        # The actual chosen path (nodes) for a request
        self.chosen_path_list: list[int] = []
        self.core_index = None  # Index of the last core chosen for a request

        # Additional attributes used by SimEnv
        self.super_channel_space: int = 0  # Super channel space configuration
        self.arrival_count: int = 0  # Count of processed arrivals

    def __repr__(self) -> str:
        return f"RLProps({self.__dict__})"


class QProps:
    """
    Properties object used in the Q-learning algorithm.
    """

    def __init__(self) -> None:
        # Current epsilon used at a certain point in time
        self.epsilon: float | None = None
        # Starting value of epsilon
        self.epsilon_start: float | None = None
        # Ending value of epsilon to be linearly decayed
        self.epsilon_end: float | None = None
        # A list of every value at each time step
        self.epsilon_list: list[float] = []

        # Flag to determine whether to load a trained agent
        self.is_training: bool | None = None

        # Rewards for the core and path q-learning agents
        self.rewards_dict: dict[str, dict[str, Any]] = {
            'routes_dict': {'average': [], 'min': [], 'max': [], 'rewards': {}},
            'cores_dict': {'average': [], 'min': [], 'max': [], 'rewards': {}}
        }
        # Temporal difference (TD) errors for the core and path agents
        self.errors_dict: dict[str, dict[str, Any]] = {
            'routes_dict': {'average': [], 'min': [], 'max': [], 'errors': {}},
            'cores_dict': {'average': [], 'min': [], 'max': [], 'errors': {}}
        }
        # Total sum of rewards for each episode (episode as key, sum as value)
        self.sum_rewards_dict: dict[int, float] = {}
        # Total sum of TD errors each episode
        self.sum_errors_dict: dict[int, float] = {}

        # Main routing q-table for path agent
        self.routes_matrix: np.ndarray | None = None
        self.cores_matrix: np.ndarray | None = None  # Main core q-table for core agent
        self.num_nodes: int | None = None  # Total number of nodes in the topology

        # All important parameters to be saved in a QL simulation run
        self.save_params_dict = {
            'q_params_list': [
                'rewards_dict', 'errors_dict', 'epsilon_list',
                'sum_rewards_dict', 'sum_errors_dict'
            ],
            'engine_params_list': [
                'epsilon_start', 'epsilon_end', 'max_iters', 'alpha_start',
                'alpha_end', 'gamma', 'epsilon_update', 'alpha_update'
            ]
        }

    def get_data(self, key: str) -> Any:
        """
        Retrieve a property of the object.

        :param key: The property name
        :type key: str
        :return: The value of the property
        :rtype: Any
        :raises AttributeError: If the property doesn't exist
        """
        if hasattr(self, key):
            return getattr(self, key)

        raise AttributeError(f"'RLProps' object has no attribute '{key}'")

    def __repr__(self) -> str:
        return f"QProps({self.__dict__})"


class BanditProps:  # pylint: disable=too-few-public-methods
    """
    Properties object used in the bandit algorithms.
    """

    def __init__(self) -> None:
        # Total sum of rewards for each episode
        self.rewards_matrix: list[list[float]] = []
        # Total number of counts for each action taken for every episode
        self.counts_list: list[Any] = []
        self.state_values_list: list[Any] = []  # Every possible V(s)

    def __repr__(self) -> str:
        return f"BanditProps({self.__dict__})"


class PPOProps:  # pylint: disable=too-few-public-methods
    """
    Properties object for PPO algorithm.

    Currently not implemented. Will be added when PPO-specific
    properties are needed beyond the base DRL functionality.
    """
