"""Model persistence classes for RL algorithms."""

import os
import json
from typing import Dict, Any
import numpy as np

from fusion.utils.os import create_dir  # pylint: disable=unused-import
from fusion.modules.rl.errors import AlgorithmNotFoundError


class BanditModelPersistence:
    """Handles model persistence for bandit algorithms."""

    @staticmethod
    def load_model(train_fp: str) -> Dict[str, Any]:
        """
        Load a pre-trained bandit model.

        :param train_fp: File path where the model has been saved
        :type train_fp: str
        :return: The state-value functions V(s, a)
        :rtype: Dict[str, Any]
        """
        train_fp = os.path.join('logs', train_fp)
        with open(train_fp, 'r', encoding='utf-8') as file_obj:
            state_vals_dict = json.load(file_obj)
        return state_vals_dict

    @staticmethod
    def save_model(state_values_dict: Dict[str, Any], erlang: float, cores_per_link: int,
                   save_dir: str, is_path: bool, trial: int) -> None:
        """
        Save bandit model state values.

        :param state_values_dict: Dictionary of state values to save
        :type state_values_dict: Dict[str, Any]
        :param erlang: Erlang traffic value
        :type erlang: float
        :param cores_per_link: Number of cores per link
        :type cores_per_link: int
        :param save_dir: Directory to save the model
        :type save_dir: str
        :param is_path: Whether this is a path agent model
        :type is_path: bool
        :param trial: Trial number
        :type trial: int
        :raises AlgorithmNotFoundError: If core agent saving is attempted
        """
        if state_values_dict is None:
            return

        # Convert tuples to strings and arrays to lists for JSON format
        state_values_dict = {str(key): value.tolist() for key, value in state_values_dict.items()}

        if is_path:
            state_vals_fp = f"state_vals_e{erlang}_routes_c{cores_per_link}_t{trial + 1}.json"
        else:
            raise AlgorithmNotFoundError(
                "Core agent bandit model saving is not yet implemented. "
                "Only path agent bandit models are currently supported."
            )

        save_fp = os.path.join(os.getcwd(), save_dir, state_vals_fp)
        with open(save_fp, 'w', encoding='utf-8') as file_obj:
            json.dump(state_values_dict, file_obj)


class QLearningModelPersistence:  # pylint: disable=too-few-public-methods
    """Handles model persistence for Q-learning algorithms."""

    @staticmethod
    def save_model(q_dict: Dict[str, list], rewards_avg: np.ndarray, erlang: float,
                   cores_per_link: int, base_str: str, trial: int, iteration: int,
                   save_dir: str) -> None:
        """
        Save Q-learning model data.

        :param q_dict: Q-values dictionary to save
        :type q_dict: Dict[str, list]
        :param rewards_avg: Average rewards array
        :type rewards_avg: np.ndarray
        :param erlang: Erlang traffic value
        :type erlang: float
        :param cores_per_link: Number of cores per link
        :type cores_per_link: int
        :param base_str: Base string for file naming ('routes' or 'cores')
        :type base_str: str
        :param trial: Trial number
        :type trial: int
        :param iteration: Current iteration
        :type iteration: int
        :param save_dir: Directory to save the model
        :type save_dir: str
        :raises AlgorithmNotFoundError: If core model saving is attempted
        """
        if 'cores' in base_str:
            raise AlgorithmNotFoundError(
                "Core Q-learning model saving is not yet implemented. "
                "Only routes Q-learning models are currently supported."
            )

        # Save numpy array
        filename_npy = f"rewards_e{erlang}_{base_str}_c{cores_per_link}_t{trial + 1}_iter_{iteration}.npy"
        save_path_npy = os.path.join(save_dir, filename_npy)
        np.save(save_path_npy, rewards_avg)

        # Save JSON dictionary
        json_filename = f"state_vals_e{erlang}_{base_str}_c{cores_per_link}_t{trial + 1}.json"
        save_path_json = os.path.join(save_dir, json_filename)
        with open(save_path_json, 'w', encoding='utf-8') as file_obj:
            json.dump(q_dict, file_obj)
