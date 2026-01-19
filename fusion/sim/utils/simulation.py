"""
Simulation setup and management utilities.

This module provides functions for setting up simulations, managing Erlang
values, saving study results, and handling logging.
"""

import os
import pickle
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def log_message(message: str, log_queue: Any) -> None:
    """
    Log a message to queue or logger.

    :param message: Message to log
    :type message: str
    :param log_queue: Queue for message logging (None to use logger)
    :type log_queue: Any
    """
    if log_queue:
        log_queue.put(message)
    else:
        logger.info(message)


def get_start_time(sim_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Get a unique start time for a simulation, ensuring it does not already exist.

    :param sim_dict: Holds the simulation parameters
    :type sim_dict: dict[str, Any]
    :return: Updated simulation dictionary with unique start time
    :rtype: dict[str, Any]
    """
    base_path = Path(f"data/input/{sim_dict['s1']['network']}/")

    while True:
        time.sleep(0.1)

        sim_start = datetime.now().strftime("%m%d_%H_%M_%S_%f")
        temp_list = sim_start.split("_")
        date = temp_list[0]
        time_string = f"{temp_list[1]}_{temp_list[2]}_{temp_list[3]}_{temp_list[4]}"

        full_path = base_path / date / time_string
        if not full_path.exists():
            break

        logger.warning("Duplicate start times, picking a new start!")

    sim_dict["s1"]["date"] = date
    sim_dict["s1"]["sim_start"] = time_string

    return sim_dict


def get_erlang_values(sim_dict: dict[str, Any]) -> list[int]:
    """
    Generate a list of arrival rates based on the configuration dictionary.

    :param sim_dict: Simulation param dictionary
    :type sim_dict: dict[str, Any]
    :return: List of arrival rates generated from the configuration
    :rtype: list[int]
    """
    start = int(sim_dict["erlang_start"])
    stop = int(sim_dict["erlang_stop"])
    step = int(sim_dict["erlang_step"])

    return list(range(start, stop, step))


def run_simulation_for_erlangs(
    env: Any,
    erlang_list: list[float],
    sim_dict: dict[str, Any],
    run_func: Callable,
    callback_list: Any,
    trial: Any,
) -> float:
    """
    Run the simulation for each arrival rate in the given list.

    :param env: Simulation environment instance
    :type env: Any
    :param erlang_list: List of traffic volumes (erlangs) to simulate
    :type erlang_list: list[float]
    :param sim_dict: Simulation properties dictionary
    :type sim_dict: dict[str, Any]
    :param run_func: Function to run a simulation
    :type run_func: Callable
    :param callback_list: Callback list for simulation
    :type callback_list: Any
    :param trial: Trial object
    :type trial: Any
    :return: Mean of total rewards from all simulations
    :rtype: float
    """
    total_rewards = []

    for erlang in erlang_list:
        env.engine_obj.engine_props["erlang"] = erlang
        arrival_rate = sim_dict["cores_per_link"] * erlang
        env.engine_obj.engine_props["arrival_rate"] = arrival_rate
        env.engine_obj.engine_props["arrival_rate"] /= sim_dict["holding_time"]

        sum_returns = run_func(
            env=env, sim_dict=env.sim_dict, callback_list=callback_list, trial=trial
        )
        total_rewards.append(sum_returns)

    return float(np.mean(total_rewards))


def save_study_results(
    study: Any,
    env: Any,
    study_name: str,
    best_params: dict[str, Any],
    best_reward: float,
    best_sim_start: int,
) -> None:
    """
    Save the results of the study.

    Includes the best hyperparameters and the best reward value.

    :param study: Optuna study object containing the results
    :type study: Any
    :param env: Simulation environment instance
    :type env: Any
    :param study_name: Name of the study file to save
    :type study_name: str
    :param best_params: Best hyperparameters found by Optuna
    :type best_params: dict[str, Any]
    :param best_reward: Best reward value from the study
    :type best_reward: float
    :param best_sim_start: Start time of the best simulation
    :type best_sim_start: int
    """
    date_time = os.path.join(
        env.engine_obj.engine_props["network"],
        env.engine_obj.engine_props["date"],
        env.engine_obj.engine_props["sim_start"],
    )
    save_dir = os.path.join(
        "logs", env.engine_obj.engine_props["path_algorithm"], date_time
    )
    os.makedirs(save_dir, exist_ok=True)

    save_filepath = os.path.join(save_dir, study_name)
    with open(save_filepath, "wb") as file_path:
        pickle.dump(study, file_path)

    save_filepath = os.path.join(save_dir, "best_hyperparams.txt")
    with open(save_filepath, "w", encoding="utf-8") as file_path:
        file_path.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            file_path.write(f"{key}: {value}\n")
        file_path.write(f"\nBest Trial Reward: {best_reward}\n")
        file_path.write(f"\nBest Simulation Start Time: {best_sim_start}\n")
