"""
Workflow runner for reinforcement learning training and testing.

This module orchestrates the training and evaluation of RL agents, including
support for hyperparameter optimization with Optuna and memory usage monitoring.
"""

import os
from typing import Any

import numpy as np
import optuna
import psutil
from optuna.pruners import HyperbandPruner

from fusion.modules.rl.args.general_args import (
    VALID_CORE_ALGORITHMS,
    VALID_DRL_ALGORITHMS,
    VALID_PATH_ALGORITHMS,
)
from fusion.modules.rl.errors import TrainingError
from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
from fusion.modules.rl.model_manager import get_model, save_model
from fusion.modules.rl.utils.general_utils import save_arr
from fusion.modules.rl.utils.hyperparams import get_optuna_hyperparams
from fusion.modules.rl.utils.setup import print_info, setup_rl_sim
from fusion.sim.utils.data import update_dict_from_list
from fusion.sim.utils.io import modify_multiple_json_values
from fusion.sim.utils.simulation import (
    get_erlang_values,
    run_simulation_for_erlangs,
    save_study_results,
)
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


# TODO moved to TODO.md: Support for picking up where you left off (testing)


def _run_drl_training(
    env: Any, sim_dict: dict[str, Any], yaml_dict: dict[str, Any] | None = None
) -> None:
    """
    Train a deep reinforcement learning model using Stable-Baselines3.

    Handles the complete training pipeline including model creation,
    training execution, and model saving.

    :param env: The reinforcement learning environment
    :type env: Any
    :param sim_dict: Simulation configuration dictionary
    :type sim_dict: Dict[str, Any]
    :param yaml_dict: Optional hyperparameters dictionary
    :type yaml_dict: Optional[Dict[str, Any]]
    """
    model, yaml_dict = get_model(
        sim_dict=sim_dict, device=sim_dict["device"], env=env, yaml_dict=yaml_dict
    )
    model.learn(
        total_timesteps=yaml_dict["n_timesteps"],
        log_interval=sim_dict["print_step"],
        callback=sim_dict["callback"],
    )

    save_model(sim_dict=sim_dict, env=env, model=model)


def _setup_callbacks(callback_list: Any, sim_dict: dict[str, Any]) -> None:
    """
    Initialize callback attributes that depend on simulation settings.

    Sets up callback objects with simulation-specific parameters like
    maximum iterations and simulation dictionary.

    :param callback_list: List of callback objects
    :type callback_list: Any
    :param sim_dict: Simulation configuration dictionary
    :type sim_dict: Dict[str, Any]
    """
    if callback_list:
        for callback in callback_list.callbacks:
            callback.max_iters = sim_dict["max_iters"]
            callback.sim_dict = sim_dict


def _train_drl_trial(
    env: Any,
    sim_dict: dict[str, Any],
    callback_list: Any,
    completed_trials: int,
    rewards_matrix: np.ndarray,
) -> tuple[Any, int]:
    """
    Execute one complete DRL training trial.

    Runs a full training trial including all episodes, updates reward tracking,
    and prepares for the next trial.

    :param env: The reinforcement learning environment
    :type env: Any
    :param sim_dict: Simulation configuration dictionary
    :type sim_dict: Dict[str, Any]
    :param callback_list: List of callback objects
    :type callback_list: Any
    :param completed_trials: Number of completed trials
    :type completed_trials: int
    :param rewards_matrix: Matrix to store episode rewards
    :type rewards_matrix: np.ndarray
    :return: New observation and updated completed trials count
    :rtype: Tuple[Any, int]
    """
    _run_drl_training(env=env, sim_dict=sim_dict)
    rewards_matrix[completed_trials] = callback_list.callbacks[0].episode_rewards
    callback_list.callbacks[0].episode_rewards = np.array([])

    completed_trials += 1
    env.trial = completed_trials

    for callback in callback_list.callbacks:
        callback.trial += 1

    callback_list.callbacks[1].current_entropy = sim_dict["epsilon_start"]
    callback_list.callbacks[1].current_learning_rate = sim_dict["alpha_start"]
    callback_list.callbacks[1].iter = 0
    env.iteration = 0

    logger.info(
        "%d trials completed out of %d.", completed_trials, sim_dict["n_trials"]
    )
    observation, _ = env.reset(seed=completed_trials)
    return observation, completed_trials


def _update_episode_stats(
    observation: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    episodic_reward: float,
    episodic_reward_array: np.ndarray,
    completed_episodes: int,
    completed_trials: int,
    env: Any,
    sim_dict: dict[str, Any],
    rewards_matrix: np.ndarray,
    trial: optuna.Trial | None,
) -> tuple[Any, float, np.ndarray, int, int]:
    """
    Consolidates the bookkeeping that happens whenever an episode ends.

    Updates episode statistics, handles episode completion logic, and manages
    trial progression. Returns the updated state so the caller's loop stays
    perfectly in sync.

    :param observation: Current environment observation
    :type observation: Any
    :param reward: Reward received from the last step
    :type reward: float
    :param terminated: Whether the episode terminated naturally
    :type terminated: bool
    :param truncated: Whether the episode was truncated
    :type truncated: bool
    :param episodic_reward: Cumulative reward for the current episode
    :type episodic_reward: float
    :param episodic_reward_array: Array of rewards for completed episodes
    :type episodic_reward_array: np.ndarray
    :param completed_episodes: Number of completed episodes in current trial
    :type completed_episodes: int
    :param completed_trials: Number of completed trials
    :type completed_trials: int
    :param env: The reinforcement learning environment
    :type env: Any
    :param sim_dict: Simulation configuration dictionary
    :type sim_dict: Dict[str, Any]
    :param rewards_matrix: Matrix storing rewards for all trials and episodes
    :type rewards_matrix: np.ndarray
    :param trial: Optional Optuna trial for hyperparameter optimization
    :type trial: Optional[optuna.Trial]
    :return: Updated observation, episodic reward, reward array, episode count,
        and trial count
    :rtype: Tuple[Any, float, np.ndarray, int, int]
    """
    episodic_reward += reward
    if not (terminated or truncated):
        return (
            observation,
            episodic_reward,
            episodic_reward_array,
            completed_episodes,
            completed_trials,
        )

    episodic_reward_array = np.append(episodic_reward_array, episodic_reward)
    episodic_reward = 0
    completed_episodes += 1

    if trial is not None:
        current_mean_reward = (
            np.mean(episodic_reward_array) if episodic_reward_array.size else 0
        )
        trial.report(current_mean_reward, completed_episodes)
        if trial.should_prune():
            raise optuna.TrialPruned()

    logger.info(
        "%d episodes completed out of %d.", completed_episodes, sim_dict["max_iters"]
    )

    if completed_episodes == sim_dict["max_iters"]:
        env.iteration = 0
        env.trial += 1
        rewards_matrix[completed_trials] = episodic_reward_array
        episodic_reward_array = np.array([])
        completed_trials += 1
        completed_episodes = 0
        logger.info(
            "%d trials completed out of %d.", completed_trials, sim_dict["n_trials"]
        )

    # Only reset if we haven't completed all trials yet
    if completed_trials < sim_dict["n_trials"]:
        observation, _ = env.reset(seed=completed_trials)

    return (
        observation,
        episodic_reward,
        episodic_reward_array,
        completed_episodes,
        completed_trials,
    )


def _initialize_training_state(
    sim_dict: dict[str, Any],
) -> tuple[int, int, float, np.ndarray, np.ndarray, list[float], Any]:
    """
    Initialize the training state variables and data structures.

    Sets up all necessary variables for tracking training progress including
    episode/trial counters, reward tracking matrices, and memory monitoring.

    :param sim_dict: Simulation configuration dictionary containing training parameters
    :type sim_dict: Dict[str, Any]
    :return: Tuple containing completed episodes, completed trials, episodic reward,
             rewards matrix, episodic reward array, memory usage list,
             and process object
    :rtype: Tuple[int, int, float, np.ndarray, np.ndarray, List[float], Any]
    """
    completed_episodes = 0
    completed_trials = 0
    episodic_reward = 0
    rewards_matrix = np.zeros((sim_dict["n_trials"], sim_dict["max_iters"]))
    episodic_reward_array = np.array([])
    memory_usage_list: list[float] = []
    process = psutil.Process()

    return (
        completed_episodes,
        completed_trials,
        episodic_reward,
        rewards_matrix,
        episodic_reward_array,
        memory_usage_list,
        process,
    )


def _process_episode_step(
    env: Any,
    is_training: bool,
    model: Any | None = None,
    observation: Any | None = None,
) -> tuple[Any, float, bool, bool]:
    """
    Process a single episode step based on training/testing mode.

    Executes one step in the environment, either using random actions for training
    or model predictions for testing, and returns the resulting state information.

    :param env: The reinforcement learning environment
    :type env: Any
    :param is_training: Whether in training mode (uses random actions) or testing mode
    :type is_training: bool
    :param model: The trained model used for action prediction in testing mode
    :type model: Optional[Any]
    :param observation: Current observation from the environment
    :type observation: Optional[Any]
    :return: Tuple containing new observation, reward, termination flag,
        and truncation flag
    :rtype: Tuple[Any, float, bool, bool]
    """
    if is_training:
        observation, reward, terminated, truncated, _ = env.step(0)
    else:
        if model is None:
            raise ValueError("Model cannot be None when not in training mode")
        action, _state = model.predict(observation)
        observation, reward, terminated, truncated, _ = env.step(action)

    return observation, reward, terminated, truncated


def _handle_training_completion(
    is_training: bool,
    rewards_matrix: np.ndarray,
    memory_usage_list: list[float],
    sim_dict: dict[str, Any],
) -> float:
    """
    Handle the completion of training, saving results and calculating metrics.

    Processes training results by calculating mean rewards, saving performance data,
    and returning the total reward sum for optimization purposes.

    :param is_training: Whether in training mode (raises error if False)
    :type is_training: bool
    :param rewards_matrix: Matrix of rewards per trial and episode
    :type rewards_matrix: np.ndarray
    :param memory_usage_list: List of memory usage measurements during training
    :type memory_usage_list: List[float]
    :param sim_dict: Simulation configuration dictionary with file paths
    :type sim_dict: Dict[str, Any]
    :return: Sum of mean rewards per iteration across all trials
    :rtype: float
    """
    if is_training:
        mean_per_iter = np.mean(rewards_matrix, axis=0)
        sum_reward = float(np.sum(mean_per_iter))
        save_arr(mean_per_iter, sim_dict, "average_rewards.npy")
        save_arr(np.array(memory_usage_list), sim_dict, "memory_usage.npy")
        return sum_reward

    raise TrainingError(
        "Testing mode is not yet implemented. Please use training mode."
    )


def run_iters(
    env: Any,
    sim_dict: dict[str, Any],
    is_training: bool,
    drl_agent: bool,
    model: Any | None = None,
    callback_list: Any | None = None,
    trial: optuna.Trial | None = None,
) -> float:
    """
    Runs the specified number of episodes/trials in the reinforcement learning
    environment.

    Orchestrates the main training or testing loop, handling episode execution,
    reward tracking, memory monitoring, and trial management for both DRL and
    traditional RL algorithms.

    :param env: The reinforcement learning environment
    :type env: Any
    :param sim_dict: Simulation configuration dictionary containing training parameters
    :type sim_dict: Dict[str, Any]
    :param is_training: Whether in training mode or testing mode
    :type is_training: bool
    :param drl_agent: Whether using deep reinforcement learning agent (SB3)
    :type drl_agent: bool
    :param model: Pre-trained model for testing mode or evaluation
    :type model: Optional[Any]
    :param callback_list: List of callbacks for monitoring training progress
    :type callback_list: Optional[Any]
    :param trial: Optuna trial object for hyperparameter optimization
    :type trial: Optional[optuna.Trial]
    :return: Sum of rewards across all iterations for optimization
    :rtype: float
    """
    # Initialize training state
    (
        completed_episodes,
        completed_trials,
        episodic_reward,
        rewards_matrix,
        episodic_reward_array,
        memory_usage_list,
        process,
    ) = _initialize_training_state(sim_dict)

    _setup_callbacks(callback_list, sim_dict)
    observation, _ = env.reset(seed=completed_trials)

    while completed_trials < sim_dict["n_trials"]:
        memory_usage_list.append(process.memory_info().rss / (1024 * 1024))

        if is_training and drl_agent:
            observation, completed_trials = _train_drl_trial(
                env, sim_dict, callback_list, completed_trials, rewards_matrix
            )
            continue

        # Process single step
        observation, reward, terminated, truncated = _process_episode_step(
            env, is_training, model, observation
        )

        # Update episode statistics
        (
            observation,
            episodic_reward,
            episodic_reward_array,
            completed_episodes,
            completed_trials,
        ) = _update_episode_stats(
            observation,
            reward,
            terminated,
            truncated,
            episodic_reward,
            episodic_reward_array,
            completed_episodes,
            completed_trials,
            env,
            sim_dict,
            rewards_matrix,
            trial,
        )

    return _handle_training_completion(
        is_training, rewards_matrix, memory_usage_list, sim_dict
    )


def run_testing() -> None:
    """
    Runs pre-trained RL model evaluation in the environment for the number of
    episodes specified in sim_dict.

    This function is currently not implemented and raises a TrainingError when called.
    Future implementation will handle loading and evaluating pre-trained models.

    :raises TrainingError: Always raised as testing functionality is not yet implemented
    :rtype: None
    """
    raise TrainingError(
        "Testing functionality is not yet implemented. Please use training mode."
    )


def run(
    env: Any,
    sim_dict: dict[str, Any],
    callback_list: Any | None = None,
    trial: optuna.Trial | None = None,
) -> float:
    """
    Manages the execution of simulations for training or testing RL models.

    Delegates to either training or testing based on flags within the simulation
    configuration. Validates algorithm configuration and orchestrates the appropriate
    training workflow.

    :param env: The reinforcement learning environment
    :type env: Any
    :param sim_dict: Simulation configuration dictionary containing paths,
        algorithms, and statistical parameters
    :type sim_dict: Dict[str, Any]
    :param callback_list: Custom callback list to monitor episodic rewards from SB3
    :type callback_list: Optional[Any]
    :param trial: Optuna trial object for hyperparameter optimization
    :type trial: Optional[optuna.Trial]
    :return: Sum of returns from the training process for optimization
    :rtype: float
    :raises TrainingError: If testing mode is used or invalid algorithm configuration
    """
    print_info(sim_dict=sim_dict)

    if sim_dict["is_training"]:
        # Print info function should already error check valid input,
        # no need to raise an error here
        if (
            sim_dict["path_algorithm"] in VALID_PATH_ALGORITHMS
            or sim_dict["core_algorithm"] in VALID_CORE_ALGORITHMS
        ):
            sum_returns = run_iters(
                env=env,
                sim_dict=sim_dict,
                is_training=True,
                drl_agent=sim_dict["path_algorithm"] in VALID_DRL_ALGORITHMS,
                callback_list=callback_list,
                trial=trial,
            )
        else:
            raise TrainingError(
                f"Invalid algorithm configuration. Path algorithm: "
                f"{sim_dict.get('path_algorithm')}, "
                f"Core algorithm: {sim_dict.get('core_algorithm')}"
            )
    else:
        raise TrainingError(
            "Testing mode is not yet implemented. Please set 'is_training' to True."
        )

    return sum_returns


def run_optuna_study(sim_dict: dict[str, Any], callback_list: Any) -> None:
    """
    Runs Optuna study for hyperparameter optimization.

    Creates and executes an Optuna optimization study to find the best hyperparameters
    for the reinforcement learning model. Uses Hyperband pruning to efficiently
    eliminate poor-performing trials early.

    :param sim_dict: Simulation configuration dictionary containing paths,
        algorithms, and statistical parameters
    :type sim_dict: Dict[str, Any]
    :param callback_list: Callback list for monitoring training progress
        during optimization
    :type callback_list: Any
    :rtype: None
    """

    # Define the Optuna objective function
    def objective(trial: optuna.Trial) -> float:
        env = SimEnv(
            render_mode=None, custom_callback=callback_list, sim_dict=setup_rl_sim()
        )

        for callback_obj in callback_list.callbacks:
            callback_obj.sim_dict = env.sim_dict
            callback_obj.max_iters = sim_dict["max_iters"]
        env.sim_dict["callback"] = callback_list.callbacks

        hyperparam_dict = get_optuna_hyperparams(sim_dict=sim_dict, trial=trial)
        update_list = [
            (param, value)
            for param, value in hyperparam_dict.items()
            if param in sim_dict
        ]

        modify_multiple_json_values(
            trial_num=trial.number, file_path=file_path, update_list=update_list
        )
        env.sim_dict = update_dict_from_list(
            input_dict=env.sim_dict, updates_list=update_list
        )
        erlang_list = get_erlang_values(sim_dict=sim_dict)

        mean_reward = run_simulation_for_erlangs(
            env=env,
            erlang_list=[float(x) for x in erlang_list],
            sim_dict=sim_dict,
            run_func=run,
            callback_list=callback_list,
            trial=trial,
        )
        mean_reward = mean_reward / sim_dict["max_iters"]
        if "callback" in env.sim_dict:
            del env.sim_dict["callback"]
            del env.callback

        trial.set_user_attr("sim_start_time", sim_dict["sim_start"])
        trial.set_user_attr("env", env)

        return float(mean_reward)

    # Run the optimization
    study_name = "hyperparam_study.pkl"
    file_path = os.path.join(
        "data", "input", sim_dict["network"], sim_dict["date"], sim_dict["sim_start"]
    )
    pruner = HyperbandPruner(
        min_resource=20, max_resource=sim_dict["max_iters"], reduction_factor=3
    )
    study = optuna.create_study(
        direction="maximize", study_name=study_name, pruner=pruner
    )
    n_trials = sim_dict["optuna_trials"]
    study.optimize(objective, n_trials=n_trials)

    # Save study results
    best_trial = study.best_trial
    # Validate required values before saving
    best_reward = best_trial.value
    if best_reward is None:
        raise ValueError("Best trial value cannot be None")

    best_sim_start = best_trial.user_attrs.get("sim_start_time")
    if not isinstance(best_sim_start, int):
        raise ValueError("Best trial sim_start_time must be an integer")

    save_study_results(  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        study=study,
        env=best_trial.user_attrs.get("env"),
        study_name=study_name,
        best_params=best_trial.params,
        best_reward=best_reward,
        best_sim_start=best_sim_start,
    )
