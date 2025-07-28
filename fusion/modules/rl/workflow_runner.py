import os

import optuna
from optuna.pruners import HyperbandPruner
import numpy as np

import psutil

from fusion.sim.sim_helpers import modify_multiple_json_values, update_dict_from_list
from fusion.sim.sim_helpers import get_erlang_vals, run_simulation_for_erlangs, save_study_results
from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
from fusion.modules.rl.utils.setup import print_info, setup_rl_sim
from fusion.modules.rl.model_manager import get_model, save_model

from fusion.modules.rl.utils.hyperparams import get_optuna_hyperparams
from fusion.modules.rl.utils.general_utils import save_arr

from fusion.modules.rl.args.general_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS, VALID_DRL_ALGORITHMS


# TODO: (version 5.5-6) Put support for picking up where you left off (testing)

def _run_drl_training(env: object, sim_dict: dict, yaml_dict: dict = None):
    """
    Trains a deep reinforcement learning model with StableBaselines3.
    """
    model, yaml_dict = get_model(sim_dict=sim_dict, device=sim_dict['device'], env=env, yaml_dict=yaml_dict)
    model.learn(total_timesteps=yaml_dict['n_timesteps'], log_interval=sim_dict['print_step'],
                callback=sim_dict['callback'])

    save_model(sim_dict=sim_dict, env=env, model=model)


def _setup_callbacks(callback_list, sim_dict):
    """Initialise callback attributes that depend on the simulation settings."""
    if callback_list:
        for cb in callback_list.callbacks:
            cb.max_iters = sim_dict['max_iters']
            cb.sim_dict = sim_dict


def _train_drl_trial(env, sim_dict, callback_list, completed_trials, rewards_matrix):
    """
    Runs one full DRL training trial (all episodes) and returns the new
    observation and updated completed_trials counter.
    """
    _run_drl_training(env=env, sim_dict=sim_dict)
    rewards_matrix[completed_trials] = callback_list.callbacks[0].episode_rewards
    callback_list.callbacks[0].episode_rewards = np.array([])

    completed_trials += 1
    env.trial = completed_trials

    for cb in callback_list.callbacks:
        cb.trial += 1

    callback_list.callbacks[1].current_ent = sim_dict['epsilon_start']
    callback_list.callbacks[1].current_lr = sim_dict['alpha_start']
    callback_list.callbacks[1].iter = 0
    env.iteration = 0

    print(f"{completed_trials} trials completed out of {sim_dict['n_trials']}.")
    obs, _ = env.reset(seed=completed_trials)
    return obs, completed_trials


def _update_episode_stats(obs, reward, terminated, truncated, episodic_reward, episodic_rew_arr, completed_episodes,
                          completed_trials, env, sim_dict, rewards_matrix, trial):
    """
    Consolidates the bookkeeping that happens whenever an episode ends.
    Returns the updated state so the caller’s loop stays perfectly in sync.
    """
    episodic_reward += reward
    if not (terminated or truncated):
        return obs, episodic_reward, episodic_rew_arr, completed_episodes, completed_trials

    episodic_rew_arr = np.append(episodic_rew_arr, episodic_reward)
    episodic_reward = 0
    completed_episodes += 1

    if trial is not None:
        current_mean_reward = np.mean(episodic_rew_arr) if episodic_rew_arr.size else 0
        trial.report(current_mean_reward, completed_episodes)
        if trial.should_prune():
            raise optuna.TrialPruned()

    print(f"{completed_episodes} episodes completed out of {sim_dict['max_iters']}.")

    if completed_episodes == sim_dict['max_iters']:
        env.iteration = 0
        env.trial += 1
        rewards_matrix[completed_trials] = episodic_rew_arr
        episodic_rew_arr = np.array([])
        completed_trials += 1
        completed_episodes = 0
        print(f"{completed_trials} trials completed out of {sim_dict['n_trials']}.")

    obs, _ = env.reset(seed=completed_trials)
    return obs, episodic_reward, episodic_rew_arr, completed_episodes, completed_trials


def run_iters(env: object, sim_dict: dict, is_training: bool, drl_agent: bool,
              model=None, callback_list: list = None, trial=None):
    """
    Runs the specified number of episodes/trials in the reinforcement‑learning
    environment, exactly as before – just with the heavy lifting delegated to
    helpers for clarity.
    """
    process = psutil.Process()
    memory_usage_list = []

    completed_episodes = 0
    completed_trials = 0
    episodic_reward = 0
    rewards_matrix = np.zeros((sim_dict['n_trials'], sim_dict['max_iters']))
    episodic_rew_arr = np.array([])

    _setup_callbacks(callback_list, sim_dict)

    obs, _ = env.reset(seed=completed_trials)

    while completed_trials < sim_dict['n_trials']:
        memory_usage_list.append(process.memory_info().rss / (1024 * 1024))

        if is_training and drl_agent:
            obs, completed_trials = _train_drl_trial(
                env, sim_dict, callback_list, completed_trials, rewards_matrix
            )
            continue

        if is_training:
            obs, reward, term, trunc, _ = env.step(0)
        else:
            action, _st = model.predict(obs)
            obs, reward, term, trunc, _ = env.step(action)

        obs, episodic_reward, episodic_rew_arr, completed_episodes, completed_trials = \
            _update_episode_stats(
                obs, reward, term, trunc,
                episodic_reward, episodic_rew_arr,
                completed_episodes, completed_trials,
                env, sim_dict, rewards_matrix, trial
            )

    if is_training:
        mean_per_iter = np.mean(rewards_matrix, axis=0)
        sum_reward = np.sum(mean_per_iter)
        save_arr(mean_per_iter, sim_dict, "average_rewards.npy")
        save_arr(memory_usage_list, sim_dict, "memory_usage.npy")
    else:
        raise NotImplementedError

    return sum_reward


def run_testing():
    """
    Runs pre-trained RL model evaluation in the environment for the number of episodes specified in `sim_dict`.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters (e.g., model type, paths).
    """
    raise NotImplementedError


def run(env: object, sim_dict: dict, callback_list: list = None, trial=None):
    """
    Manages the execution of simulations for training or testing RL models.

    Delegates to either training or testing based on flags within the simulation configuration.

    :param env: The reinforcement learning environment.
    :param sim_dict: The simulation configuration dictionary containing paths, algorithms, and statistical parameters.
    :param callback_obj: The custom callback to monitor episodic rewards from SB3.
    """
    print_info(sim_dict=sim_dict)

    if sim_dict['is_training']:
        # Print info function should already error check valid input, no need to raise an error here
        if sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS or sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
            sum_returns = run_iters(env=env, sim_dict=sim_dict, is_training=True,
                                    drl_agent=sim_dict['path_algorithm'] in VALID_DRL_ALGORITHMS,
                                    callback_list=callback_list, trial=trial)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return sum_returns


def run_optuna_study(sim_dict, callback_list):
    """
    Runs Optuna study for hyperparameter optimization.
    
    :param env: The reinforcement learning environment.
    :param sim_dict: The simulation configuration dictionary containing paths, algorithms, and statistical parameters.
    """

    # Define the Optuna objective function
    def objective(trial: optuna.Trial):
        env = SimEnv(render_mode=None, custom_callback=callback_list, sim_dict=setup_rl_sim())

        for callback_obj in callback_list.callbacks:
            callback_obj.sim_dict = env.sim_dict
            callback_obj.max_iters = sim_dict['max_iters']
        env.sim_dict['callback'] = callback_list.callbacks

        hyperparam_dict = get_optuna_hyperparams(sim_dict=sim_dict, trial=trial)
        update_list = [(param, value) for param, value in hyperparam_dict.items() if param in sim_dict]

        modify_multiple_json_values(trial_num=trial.number, file_path=file_path, update_list=update_list)
        env.sim_dict = update_dict_from_list(input_dict=env.sim_dict, updates_list=update_list)
        erlang_list = get_erlang_vals(sim_dict=sim_dict)

        mean_reward = run_simulation_for_erlangs(env=env, erlang_list=erlang_list, sim_dict=sim_dict, run_func=run,
                                                 callback_list=callback_list, trial=trial)
        mean_reward = mean_reward / sim_dict['max_iters']
        if "callback" in env.sim_dict:
            del env.sim_dict["callback"]
            del env.callback

        trial.set_user_attr("sim_start_time", sim_dict['sim_start'])
        trial.set_user_attr("env", env)

        return mean_reward

    # Run the optimization
    study_name = "hyperparam_study.pkl"
    file_path = os.path.join('data', 'input', sim_dict['network'], sim_dict['date'],
                             sim_dict['sim_start'])
    pruner = HyperbandPruner(
        min_resource=20,
        max_resource=sim_dict['max_iters'],
        reduction_factor=3
    )
    study = optuna.create_study(direction='maximize', study_name=study_name, pruner=pruner)
    n_trials = sim_dict['optuna_trials']
    study.optimize(objective, n_trials=n_trials)

    # Save study results
    best_trial = study.best_trial
    save_study_results(  # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        study=study,
        env=best_trial.user_attrs.get("env"),
        study_name=study_name,
        best_params=best_trial.params,
        best_reward=best_trial.value,
        best_sim_start=best_trial.user_attrs.get("sim_start_time"),
    )
