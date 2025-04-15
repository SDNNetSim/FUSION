import os

import optuna
from optuna.pruners import HyperbandPruner
import numpy as np

import psutil

from helper_scripts.sim_helpers import modify_multiple_json_values, update_dict_from_list
from helper_scripts.sim_helpers import get_erlang_vals, run_simulation_for_erlangs, save_study_results
from reinforcement_learning.utils.rl_zoo import run_rl_zoo
from reinforcement_learning.gymnasium_envs.general_sim_env import SimEnv
from reinforcement_learning.utils.setup import print_info, setup_rl_sim
from reinforcement_learning.utils.callbacks import EpisodicRewardCallback
from stable_baselines3.common.callbacks import CallbackList
from reinforcement_learning.model_manager import get_trained_model, get_model, save_model

from reinforcement_learning.utils.hyperparams import get_optuna_hyperparams
from reinforcement_learning.utils.general_utils import save_arr

from reinforcement_learning.args.general_args import VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS, VALID_DRL_ALGORITHMS


def _run_drl_training(env: object, sim_dict: dict, yaml_dict: dict = None):
    """
    Trains a deep reinforcement learning model with StableBaselines3.
    """
    # if sim_dict['optimize_hyperparameters'] or sim_dict['optimize']:
    #     run_rl_zoo(sim_dict=sim_dict)
    # else:
    model, yaml_dict = get_model(sim_dict=sim_dict, device=sim_dict['device'], env=env, yaml_dict=yaml_dict)
    model.learn(total_timesteps=yaml_dict['n_timesteps'], log_interval=sim_dict['print_step'],
                callback=sim_dict['callback'])

    save_model(sim_dict=sim_dict, env=env, model=model)


# TODO: (drl_path_agents) Break this function up for organizational purposes
#   - You have repeat logic
def run_iters(env: object, sim_dict: dict, is_training: bool, drl_agent: bool, model=None, callback_list: list = None,
              trial=None):
    """
    Runs the specified number of episodes in the reinforcement learning environment.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation settings, such as maximum iterations.
    :param is_training: A boolean flag indicating whether the model should train or evaluate.
    :param drl_agent: A boolean flag indicating whether the model is a DRL agent.
    :param model: The RL model to be used; required only if not in training mode.
    :param callback_obj: A callback object to be passed to the callback function.
    """
    process = psutil.Process()
    memory_usage_list = []

    completed_episodes = 0
    completed_trials = 0
    episodic_reward = 0
    rewards_matrix = np.zeros((sim_dict['n_trials'], sim_dict['max_iters']))
    episodic_rew_arr = np.array([])

    if callback_list:
        for callback_obj in callback_list.callbacks:
            callback_obj.max_iters = sim_dict['max_iters']
            callback_obj.sim_dict = sim_dict

    obs, _ = env.reset(seed=completed_trials)

    while completed_trials < sim_dict['n_trials']:
        mem_usage = process.memory_info().rss / (1024 * 1024)
        memory_usage_list.append(mem_usage)

        if is_training:
            if drl_agent:
                _run_drl_training(env=env, sim_dict=sim_dict)
                rewards_matrix[completed_trials] = callback_list.callbacks[0].episode_rewards

                callback_list.callbacks[0].episode_rewards = np.array([])
                completed_trials += 1
                env.trial = completed_trials

                for callback_obj in callback_list.callbacks:
                    callback_obj.trial += 1

                # Reset dynamic params after a trial
                # TODO (drl_path_agents) this may not be needed
                callback_list.callbacks[1].current_ent = sim_dict['epsilon_start']
                callback_list.callbacks[1].current_lr = sim_dict['alpha_start']
                callback_list.callbacks[1].iter = 0
                env.iteration = 0

                print(f"{completed_trials} trials completed out of {sim_dict['n_trials']}.")
                obs, _ = env.reset(seed=completed_trials)
                continue

            obs, reward, is_terminated, is_truncated, _ = env.step(0)
        else:
            action, _states = model.predict(obs)
            obs, reward, is_terminated, is_truncated, _ = env.step(action)

        episodic_reward += reward
        if is_terminated or is_truncated:
            episodic_rew_arr = np.append(episodic_rew_arr, episodic_reward)
            episodic_reward = 0
            completed_episodes += 1

            if trial is not None:
                current_mean_reward = np.mean(episodic_rew_arr) if episodic_rew_arr.size > 0 else 0
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

    if is_training:
        means_arr = np.mean(rewards_matrix, axis=0)
        sum_reward = np.sum(means_arr)
        save_arr(arr=means_arr, sim_dict=sim_dict, file_name="average_rewards.npy")
        save_arr(arr=memory_usage_list, sim_dict=sim_dict, file_name="memory_usage.npy")
    else:
        raise NotImplementedError

    return sum_reward


def run_testing(env: object, sim_dict: dict):
    """
    Runs pre-trained RL model evaluation in the environment for the number of episodes specified in `sim_dict`.

    :param env: The reinforcement learning environment.
    :param sim_dict: A dictionary containing simulation-specific parameters (e.g., model type, paths).
    """
    model = get_trained_model(env=env, sim_dict=sim_dict)
    run_iters(env=env, sim_dict=sim_dict, is_training=False, model=model)


def run(env: object, sim_dict: dict, callback_list: list = None, trial = None):
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
        sum_returns = run_testing(sim_dict=sim_dict, env=env)

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
