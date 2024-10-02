import os
import copy
import subprocess
import optuna
import pickle

from torch import nn  # pylint: disable=unused-import
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from src.engine import Engine
from src.routing import Routing
from helper_scripts.rl_setup_helpers import setup_rl_sim, print_info, setup_ppo
from helper_scripts.setup_helpers import create_input, save_input
from helper_scripts.rl_helpers import RLHelpers
from helper_scripts.callback_helpers import GetModelParams
from helper_scripts.sim_helpers import get_start_time, find_path_len, get_path_mod
from helper_scripts.multi_agent_helpers import PathAgent, CoreAgent, SpectrumAgent
from arg_scripts.rl_args import RLProps, LOCAL_RL_COMMANDS_LIST, VALID_PATH_ALGORITHMS, VALID_CORE_ALGORITHMS
from arg_scripts.rl_args import VALID_SPECTRUM_ALGORITHMS, get_optuna_hyperparams


# TODO: No support for core or spectrum assignment
# TODO: Update tests
# TODO: Need to print traffic volume for every iteration
# TODO: Multiple q tables didn't save?

class SimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Controls all reinforcement learning assisted simulations.
    """
    metadata = dict()

    def __init__(self, render_mode: str = None, custom_callback: object = None, sim_dict: dict = None,
                 **kwargs):  # pylint: disable=unused-argument
        super().__init__()

        self.rl_props = RLProps()

        if sim_dict is None:
            self.sim_dict = setup_rl_sim()['s1']
        else:
            self.sim_dict = sim_dict['s1']
        self.rl_props.super_channel_space = self.sim_dict['super_channel_space']

        self.iteration = 0
        self.options = None
        self.optimize = None
        self.callback = custom_callback
        self.render_mode = render_mode

        self.engine_obj = None
        self.route_obj = None

        # TODO: Change all inputs to account for the new object
        self.rl_help_obj = RLHelpers(rl_props=self.rl_props, engine_obj=self.engine_obj, route_obj=self.route_obj)
        self._setup_agents()

        self.modified_props = None
        self.sim_props = None
        # Used to get config variables into the observation space
        self.reset(options={'save_sim': False})
        self.observation_space = self.spectrum_agent.get_obs_space()
        self.action_space = self.spectrum_agent.get_action_space()

    def _setup_agents(self):
        self.path_agent = PathAgent(path_algorithm=self.sim_dict['path_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.core_agent = CoreAgent(core_algorithm=self.sim_dict['core_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.spectrum_agent = SpectrumAgent(spectrum_algorithm=self.sim_dict['spectrum_algorithm'],
                                            rl_props=self.rl_props)

    def _check_terminated(self):
        if self.rl_props.arrival_count == (self.engine_obj.engine_props['num_requests']):
            terminated = True
            base_fp = os.path.join('data')
            # The spectrum agent is handled by SB3 automatically
            if self.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS and self.sim_dict['is_training']:
                self.path_agent.end_iter()
            elif self.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS and self.sim_dict['is_training']:
                self.core_agent.end_iter()
            self.engine_obj.end_iter(iteration=self.iteration, print_flag=False, base_fp=base_fp)
            self.iteration += 1
        else:
            terminated = False

        return terminated

    def _update_helper_obj(self, action: list, bandwidth: str):
        self.rl_help_obj.path_index = self.rl_props.path_index
        self.rl_help_obj.core_num = self.rl_props.core_index

        if self.sim_dict['spectrum_algorithm'] in ('dqn', 'ppo', 'a2c'):
            self.rl_help_obj.rl_props.forced_index = action
        else:
            self.rl_help_obj.rl_props.forced_index = None

        # TODO: Definitely make sure this is a pointer (comparing results)
        self.rl_help_obj.rl_props = self.rl_props
        self.rl_help_obj.engine_obj = self.engine_obj
        self.rl_help_obj.handle_releases()
        self.rl_help_obj.update_route_props(chosen_path=self.rl_props.chosen_path_list, bandwidth=bandwidth)

    def _handle_test_train_step(self, was_allocated: bool, path_length: int):
        if self.sim_dict['is_training']:
            if self.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self.path_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                       iteration=self.iteration, path_length=path_length)
            elif self.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
                self.core_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                       iteration=self.iteration)
            else:
                raise NotImplementedError
        else:
            self.path_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                   iteration=self.iteration, path_length=path_length)
            self.core_agent.update(was_allocated=was_allocated, net_spec_dict=self.engine_obj.net_spec_dict,
                                   iteration=self.iteration)

    def step(self, action: list):
        """
        Handles a single time step in the simulation.

        :param action: A list of actions from the DRL agent.
        :return: The new observation, reward, if terminated, if truncated, and misc. info.
        :rtype: tuple
        """
        req_info_dict = self.rl_props.arrival_list[self.rl_props.arrival_count]
        req_id = req_info_dict['req_id']
        bandwidth = req_info_dict['bandwidth']

        self._update_helper_obj(action=action, bandwidth=bandwidth)
        self.rl_help_obj.allocate()
        reqs_status_dict = self.engine_obj.reqs_status_dict

        was_allocated = req_id in reqs_status_dict
        path_length = self.route_obj.route_props.weights_list[0]
        self._handle_test_train_step(was_allocated=was_allocated, path_length=path_length)
        self.rl_help_obj.update_snapshots()
        drl_reward = self.spectrum_agent.get_reward(was_allocated=was_allocated)

        self.rl_props.arrival_count += 1
        terminated = self._check_terminated()
        new_obs = self._get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, drl_reward, terminated, truncated, info

    @staticmethod
    def _get_info():
        return dict()

    def _handle_path_train_test(self):
        if 'bandit' in self.sim_dict['path_algorithm']:
            self.route_obj.sdn_props = self.rl_props.mock_sdn_dict
            self.route_obj.engine_props['route_method'] = 'k_shortest_path'
            self.route_obj.get_route()

        self.path_agent.get_route(route_obj=self.route_obj)
        # TODO: Update to chosen path list, be very careful as this affects results easily for the agents if done wrong
        self.rl_help_obj.rl_props.chosen_path_list = [self.rl_props.chosen_path_list]
        self.route_obj.route_props.paths_matrix = self.rl_help_obj.rl_props.chosen_path_list
        self.rl_props.core_index = None
        self.rl_props.forced_index = None

    def _determine_core_penalty(self):
        # Default to first fit if all paths fail
        self.rl_props.chosen_path = [self.route_obj.route_props.paths_matrix[0]]
        self.rl_props.chosen_path_index = 0
        for path_index, path_list in enumerate(self.route_obj.route_props.paths_matrix):
            mod_format_list = self.route_obj.route_props.mod_formats_matrix[path_index]

            was_allocated = self.rl_help_obj.mock_handle_arrival(engine_props=self.engine_obj.engine_props,
                                                                 sdn_props=self.rl_props.mock_sdn_dict,
                                                                 mod_format_list=mod_format_list, path_list=path_list)

            if was_allocated:
                self.rl_props.chosen_path_list = [path_list]
                self.rl_props.chosen_path_index = path_index
                self.core_agent.no_penalty = False
                break

            self.core_agent.no_penalty = True

    def _handle_core_train(self):
        self.route_obj.sdn_props = self.rl_props.mock_sdn_dict
        self.route_obj.engine_props['route_method'] = 'k_shortest_path'
        self.route_obj.get_route()
        self._determine_core_penalty()

        self.rl_props.forced_index = None
        # TODO: Check to make sure this doesn't affect anything
        # try:
        #     req_info_dict = self.rl_props.arrival_list[self.rl_props.arrival_count]
        # except IndexError:
        #     req_info_dict = self.rl_props.arrival_list[self.rl_props.arrival_count - 1]

        self.core_agent.get_core()

    def _handle_spectrum_train(self):
        self.route_obj.sdn_props = self.rl_props.mock_sdn_dict
        self.route_obj.engine_props['route_method'] = 'shortest_path'
        self.route_obj.get_route()
        # TODO: Change name in rl props eventually
        self.rl_props.paths_list = self.route_obj.route_props.paths_matrix
        self.rl_props.chosen_path = self.route_obj.route_props.paths_matrix
        self.rl_props.path_index = 0
        self.rl_props.core_index = None

    def _handle_test_train_obs(self, curr_req: dict):
        if self.sim_dict['is_training']:
            if self.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS:
                self._handle_path_train_test()
            elif self.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
                self._handle_core_train()
            elif self.sim_dict['spectrum_algorithm'] not in ('first_fit', 'best_fit', ' last_fit'):
                self._handle_spectrum_train()
            else:
                raise NotImplementedError
        else:
            self._handle_path_train_test()
            self.core_agent.get_core()

        path_len = find_path_len(path_list=self.rl_props.chosen_path_list[0],
                                 topology=self.engine_obj.topology)
        path_mod = get_path_mod(mods_dict=curr_req['mod_formats'], path_len=path_len)

        return path_mod

    # fixme
    def _get_spectrum_obs(self, curr_req: dict):  # pylint: disable=unused-argument
        # path_mod = self._handle_test_train_obs(curr_req=curr_req)
        # if path_mod is not False:
        #     slots_needed = curr_req['mod_formats'][path_mod]['slots_needed']
        # super_channels, no_penalty = self.rl_help_obj.get_super_channels(slots_needed=slots_needed,
        #                                                                  num_channels=self.rl_props[
        #                                                                      'super_channel_space'])
        # No penalty for DRL agent, mistake not made by it
        # else:
        slots_needed = -1
        no_penalty = True
        super_channels = np.array([100.0, 100.0, 100.0])

        self.spectrum_agent.no_penalty = no_penalty
        source_obs = np.zeros(self.rl_props.num_nodes)
        source_obs[self.rl_props.source] = 1.0
        dest_obs = np.zeros(self.rl_props.num_nodes)
        dest_obs[self.rl_props.destination] = 1.0

        return slots_needed, source_obs, dest_obs, super_channels

    def _get_obs(self):
        # Used when we reach a reset after a simulation has finished (reset automatically called by gymnasium, use
        # placeholder variable)
        if self.rl_props.arrival_count == self.engine_obj.engine_props['num_requests']:
            curr_req = self.rl_props.arrival_list[self.rl_props.arrival_count - 1]
        else:
            curr_req = self.rl_props.arrival_list[self.rl_props.arrival_count]

        self.rl_help_obj.handle_releases()
        self.rl_props.source = int(curr_req['source'])
        self.rl_props.destination = int(curr_req['destination'])
        self.rl_props.mock_sdn_dict = self.rl_help_obj.update_mock_sdn(curr_req=curr_req)

        _ = self._handle_test_train_obs(curr_req=curr_req)
        slots_needed, source_obs, dest_obs, super_channels = self._get_spectrum_obs(curr_req=curr_req)
        obs_dict = {
            'slots_needed': slots_needed,
            'source': source_obs,
            'destination': dest_obs,
            'super_channels': super_channels,
        }
        return obs_dict

    def _init_envs(self):
        # SB3 will init the environment for us, but not for non-DRL algorithms we've added
        if self.sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS and self.sim_dict['is_training']:
            self.path_agent.engine_props = self.engine_obj.engine_props
            self.path_agent.setup_env()
        elif self.sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS and self.sim_dict['is_training']:
            self.core_agent.engine_props = self.engine_obj.engine_props
            self.core_agent.setup_env()

    def _create_input(self):
        base_fp = os.path.join('data')
        self.sim_dict['thread_num'] = 's1'
        # fixme
        # Added only for structure consistency
        # time.sleep(20)
        get_start_time(sim_dict={'s1': self.sim_dict})
        file_name = "sim_input_s1.json"

        self.engine_obj = Engine(engine_props=self.sim_dict)
        self.route_obj = Routing(engine_props=self.engine_obj.engine_props,
                                 sdn_props=self.rl_props.mock_sdn_dict)

        # fixme
        # time.sleep(30)
        self.sim_props = create_input(base_fp=base_fp, engine_props=self.sim_dict)
        self.modified_props = copy.deepcopy(self.sim_props)
        if 'topology' in self.sim_props:
            self.modified_props.pop('topology')
            try:
                self.modified_props.pop('callback')
            except KeyError:
                print('Callback does not exist, skipping.')

        save_input(base_fp=base_fp, properties=self.modified_props, file_name=file_name,
                   data_dict=self.modified_props)

    # TODO: Options to have select AI agents
    def _load_models(self):
        self.path_agent.engine_props = self.engine_obj.engine_props
        self.path_agent.rl_props = self.rl_props
        self.path_agent.load_model(model_path=self.sim_dict['path_model'], erlang=self.sim_dict['erlang'],
                                   num_cores=self.sim_dict['cores_per_link'])

        self.core_agent.engine_props = self.engine_obj.engine_props
        self.core_agent.rl_props = self.rl_props
        self.core_agent.load_model(model_path=self.sim_dict['core_model'], erlang=self.sim_dict['erlang'],
                                   num_cores=self.sim_dict['cores_per_link'])

    def setup(self):
        """
        Sets up this class.
        """
        self.optimize = self.sim_dict['optimize']
        self.rl_props.k_paths = self.sim_dict['k_paths']
        self.rl_props.cores_per_link = self.sim_dict['cores_per_link']
        # TODO: Only support for 'c' band...Maybe add multi-band
        self.rl_props.spectral_slots = self.sim_dict['c_band']

        self._create_input()

        # fixme
        try:
            start_arr_rate = float(self.sim_dict['arrival_rate']['start'])
        except TypeError:
            start_arr_rate = self.sim_dict['arrival_rate'] / self.sim_dict['cores_per_link']

        self.engine_obj.engine_props['erlang'] = start_arr_rate / self.sim_dict['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = start_arr_rate * self.sim_dict['cores_per_link']

    def _init_props_envs(self):
        self.rl_props.arrival_count = 0
        self.engine_obj.init_iter(iteration=self.iteration)
        self.engine_obj.create_topology()
        self.rl_help_obj.topology = self.engine_obj.topology
        self.rl_props.num_nodes = len(self.engine_obj.topology.nodes)

        if self.iteration == 0:
            self._init_envs()

        self.rl_help_obj.rl_props = self.rl_props
        self.rl_help_obj.engine_obj = self.engine_obj
        self.rl_help_obj.route_obj = self.route_obj

    def reset(self, seed: int = None, options: dict = None):  # pylint: disable=arguments-differ
        """
        Resets necessary variables after each iteration of the simulation.

        :param seed: Seed for random generation.
        :param options: Custom option input.
        :return: The first observation and misc. information.
        :rtype: tuple
        """
        super().reset(seed=seed)
        self.rl_props.arrival_list = list()
        self.rl_props.depart_list = list()

        if self.optimize or self.optimize is None:
            self.iteration = 0
            self.setup()

        self._init_props_envs()
        if not self.sim_dict['is_training'] and self.iteration == 0:
            self._load_models()
        if seed is None:
            # fixme
            # seed = self.iteration + 1
            seed = 0

        self.rl_help_obj.reset_reqs_dict(seed=seed)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info


def _run_iters(env: object, sim_dict: dict, is_training: bool, model=None):
    completed_episodes = 0
    obs, _ = env.reset()
    while True:
        if is_training:
            obs, _, is_terminated, is_truncated, _ = env.step([0])
        else:
            # TODO: Implement
            action, _states = model.predict(obs)
            # action = [0]
            obs, _, is_terminated, is_truncated, _ = env.step(action)

        if completed_episodes >= sim_dict['max_iters']:
            break
        if is_terminated or is_truncated:
            obs, _ = env.reset()
            completed_episodes += 1
            print(f'{completed_episodes} episodes completed out of {sim_dict["max_iters"]}.')


def _get_model(algorithm: str, device: str, env: object):
    model = None
    yaml_dict = None
    env_name = None

    if algorithm == 'dqn':
        model = None
    elif algorithm == 'ppo':
        model = setup_ppo(env=env, device=device)
    elif algorithm == 'a2c':
        model = None

    return model, yaml_dict[env_name]


def _get_trained_model(env: object, sim_dict: dict):
    if sim_dict['spectrum_algorithm'] == 'ppo':
        model = PPO.load(os.path.join('logs', sim_dict['spectrum_model'], 'ppo_model.zip'), env=env)
    else:
        model = None

    return model


def _run_rl_zoo(sim_dict: dict):
    # TODO: Detect if working locally or on the cluster
    for command in LOCAL_RL_COMMANDS_LIST:
        subprocess.run(command, shell=True, check=True)

    if sim_dict['spectrum_algorithm'] == 'ppo':
        subprocess.run('python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file '
                       './sb3_scripts/yml/ppo.yml -optimize --n-trials 5 --n-timesteps 20000', shell=True, check=True)
    else:
        raise NotImplementedError(f"Spectrum Algorithm has not been implemented: {sim_dict['spectrum_algorithm']}")


def _run_testing(env: object, sim_dict: dict):
    model = _get_trained_model(env=env, sim_dict=sim_dict)
    _run_iters(env=env, sim_dict=sim_dict, is_training=False, model=model)
    # TODO: Hard coded
    save_fp = os.path.join('logs', 'ppo', env.modified_props['network'], env.modified_props['date'],
                           env.modified_props['sim_start'], 'ppo_model.zip')
    model.save(save_fp)


def _run_spectrum(sim_dict: dict, env: object):
    if sim_dict['optimize_hyperparameters']:
        _run_rl_zoo(sim_dict=sim_dict)
    else:
        model, yaml_dict = _get_model(algorithm=sim_dict['spectrum_algorithm'], device=sim_dict['device'],
                                      env=env)
        model.learn(total_timesteps=yaml_dict['n_timesteps'], log_interval=sim_dict['print_step'],
                    callback=sim_dict['callback'])

        save_fp = os.path.join('logs', 'ppo', env.modified_props['network'], env.modified_props['date'],
                               env.modified_props['sim_start'], 'ppo_model.zip')
        model.save(save_fp)


def _run(env: object, sim_dict: dict):
    print_info(sim_dict=sim_dict)

    if sim_dict['is_training']:
        # Print info function should already error check valid input, no need to raise an error here
        if sim_dict['path_algorithm'] in VALID_PATH_ALGORITHMS or sim_dict['core_algorithm'] in VALID_CORE_ALGORITHMS:
            _run_iters(env=env, sim_dict=sim_dict, is_training=True)
        elif sim_dict['spectrum_algorithm'] in VALID_SPECTRUM_ALGORITHMS:
            _run_spectrum(sim_dict=sim_dict, env=env)
    else:
        _run_testing(sim_dict=sim_dict, env=env)


def run_rl_sim():
    """
    The main function that controls reinforcement learning simulations.
    """
    callback = GetModelParams()
    env = SimEnv(render_mode=None, custom_callback=callback, sim_dict=setup_rl_sim())
    env.sim_dict['callback'] = callback

    def objective(trial: optuna.trial):
        hyperparam_dict = get_optuna_hyperparams(trial)
        for param, value in hyperparam_dict.items():
            if param not in list(env.sim_dict.keys()):
                raise NotImplementedError(f'Param: {param} does not exist in simulation dictionary.')
            env.sim_dict[param] = value

        total_rewards = []
        # TODO: Integrate with configuration file somehow
        arrival_list = [80, 100]
        for arrival_rate in arrival_list:
            env.engine_obj.engine_props['erlang'] = arrival_rate / env.sim_dict['holding_time']
            env.engine_obj.engine_props['arrival_rate'] = arrival_rate * env.sim_dict['cores_per_link']
            # TODO: How much else should be reset? Shouldn't we just redo the object in the loop...
            env.iteration = 0
            _run(env=env, sim_dict=env.sim_dict)
            # TODO: Only works for path agent
            sum_returns = np.sum(env.path_agent.reward_penalty_list)
            total_rewards.append(sum_returns)

        return np.mean(total_rewards)

    # TODO: Need something here to optimize or not without interfere with DRL
    # TODO: Update name to something more specific
    # TODO: Add study name to config?
    study_name = "hyperparam_study.pkl"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    n_trials = env.sim_dict['n_trials']
    study.optimize(objective, n_trials=n_trials)

    # TODO: Repeat code, only work for path agent
    date_time = os.path.join(env.engine_obj.engine_props['network'], env.engine_obj.engine_props['date'],
                             env.engine_obj.engine_props['sim_start'])
    save_dir = os.path.join('logs', env.engine_obj.engine_props['path_algorithm'], date_time)
    save_fp = os.path.join(save_dir, study_name)
    with open(save_fp, 'wb') as curr_f:
        pickle.dump(study, curr_f)

    best_params = study.best_params
    best_reward = study.best_value
    save_fp = os.path.join(save_dir, 'best_hyperparams.txt')
    with open(save_fp, 'w') as curr_f:
        curr_f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            curr_f.write(f"{key}: {value}\n")

        curr_f.write(f"\nBest Trial Reward: {best_reward}\n")


if __name__ == '__main__':
    run_rl_sim()
