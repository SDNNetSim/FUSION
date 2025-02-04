import gymnasium as gym

from rl_scripts.utils.env_util import SimEnvUtils
from rl_scripts.helpers.setup_helpers import setup_rl_sim, SetupHelper
from rl_scripts.helpers.general_helpers import CoreUtilHelpers, SimEnvHelpers
from rl_scripts.agents.path_agent import PathAgent
from rl_scripts.agents.core_agent import CoreAgent
from rl_scripts.agents.spectrum_agent import SpectrumAgent

from arg_scripts.rl_args import RLProps


class SimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Defines a reinforcement learning-enabled simulation environment (SimEnv) that integrates multi-agent helpers,
    setup utilities, and step management for dynamic, iterative simulations.
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

        self.rl_help_obj = CoreUtilHelpers(rl_props=self.rl_props, engine_obj=self.engine_obj, route_obj=self.route_obj)
        self._setup_agents()

        self.modified_props = None
        self.sim_props = None
        self.setup_helper = SetupHelper(sim_env=self)
        self.sim_env_helper = SimEnvHelpers(sim_env=self)
        self.step_helper = SimEnvUtils(sim_env=self)

        # Used to get config variables into the observation space
        self.reset(options={'save_sim': False})

        # TODO: (drl_path_agent) Here
        self.observation_space = self.spectrum_agent.get_obs_space()
        self.action_space = self.spectrum_agent.get_action_space()

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

        # TODO: fixme statement breaks for DRL (drl_path_agents)
        if self.optimize is None:
            self.iteration = 0
            self.setup()

        self._init_props_envs()
        if not self.sim_dict['is_training'] and self.iteration == 0:
            self._load_models()
        if seed is None:
            seed = self.iteration + 1

        self.rl_help_obj.reset_reqs_dict(seed=seed)
        obs = self.step_helper.get_obs()
        info = self._get_info()
        return obs, info

    def _init_envs(self):
        self.setup_helper.init_envs()

    def _create_input(self):
        self.setup_helper.create_input()

    def _load_models(self):
        self.setup_helper.load_models()

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

        self.sim_env_helper.update_helper_obj(action=action, bandwidth=bandwidth)
        self.rl_help_obj.allocate()
        reqs_status_dict = self.engine_obj.reqs_status_dict

        was_allocated = req_id in reqs_status_dict
        path_length = self.route_obj.route_props.weights_list[0]
        self.step_helper.handle_test_train_step(was_allocated=was_allocated, path_length=path_length)
        self.rl_help_obj.update_snapshots()

        # TODO (drl_path_agents) Spectrum no longer relates to only the drl reward
        # drl_reward = self.spectrum_agent.get_reward(was_allocated=was_allocated)
        drl_reward = 1.0

        self.rl_props.arrival_count += 1
        terminated = self.step_helper.check_terminated()
        new_obs = self.step_helper.get_obs()
        truncated = False
        info = self._get_info()

        return new_obs, drl_reward, terminated, truncated, info

    @staticmethod
    def _get_info():
        return dict()

    def setup(self):
        """
        Sets up this class.
        """
        self.optimize = self.sim_dict['optimize']
        self.rl_props.k_paths = self.sim_dict['k_paths']
        self.rl_props.cores_per_link = self.sim_dict['cores_per_link']
        # TODO: Only support for 'c' band (drl_path_agents)
        self.rl_props.spectral_slots = self.sim_dict['c_band']

        self._create_input()

        self.sim_dict['arrival_dict'] = {
            'start': self.sim_dict['erlang_start'],
            'stop': self.sim_dict['erlang_stop'],
            'step': self.sim_dict['erlang_step'],
        }

        self.engine_obj.engine_props['erlang'] = float(self.sim_dict['erlang_start'])
        cores_per_link = self.engine_obj.engine_props['cores_per_link']
        erlang = self.engine_obj.engine_props['erlang']
        holding_time = self.engine_obj.engine_props['holding_time']
        self.engine_obj.engine_props['arrival_rate'] = (cores_per_link * erlang) / holding_time

        self.engine_obj.engine_props['band_list'] = ['c']

    def _setup_agents(self):
        self.path_agent = PathAgent(path_algorithm=self.sim_dict['path_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.core_agent = CoreAgent(core_algorithm=self.sim_dict['core_algorithm'], rl_props=self.rl_props,
                                    rl_help_obj=self.rl_help_obj)
        self.spectrum_agent = SpectrumAgent(spectrum_algorithm=self.sim_dict['spectrum_algorithm'],
                                            rl_props=self.rl_props)
