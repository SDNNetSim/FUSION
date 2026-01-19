"""
Gymnasium environment implementation for FUSION simulation.

This module provides the main SimEnv class that wraps the FUSION simulation
in a Gymnasium-compatible interface for reinforcement learning training and evaluation.
The environment handles simulation setup, state management, and reward calculation
for network resource allocation problems.

.. deprecated:: 4.0
    SimEnv (GeneralSimEnv) is deprecated and will be removed in v5.0.
    Use UnifiedSimEnv instead for better accuracy and unified code paths.
    See migration guide: docs/migration/rl_to_unified_env.md
"""

import os
import warnings
from typing import Any

import gymnasium as gym

from fusion.modules.rl.agents.path_agent import PathAgent
from fusion.modules.rl.algorithms.algorithm_props import RLProps
from fusion.modules.rl.gymnasium_envs.constants import (
    ARRIVAL_DICT_KEYS,
    DEFAULT_ARRIVAL_COUNT,
    DEFAULT_ITERATION,
    DEFAULT_SAVE_SIMULATION,
    DEFAULT_SIMULATION_KEY,
    SUPPORTED_SPECTRAL_BANDS,
)
from fusion.modules.rl.utils.deep_rl import get_action_space, get_obs_space
from fusion.modules.rl.utils.general_utils import CoreUtilHelpers
from fusion.modules.rl.utils.setup import SetupHelper, setup_rl_sim
from fusion.modules.rl.utils.sim_env import SimEnvObs, SimEnvUtils


class SimEnv(gym.Env):  # pylint: disable=abstract-method
    """
    Defines a reinforcement learning-enabled simulation environment (SimEnv).

    Integrates multi-agent helpers, setup utilities, and step management
    for dynamic, iterative simulations.
    """

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        render_mode: str | None = None,
        custom_callback: object | None = None,
        sim_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the simulation environment.

        Sets up the reinforcement learning environment with simulation components,
        agents, and helper objects for managing the network simulation.

        :param render_mode: Rendering mode for visualization (currently unused)
        :type render_mode: str | None
        :param custom_callback: Optional callback for custom functionality
        :type custom_callback: object | None
        :param sim_dict: Configuration dictionary for simulation setup
        :type sim_dict: dict[str, Any] | None
        :param kwargs: Additional keyword arguments
        :type kwargs: Any
        """
        super().__init__()

        # Emit deprecation warning unless suppressed
        if not os.environ.get("SUPPRESS_SIMENV_DEPRECATION", "").lower() in (
            "1",
            "true",
            "yes",
        ):
            warnings.warn(
                "SimEnv (GeneralSimEnv) is deprecated and will be removed in v5.0. "
                "Use UnifiedSimEnv instead via:\n"
                "  from fusion.modules.rl.environments import UnifiedSimEnv\n"
                "  env = UnifiedSimEnv(config)\n"
                "Or use the factory function:\n"
                "  from fusion.modules.rl.gymnasium_envs import create_sim_env\n"
                "  env = create_sim_env(config, env_type='unified')\n"
                "See: docs/migration/rl_to_unified_env.md\n"
                "To suppress this warning, set SUPPRESS_SIMENV_DEPRECATION=1",
                DeprecationWarning,
                stacklevel=2,
            )

        # Acknowledge kwargs parameter (required by Gymnasium interface)
        _ = kwargs

        self.rl_props = RLProps()
        if sim_dict is None:
            self.sim_dict = setup_rl_sim()[DEFAULT_SIMULATION_KEY]
        else:
            self.sim_dict = sim_dict[DEFAULT_SIMULATION_KEY]
        # Set super_channel_space from simulation dictionary
        self.rl_props.super_channel_space = self.sim_dict["super_channel_space"]
        # Initialize arrival_count
        self.rl_props.arrival_count = DEFAULT_ARRIVAL_COUNT

        self.iteration = DEFAULT_ITERATION
        self.trial: int | None = None
        self.options: dict[str, Any] | None = None
        self.optimize: bool | None = None
        self.callback = custom_callback
        self.render_mode = render_mode

        # These will be initialized during setup - using Any for now to satisfy mypy
        self.engine_obj: Any = None
        self.route_obj: Any = None

        self.rl_help_obj = CoreUtilHelpers(
            rl_props=self.rl_props,
            engine_props=self.engine_obj,
            route_obj=self.route_obj,
        )
        self._setup_agents()

        self.modified_props = None
        self.sim_props = None
        self.setup_helper = SetupHelper(sim_env=self)
        self.sim_env_helper = SimEnvObs(sim_env=self)
        self.step_helper = SimEnvUtils(sim_env=self)

        # Used to get config variables into the observation space
        self.reset(options={"save_sim": DEFAULT_SAVE_SIMULATION})

        obs_space = get_obs_space(
            sim_dict=self.sim_dict, rl_props=self.rl_props, engine_props=self.engine_obj
        )
        # Ensure observation_space is properly typed
        self.observation_space: gym.Space[Any] = (
            obs_space
            if obs_space is not None
            else gym.spaces.Box(low=0, high=1, shape=(1,))
        )

        action_space = get_action_space(
            sim_dict=self.sim_dict, rl_props=self.rl_props, engine_props=self.engine_obj
        )
        # Ensure action_space is properly typed
        self.action_space: gym.Space[Any] = (
            action_space if action_space is not None else gym.spaces.Discrete(2)
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:  # pylint: disable=arguments-differ
        """
        Resets necessary variables after each iteration of the simulation.

        :param seed: Seed for random generation.
        :param options: Custom option input.
        :return: The first observation and misc. information.
        :rtype: tuple
        """
        super().reset(seed=seed)
        self.trial = seed if seed is not None else 0
        self.rl_props.arrival_list = []
        self.rl_props.depart_list = []

        if self.optimize is None:
            self.setup()
            print_flag = False
        else:
            print_flag = True

        # Ensure seed is not None for _init_props_envs
        actual_seed = seed if seed is not None else 0
        self._init_props_envs(seed=actual_seed, print_flag=print_flag)

        if not self.sim_dict["is_training"] and self.iteration == 0:
            self._load_models()
        if seed is None:
            seed = self.iteration

        self.rl_help_obj.reset_reqs_dict(seed=seed)

        # Ensure arrival_list is properly typed and arrival_count exists
        arrival_count = getattr(self.rl_props, "arrival_count", 0)
        if (
            hasattr(self.rl_props, "arrival_list")
            and len(self.rl_props.arrival_list) > arrival_count
        ):
            req_info_dict = self.rl_props.arrival_list[arrival_count]
            if isinstance(req_info_dict, dict):
                bandwidth = req_info_dict["bandwidth"]
                holding_time = req_info_dict["depart"] - req_info_dict["arrive"]
            else:
                bandwidth = 0.0
                holding_time = 0.0
        else:
            # Fallback values if arrival_list is not properly initialized
            bandwidth = 0.0
            holding_time = 0.0
        obs = self.step_helper.get_obs(
            bandwidth=str(bandwidth), holding_time=holding_time
        )
        info = self._get_info()
        return obs, info

    def _init_envs(self) -> None:
        """
        Initialize simulation environments.

        Delegates to the setup helper to initialize required environment components.
        """
        self.setup_helper.init_envs()

    def _create_input(self) -> None:
        """
        Create simulation input parameters.

        Delegates to the setup helper to create necessary input configurations.
        """
        self.setup_helper.create_input()

    def _load_models(self) -> None:
        """
        Load pre-trained models for inference.

        Delegates to the setup helper to load saved model weights and configurations.
        """
        self.setup_helper.load_models()

    def _init_props_envs(self, seed: int, print_flag: bool) -> None:
        """
        Initialize properties and environments for the simulation iteration.

        Sets up the simulation engine, creates network topology, and initializes
        reinforcement learning properties for the current iteration.

        :param seed: Random seed for reproducibility
        :type seed: int
        :param print_flag: Whether to enable printing during initialization
        :type print_flag: bool
        """
        # Reset arrival_count
        self.rl_props.arrival_count = DEFAULT_ARRIVAL_COUNT

        # Ensure engine_obj is properly initialized before use
        if self.engine_obj is not None:
            self.engine_obj.init_iter(
                seed=seed,
                trial=self.trial,
                iteration=self.iteration,
                print_flag=print_flag,
            )
            self.engine_obj.create_topology()
            self.rl_help_obj.topology = self.engine_obj.topology
            # Ensure num_nodes attribute exists
            if not hasattr(self.rl_props, "num_nodes"):
                self.rl_props.num_nodes = len(self.engine_obj.topology.nodes)  # type: ignore[attr-defined,assignment]
            else:
                self.rl_props.num_nodes = len(self.engine_obj.topology.nodes)  # type: ignore[attr-defined,assignment]
        else:
            # Initialize with default values if engine_obj is None
            if not hasattr(self.rl_props, "num_nodes"):
                self.rl_props.num_nodes = 0  # type: ignore[attr-defined,assignment]
            else:
                self.rl_props.num_nodes = 0  # type: ignore[attr-defined,assignment]

        if self.iteration == 0:
            self._init_envs()

        self.rl_help_obj.rl_props = self.rl_props
        self.rl_help_obj.engine_props = self.engine_obj
        self.rl_help_obj.route_obj = self.route_obj

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Handles a single time step in the simulation.

        :param action: An int representing the action to take.
        :return: The new observation, reward, if terminated, if truncated, and misc.
            info.
        :rtype: tuple
        """
        # Ensure engine_obj is available
        if self.engine_obj is not None:
            self.step_helper.handle_step(
                action=action, is_drl_agent=self.engine_obj.engine_props["is_drl_agent"]
            )

        # Get arrival_count safely
        arrival_count = getattr(self.rl_props, "arrival_count", 0)
        if (
            hasattr(self.rl_props, "arrival_list")
            and len(self.rl_props.arrival_list) > arrival_count
        ):
            req_info_dict = self.rl_props.arrival_list[arrival_count]
            if isinstance(req_info_dict, dict):
                req_id = req_info_dict["req_id"]
                bandwidth = req_info_dict["bandwidth"]
                holding_time = req_info_dict["depart"] - req_info_dict["arrive"]
            else:
                req_id = 0
                bandwidth = 0.0
                holding_time = 0.0
        else:
            # Fallback values
            req_id = 0
            bandwidth = 0.0
            holding_time = 0.0

        self.sim_env_helper.update_helper_obj(action=action, bandwidth=str(bandwidth))
        self.rl_help_obj.allocate()

        # Safely access engine_obj attributes
        if self.engine_obj is not None:
            reqs_status_dict = self.engine_obj.reqs_status_dict
            was_allocated = req_id in reqs_status_dict
        else:
            was_allocated = False

        # Safely access route_obj attributes
        if self.route_obj is not None and hasattr(self.route_obj, "route_props"):
            path_length = self.route_obj.route_props.weights_list[0]
        else:
            path_length = 0.0

        # Ensure trial is not None
        trial_value = self.trial if self.trial is not None else 0
        self.step_helper.handle_test_train_step(
            was_allocated=was_allocated, path_length=path_length, trial=trial_value, req_id=req_id
        )
        self.rl_help_obj.update_snapshots()

        # Safely get reward values
        if self.engine_obj is not None:
            if was_allocated:
                reward = self.engine_obj.engine_props["reward"]
            else:
                reward = self.engine_obj.engine_props["penalty"]
        else:
            reward = 0.0

        # Safely increment arrival_count
        current_count = getattr(self.rl_props, "arrival_count", 0)
        self.rl_props.arrival_count = current_count + 1  # type: ignore[attr-defined]
        terminated = self.step_helper.check_terminated()
        new_obs = self.step_helper.get_obs(
            bandwidth=str(bandwidth), holding_time=holding_time
        )
        truncated = False
        info = self._get_info()

        return new_obs, reward, terminated, truncated, info

    @staticmethod
    def _get_info() -> dict[str, Any]:
        """
        Return simulation metadata information.

        Currently returns an empty dictionary. Can be extended to include
        simulation statistics, debug information, or other metadata.

        :return: Dictionary containing simulation metadata
        :rtype: dict[str, Any]
        """
        return {}

    def setup(self) -> None:
        """
        Sets up this class.
        """
        if self.sim_dict["optimize"] or self.sim_dict["optimize_hyperparameters"]:
            self.optimize = True
        else:
            self.optimize = False

        self.rl_props.k_paths = self.sim_dict["k_paths"]
        self.rl_props.cores_per_link = self.sim_dict["cores_per_link"]
        # Currently only C-band is supported - see SUPPORTED_SPECTRAL_BANDS
        self.rl_props.spectral_slots = self.sim_dict["c_band"]

        self._create_input()

        self.sim_dict["arrival_dict"] = {
            ARRIVAL_DICT_KEYS["start"]: self.sim_dict["erlang_start"],
            ARRIVAL_DICT_KEYS["stop"]: self.sim_dict["erlang_stop"],
            ARRIVAL_DICT_KEYS["step"]: self.sim_dict["erlang_step"],
        }

        # Ensure engine_obj is available before accessing its properties
        if self.engine_obj is not None:
            self.engine_obj.engine_props["erlang"] = float(
                self.sim_dict["erlang_start"]
            )
            cores_per_link = self.engine_obj.engine_props["cores_per_link"]
            erlang = self.engine_obj.engine_props["erlang"]
            holding_time = self.engine_obj.engine_props["holding_time"]
            self.engine_obj.engine_props["arrival_rate"] = (
                cores_per_link * erlang
            ) / holding_time
            self.engine_obj.engine_props["band_list"] = SUPPORTED_SPECTRAL_BANDS

    def _setup_agents(self) -> None:
        """
        Initialize RL agents for the simulation.

        Creates the path agent using the specified algorithm and sets up
        placeholders for core and spectrum agents (currently unused).
        """
        self.path_agent = PathAgent(
            path_algorithm=self.sim_dict["path_algorithm"],
            rl_props=self.rl_props,
            rl_help_obj=self.rl_help_obj,
        )
        self.core_agent = None
        self.spectrum_agent = None
