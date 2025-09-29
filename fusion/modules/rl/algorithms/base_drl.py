"""Base class for Deep Reinforcement Learning algorithms."""

from typing import Any

from gymnasium import spaces

from fusion.modules.rl.utils.observation_space import get_observation_space


class BaseDRLAlgorithm:
    """
    Base class for Deep Reinforcement Learning algorithms.

    Provides common functionality for observation space and action space
    handling across different DRL frameworks (PPO, A2C, DQN, QR-DQN).
    """

    def __init__(self, rl_props: Any, engine_obj: Any) -> None:
        """
        Initialize the DRL algorithm.

        :param rl_props: Object containing reinforcement learning-specific properties
        :type rl_props: Any
        :param engine_obj: Object containing engine-specific properties for env
        :type engine_obj: Any
        """
        self.rl_props = rl_props
        self.engine_obj = engine_obj

    def get_obs_space(self) -> spaces.Dict:
        """
        Get the observation space for the reinforcement learning framework.

        Creates a dictionary-based observation space using the configured
        RL and engine properties.

        :return: Dictionary observation space compatible with Gymnasium
        :rtype: spaces.Dict
        """
        obs_space_dict = get_observation_space(
            rl_props=self.rl_props, engine_props=self.engine_obj
        )
        return spaces.Dict(obs_space_dict)

    def get_action_space(self) -> spaces.Discrete:
        """
        Get the action space for the DRL environment.

        By default, uses a discrete action space where the number of actions
        corresponds to the number of valid paths (k_paths) in the environment.

        :return: Discrete action space object compatible with Gymnasium
        :rtype: spaces.Discrete
        """
        action_space = spaces.Discrete(self.engine_obj.engine_props['k_paths'])
        return action_space
