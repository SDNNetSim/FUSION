# pylint: disable=duplicate-code
# TODO: (version 5.5-6) Address all duplicate code if you can

from gymnasium import spaces

from fusion.modules.rl.utils.observation_space import get_observation_space


class A2C:
    """
    Facilitates Advantage Actor-Critic (A2C) for reinforcement learning.

    This class provides functionalities for handling observation space and action
    space specific to the A2C framework for reinforcement learning. It is driven by
    the properties passed during initialization to define the behavior and attributes
    of the RL environment and its engine.
    """

    def __init__(self, rl_props: object, engine_obj: object):
        """
        A2C initialization function.

        :param rl_props: Object containing reinforcement learning-specific properties.
        :param engine_obj: Object containing engine-specific properties for the environment.
        """
        self.rl_props = rl_props
        self.engine_obj = engine_obj

    def get_obs_space(self):
        """
        Gets the observation space for the a2c reinforcement learning framework.
        """
        obs_space_dict = get_observation_space(rl_props=self.rl_props, engine_obj=self.engine_obj)

        return spaces.Dict(obs_space_dict)

    def get_action_space(self):
        """
        Gets the action space for the A2C-based environment.

        By default, we use a discrete action space, where the number of actions corresponds
        to valid paths or nodes within the environment.

        :return: An action space object compatible with Gymnasium.
        """
        action_space = spaces.Discrete(self.engine_obj.engine_props['k_paths'])
        return action_space
