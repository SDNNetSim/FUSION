from gymnasium import spaces

from reinforcement_learning.utils.observation_space import get_observation_space


class QrDQN:
    """
    Facilitates DQN for reinforcement learning.
    """

    def __init__(self, rl_props: object, engine_obj: object):
        self.rl_props = rl_props
        self.engine_obj = engine_obj

    def get_obs_space(self):
        """
        Gets the observation space for the ppo reinforcement learning framework.
        """
        obs_space_dict = get_observation_space(rl_props=self.rl_props, engine_obj=self.engine_obj)

        return spaces.Dict(obs_space_dict)

    def get_action_space(self):
        """
        Get the action space for the environment.
        """
        action_space = spaces.Discrete(self.engine_obj.engine_props['k_paths'])
        return action_space
