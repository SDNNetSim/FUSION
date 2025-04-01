import numpy as np

from gymnasium import spaces


class PPO:
    """
    Facilitates Proximal Policy Optimization (PPO) for reinforcement learning.

    This class provides functionalities for handling observation space, action space,
    and rewards specific to the PPO framework for reinforcement learning. It's driven
    by the properties passed during initialization to define the behavior and attributes
    of the reinforcement learning environment and its engine.
    """

    def __init__(self, rl_props: object, engine_obj: object):
        self.rl_props = rl_props
        self.engine_obj = engine_obj

    def get_obs_space(self):
        """
        Gets the observation space for the ppo reinforcement learning framework.
        """
        bw_set = {d["bandwidth"] for d in self.rl_props.arrival_list}
        resp_obs = spaces.Dict({
            'source': spaces.MultiBinary(self.rl_props.num_nodes),
            'destination': spaces.MultiBinary(self.rl_props.num_nodes),
            'request_bandwidth': spaces.MultiBinary(len(bw_set)),
            'holding_time': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            #'fragmentation': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            #'path_frag': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        return resp_obs

    def get_action_space(self):
        """
        Get the action space for the environment.
        """
        action_space = spaces.Discrete(self.engine_obj.engine_props['k_paths'])
        return action_space
