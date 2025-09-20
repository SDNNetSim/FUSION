import os
from typing import Optional
import numpy as np

from fusion.modules.rl.utils.hyperparams import HyperparamConfig
from fusion.modules.rl.errors import AlgorithmNotFoundError

from fusion.modules.rl.algorithms.q_learning import QLearning
from fusion.modules.rl.algorithms.bandits import EpsilonGreedyBandit, UCBBandit
from fusion.modules.rl.algorithms.ppo import PPO
from fusion.modules.rl.algorithms.a2c import A2C
from fusion.modules.rl.algorithms.dqn import DQN
from fusion.modules.rl.algorithms.qr_dqn import QrDQN


class BaseAgent:
    """
    A base agent to be used for path, core, and spectrum agents.
    """

    def __init__(self, algorithm: str, rl_props: object, rl_help_obj: object) -> None:
        """
        Common initializer for all agents.
        """
        self.algorithm = algorithm
        self.rl_props = rl_props
        self.rl_help_obj = rl_help_obj
        self.algorithm_obj = None
        self.engine_props = None

        self.reward_penalty_list = None
        self.hyperparam_obj = None

    def setup_env(self, is_path: bool) -> None:
        """
        Sets up the environment for both core or path agents, depending on the algorithm.
        """
        self.reward_penalty_list = np.zeros(self.engine_props['max_iters'])
        self.hyperparam_obj = HyperparamConfig(engine_props=self.engine_props, rl_props=self.rl_props, is_path=True)

        if self.algorithm == 'q_learning':
            self.algorithm_obj = QLearning(rl_props=self.rl_props, engine_props=self.engine_props)
        elif self.algorithm == 'epsilon_greedy_bandit':
            self.algorithm_obj = EpsilonGreedyBandit(rl_props=self.rl_props, engine_props=self.engine_props,
                                                     is_path=is_path)
        elif self.algorithm == 'ucb_bandit':
            self.algorithm_obj = UCBBandit(rl_props=self.rl_props, engine_props=self.engine_props, is_path=is_path)
        elif self.algorithm == 'ppo':
            self.algorithm_obj = PPO(rl_props=self.rl_props, engine_obj=self.engine_props)
        elif self.algorithm == 'a2c':
            self.algorithm_obj = A2C(rl_props=self.rl_props, engine_obj=self.engine_props)
        elif self.algorithm == 'dqn':
            self.algorithm_obj = DQN(rl_props=self.rl_props, engine_obj=self.engine_props)
        elif self.algorithm == 'qr_dqn':
            self.algorithm_obj = QrDQN(rl_props=self.rl_props, engine_obj=self.engine_props)
        else:
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.algorithm}' is not supported. "
                f"Supported algorithms: q_learning, epsilon_greedy_bandit, ucb_bandit, ppo, a2c, dqn, qr_dqn"
            )

    def calculate_dynamic_penalty(self, core_index: float, req_id: float) -> float:
        """
        Calculate a dynamic penalty after every action.
        """
        return self.engine_props['penalty'] * (1 + self.engine_props['gamma'] * core_index / req_id)

    def calculate_dynamic_reward(self, core_index: float, req_id: float) -> float:
        """
        Calculates a dynamic reward after every action.
        """
        core_decay = self.engine_props['reward'] / (1 + self.engine_props['decay_factor'] * core_index)
        request_weight = ((self.engine_props['num_requests'] - req_id) /
                          self.engine_props['num_requests']) ** self.engine_props['core_beta']
        return core_decay * request_weight

    def get_reward(self, was_allocated: bool, dynamic: bool, core_index: Optional[float], req_id: Optional[float]) -> float:
        """
        Generalized reward calculation for both path and core agents.
        """
        if was_allocated:
            if dynamic:
                return self.calculate_dynamic_reward(core_index, req_id)

            return self.engine_props['reward']

        if dynamic:
            return self.calculate_dynamic_penalty(core_index, req_id)

        return self.engine_props['penalty']

    def load_model(self, model_path: str, file_prefix: str, **kwargs) -> None:
        """
        Loads a previously-trained model for either a core or path agent.
        """
        self.setup_env(is_path=kwargs.get('is_path', False))
        if self.algorithm == 'q_learning':
            # Assumes similar directory logic
            model_path = os.path.join('logs', model_path,
                                      f"{file_prefix}_e{kwargs['erlang']}_c{kwargs['num_cores']}.npy")
            self.algorithm_obj.props.cores_matrix = np.load(model_path, allow_pickle=True)
