import os
from typing import Any

import numpy as np

from fusion.modules.rl.algorithms.a2c import A2C
from fusion.modules.rl.algorithms.bandits import EpsilonGreedyBandit, UCBBandit
from fusion.modules.rl.algorithms.dqn import DQN
from fusion.modules.rl.algorithms.ppo import PPO
from fusion.modules.rl.algorithms.q_learning import QLearning
from fusion.modules.rl.algorithms.qr_dqn import QrDQN
from fusion.modules.rl.errors import AlgorithmNotFoundError
from fusion.modules.rl.utils.hyperparams import HyperparamConfig

# Type alias for algorithm objects
AlgorithmType = QLearning | EpsilonGreedyBandit | UCBBandit | PPO | A2C | DQN | QrDQN


class BaseAgent:
    """
    Base agent class for path, core, and spectrum RL agents.

    Provides common functionality for reinforcement learning agents including
    algorithm initialization, reward calculation, and model loading.
    """

    def __init__(self, algorithm: str, rl_props: Any, rl_help_obj: Any) -> None:
        """
        Common initializer for all agents.

        :param algorithm: RL algorithm to use (e.g., 'q_learning', 'ppo')
        :type algorithm: str
        :param rl_props: Reinforcement learning properties object
        :type rl_props: Any
        :param rl_help_obj: RL helper object for utility functions
        :type rl_help_obj: Any
        """
        self.algorithm: str = algorithm
        self.rl_props: Any = rl_props
        self.rl_help_obj: Any = rl_help_obj
        self.algorithm_obj: Any | None = None  # Will be one of the RL algorithms
        self.engine_props: dict[str, Any] | None = None

        self.reward_penalty_list: np.ndarray | None = None
        self.hyperparam_obj: HyperparamConfig | None = None

    def setup_env(self, is_path: bool) -> None:
        """
        Set up the environment for core or path agents.

        :param is_path: Whether this is a path agent (True) or core agent (False)
        :type is_path: bool
        :raises AlgorithmNotFoundError: If the algorithm is not supported
        """
        if self.engine_props is None:
            raise ValueError("engine_props must be set before calling setup_env")

        self.reward_penalty_list = np.zeros(self.engine_props['max_iters'])
        self.hyperparam_obj = HyperparamConfig(
            engine_props=self.engine_props, rl_props=self.rl_props, is_path=True
        )

        if self.algorithm == 'q_learning':
            self.algorithm_obj = QLearning(
                rl_props=self.rl_props, engine_props=self.engine_props
            )
        elif self.algorithm == 'epsilon_greedy_bandit':
            self.algorithm_obj = EpsilonGreedyBandit(
                rl_props=self.rl_props,
                engine_props=self.engine_props,
                is_path=is_path
            )
        elif self.algorithm == 'ucb_bandit':
            self.algorithm_obj = UCBBandit(
                rl_props=self.rl_props, engine_props=self.engine_props, is_path=is_path
            )
        elif self.algorithm == 'ppo':
            self.algorithm_obj = PPO(
                rl_props=self.rl_props, engine_obj=self.engine_props
            )
        elif self.algorithm == 'a2c':
            self.algorithm_obj = A2C(
                rl_props=self.rl_props, engine_obj=self.engine_props
            )
        elif self.algorithm == 'dqn':
            self.algorithm_obj = DQN(
                rl_props=self.rl_props, engine_obj=self.engine_props
            )
        elif self.algorithm == 'qr_dqn':
            self.algorithm_obj = QrDQN(
                rl_props=self.rl_props, engine_obj=self.engine_props
            )
        else:
            raise AlgorithmNotFoundError(
                f"Algorithm '{self.algorithm}' is not supported. "
                f"Supported algorithms: q_learning, epsilon_greedy_bandit, "
                f"ucb_bandit, ppo, a2c, dqn, qr_dqn"
            )

    def calculate_dynamic_penalty(self, core_index: float, req_id: float) -> float:
        """
        Calculate dynamic penalty based on core and request indices.

        :param core_index: Index of the core being used
        :type core_index: float
        :param req_id: Request identifier
        :type req_id: float
        :return: Calculated dynamic penalty value
        :rtype: float
        """
        if self.engine_props is None:
            raise ValueError("engine_props must be set before calculating penalty")
        penalty_factor = 1 + self.engine_props['gamma'] * core_index / req_id
        return float(self.engine_props['penalty'] * penalty_factor)

    def calculate_dynamic_reward(self, core_index: float, req_id: float) -> float:
        """
        Calculate dynamic reward based on core and request indices.

        :param core_index: Index of the core being used
        :type core_index: float
        :param req_id: Request identifier
        :type req_id: float
        :return: Calculated dynamic reward value
        :rtype: float
        """
        if self.engine_props is None:
            raise ValueError("engine_props must be set before calculating reward")
        decay_factor = 1 + self.engine_props['decay_factor'] * core_index
        core_decay = self.engine_props['reward'] / decay_factor
        request_ratio = (
            (self.engine_props['num_requests'] - req_id) /
            self.engine_props['num_requests']
        )
        request_weight = request_ratio ** self.engine_props['core_beta']
        return float(core_decay * request_weight)

    def get_reward(
        self,
        was_allocated: bool,
        dynamic: bool,
        core_index: float | None,
        req_id: float | None
    ) -> float:
        """
        Calculate reward based on allocation success and dynamic settings.

        :param was_allocated: Whether the request was successfully allocated
        :type was_allocated: bool
        :param dynamic: Whether to use dynamic reward/penalty calculation
        :type dynamic: bool
        :param core_index: Index of the core (for dynamic calculation)
        :type core_index: Optional[float]
        :param req_id: Request identifier (for dynamic calculation)
        :type req_id: Optional[float]
        :return: Calculated reward or penalty value
        :rtype: float
        """
        if self.engine_props is None:
            raise ValueError("engine_props must be set before calculating reward")

        if was_allocated:
            if dynamic:
                if core_index is None or req_id is None:
                    raise ValueError(
                        "core_index and req_id must be provided for dynamic "
                        "reward calculation"
                    )
                return self.calculate_dynamic_reward(core_index, req_id)

            return float(self.engine_props['reward'])

        if dynamic:
            if core_index is None or req_id is None:
                raise ValueError(
                    "core_index and req_id must be provided for dynamic "
                    "penalty calculation"
                )
            return self.calculate_dynamic_penalty(core_index, req_id)

        return float(self.engine_props['penalty'])

    def load_model(self, model_path: str, file_prefix: str, **kwargs: Any) -> None:
        """
        Load a previously trained model from disk.

        :param model_path: Path to the directory containing the model
        :type model_path: str
        :param file_prefix: Prefix for the model file name
        :type file_prefix: str
        :param kwargs: Additional parameters (is_path, erlang, num_cores)
        :type kwargs: dict
        """
        self.setup_env(is_path=kwargs.get('is_path', False))
        if self.algorithm == 'q_learning':
            # Construct model file path
            model_file = f"{file_prefix}_e{kwargs['erlang']}_c{kwargs['num_cores']}.npy"
            full_path = os.path.join('logs', model_path, model_file)
            if self.algorithm_obj is None:
                raise ValueError(
                    "algorithm_obj must be initialized before loading model"
                )
            # Type narrowing: for q_learning, algorithm_obj is QLearning
            if hasattr(self.algorithm_obj, 'props'):
                self.algorithm_obj.props.cores_matrix = np.load(
                    full_path, allow_pickle=True
                )
