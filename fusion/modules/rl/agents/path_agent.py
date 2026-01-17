from typing import TYPE_CHECKING, Any

import numpy as np

from fusion.modules.rl.agents.base_agent import BaseAgent
from fusion.modules.rl.args.general_args import (
    EPISODIC_STRATEGIES,
    VALID_DRL_ALGORITHMS,
)
from fusion.modules.rl.errors import AgentError, InvalidActionError, RouteSelectionError

if TYPE_CHECKING:
    pass


class PathAgent(BaseAgent):
    """
    Path agent for intelligent route selection in RL simulations.

    Handles path assignment using various RL algorithms including Q-learning,
    bandits, and deep RL methods (PPO, A2C, DQN).
    """

    def __init__(self, path_algorithm: str, rl_props: Any, rl_help_obj: Any) -> None:
        super().__init__(path_algorithm, rl_props, rl_help_obj)

        self.iteration: int | None = None
        self.hyperparam_obj: Any | None = None  # HyperparamConfig
        self.reward_penalty_list: np.ndarray | None = None
        self.level_index: int | None = None
        self.congestion_list: list[Any] | None = None

        self.state_action_pair: tuple[Any, Any] | None = None
        self.action_index: int | None = None

    def _ensure_initialized(self) -> None:
        """
        Ensure all required attributes are initialized.

        :raises ValueError: If any required attribute is not initialized
        """
        if self.engine_props is None:
            raise ValueError("engine_props must be set before using PathAgent")
        if self.hyperparam_obj is None:
            raise ValueError(
                "hyperparam_obj must be initialized - call setup_env first"
            )
        if self.reward_penalty_list is None:
            raise ValueError(
                "reward_penalty_list must be initialized - call setup_env first"
            )
        if self.algorithm_obj is None:
            raise ValueError("algorithm_obj must be initialized - call setup_env first")

    def end_iter(self) -> None:
        """
        End current iteration and update hyperparameters.

        Updates iteration counter and episodic hyperparameters (alpha, epsilon)
        based on the configured strategy.
        """
        self._ensure_initialized()
        assert self.hyperparam_obj is not None  # For mypy
        assert self.engine_props is not None  # For mypy

        self.hyperparam_obj.iteration += 1
        if self.hyperparam_obj.alpha_strategy in EPISODIC_STRATEGIES:
            if "bandit" not in self.engine_props["path_algorithm"]:
                self.hyperparam_obj.update_alpha()
        if self.hyperparam_obj.epsilon_strategy in EPISODIC_STRATEGIES:
            if "ucb" not in self.engine_props["path_algorithm"]:
                self.hyperparam_obj.update_eps()

    def _handle_hyperparams(self) -> None:
        """Update hyperparameters based on current timestep and strategy."""
        # Validation already done in calling method
        assert self.hyperparam_obj is not None  # For mypy
        assert self.engine_props is not None  # For mypy

        if not self.hyperparam_obj.fully_episodic:
            self.state_action_pair = (self.rl_props.source, self.rl_props.destination)
            self.action_index = self.rl_props.chosen_path_index
            self.hyperparam_obj.update_timestep_data(
                state_action_pair=self.state_action_pair, action_index=self.action_index
            )
        if self.hyperparam_obj.alpha_strategy not in EPISODIC_STRATEGIES:
            if "bandit" not in self.engine_props["path_algorithm"]:
                self.hyperparam_obj.update_alpha()
        if self.hyperparam_obj.epsilon_strategy not in EPISODIC_STRATEGIES:
            self.hyperparam_obj.update_eps()

    def update(
        self,
        was_allocated: bool,
        network_spectrum_dict: dict[str, Any],
        iteration: int,
        path_length: int,
        trial: int,
    ) -> None:  # pylint: disable=unused-argument
        """
        Update agent state based on action outcome.

        :param was_allocated: Whether the request was successfully allocated
        :type was_allocated: bool
        :param network_spectrum_dict: Current network spectrum database
        :type network_spectrum_dict: dict
        :param iteration: Current iteration number
        :type iteration: int
        :param path_length: Length of the selected path
        :type path_length: int
        :param trial: Current trial number
        :type trial: int
        :raises AgentError: If iteration exceeds maximum allowed
        :raises InvalidActionError: If algorithm is not supported
        """
        # StableBaselines3 will handle all algorithm updates
        if self.algorithm in VALID_DRL_ALGORITHMS:
            return

        self._ensure_initialized()
        assert self.hyperparam_obj is not None  # For mypy
        assert self.engine_props is not None
        assert self.reward_penalty_list is not None
        assert self.algorithm_obj is not None

        if self.hyperparam_obj.iteration >= self.engine_props["max_iters"]:
            raise AgentError(
                f"Iteration {self.hyperparam_obj.iteration} exceeds maximum "
                f"allowed iterations {self.engine_props['max_iters']}. "
                f"Check iteration management logic."
            )

        reward = self.get_reward(
            was_allocated=was_allocated,
            dynamic=self.engine_props["dynamic_reward"],
            core_index=None,
            req_id=None,
        )
        self.reward_penalty_list[self.hyperparam_obj.iteration] += reward
        self.hyperparam_obj.current_reward = reward
        self.iteration = iteration

        if self.algorithm_obj is None:
            raise ValueError("algorithm_obj must be initialized")

        # Set common attributes if they exist
        if hasattr(self.algorithm_obj, "learn_rate"):
            self.algorithm_obj.learn_rate = self.hyperparam_obj.current_alpha
        if hasattr(self.algorithm_obj, "iteration"):
            self.algorithm_obj.iteration = iteration

        self._handle_hyperparams()
        if self.algorithm == "q_learning":
            if hasattr(self.algorithm_obj, "learn_rate"):
                self.algorithm_obj.learn_rate = self.hyperparam_obj.current_alpha
            if hasattr(self.algorithm_obj, "update_q_matrix"):
                self.algorithm_obj.update_q_matrix(
                    reward=reward,
                    level_index=self.level_index,
                    network_spectrum_dict=network_spectrum_dict,
                    flag="path",
                    trial=trial,
                    iteration=iteration,
                )
        elif self.algorithm == "epsilon_greedy_bandit":
            if hasattr(self.algorithm_obj, "update"):
                self.algorithm_obj.update(
                    reward=reward,
                    arm=self.rl_props.chosen_path_index,
                    iteration=iteration,
                    trial=trial,
                )
        elif self.algorithm == "ucb_bandit":
            if hasattr(self.algorithm_obj, "update"):
                self.algorithm_obj.update(
                    reward=reward,
                    arm=self.rl_props.chosen_path_index,
                    iteration=iteration,
                    trial=trial,
                )
        else:
            raise InvalidActionError(
                f"Algorithm '{self.algorithm}' is not supported for agent "
                f"updates. Supported algorithms: q_learning, "
                f"epsilon_greedy_bandit, ucb_bandit, or DRL algorithms."
            )

    def _select_ql_route(self, random_float: float) -> None:
        """
        Select route using epsilon-greedy strategy for Q-learning.

        :param random_float: Random value for epsilon-greedy decision
        :type random_float: float
        """
        assert self.hyperparam_obj is not None  # Validated by caller
        assert self.congestion_list is not None

        if random_float < self.hyperparam_obj.current_epsilon:
            self.rl_props.chosen_path_index = np.random.choice(self.rl_props.k_paths)
            # The level will always be the last index
            self.level_index = self.congestion_list[self.rl_props.chosen_path_index][-1]

            if self.rl_props.chosen_path_index == 1 and self.rl_props.k_paths == 1:
                self.rl_props.chosen_path_index = 0
            self.rl_props.chosen_path_list = self.rl_props.paths_list[
                self.rl_props.chosen_path_index
            ]
        else:
            if self.algorithm_obj is None:
                raise ValueError("algorithm_obj must be initialized")
            # For Q-learning, we know algorithm_obj has get_max_curr_q method
            if hasattr(self.algorithm_obj, "get_max_curr_q"):
                result = self.algorithm_obj.get_max_curr_q(
                    cong_list=self.congestion_list, matrix_flag="routes_matrix"
                )
                self.rl_props.chosen_path_index, self.rl_props.chosen_path_list = result
            else:
                raise ValueError(
                    f"Algorithm {self.algorithm} does not have get_max_curr_q method"
                )
            self.level_index = self.congestion_list[self.rl_props.chosen_path_index][-1]

    def _ql_route(self) -> None:
        """
        Execute Q-learning route selection process.

        :raises RouteSelectionError: If no valid path is selected
        """
        random_float = float(np.round(np.random.uniform(0, 1), decimals=1))
        if self.algorithm_obj is None:
            raise ValueError("algorithm_obj must be initialized")
        # For Q-learning, we know algorithm_obj has props attribute
        if hasattr(self.algorithm_obj, "props"):
            props_matrix = self.algorithm_obj.props.routes_matrix
            routes_matrix = props_matrix[
                self.rl_props.source, self.rl_props.destination
            ]["path"]
            self.rl_props.paths_list = routes_matrix
        else:
            raise ValueError(
                f"Algorithm {self.algorithm} does not have props attribute"
            )

        self.congestion_list = self.rl_help_obj.classify_paths(
            paths_list=self.rl_props.paths_list
        )
        if self.rl_props.paths_list.ndim != 1:
            self.rl_props.paths_list = self.rl_props.paths_list[:, 0]

        self._select_ql_route(random_float=random_float)

        if len(self.rl_props.chosen_path_list) == 0:
            raise RouteSelectionError(
                f"Failed to select a valid path for source "
                f"{self.rl_props.source} to destination "
                f"{self.rl_props.destination}. "
                f"Available paths: {len(self.rl_props.paths_list)}. "
                f"Check path availability and routing logic."
            )

    def _bandit_route(self, route_obj: Any) -> None:
        """
        Select route using bandit algorithm.

        :param route_obj: Route object containing path information
        :type route_obj: object
        """
        paths_list = route_obj.route_props.paths_matrix
        source = paths_list[0][0]
        dest = paths_list[0][-1]

        assert self.algorithm_obj is not None  # Validated by setup_env
        assert self.hyperparam_obj is not None  # Validated by setup_env

        if hasattr(self.algorithm_obj, "epsilon"):
            self.algorithm_obj.epsilon = self.hyperparam_obj.current_epsilon
        if hasattr(self.algorithm_obj, "select_path_arm"):
            self.rl_props.chosen_path_index = self.algorithm_obj.select_path_arm(
                source=int(source), dest=int(dest)
            )
        else:
            raise ValueError(
                f"Algorithm {self.algorithm} does not have select_path_arm method"
            )
        paths_matrix = route_obj.route_props.paths_matrix
        self.rl_props.chosen_path_list = paths_matrix[self.rl_props.chosen_path_index]

    def _drl_route(self, route_obj: Any, action: int) -> None:
        """
        Select route using deep RL action.

        :param route_obj: Route object containing path information
        :type route_obj: object
        :param action: Action selected by DRL algorithm
        :type action: int
        :raises InvalidActionError: If algorithm is not a supported DRL algorithm
        """
        if self.algorithm in ("ppo", "a2c", "dqn", "qr_dqn"):
            self.rl_props.chosen_path_index = action
            self.rl_props.chosen_path_list = route_obj.route_props.paths_matrix[action]
        else:
            raise InvalidActionError(
                f"Algorithm '{self.algorithm}' is not supported for DRL routing. "
                f"Supported DRL algorithms: ppo, a2c, dqn, qr_dqn"
            )

    def get_route(self, **kwargs: Any) -> None:
        """
        Select a route for the current request using the configured algorithm.

        :param kwargs: Additional parameters (route_obj for bandits, action for DRL)
        :type kwargs: dict
        :raises InvalidActionError: If algorithm is not supported
        :raises RouteSelectionError: If no valid path can be selected
        """
        if self.algorithm == "q_learning":
            self._ql_route()
        elif self.algorithm in (
            "epsilon_greedy_bandit",
            "thompson_sampling_bandit",
            "ucb_bandit",
        ):
            self._bandit_route(route_obj=kwargs["route_obj"])
        elif self.algorithm in ("ppo", "a2c", "dqn", "qr_dqn"):
            self._drl_route(route_obj=kwargs["route_obj"], action=kwargs["action"])
        else:
            raise InvalidActionError(
                f"Algorithm '{self.algorithm}' is not supported for routing. "
                f"Supported algorithms: q_learning, epsilon_greedy_bandit, "
                f"thompson_sampling_bandit, ucb_bandit, ppo, a2c, dqn, qr_dqn"
            )
