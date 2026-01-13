"""
Abstract base class for reinforcement learning agents in FUSION.
"""

from abc import ABC, abstractmethod
from typing import Any


class AgentInterface(ABC):
    """
    Base interface for all reinforcement learning agents in FUSION.

    This interface defines the contract that all RL agents must follow
    to ensure compatibility with the FUSION simulation framework.
    """

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """
        Return the name of the RL algorithm.

        :return: String identifier for this RL algorithm
        :rtype: str
        """

    @property
    @abstractmethod
    def action_space_type(self) -> str:
        """
         Return the type of action space.

        :return: 'discrete' or 'continuous'
        :rtype: str
        """

    @property
    @abstractmethod
    def observation_space_shape(self) -> tuple[int, ...]:
        """
        Return the shape of the observation space.

        :return: Tuple describing observation dimensions
        :rtype: Tuple[int, ...]
        """

    @abstractmethod
    def act(self, observation: Any, _deterministic: bool = False) -> int | Any:
        """
        Select an action based on the current observation.

        :param observation: Current environment observation
        :type observation: Any
        :param deterministic: If True, select action deterministically (no exploration)
        :type deterministic: bool
        :return: Action to take (int for discrete, array for continuous)
        :rtype: Union[int, Any]
        """

    @abstractmethod
    def train(self, env: Any, _total_timesteps: int, **kwargs: Any) -> dict[str, Any]:
        """
        Train the agent on the given environment.

        :param env: Training environment (e.g., Gym environment)
        :type env: Any
        :param total_timesteps: Total number of timesteps to train for
        :type total_timesteps: int
        :param kwargs: Additional training parameters
        :type kwargs: dict
        :return: Dictionary containing training metrics and results
        :rtype: Dict[str, Any]
        """

    @abstractmethod
    def learn_from_experience(
        self,
        observation: Any,
        action: int | Any,
        reward: float,
        _next_observation: Any,
        done: bool,
    ) -> dict[str, float] | None:
        """
        Learn from a single experience tuple.

        :param observation: Current observation
        :type observation: Any
        :param action: Action taken
        :type action: Union[int, Any]
        :param reward: Reward received
        :type reward: float
        :param next_observation: Resulting observation
        :type next_observation: Any
        :param done: Whether episode terminated
        :type done: bool
        :return: Optional dictionary containing learning metrics (e.g., loss values)
        :rtype: Optional[Dict[str, float]]
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the agent's model/parameters to disk.

        :param path: Path where to save the model
        :type path: str
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the agent's model/parameters from disk.

        :param path: Path from where to load the model
        :type path: str
        """

    @abstractmethod
    def get_reward(
        self,
        state: dict[str, Any],
        action: int | Any,
        _next_state: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        """
        Calculate reward for a state-action-next_state transition.

        :param state: Current state information
        :type state: Dict[str, Any]
        :param action: Action taken
        :type action: Union[int, Any]
        :param next_state: Resulting state information
        :type next_state: Dict[str, Any]
        :param info: Additional information from environment
        :type info: Dict[str, Any]
        :return: Calculated reward value
        :rtype: float
        """

    @abstractmethod
    def update_exploration_params(self, _timestep: int, _total_timesteps: int) -> None:
        """
        Update exploration parameters based on training progress.

        :param timestep: Current training timestep
        :type timestep: int
        :param total_timesteps: Total training timesteps
        :type total_timesteps: int
        """

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """
        Get agent configuration parameters.

        :return: Dictionary containing agent configuration
        :rtype: Dict[str, Any]
        """

    @abstractmethod
    def set_config(self, config: dict[str, Any]) -> None:
        """
        Set agent configuration parameters.

        :param config: Dictionary containing agent configuration
        :type config: Dict[str, Any]
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """
        Get agent performance metrics.

        :return: Dictionary containing agent-specific metrics
        :rtype: Dict[str, Any]
        """

    def reset(self) -> None:  # noqa: B027
        """
        Reset the agent's internal state.

        This method can be overridden by subclasses that maintain episode state.
        """
        pass

    def on_episode_start(self) -> None:  # noqa: B027
        """
        Called at the beginning of each episode.

        This method can be overridden by subclasses for episode initialization.
        """
        pass

    def on_episode_end(self) -> None:  # noqa: B027
        """
        Called at the end of each episode.

        This method can be overridden by subclasses for episode cleanup.
        """
        pass
