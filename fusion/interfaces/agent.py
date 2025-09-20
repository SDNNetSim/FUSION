"""
Abstract base class for reinforcement learning agents in FUSION.
"""

# pylint: disable=duplicate-code

from abc import ABC, abstractmethod
from typing import Any


class AgentInterface(ABC):
    """Base interface for all reinforcement learning agents in FUSION.

    This interface defines the contract that all RL agents must follow
    to ensure compatibility with the FUSION simulation framework.
    """

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the name of the RL algorithm.

        Returns:
            String identifier for this RL algorithm
        """

    @property
    @abstractmethod
    def action_space_type(self) -> str:
        """Return the type of action space.

        Returns:
            'discrete' or 'continuous'
        """

    @property
    @abstractmethod
    def observation_space_shape(self) -> tuple[int, ...]:
        """Return the shape of the observation space.

        Returns:
            Tuple describing observation dimensions
        """

    @abstractmethod
    def act(self, observation: Any, deterministic: bool = False) -> int | Any:
        """Select an action based on the current observation.

        Args:
            observation: Current environment observation
            deterministic: If True, select action deterministically (no exploration)

        Returns:
            Action to take (int for discrete, array for continuous)
        """

    @abstractmethod
    def train(self, env: Any, total_timesteps: int, **kwargs) -> dict[str, Any]:
        """Train the agent on the given environment.

        Args:
            env: Training environment (e.g., Gym environment)
            total_timesteps: Total number of timesteps to train for
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training metrics and results
        """

    @abstractmethod
    def learn_from_experience(
        self,
        observation: Any,
        action: int | Any,
        reward: float,
        next_observation: Any,
        done: bool,
    ) -> dict[str, float] | None:
        """Learn from a single experience tuple.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Resulting observation
            done: Whether episode terminated

        Returns:
            Optional dictionary containing learning metrics (e.g., loss values)
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent's model/parameters to disk.

        Args:
            path: Path where to save the model
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the agent's model/parameters from disk.

        Args:
            path: Path from where to load the model
        """

    @abstractmethod
    def get_reward(
        self,
        state: dict[str, Any],
        action: int | Any,
        next_state: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        """Calculate reward for a state-action-next_state transition.

        Args:
            state: Current state information
            action: Action taken
            next_state: Resulting state information
            info: Additional information from environment

        Returns:
            Calculated reward value
        """

    @abstractmethod
    def update_exploration_params(self, timestep: int, total_timesteps: int) -> None:
        """Update exploration parameters based on training progress.

        Args:
            timestep: Current training timestep
            total_timesteps: Total training timesteps
        """

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Get agent configuration parameters.

        Returns:
            Dictionary containing agent configuration
        """

    @abstractmethod
    def set_config(self, config: dict[str, Any]) -> None:
        """Set agent configuration parameters.

        Args:
            config: Dictionary containing agent configuration
        """

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get agent performance metrics.

        Returns:
            Dictionary containing agent-specific metrics
        """
