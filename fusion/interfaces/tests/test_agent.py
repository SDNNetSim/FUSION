"""
Unit tests for fusion.interfaces.agent module.

Tests the AgentInterface abstract base class for RL agents.
"""

import inspect
from typing import Any

import pytest

from fusion.interfaces.agent import AgentInterface

# ============================================================================
# Test Abstract Interface Instantiation
# ============================================================================


class TestAgentInterfaceInstantiation:
    """Tests that AgentInterface cannot be directly instantiated."""

    def test_agent_interface_cannot_be_instantiated(self) -> None:
        """Test that AgentInterface cannot be directly instantiated."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError):
            AgentInterface()  # type: ignore[abstract]


# ============================================================================
# Test Abstract Methods
# ============================================================================


class TestAgentInterfaceAbstractMethods:
    """Tests that required methods are marked as abstract."""

    def test_agent_interface_has_correct_abstract_methods(self) -> None:
        """Test that AgentInterface has correct abstract methods."""
        # Arrange
        expected_methods = {
            "algorithm_name",
            "action_space_type",
            "observation_space_shape",
            "act",
            "train",
            "learn_from_experience",
            "save",
            "load",
            "get_reward",
            "update_exploration_params",
            "get_config",
            "set_config",
            "get_metrics",
        }

        # Act
        abstract_methods = {
            method
            for method in dir(AgentInterface)
            if hasattr(getattr(AgentInterface, method), "__isabstractmethod__")
            and getattr(AgentInterface, method).__isabstractmethod__
        }

        # Assert
        assert abstract_methods == expected_methods


# ============================================================================
# Test Interface Method Signatures
# ============================================================================


class TestAgentInterfaceMethodSignatures:
    """Tests that AgentInterface method signatures are consistent."""

    def test_act_method_signature(self) -> None:
        """Test AgentInterface.act method signature."""
        # Arrange & Act
        sig = inspect.signature(AgentInterface.act)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "observation", "_deterministic"]
        assert sig.parameters["_deterministic"].default is False

    def test_train_method_signature(self) -> None:
        """Test AgentInterface.train method signature."""
        # Arrange & Act
        sig = inspect.signature(AgentInterface.train)

        # Assert
        assert "env" in sig.parameters
        assert "_total_timesteps" in sig.parameters
        assert "kwargs" in sig.parameters

    def test_learn_from_experience_method_signature(self) -> None:
        """Test AgentInterface.learn_from_experience method signature."""
        # Arrange & Act
        sig = inspect.signature(AgentInterface.learn_from_experience)
        params = list(sig.parameters.keys())

        # Assert
        assert params == [
            "self",
            "observation",
            "action",
            "reward",
            "_next_observation",
            "done",
        ]
        annotation_str = str(sig.return_annotation)
        assert "None" in annotation_str or "Optional" in annotation_str

    def test_get_reward_method_signature(self) -> None:
        """Test AgentInterface.get_reward method signature."""
        # Arrange & Act
        sig = inspect.signature(AgentInterface.get_reward)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "state", "action", "_next_state", "info"]
        assert sig.return_annotation is float

    def test_update_exploration_params_method_signature(self) -> None:
        """Test AgentInterface.update_exploration_params method signature."""
        # Arrange & Act
        sig = inspect.signature(AgentInterface.update_exploration_params)
        params = list(sig.parameters.keys())

        # Assert
        assert params == ["self", "_timestep", "_total_timesteps"]


# ============================================================================
# Test Required Methods
# ============================================================================


class TestAgentInterfaceRequiredMethods:
    """Tests that AgentInterface has all required methods."""

    def test_agent_interface_has_all_required_methods(self) -> None:
        """Test AgentInterface has all required methods."""
        # Arrange
        expected_methods = [
            "algorithm_name",
            "action_space_type",
            "observation_space_shape",
            "act",
            "train",
            "learn_from_experience",
            "save",
            "load",
            "get_reward",
            "update_exploration_params",
            "get_config",
            "set_config",
            "get_metrics",
            "reset",
            "on_episode_start",
            "on_episode_end",
        ]

        # Act & Assert
        for method in expected_methods:
            assert hasattr(AgentInterface, method)


# ============================================================================
# Test Concrete Implementation
# ============================================================================


class TestConcreteAgentImplementation:
    """Tests for concrete implementation of AgentInterface."""

    def test_concrete_agent_can_be_instantiated_with_all_methods(
        self,
    ) -> None:
        """Test concrete implementation with all methods can be instantiated."""

        # Arrange
        class ConcreteAgent(AgentInterface):
            @property
            def algorithm_name(self) -> str:
                return "test_agent"

            @property
            def action_space_type(self) -> str:
                return "discrete"

            @property
            def observation_space_shape(self) -> tuple[int, ...]:
                return (10,)

            def act(self, observation: Any, _deterministic: bool = False) -> int | Any:
                return 0

            def train(
                self, env: Any, _total_timesteps: int, **kwargs: Any
            ) -> dict[str, Any]:
                return {"loss": 0.5}

            def learn_from_experience(
                self,
                observation: Any,
                action: int | Any,
                reward: float,
                _next_observation: Any,
                done: bool,
            ) -> dict[str, float] | None:
                return {"loss": 0.1}

            def save(self, path: str) -> None:
                pass

            def load(self, path: str) -> None:
                pass

            def get_reward(
                self,
                state: dict[str, Any],
                action: int | Any,
                _next_state: dict[str, Any],
                info: dict[str, Any],
            ) -> float:
                return 1.0

            def update_exploration_params(
                self, _timestep: int, _total_timesteps: int
            ) -> None:
                pass

            def get_config(self) -> dict[str, Any]:
                return {}

            def set_config(self, config: dict[str, Any]) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        # Act
        agent = ConcreteAgent()

        # Assert
        assert agent.algorithm_name == "test_agent"
        assert agent.action_space_type == "discrete"
        assert agent.observation_space_shape == (10,)

    def test_concrete_agent_missing_abstract_method_cannot_be_instantiated(
        self,
    ) -> None:
        """Test concrete implementation missing abstract methods."""

        # Arrange
        class IncompleteAgent(AgentInterface):
            @property
            def algorithm_name(self) -> str:
                return "incomplete"

            # Missing other required abstract methods

        # Act & Assert
        with pytest.raises(TypeError):
            IncompleteAgent()  # type: ignore[abstract]


# ============================================================================
# Test Optional Methods
# ============================================================================


class TestAgentInterfaceOptionalMethods:
    """Tests for optional methods with default implementations."""

    def test_reset_method_has_default_implementation(self) -> None:
        """Test that reset method has default implementation."""

        # Arrange
        class MinimalAgent(AgentInterface):
            @property
            def algorithm_name(self) -> str:
                return "minimal"

            @property
            def action_space_type(self) -> str:
                return "discrete"

            @property
            def observation_space_shape(self) -> tuple[int, ...]:
                return (10,)

            def act(self, observation: Any, _deterministic: bool = False) -> int | Any:
                return 0

            def train(
                self, env: Any, _total_timesteps: int, **kwargs: Any
            ) -> dict[str, Any]:
                return {}

            def learn_from_experience(
                self,
                observation: Any,
                action: int | Any,
                reward: float,
                _next_observation: Any,
                done: bool,
            ) -> dict[str, float] | None:
                return None

            def save(self, path: str) -> None:
                pass

            def load(self, path: str) -> None:
                pass

            def get_reward(
                self,
                state: dict[str, Any],
                action: int | Any,
                _next_state: dict[str, Any],
                info: dict[str, Any],
            ) -> float:
                return 0.0

            def update_exploration_params(
                self, _timestep: int, _total_timesteps: int
            ) -> None:
                pass

            def get_config(self) -> dict[str, Any]:
                return {}

            def set_config(self, config: dict[str, Any]) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        # Act
        agent = MinimalAgent()
        agent.reset()

        # Assert - reset doesn't return a value
        assert True

    def test_on_episode_start_has_default_implementation(self) -> None:
        """Test that on_episode_start has default implementation."""

        # Arrange
        class MinimalAgent(AgentInterface):
            @property
            def algorithm_name(self) -> str:
                return "minimal"

            @property
            def action_space_type(self) -> str:
                return "discrete"

            @property
            def observation_space_shape(self) -> tuple[int, ...]:
                return (10,)

            def act(self, observation: Any, _deterministic: bool = False) -> int | Any:
                return 0

            def train(
                self, env: Any, _total_timesteps: int, **kwargs: Any
            ) -> dict[str, Any]:
                return {}

            def learn_from_experience(
                self,
                observation: Any,
                action: int | Any,
                reward: float,
                _next_observation: Any,
                done: bool,
            ) -> dict[str, float] | None:
                return None

            def save(self, path: str) -> None:
                pass

            def load(self, path: str) -> None:
                pass

            def get_reward(
                self,
                state: dict[str, Any],
                action: int | Any,
                _next_state: dict[str, Any],
                info: dict[str, Any],
            ) -> float:
                return 0.0

            def update_exploration_params(
                self, _timestep: int, _total_timesteps: int
            ) -> None:
                pass

            def get_config(self) -> dict[str, Any]:
                return {}

            def set_config(self, config: dict[str, Any]) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        # Act
        agent = MinimalAgent()
        agent.on_episode_start()

        # Assert - on_episode_start doesn't return a value
        assert True

    def test_on_episode_end_has_default_implementation(self) -> None:
        """Test that on_episode_end has default implementation."""

        # Arrange
        class MinimalAgent(AgentInterface):
            @property
            def algorithm_name(self) -> str:
                return "minimal"

            @property
            def action_space_type(self) -> str:
                return "discrete"

            @property
            def observation_space_shape(self) -> tuple[int, ...]:
                return (10,)

            def act(self, observation: Any, _deterministic: bool = False) -> int | Any:
                return 0

            def train(
                self, env: Any, _total_timesteps: int, **kwargs: Any
            ) -> dict[str, Any]:
                return {}

            def learn_from_experience(
                self,
                observation: Any,
                action: int | Any,
                reward: float,
                _next_observation: Any,
                done: bool,
            ) -> dict[str, float] | None:
                return None

            def save(self, path: str) -> None:
                pass

            def load(self, path: str) -> None:
                pass

            def get_reward(
                self,
                state: dict[str, Any],
                action: int | Any,
                _next_state: dict[str, Any],
                info: dict[str, Any],
            ) -> float:
                return 0.0

            def update_exploration_params(
                self, _timestep: int, _total_timesteps: int
            ) -> None:
                pass

            def get_config(self) -> dict[str, Any]:
                return {}

            def set_config(self, config: dict[str, Any]) -> None:
                pass

            def get_metrics(self) -> dict[str, Any]:
                return {}

        # Act
        agent = MinimalAgent()
        agent.on_episode_end()

        # Assert - on_episode_end doesn't return a value
        assert True


# ============================================================================
# Test Property Return Types
# ============================================================================


class TestAgentInterfacePropertyReturnTypes:
    """Tests for property return types."""

    def test_algorithm_name_returns_string(self) -> None:
        """Test that algorithm_name property returns string."""
        # Arrange
        sig = inspect.signature(
            AgentInterface.algorithm_name.fget  # type: ignore[attr-defined]
        )

        # Assert
        assert sig.return_annotation is str

    def test_action_space_type_returns_string(self) -> None:
        """Test that action_space_type property returns string."""
        # Arrange
        sig = inspect.signature(
            AgentInterface.action_space_type.fget  # type: ignore[attr-defined]
        )

        # Assert
        assert sig.return_annotation is str

    def test_observation_space_shape_returns_tuple(self) -> None:
        """Test that observation_space_shape property returns tuple."""
        # Arrange
        sig = inspect.signature(
            AgentInterface.observation_space_shape.fget  # type: ignore[attr-defined]
        )
        annotation_str = str(sig.return_annotation)

        # Assert
        assert "tuple" in annotation_str
