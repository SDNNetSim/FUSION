"""Unit tests for fusion/sim/train_pipeline.py module."""

from typing import Any
from unittest.mock import MagicMock, patch

from fusion.sim.train_pipeline import run_training_pipeline, train_rl_agent


class TestTrainRLAgent:
    """Tests for train_rl_agent function."""

    @patch("fusion.sim.train_pipeline.workflow_runner.run")
    @patch("fusion.sim.train_pipeline.create_environment")
    @patch("fusion.sim.train_pipeline.logger")
    def test_train_rl_agent_creates_env_and_runs_workflow(
        self,
        mock_logger: MagicMock,
        mock_create_env: MagicMock,
        mock_workflow: MagicMock,
    ) -> None:
        """Test that train_rl_agent creates environment and runs workflow."""
        # Arrange
        mock_config = MagicMock()
        mock_args = MagicMock()
        mock_args.config_path = "/path/to/config.yaml"
        mock_config.get_args.return_value = mock_args

        mock_env = MagicMock()
        mock_sim_dict = {"s1": {"network": "test_net"}}
        mock_callbacks: list[Any] = []
        mock_create_env.return_value = (mock_env, mock_sim_dict, mock_callbacks)

        # Act
        train_rl_agent(mock_config)

        # Assert
        mock_create_env.assert_called_once_with(config_path="/path/to/config.yaml")
        mock_workflow.assert_called_once_with(
            env=mock_env,
            sim_dict={"network": "test_net", "callback": mock_callbacks},
            callback_list=mock_callbacks,
        )
        mock_logger.info.assert_called()

    @patch("fusion.sim.train_pipeline.workflow_runner.run")
    @patch("fusion.sim.train_pipeline.create_environment")
    def test_train_rl_agent_flattens_sim_dict_correctly(
        self, mock_create_env: MagicMock, mock_workflow: MagicMock
    ) -> None:
        """Test that sim_dict is correctly flattened from s1 key."""
        # Arrange
        mock_config = MagicMock()
        mock_args = MagicMock()
        mock_args.config_path = "/path/to/config.yaml"
        mock_config.get_args.return_value = mock_args

        mock_env = MagicMock()
        mock_sim_dict = {
            "s1": {"network": "test_net", "erlang": 300, "other": "value"}
        }
        mock_callbacks = ["callback1"]
        mock_create_env.return_value = (mock_env, mock_sim_dict, mock_callbacks)

        # Act
        train_rl_agent(mock_config)

        # Assert
        # Check that the flat_dict passed to workflow contains correct data
        call_kwargs = mock_workflow.call_args[1]
        assert call_kwargs["sim_dict"]["network"] == "test_net"
        assert call_kwargs["sim_dict"]["erlang"] == 300
        assert call_kwargs["sim_dict"]["callback"] == ["callback1"]

    @patch("fusion.sim.train_pipeline.workflow_runner.run")
    @patch("fusion.sim.train_pipeline.create_environment")
    def test_train_rl_agent_uses_sim_dict_directly_when_no_s1_key(
        self, mock_create_env: MagicMock, mock_workflow: MagicMock
    ) -> None:
        """Test that sim_dict is used directly when no s1 key exists."""
        # Arrange
        mock_config = MagicMock()
        mock_args = MagicMock()
        mock_args.config_path = "/path/to/config.yaml"
        mock_config.get_args.return_value = mock_args

        mock_env = MagicMock()
        mock_sim_dict = {"network": "test_net", "erlang": 300}
        mock_callbacks: list[Any] = []
        mock_create_env.return_value = (mock_env, mock_sim_dict, mock_callbacks)

        # Act
        train_rl_agent(mock_config)

        # Assert
        call_kwargs = mock_workflow.call_args[1]
        assert call_kwargs["sim_dict"]["network"] == "test_net"


class TestRunTrainingPipeline:
    """Tests for run_training_pipeline function."""

    @patch("fusion.sim.train_pipeline.train_rl_agent")
    def test_run_training_pipeline_wraps_args_and_calls_train(
        self, mock_train: MagicMock
    ) -> None:
        """Test that run_training_pipeline wraps args and calls train_rl_agent."""
        # Arrange
        mock_args = MagicMock()
        mock_args.config_path = "/path/to/config.yaml"

        # Act
        run_training_pipeline(mock_args)

        # Assert
        mock_train.assert_called_once()
        # Check that config wrapper was created correctly
        call_args = mock_train.call_args[0]
        config_wrapper = call_args[0]
        assert config_wrapper.get_args() == mock_args

    @patch("fusion.sim.train_pipeline.train_rl_agent")
    def test_run_training_pipeline_config_wrapper_has_get_args_method(
        self, mock_train: MagicMock
    ) -> None:
        """Test that ConfigWrapper has get_args method."""
        # Arrange
        mock_args = MagicMock()

        # Act
        run_training_pipeline(mock_args)

        # Assert
        call_args = mock_train.call_args[0]
        config_wrapper = call_args[0]
        assert hasattr(config_wrapper, "get_args")
        assert callable(config_wrapper.get_args)
