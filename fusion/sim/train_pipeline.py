"""
RL training pipeline for FUSION simulations.

This module provides reinforcement learning training capabilities,
integrating with the legacy RL workflow while adapting to the new
CLI and configuration system.
"""

from pathlib import Path
from typing import Any

from fusion.modules.rl import workflow_runner
from fusion.modules.rl.utils.gym_envs import create_environment
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def train_rl_agent(config: Any) -> None:
    """
    Launch RL training using legacy workflow via new CLI and config system.

    This function bridges the new configuration system with the existing
    RL training workflow, maintaining compatibility while providing improved
    configuration management.

    :param config: Configuration object containing training parameters
    :type config: Any
    """
    config_path = Path(config.get_args().config_path)

    env, sim_dict, callback_list = create_environment(config_path=config_path)

    # Legacy runner expects flat sim_dict
    flat_dict = sim_dict.get("s1", sim_dict)
    flat_dict["callback"] = callback_list

    workflow_runner.run(env=env, sim_dict=flat_dict, callback_list=callback_list)

    logger.info("✅ RL training pipeline ran successfully using legacy logic.")


def run_training_pipeline(args: Any) -> None:
    """
    Pipeline function for running RL training from CLI.

    :param args: Parsed command line arguments
    :type args: Any
    """

    # Create config object with args
    class ConfigWrapper:  # pylint: disable=too-few-public-methods
        """Wrapper class to adapt args for legacy train_rl_agent function."""

        def __init__(self, args: Any) -> None:
            """Initialize with command line arguments.

            :param args: Command line arguments
            :type args: Any
            """
            self.args = args

        def get_args(self) -> Any:
            """Return the arguments.

            :return: Command line arguments
            :rtype: Any
            """
            return self.args

    config = ConfigWrapper(args)
    train_rl_agent(config)
