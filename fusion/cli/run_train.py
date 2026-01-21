"""
CLI entry point for training FUSION agents.

This module provides the command-line interface for training reinforcement
learning (RL) and supervised learning (SL) agents. It supports multiple
training algorithms and provides detailed error handling with helpful
guidance for common issues.
"""

from fusion.cli.constants import ERROR_EXIT_CODE, INTERRUPT_EXIT_CODE, SUCCESS_EXIT_CODE
from fusion.cli.main_parser import create_training_argument_parser
from fusion.cli.utils import create_entry_point_wrapper
from fusion.sim.train_pipeline import run_training_pipeline
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def main() -> int:
    """
    Train FUSION agents using RL or SL algorithms.

    Parses command line arguments and delegates training execution to the
    appropriate training pipeline module. Supports both reinforcement learning (RL)
    and supervised learning (SL) workflows with proper error handling and user feedback.

    :return: Exit code (0 for success, 1 for error or interruption)
    :rtype: int
    :raises SystemExit: On argument parsing errors (handled by argparse)
    """
    try:
        training_arguments = create_training_argument_parser()
        run_training_pipeline(training_arguments)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Training progress has been saved where possible.")
        return INTERRUPT_EXIT_CODE
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Missing training dependencies: {e}. Try: pip install -e .[rl]")
        return ERROR_EXIT_CODE
    except OSError as e:
        logger.error(f"File system error during training: {e}. Check file permissions and disk space.")
        return ERROR_EXIT_CODE
    except (ValueError, TypeError) as e:
        logger.error(f"Training configuration error: {e}. Check training parameters and agent configuration.")
        return ERROR_EXIT_CODE
    except (RuntimeError, MemoryError) as e:
        logger.error(f"Training runtime error: {e}. Consider reducing batch size or model complexity.")
        return ERROR_EXIT_CODE

    return SUCCESS_EXIT_CODE


# Create entry point functions using shared utilities
train_main, run_train_main = create_entry_point_wrapper(
    main,
    "training",
    "Convenience function that handles the sys.exit call for the main training "
    "entry point. Provides clean separation between main logic and process "
    "exit handling.",
)


if __name__ == "__main__":
    run_train_main()
