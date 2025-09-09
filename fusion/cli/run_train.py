"""CLI entry point for training FUSION agents (RL or ML).

This module provides the command-line interface for training machine learning
and reinforcement learning agents. It supports multiple training algorithms
and provides detailed error handling with helpful guidance for common issues.
"""

import sys

from fusion.cli.constants import SUCCESS_EXIT_CODE, ERROR_EXIT_CODE, INTERRUPT_EXIT_CODE
from fusion.cli.main_parser import create_training_argument_parser
from fusion.sim.train_pipeline import run_training_pipeline


def main() -> int:
    """
    Main entry point for training FUSION agents (RL or ML).
    
    Parses command line arguments and delegates training execution to the
    appropriate training pipeline module. Supports both reinforcement learning
    and machine learning workflows with proper error handling and user feedback.

    :return: Exit code (0 for success, 1 for error or interruption)
    :rtype: int
    :raises SystemExit: On argument parsing errors (handled by argparse)
    """
    try:
        training_arguments = create_training_argument_parser()

        run_training_pipeline(training_arguments)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        print("💾 Training progress has been saved where possible")
        return INTERRUPT_EXIT_CODE
    except (ImportError, ModuleNotFoundError) as e:
        print(f"❌ Missing training dependencies: {e}")
        print("💡 Try installing ML/RL dependencies with: pip install -e .[ml,rl]")
        return ERROR_EXIT_CODE
    except (OSError, IOError) as e:
        print(f"❌ File system error during training: {e}")
        print("💡 Check file permissions and available disk space for model storage")
        return ERROR_EXIT_CODE
    except (ValueError, TypeError) as e:
        print(f"❌ Training configuration error: {e}")
        print("💡 Check your training parameters and agent configuration")
        return ERROR_EXIT_CODE
    except (RuntimeError, MemoryError) as e:
        print(f"❌ Training runtime error: {e}")
        print("💡 Consider reducing batch size, model complexity, or check system resources")
        return ERROR_EXIT_CODE

    return SUCCESS_EXIT_CODE


# Backward compatibility function alias
def train_main() -> int:
    """
    Legacy function name for main training entry point.
    
    :return: Exit code from main function
    :rtype: int
    :deprecated: Use main() directly instead
    """
    return main()


def run_train_main() -> None:
    """
    Execute the training main function and exit with appropriate code.
    
    Convenience function that handles the sys.exit call for the main
    training entry point. Provides clean separation between main logic
    and process exit handling.

    :raises SystemExit: Always exits with code from main() function
    """
    sys.exit(main())


if __name__ == "__main__":
    run_train_main()
