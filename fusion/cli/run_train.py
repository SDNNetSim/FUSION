# fusion/cli/run_train.py

"""
CLI entry point for training agents (RL or ML).
Follows architecture best practice: entry points should have no logic.
"""

from fusion.cli.main_parser import get_train_args
from fusion.sim.train_pipeline import run_training_pipeline


def main():
    """
    Entry point for training agents from the command line.
    Delegates all logic to appropriate modules.
    """
    try:
        args = get_train_args()

        # Delegate to training pipeline - no business logic here
        run_training_pipeline(args)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
        return 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"❌ Error during training: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
