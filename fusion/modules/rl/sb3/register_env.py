"""
StableBaselines3 environment registration utilities.

This module provides functionality to register custom Gymnasium environments
with StableBaselines3 and manage algorithm configuration files for RL training.
The registration process is critical for enabling custom environments to work
with the SB3 ecosystem and RLZoo3 hyperparameter management.
"""

# Standard library imports
import argparse
import shutil
from pathlib import Path

# Third-party imports
from gymnasium.envs.registration import register

__all__ = [
    "copy_yml_file",
    "main",
]


def copy_yml_file(algorithm: str) -> None:
    """
    Copy algorithm configuration file to RLZoo3 hyperparameters directory.

    This function copies YAML configuration files from the local sb3_scripts
    directory to the RLZoo3 installation, enabling custom algorithm training
    with proper hyperparameter configurations.

    :param algorithm: Algorithm name for configuration file lookup
    :type algorithm: str
    :raises FileNotFoundError: If source configuration file doesn't exist
    :raises PermissionError: If destination directory is not writable
    :raises OSError: If file copy operation fails

    Example:
        >>> copy_yml_file("PPO")
        # Copies sb3_scripts/yml/PPO.yml to RLZoo3 hyperparams directory
    """
    # Note: Preserving exact hard-coded paths as required for SB3 integration
    source_file = Path(f"sb3_scripts/yml/{algorithm}.yml")
    destination_file = Path(f"venvs/unity_venv/venv/lib/python3.11/site-packages/rl_zoo3/hyperparams/{algorithm}.yml")

    try:
        shutil.copy(str(source_file), str(destination_file))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Configuration file not found: {source_file}. Ensure the algorithm configuration exists in sb3_scripts/yml/"
        ) from exc
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to RLZoo3 directory: {destination_file}. Check file permissions and virtual environment access."
        ) from exc
    except OSError as exc:
        raise OSError(f"Failed to copy configuration file: {exc}") from exc


def main() -> None:
    """
    Main entry point for environment registration script.

    Parses command-line arguments and orchestrates the registration of custom
    Gymnasium environments with StableBaselines3. This includes both the
    environment registration with Gymnasium and copying algorithm configuration
    files to the RLZoo3 hyperparameters directory.

    Command-line Arguments:
        --algo: Algorithm name for configuration file (e.g., 'PPO', 'DQN')
        --env-name: Environment class name to register (e.g., 'SimEnv')

    :raises ValueError: If required command-line arguments are missing
    :raises RuntimeError: If environment registration fails

    Example:
        $ python register_env.py --algo PPO --env-name SimEnv
    """
    parser = argparse.ArgumentParser(
        description="Register custom Gymnasium environment with StableBaselines3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python register_env.py --algo PPO --env-name SimEnv",
    )
    parser.add_argument(
        "--algo",
        required=True,
        help="Algorithm name for configuration file (e.g., PPO, DQN)",
    )
    parser.add_argument(
        "--env-name",
        required=True,
        help="Environment class name to register (e.g., SimEnv)",
    )
    args = parser.parse_args()

    try:
        # Register environment with Gymnasium
        # CRITICAL: Preserving exact entry point string for SB3 compatibility
        register(
            id=args.env_name,
            entry_point=(f"reinforcement_learning.gymnasium_envs.general_sim_env:{args.env_name}"),
        )

        print("\n=== Registered Environments with Gymnasium ===\n")
        # Note: gymnasium.pprint_registry() was removed in newer versions
        print(f"Environment '{args.env_name}' registered successfully")
        print("\n")

        # Copy algorithm configuration to RLZoo3
        copy_yml_file(algorithm=args.algo)

        print(f"Successfully registered environment '{args.env_name}' with algorithm '{args.algo}'")

    except Exception as exc:
        raise RuntimeError(f"Failed to register environment: {exc}") from exc


if __name__ == "__main__":
    main()
