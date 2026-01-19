"""
CLI arguments for training configuration (RL and SL).

This module provides arguments for reinforcement learning (RL) and supervised
learning (SL) training pipelines. It consolidates algorithm selection, model
configuration, training parameters, and optimization settings.

TODO (v6.1.0): Rename ml_* arguments to sl_* for consistency:
  --ml_training -> --sl_training
  --ml_model -> --sl_model
  Also rename add_machine_learning_args to add_supervised_learning_args.

TODO (v6.1.0): Add support for unsupervised learning (UL) methods such as
clustering and dimensionality reduction for network state analysis.
"""

import argparse


def add_reinforcement_learning_args(parser: argparse.ArgumentParser) -> None:
    """
    Add reinforcement learning specific arguments.

    Configures RL algorithms, model paths, training parameters,
    exploration strategies, and reward functions for RL-based optimization.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    rl_group = parser.add_argument_group("Reinforcement Learning Configuration")

    # Algorithm selection
    rl_group.add_argument(
        "--path_algorithm",
        type=str,
        choices=["dqn", "ppo", "a2c", "q_learning", "bandits", "epsilon_greedy_bandit"],
        help="Path selection RL algorithm",
    )
    rl_group.add_argument(
        "--core_algorithm",
        type=str,
        choices=[
            "dqn",
            "ppo",
            "a2c",
            "q_learning",
            "bandits",
            "epsilon_greedy_bandit",
            "first_fit",
        ],
        help="Core selection RL algorithm",
    )
    rl_group.add_argument(
        "--spectrum_algorithm",
        type=str,
        choices=[
            "dqn",
            "ppo",
            "a2c",
            "q_learning",
            "bandits",
            "epsilon_greedy_bandit",
            "first_fit",
        ],
        help="Spectrum allocation RL algorithm",
    )

    # Model configuration
    rl_group.add_argument("--path_model", type=str, help="Path to pre-trained path selection model")
    rl_group.add_argument("--core_model", type=str, help="Path to pre-trained core selection model")
    rl_group.add_argument(
        "--spectrum_model",
        type=str,
        help="Path to pre-trained spectrum allocation model",
    )

    # Training parameters
    rl_group.add_argument(
        "--is_training",
        action="store_true",
        help="Enable training mode (vs. inference mode)",
    )
    rl_group.add_argument(
        "--learn_rate",
        type=float,
        default=None,
        help="Learning rate for RL algorithms",
    )
    rl_group.add_argument("--gamma", type=float, default=None, help="Discount factor for future rewards")
    rl_group.add_argument(
        "--epsilon_start",
        type=float,
        default=None,
        help="Initial epsilon value for epsilon-greedy exploration",
    )
    rl_group.add_argument(
        "--epsilon_end",
        type=float,
        default=None,
        help="Final epsilon value for epsilon-greedy exploration",
    )
    rl_group.add_argument(
        "--epsilon_update",
        type=str,
        choices=["linear", "exponential", "step", "linear_decay", "exp_decay"],
        default=None,
        help="Epsilon decay strategy",
    )

    # Reward configuration
    rl_group.add_argument("--reward", type=float, default=None, help="Reward value for successful actions")
    rl_group.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="Penalty value for unsuccessful actions",
    )
    rl_group.add_argument(
        "--dynamic_reward",
        action="store_true",
        help="Enable dynamic reward calculation",
    )


def add_feature_extraction_args(parser: argparse.ArgumentParser) -> None:
    """
    Add feature extraction and neural network arguments to the parser.

    Configures feature extraction methods, neural network architectures,
    and observation space representations for RL and SL model training.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    feature_group = parser.add_argument_group("Feature Extraction Configuration")
    feature_group.add_argument(
        "--feature_extractor",
        type=str,
        choices=["graphormer", "path_gnn"],
        help="Feature extraction method",
    )
    feature_group.add_argument(
        "--gnn_type",
        type=str,
        choices=["gcn", "gat", "sage", "graphconv"],
        help="Graph Neural Network architecture type",
    )
    feature_group.add_argument("--layers", type=int, default=None, help="Number of layers in neural network")
    feature_group.add_argument(
        "--emb_dim",
        type=int,
        default=None,
        help="Embedding dimension for neural networks",
    )
    feature_group.add_argument(
        "--heads",
        type=int,
        default=None,
        help="Number of attention heads (for attention-based models)",
    )
    feature_group.add_argument(
        "--obs_space",
        type=str,
        choices=["graph", "vector", "matrix", "hybrid"],
        help="Observation space representation",
    )


# TODO (v6.1.0): Rename to add_supervised_learning_args
def add_machine_learning_args(parser: argparse.ArgumentParser) -> None:
    """
    Add supervised learning (SL) arguments to the parser.

    Configures classical supervised learning algorithms, training data paths,
    test/train splits, and model deployment options.

    NOTE: This function and its arguments use "ml" naming for backward
    compatibility. Will be renamed to sl_* in v6.1.0.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    ml_group = parser.add_argument_group("Supervised Learning Configuration")
    ml_group.add_argument("--ml_training", action="store_true", help="Enable SL training mode")
    ml_group.add_argument(
        "--ml_model",
        type=str,
        choices=[
            "random_forest",
            "svm",
            "linear_regression",
            "neural_network",
            "decision_tree",
        ],
        help="Supervised learning model type",
    )
    ml_group.add_argument("--train_file_path", type=str, help="Path to training data file")
    ml_group.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="Fraction of data to use for testing (0.0-1.0)",
    )
    ml_group.add_argument("--output_train_data", action="store_true", help="Save training data to file")
    ml_group.add_argument("--deploy_model", action="store_true", help="Deploy trained model for inference")


def add_optimization_args(parser: argparse.ArgumentParser) -> None:
    """
    Add hyperparameter optimization arguments.

    Configures automated hyperparameter tuning, optimization trials,
    and computing device selection for training acceleration.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    opt_group = parser.add_argument_group("Optimization Configuration")
    opt_group.add_argument("--optimize", action="store_true", help="Enable hyperparameter optimization")
    opt_group.add_argument(
        "--optimize_hyperparameters",
        action="store_true",
        help="Enable automated hyperparameter tuning",
    )
    opt_group.add_argument(
        "--optuna_trials",
        type=int,
        default=None,
        help="Number of optimization trials for Optuna",
    )
    opt_group.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Number of trials for grid search or random search",
    )
    opt_group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default=None,
        help="Computing device for training (cpu/gpu)",
    )


def add_all_training_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all training-related argument groups to the parser.

    Convenience function that combines reinforcement learning (RL),
    feature extraction, supervised learning (SL), and optimization
    arguments in a single call.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    add_reinforcement_learning_args(parser)
    add_feature_extraction_args(parser)
    add_machine_learning_args(parser)
    add_optimization_args(parser)
