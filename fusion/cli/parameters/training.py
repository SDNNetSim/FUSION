"""
CLI arguments for training configuration (RL and ML).
Consolidates machine learning and reinforcement learning arguments.
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
    rl_group.add_argument(
        "--path_model", type=str, help="Path to pre-trained path selection model"
    )
    rl_group.add_argument(
        "--core_model", type=str, help="Path to pre-trained core selection model"
    )
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
        default=0.001,
        help="Learning rate for RL algorithms",
    )
    rl_group.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for future rewards"
    )
    rl_group.add_argument(
        "--epsilon_start",
        type=float,
        default=1.0,
        help="Initial epsilon value for epsilon-greedy exploration",
    )
    rl_group.add_argument(
        "--epsilon_end",
        type=float,
        default=0.01,
        help="Final epsilon value for epsilon-greedy exploration",
    )
    rl_group.add_argument(
        "--epsilon_update",
        type=str,
        choices=["linear", "exponential", "step", "linear_decay", "exp_decay"],
        default="linear",
        help="Epsilon decay strategy",
    )

    # Reward configuration
    rl_group.add_argument(
        "--reward", type=float, default=1.0, help="Reward value for successful actions"
    )
    rl_group.add_argument(
        "--penalty",
        type=float,
        default=-1.0,
        help="Penalty value for unsuccessful actions",
    )
    rl_group.add_argument(
        "--dynamic_reward",
        action="store_true",
        help="Enable dynamic reward calculation",
    )


def add_feature_extraction_args(parser: argparse.ArgumentParser) -> None:
    """
    Add feature extraction and neural network arguments.

    Configures feature extraction methods, neural network architectures,
    and observation space representations for ML model training.

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
    feature_group.add_argument(
        "--layers", type=int, default=3, help="Number of layers in neural network"
    )
    feature_group.add_argument(
        "--emb_dim",
        type=int,
        default=64,
        help="Embedding dimension for neural networks",
    )
    feature_group.add_argument(
        "--heads",
        type=int,
        default=8,
        help="Number of attention heads (for attention-based models)",
    )
    feature_group.add_argument(
        "--obs_space",
        type=str,
        choices=["graph", "vector", "matrix", "hybrid"],
        help="Observation space representation",
    )


def add_machine_learning_args(parser: argparse.ArgumentParser) -> None:
    """
    Add traditional machine learning arguments.

    Configures classical ML algorithms, training data paths,
    test/train splits, and model deployment options.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    ml_group = parser.add_argument_group("Machine Learning Configuration")
    ml_group.add_argument(
        "--ml_training", action="store_true", help="Enable ML training mode"
    )
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
        help="Machine learning model type",
    )
    ml_group.add_argument(
        "--train_file_path", type=str, help="Path to training data file"
    )
    ml_group.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (0.0-1.0)",
    )
    ml_group.add_argument(
        "--output_train_data", action="store_true", help="Save training data to file"
    )
    ml_group.add_argument(
        "--deploy_model", action="store_true", help="Deploy trained model for inference"
    )


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
    opt_group.add_argument(
        "--optimize", action="store_true", help="Enable hyperparameter optimization"
    )
    opt_group.add_argument(
        "--optimize_hyperparameters",
        action="store_true",
        help="Enable automated hyperparameter tuning",
    )
    opt_group.add_argument(
        "--optuna_trials",
        type=int,
        default=100,
        help="Number of optimization trials for Optuna",
    )
    opt_group.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of trials for grid search or random search",
    )
    opt_group.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
        help="Computing device for training (cpu/gpu)",
    )


def add_all_training_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all training-related argument groups.

    Convenience function that combines reinforcement learning,
    feature extraction, machine learning, and optimization arguments.

    :param parser: ArgumentParser instance to add arguments to
    :type parser: argparse.ArgumentParser
    :return: None
    :rtype: None
    """
    add_reinforcement_learning_args(parser)
    add_feature_extraction_args(parser)
    add_machine_learning_args(parser)
    add_optimization_args(parser)
