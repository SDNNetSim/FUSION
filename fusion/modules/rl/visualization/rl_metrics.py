"""
RL-specific metric definitions.

This module defines metrics specific to reinforcement learning:
- Episode rewards
- TD errors
- Q-values
- Policy entropy
- Value estimates
- Exploration metrics
"""

from fusion.visualization.domain.entities.metric import (
    AggregationStrategy,
    DataType,
    MetricDefinition,
)


def get_rl_metrics() -> list[MetricDefinition]:
    """
    Get all RL-specific metric definitions.

    :return: List of RL metric definitions
    :rtype: list[MetricDefinition]
    """
    return [
        # Episode rewards
        MetricDefinition(
            name="episode_reward",
            display_name="Episode Reward",
            data_type=DataType.FLOAT,
            source_path="$.training.episode_rewards",
            aggregation=AggregationStrategy.MEAN,
            unit="reward",
            description="Total reward accumulated in an episode",
        ),
        MetricDefinition(
            name="episode_reward_mean",
            display_name="Mean Episode Reward",
            data_type=DataType.FLOAT,
            source_path="$.training.episode_reward_mean",
            aggregation=AggregationStrategy.LAST,
            unit="reward",
            description="Moving average of episode rewards",
        ),
        # TD Errors
        MetricDefinition(
            name="td_error",
            display_name="TD Error",
            data_type=DataType.ARRAY,
            source_path="$.training.td_errors",
            aggregation=AggregationStrategy.MEAN,
            unit="error",
            description="Temporal difference prediction errors",
        ),
        MetricDefinition(
            name="td_error_mean",
            display_name="Mean TD Error",
            data_type=DataType.FLOAT,
            source_path="$.training.td_error_mean",
            aggregation=AggregationStrategy.MEAN,
            unit="error",
            description="Mean temporal difference error",
        ),
        # Q-Values
        MetricDefinition(
            name="q_values",
            display_name="Q-Values",
            data_type=DataType.ARRAY,
            source_path="$.training.q_values",
            aggregation=AggregationStrategy.MEAN,
            unit="value",
            description="Action-value function estimates",
        ),
        MetricDefinition(
            name="q_value_mean",
            display_name="Mean Q-Value",
            data_type=DataType.FLOAT,
            source_path="$.training.q_value_mean",
            aggregation=AggregationStrategy.MEAN,
            unit="value",
            description="Mean Q-value across all actions",
        ),
        # Value estimates
        MetricDefinition(
            name="value_estimate",
            display_name="Value Estimate",
            data_type=DataType.FLOAT,
            source_path="$.training.value_estimates",
            aggregation=AggregationStrategy.MEAN,
            unit="value",
            description="State value function estimates",
        ),
        # Policy metrics
        MetricDefinition(
            name="policy_entropy",
            display_name="Policy Entropy",
            data_type=DataType.FLOAT,
            source_path="$.training.policy_entropy",
            aggregation=AggregationStrategy.MEAN,
            unit="nats",
            description="Entropy of the policy distribution",
        ),
        MetricDefinition(
            name="policy_loss",
            display_name="Policy Loss",
            data_type=DataType.FLOAT,
            source_path="$.training.policy_loss",
            aggregation=AggregationStrategy.MEAN,
            unit="loss",
            description="Policy gradient loss",
        ),
        MetricDefinition(
            name="value_loss",
            display_name="Value Loss",
            data_type=DataType.FLOAT,
            source_path="$.training.value_loss",
            aggregation=AggregationStrategy.MEAN,
            unit="loss",
            description="Value function loss",
        ),
        # Exploration metrics
        MetricDefinition(
            name="epsilon",
            display_name="Epsilon (Exploration Rate)",
            data_type=DataType.FLOAT,
            source_path="$.training.epsilon",
            aggregation=AggregationStrategy.LAST,
            unit="probability",
            description="Epsilon-greedy exploration rate",
        ),
        MetricDefinition(
            name="exploration_rate",
            display_name="Exploration Rate",
            data_type=DataType.FLOAT,
            source_path="$.training.exploration_rate",
            aggregation=AggregationStrategy.LAST,
            unit="probability",
            description="Current exploration rate",
        ),
        # Training progress
        MetricDefinition(
            name="episode_length",
            display_name="Episode Length",
            data_type=DataType.INT,
            source_path="$.training.episode_length",
            aggregation=AggregationStrategy.MEAN,
            unit="steps",
            description="Number of steps per episode",
        ),
        MetricDefinition(
            name="total_timesteps",
            display_name="Total Timesteps",
            data_type=DataType.INT,
            source_path="$.training.total_timesteps",
            aggregation=AggregationStrategy.LAST,
            unit="steps",
            description="Total training timesteps",
        ),
        # Learning rate
        MetricDefinition(
            name="learning_rate",
            display_name="Learning Rate",
            data_type=DataType.FLOAT,
            source_path="$.training.learning_rate",
            aggregation=AggregationStrategy.LAST,
            unit="rate",
            description="Current learning rate",
        ),
        # Network performance
        MetricDefinition(
            name="blocking_probability_rl",
            display_name="Blocking Probability (RL)",
            data_type=DataType.FLOAT,
            source_path="$.performance.blocking_mean",
            aggregation=AggregationStrategy.MEAN,
            unit="probability",
            description="Network blocking probability during RL evaluation",
        ),
        MetricDefinition(
            name="acceptance_ratio",
            display_name="Acceptance Ratio",
            data_type=DataType.FLOAT,
            source_path="$.performance.acceptance_ratio",
            aggregation=AggregationStrategy.MEAN,
            unit="ratio",
            description="Ratio of accepted to total connection requests",
        ),
    ]


# Metric categories for organization
REWARD_METRICS = [
    "episode_reward",
    "episode_reward_mean",
]

ERROR_METRICS = [
    "td_error",
    "td_error_mean",
]

VALUE_METRICS = [
    "q_values",
    "q_value_mean",
    "value_estimate",
]

POLICY_METRICS = [
    "policy_entropy",
    "policy_loss",
    "value_loss",
]

EXPLORATION_METRICS = [
    "epsilon",
    "exploration_rate",
]

TRAINING_METRICS = [
    "episode_length",
    "total_timesteps",
    "learning_rate",
]

PERFORMANCE_METRICS = [
    "blocking_probability_rl",
    "acceptance_ratio",
]
