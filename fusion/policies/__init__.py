"""
Policies package for FUSION simulation framework.

This package provides policy implementations that implement the ControlPolicy
protocol defined in fusion.interfaces.control_policy.

Current implementations:

Heuristic Policies:
- FirstFeasiblePolicy: Select first feasible path
- ShortestFeasiblePolicy: Select shortest feasible path by distance
- LeastCongestedPolicy: Select least congested feasible path
- RandomFeasiblePolicy: Random selection among feasible paths
- LoadBalancedPolicy: Balance path length and congestion

ML Policies:
- MLControlPolicy: Pre-trained ML models (PyTorch, sklearn, ONNX)

RL Policies:
- RLPolicy: Wrapper for pre-trained Stable-Baselines3 models

Factory:
- PolicyFactory: Instantiate policies from configuration
- PolicyConfig: Configuration dataclass for policy creation
"""

from fusion.interfaces.control_policy import ControlPolicy, PolicyAction
from fusion.policies.heuristic_policy import (
    FirstFeasiblePolicy,
    HeuristicPolicy,
    LeastCongestedPolicy,
    LoadBalancedPolicy,
    RandomFeasiblePolicy,
    ShortestFeasiblePolicy,
)
from fusion.policies.ml_policy import FeatureBuilder, MLControlPolicy
from fusion.policies.policy_factory import PolicyConfig, PolicyFactory
from fusion.policies.rl_policy import RLControlPolicy, RLPolicy

__all__ = [
    # Protocol
    "ControlPolicy",
    "PolicyAction",
    # Base class
    "HeuristicPolicy",
    # Heuristic policies
    "FirstFeasiblePolicy",
    "ShortestFeasiblePolicy",
    "LeastCongestedPolicy",
    "RandomFeasiblePolicy",
    "LoadBalancedPolicy",
    # ML policies
    "MLControlPolicy",
    "FeatureBuilder",
    # RL policies
    "RLPolicy",
    "RLControlPolicy",
    # Factory
    "PolicyFactory",
    "PolicyConfig",
]
