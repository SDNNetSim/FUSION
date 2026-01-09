"""
Policies package for FUSION simulation framework.

This package provides policy implementations that implement the ControlPolicy
protocol defined in fusion.interfaces.control_policy.

Current implementations:
- RLPolicy: Wrapper for pre-trained Stable-Baselines3 models

Future implementations (P5.2+):
- FirstFeasiblePolicy: Simple heuristic (first feasible path)
- ShortestFeasiblePolicy: Select shortest feasible path
- LoadBalancedPolicy: Balance load across paths
- MLControlPolicy: ML-based policy
- FallbackPolicy: Composite with fallback
- TiebreakingPolicy: Composite with tiebreaking

Phase: P5.1 - ControlPolicy Protocol + RLPolicy Adapter
"""

from fusion.policies.rl_policy import RLControlPolicy, RLPolicy

__all__ = [
    "RLPolicy",
    "RLControlPolicy",  # Alias for backwards compatibility
]
