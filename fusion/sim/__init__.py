"""
Simulation orchestration module for FUSION.

This module provides the main simulation orchestration capabilities including
batch processing, pipeline management, and runtime interfaces for the
FUSION optical network simulation framework.
"""

from typing import Any

from .batch_runner import BatchRunner, run_batch_simulation
from .evaluate_pipeline import EvaluationPipeline, run_evaluation_pipeline
from .run_simulation import run_simulation, run_simulation_pipeline


# Lazy import to avoid circular dependencies with RL modules
def train_rl_agent(*args: Any, **kwargs: Any) -> None:
    """Lazy wrapper for train_rl_agent to avoid circular imports."""
    from .train_pipeline import (
        train_rl_agent as _train_rl_agent,  # pylint: disable=import-outside-toplevel
    )

    return _train_rl_agent(*args, **kwargs)


def run_training_pipeline(*args: Any, **kwargs: Any) -> None:
    """Lazy wrapper for run_training_pipeline to avoid circular imports."""
    from .train_pipeline import (
        run_training_pipeline as _run_training_pipeline,  # pylint: disable=import-outside-toplevel
    )

    return _run_training_pipeline(*args, **kwargs)


__all__ = [
    "BatchRunner",
    "run_batch_simulation",
    "EvaluationPipeline",
    "run_evaluation_pipeline",
    "run_simulation",
    "run_simulation_pipeline",
    "train_rl_agent",
    "run_training_pipeline",
]
