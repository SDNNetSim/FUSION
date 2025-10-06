"""
Unity cluster management module for FUSION.

This module provides functionality for managing simulation jobs on Unity cluster
systems, including manifest generation, job submission, and result synchronization.
"""

# Import main functionality for public API
# Import commonly used utilities
from fusion.unity.constants import RESOURCE_KEYS, RL_ALGORITHMS
from fusion.unity.errors import (
    JobSubmissionError,
    ManifestError,
    SpecificationError,
    SynchronizationError,
    UnityError,
)
from fusion.unity.fetch_results import main as fetch_results
from fusion.unity.make_manifest import main as make_manifest
from fusion.unity.submit_manifest import main as submit_manifest

# Version info
__version__ = "1.0.0"

# Public API - explicitly define what's exported
__all__ = [
    "fetch_results",
    "make_manifest",
    "submit_manifest",
    "RESOURCE_KEYS",
    "RL_ALGORITHMS",
    "UnityError",
    "ManifestError",
    "SpecificationError",
    "JobSubmissionError",
    "SynchronizationError",
]
