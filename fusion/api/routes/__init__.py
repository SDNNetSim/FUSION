"""
API route modules.

Each module defines routes for a specific resource:
- runs: Simulation run management
- configs: Configuration templates
- artifacts: Run output files
- system: Health checks and system info
- topology: Network topology visualization
"""

from . import artifacts, configs, runs, system, topology

__all__ = ["runs", "configs", "artifacts", "system", "topology"]
