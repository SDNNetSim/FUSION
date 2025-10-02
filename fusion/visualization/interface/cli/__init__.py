"""
CLI commands for the FUSION visualization system.

This module provides command-line interfaces for:
- Generating individual plots
- Batch plot generation
- Algorithm comparison
- Configuration migration
- System validation and info

Usage:
    fusion viz plot --config config.yml
    fusion viz batch --config batch.yml
    fusion viz compare -n NSFNet -a algo1 -a algo2
    fusion viz migrate --input old.yml --output new.yml
    fusion viz validate --config config.yml
    fusion viz info
"""

from fusion.visualization.interface.cli.batch_command import batch_command
from fusion.visualization.interface.cli.compare_command import compare_command
from fusion.visualization.interface.cli.migrate_command import viz_cli
from fusion.visualization.interface.cli.plot_command import plot_command

__all__ = [
    "viz_cli",
    "plot_command",
    "batch_command",
    "compare_command",
]
