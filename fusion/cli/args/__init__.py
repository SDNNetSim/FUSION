# fusion/cli/args/__init__.py
"""
Centralized CLI argument system following the FUSION architecture plan.

This module provides:
- Modular argument groups for different CLI commands
- Centralized registry system for argument management
- Common argument definitions to reduce duplication
- Clean separation between different argument categories

Registry System:
- ArgumentRegistry: Central coordinator for argument groups
- Modular argument functions: add_*_args() for each category
- Subcommand registration: register_*_args() for subcommands
"""

from .registry import args_registry, ArgumentRegistry
from .common_args import add_config_args, add_debug_args, add_output_args
from .simulation_args import register_run_sim_args, add_run_sim_args
from .simulation_args import add_simulation_args, add_network_args, add_traffic_args
from .training_args import add_all_training_args, add_reinforcement_learning_args, add_machine_learning_args
from .analysis_args import add_all_analysis_args, add_statistics_args, add_plotting_args
from .gui_args import add_gui_args

__all__ = [
    # Core registry
    'args_registry',
    'ArgumentRegistry',

    # Common arguments
    'add_config_args',
    'add_debug_args',
    'add_output_args',

    # Command registration
    'register_run_sim_args',
    'add_run_sim_args',

    # Simulation arguments
    'add_simulation_args',
    'add_network_args',
    'add_traffic_args',

    # Training arguments
    'add_all_training_args',
    'add_reinforcement_learning_args',
    'add_machine_learning_args',

    # Analysis arguments
    'add_all_analysis_args',
    'add_statistics_args',
    'add_plotting_args',

    # Interface arguments
    'add_gui_args'
]
