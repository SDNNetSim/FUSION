# fusion/cli/parameters/__init__.py
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
from .shared import add_config_args, add_debug_args, add_output_args, add_plot_format_args
from .simulation import register_run_sim_args, add_run_sim_args, add_simulation_args
from .routing import add_all_routing_args, add_routing_args, add_spectrum_args, add_modulation_args, add_sdn_args
from .snr import add_snr_args
from .network import add_network_args, add_all_network_args, add_topology_args, add_link_args, add_node_args, add_spectrum_bands_args
from .traffic import add_traffic_args, add_erlang_args, add_request_args, add_simulation_control_args
from .training import add_all_training_args, add_reinforcement_learning_args, add_machine_learning_args, add_feature_extraction_args, add_optimization_args
from .analysis import add_all_analysis_args, add_statistics_args, add_plotting_args, add_export_args, add_comparison_args, add_filtering_args
from .gui import add_gui_args, add_all_gui_args

__all__ = [
    # Core registry
    'args_registry',
    'ArgumentRegistry',

    # Common arguments
    'add_config_args',
    'add_debug_args',
    'add_output_args',
    'add_plot_format_args',

    # Command registration
    'register_run_sim_args',
    'add_run_sim_args',

    # Simulation arguments (compatibility)
    'add_simulation_args',
    'add_network_args',
    'add_traffic_args',

    # Routing arguments
    'add_all_routing_args',
    'add_routing_args',
    'add_spectrum_args',
    'add_modulation_args', 
    'add_sdn_args',

    # SNR arguments
    'add_snr_args',

    # Network arguments
    'add_network_args',
    'add_all_network_args',
    'add_topology_args',
    'add_link_args',
    'add_node_args',
    'add_spectrum_bands_args',

    # Traffic arguments
    'add_erlang_args',
    'add_request_args',
    'add_simulation_control_args',

    # Training arguments
    'add_all_training_args',
    'add_reinforcement_learning_args',
    'add_machine_learning_args',
    'add_feature_extraction_args',
    'add_optimization_args',

    # Analysis arguments
    'add_all_analysis_args',
    'add_statistics_args',
    'add_plotting_args',
    'add_export_args',
    'add_comparison_args', 
    'add_filtering_args',

    # GUI arguments
    'add_gui_args',
    'add_all_gui_args'
]
