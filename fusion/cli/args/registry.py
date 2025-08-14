"""
Centralized argument registry for CLI commands.
Provides a unified interface for managing CLI argument parsing.
"""

import argparse
from typing import Dict, Callable, List

from .common_args import add_config_args, add_debug_args, add_output_args
from .run_sim_args import add_run_sim_args, register_run_sim_args
from .simulation_args import add_simulation_args, add_network_args, add_traffic_args
from .training_args import add_all_training_args, add_reinforcement_learning_args, add_machine_learning_args
from .analysis_args import add_all_analysis_args, add_statistics_args, add_plotting_args
from .gui_args import add_gui_args

# Legacy imports for backward compatibility
from .routing_args import add_routing_args
from .spectrum_args import add_spectrum_args
from .snr_args import add_snr_args
from .sdn_args import add_sdn_args
from .stats_args import add_stats_args
from .plot_args import add_plot_args


class ArgumentRegistry:
    """
    Registry for managing CLI argument groups and parser construction.
    Implements the centralized args registry system from the architecture plan.
    """

    def __init__(self):
        self._argument_groups: Dict[str, Callable[[argparse.ArgumentParser], None]] = {}
        self._subcommand_parsers: Dict[str, Callable] = {}
        self._register_default_groups()

    def _register_default_groups(self) -> None:
        """Register default argument groups."""
        # Core argument groups
        self.register_group("config", add_config_args)
        self.register_group("debug", add_debug_args)
        self.register_group("output", add_output_args)

        # Simulation argument groups
        self.register_group("simulation", add_simulation_args)
        self.register_group("network", add_network_args)
        self.register_group("traffic", add_traffic_args)

        # Training argument groups
        self.register_group("training", add_all_training_args)
        self.register_group("rl", add_reinforcement_learning_args)
        self.register_group("ml", add_machine_learning_args)

        # Analysis argument groups
        self.register_group("analysis", add_all_analysis_args)
        self.register_group("statistics", add_statistics_args)
        self.register_group("plotting", add_plotting_args)

        # Interface-specific groups
        self.register_group("gui", add_gui_args)

        # Legacy groups (for backward compatibility)
        self.register_group("run_sim", add_run_sim_args)
        self.register_group("routing", add_routing_args)
        self.register_group("spectrum", add_spectrum_args)
        self.register_group("snr", add_snr_args)
        self.register_group("sdn", add_sdn_args)
        self.register_group("stats", add_stats_args)
        self.register_group("plot", add_plot_args)

        # Register subcommand parsers
        self.register_subcommand("run_sim", register_run_sim_args)

    def register_group(self, name: str, add_args_func: Callable[[argparse.ArgumentParser], None]) -> None:
        """
        Register an argument group.

        Args:
            name: Group identifier
            add_args_func: Function that adds arguments to parser
        """
        self._argument_groups[name] = add_args_func

    def register_subcommand(self, name: str, register_func: Callable) -> None:
        """
        Register a subcommand parser.

        Args:
            name: Subcommand name
            register_func: Function that registers the subcommand
        """
        self._subcommand_parsers[name] = register_func

    def add_groups_to_parser(self, parser: argparse.ArgumentParser, group_names: List[str]) -> None:
        """
        Add specified argument groups to a parser.

        Args:
            parser: ArgumentParser instance
            group_names: List of group names to add
        """
        for group_name in group_names:
            if group_name in self._argument_groups:
                self._argument_groups[group_name](parser)
            else:
                raise ValueError(f"Unknown argument group: {group_name}")

    def create_parser_with_groups(self, description: str, group_names: List[str]) -> argparse.ArgumentParser:
        """
        Create a parser with specified argument groups.

        Args:
            description: Parser description
            group_names: List of group names to include

        Returns:
            Configured ArgumentParser
        """
        parser = argparse.ArgumentParser(description=description)
        self.add_groups_to_parser(parser, group_names)
        return parser

    def create_main_parser(self) -> argparse.ArgumentParser:
        """
        Create the main CLI parser with subcommands.

        Returns:
            Main ArgumentParser with subcommands
        """
        parser = argparse.ArgumentParser(description="FUSION Simulator CLI")
        subparsers = parser.add_subparsers(dest="mode", required=True)

        # Register subcommands
        for _, register_func in self._subcommand_parsers.items():
            register_func(subparsers)

        return parser

    def get_available_groups(self) -> List[str]:
        """Get list of available argument groups."""
        return list(self._argument_groups.keys())

    def get_available_subcommands(self) -> List[str]:
        """Get list of available subcommands."""
        return list(self._subcommand_parsers.keys())


# Global registry instance
args_registry = ArgumentRegistry()
