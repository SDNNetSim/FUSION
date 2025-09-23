"""
Centralized argument registry for CLI commands.
Provides a unified interface for managing CLI argument parsing.
"""

import argparse
from collections.abc import Callable

from .analysis import (
    add_all_analysis_args,
    add_comparison_args,
    add_export_args,
    add_filtering_args,
    add_plotting_args,
    add_statistics_args,
)
from .gui import add_gui_args
from .network import (
    add_link_args,
    add_network_args,
    add_node_args,
    add_spectrum_bands_args,
    add_topology_args,
)
from .routing import (
    add_modulation_args,
    add_routing_args,
    add_sdn_args,
    add_spectrum_args,
)
from .shared import add_config_args, add_debug_args, add_output_args
from .simulation import add_run_sim_args, add_simulation_args, register_run_sim_args
from .snr import add_snr_args
from .traffic import (
    add_erlang_args,
    add_request_args,
    add_simulation_control_args,
    add_traffic_args,
)
from .training import (
    add_all_training_args,
    add_machine_learning_args,
    add_reinforcement_learning_args,
)


class ArgumentRegistry:
    """
    Registry for managing CLI argument groups and parser construction.

    Implements the centralized argument registry system following the FUSION
    architecture plan. Provides unified interface for argument group management,
    parser construction, and subcommand registration.

    The registry supports both focused argument groups for granular control
    and compatibility groups for backward compatibility with existing code.
    """

    def __init__(self) -> None:
        """
        Initialize the argument registry.

        Creates empty registries for argument groups and subcommands,
        then registers all default groups and subcommands.
        """
        self._argument_groups: dict[str, Callable[[argparse.ArgumentParser], None]] = {}
        self._subcommand_parsers: dict[str, Callable] = {}
        self._register_default_groups()

    def _register_default_groups(self) -> None:
        """
        Register all default argument groups.

        Organizes groups by category: core, compatibility, focused,
        training, analysis, GUI, and legacy groups.
        """
        self._register_core_groups()
        self._register_compatibility_groups()
        self._register_focused_groups()
        self._register_training_groups()
        self._register_analysis_groups()
        self._register_interface_groups()
        self._register_legacy_groups()
        self._register_subcommands()

    def _register_core_groups(self) -> None:
        """Register core argument groups used across all commands."""
        self.register_group("config", add_config_args)
        self.register_group("debug", add_debug_args)
        self.register_group("output", add_output_args)

    def _register_compatibility_groups(self) -> None:
        """Register compatibility argument groups for broad functionality."""
        self.register_group("simulation", add_simulation_args)
        self.register_group("network", add_network_args)
        self.register_group("traffic", add_traffic_args)

    def _register_focused_groups(self) -> None:
        """Register focused argument groups for granular control."""
        self.register_group("routing", add_routing_args)
        self.register_group("spectrum", add_spectrum_args)
        self.register_group("modulation", add_modulation_args)
        self.register_group("snr", add_snr_args)
        self.register_group("sdn", add_sdn_args)
        self.register_group("topology", add_topology_args)
        self.register_group("links", add_link_args)
        self.register_group("nodes", add_node_args)
        self.register_group("spectrum_bands", add_spectrum_bands_args)
        self.register_group("erlang", add_erlang_args)
        self.register_group("requests", add_request_args)
        self.register_group("sim_control", add_simulation_control_args)

    def _register_training_groups(self) -> None:
        """Register machine learning and reinforcement learning argument groups."""
        self.register_group("training", add_all_training_args)
        self.register_group("rl", add_reinforcement_learning_args)
        self.register_group("ml", add_machine_learning_args)

    def _register_analysis_groups(self) -> None:
        """Register analysis, plotting, and data export argument groups."""
        self.register_group("analysis", add_all_analysis_args)
        self.register_group("statistics", add_statistics_args)
        self.register_group("plotting", add_plotting_args)
        self.register_group("export", add_export_args)
        self.register_group("comparison", add_comparison_args)
        self.register_group("filtering", add_filtering_args)

    def _register_interface_groups(self) -> None:
        """Register user interface argument groups."""
        self.register_group("gui", add_gui_args)

    def _register_legacy_groups(self) -> None:
        """Register legacy argument groups for backward compatibility."""
        self.register_group("run_sim", add_run_sim_args)

    def _register_subcommands(self) -> None:
        """Register subcommand parsers."""
        self.register_subcommand("run_sim", register_run_sim_args)

    def register_group(
        self, name: str, add_args_func: Callable[[argparse.ArgumentParser], None]
    ) -> None:
        """
        Register an argument group with the registry.

        :param name: Unique identifier for the argument group
        :type name: str
        :param add_args_func: Function that adds arguments to parser
        :type add_args_func: Callable[[argparse.ArgumentParser], None]
        :return: None
        :rtype: None
        :raises ValueError: If group name already exists
        """
        if name in self._argument_groups:
            raise ValueError(f"Argument group '{name}' is already registered")
        self._argument_groups[name] = add_args_func

    def register_subcommand(self, name: str, register_func: Callable) -> None:
        """
        Register a subcommand parser with the registry.

        :param name: Subcommand name
        :type name: str
        :param register_func: Function that registers the subcommand
        :type register_func: Callable
        :return: None
        :rtype: None
        :raises ValueError: If subcommand name already exists
        """
        if name in self._subcommand_parsers:
            raise ValueError(f"Subcommand '{name}' is already registered")
        self._subcommand_parsers[name] = register_func

    def add_groups_to_parser(
        self, parser: argparse.ArgumentParser, group_names: list[str]
    ) -> None:
        """
        Add specified argument groups to an ArgumentParser.

        :param parser: ArgumentParser instance to modify
        :type parser: argparse.ArgumentParser
        :param group_names: List of group names to add
        :type group_names: List[str]
        :return: None
        :rtype: None
        :raises ValueError: If any group name is not registered
        """
        for group_name in group_names:
            if group_name not in self._argument_groups:
                available_groups = list(self._argument_groups.keys())
                raise ValueError(
                    f"Unknown argument group: '{group_name}'. "
                    f"Available groups: {available_groups}"
                )
            self._argument_groups[group_name](parser)

    def create_parser_with_groups(
        self, description: str, group_names: list[str]
    ) -> argparse.ArgumentParser:
        """
        Create an ArgumentParser with specified argument groups.

        :param description: Parser description text
        :type description: str
        :param group_names: List of group names to include
        :type group_names: List[str]
        :return: Configured ArgumentParser instance
        :rtype: argparse.ArgumentParser
        :raises ValueError: If any group name is not registered
        """
        parser = argparse.ArgumentParser(description=description)
        self.add_groups_to_parser(parser, group_names)
        return parser

    def create_main_parser(self) -> argparse.ArgumentParser:
        """
        Create the main CLI parser with all registered subcommands.

        :return: Main ArgumentParser with subcommands configured
        :rtype: argparse.ArgumentParser
        """
        parser = argparse.ArgumentParser(
            description="FUSION Simulator CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        subparsers = parser.add_subparsers(
            dest="mode",
            required=True,
            help="Available simulation modes"
        )

        for _, register_func in self._subcommand_parsers.items():
            register_func(subparsers)

        return parser

    def get_available_groups(self) -> list[str]:
        """
        Get list of all registered argument groups.

        :return: Sorted list of available group names
        :rtype: List[str]
        """
        return sorted(self._argument_groups.keys())

    def get_available_subcommands(self) -> list[str]:
        """
        Get list of all registered subcommands.

        :return: Sorted list of available subcommand names
        :rtype: List[str]
        """
        return sorted(self._subcommand_parsers.keys())

    def get_group_count(self) -> int:
        """
        Get total number of registered argument groups.

        :return: Number of registered groups
        :rtype: int
        """
        return len(self._argument_groups)

    def has_group(self, name: str) -> bool:
        """
        Check if an argument group is registered.

        :param name: Group name to check
        :type name: str
        :return: True if group exists, False otherwise
        :rtype: bool
        """
        return name in self._argument_groups

    def has_subcommand(self, name: str) -> bool:
        """
        Check if a subcommand is registered.

        :param name: Subcommand name to check
        :type name: str
        :return: True if subcommand exists, False otherwise
        :rtype: bool
        """
        return name in self._subcommand_parsers


args_registry = ArgumentRegistry()
