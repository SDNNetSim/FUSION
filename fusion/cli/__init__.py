"""
fusion.cli: Command-line interface entry points and argument parsing.

Provides a clean CLI architecture following the FUSION refactor plan with
modern Python practices, comprehensive error handling, and maintainable code
organization following established coding standards.

Features:
- Minimal entry scripts that delegate to appropriate modules
- Centralized argument registry system for consistent parsing
- Modular argument groups for reusability and maintainability
- Comprehensive error handling with specific exception types
- Proper type annotations and Sphinx-style documentation
- Backward compatibility for existing integrations

Entry Points:
- run_sim.py: Main network simulation runner with multiprocessing support
- run_train.py: Machine learning and reinforcement learning agent training

Core Modules:
- parameters/: Modular argument definitions and centralized registry system
- main_parser.py: Centralized parser construction with modern naming
- config_setup.py: Configuration management with proper error handling
- constants.py: Shared CLI constants including exit codes and settings

Backward Compatibility:
All legacy function names are maintained through compatibility aliases.
"""

from .config_setup import ConfigManager, setup_config_from_cli
from .constants import ERROR_EXIT_CODE, INTERRUPT_EXIT_CODE, SUCCESS_EXIT_CODE
from .main_parser import (
    build_main_argument_parser,
    build_parser,
    create_gui_argument_parser,
    create_training_argument_parser,
    get_gui_args,
    get_train_args,
)
from .parameters.registry import args_registry

# TODO (v6.1.0): Remove legacy function aliases (build_parser, get_train_args, get_gui_args)
# and their corresponding exports. Users should migrate to the modern function names.
__all__ = [
    "build_main_argument_parser",
    "create_training_argument_parser",
    "create_gui_argument_parser",
    # Legacy aliases - deprecated, remove in v6.1.0
    "build_parser",
    "get_train_args",
    "get_gui_args",
    # Core functionality
    "args_registry",
    "setup_config_from_cli",
    "ConfigManager",
    # CLI constants
    "SUCCESS_EXIT_CODE",
    "ERROR_EXIT_CODE",
    "INTERRUPT_EXIT_CODE",
]
