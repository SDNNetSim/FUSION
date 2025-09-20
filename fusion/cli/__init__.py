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
- run_gui.py: Graphical user interface launcher with dependency management

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
# Legacy function imports (backward compatibility)
# Modern function imports (recommended)
from .main_parser import (build_main_argument_parser, build_parser,
                          create_gui_argument_parser,
                          create_training_argument_parser, get_gui_args,
                          get_train_args)
# Core functionality imports
from .parameters.registry import args_registry

__all__ = [
    # Modern function names (recommended)
    "build_main_argument_parser",
    "create_training_argument_parser",
    "create_gui_argument_parser",
    # Legacy function names (backward compatibility)
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
