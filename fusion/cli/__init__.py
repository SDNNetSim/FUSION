# fusion/cli/__init__.py
"""
fusion.cli: Command-line interface entrypoints and argument parsing.

This package provides a clean CLI architecture following the FUSION refactor plan:
- Minimal entry scripts that delegate to appropriate modules
- Centralized argument registry system for consistent parsing
- Modular argument groups for reusability
- Proper error handling and validation

Entry Points:
- run_sim.py: Main simulation runner
- run_train.py: Agent training (RL/ML)
- run_gui.py: GUI launcher

Architecture:
- args/: Modular argument definitions and registry
- main_parser.py: Centralized parser construction
- config_setup.py: Configuration management
"""

from .main_parser import build_parser, get_train_args, get_gui_args
from .args.registry import args_registry
from .config_setup import setup_config_from_cli, ConfigManager

__all__ = [
    'build_parser',
    'get_train_args',
    'get_gui_args',
    'args_registry',
    'setup_config_from_cli',
    'ConfigManager'
]
