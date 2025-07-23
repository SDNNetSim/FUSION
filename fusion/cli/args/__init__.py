# fusion/cli/args/__init__.py
"""
This module centralizes all CLI argument registration functions.
Each function should register arguments for a specific subcommand.
"""

from .run_sim_args import register_run_sim_args
# from .plot_args import register_plot_args
# from .routing_args import register_routing_args
# from .spectrum_args import register_spectrum_args
# from .snr_args import register_snr_args
# from .sdn_args import register_sdn_args
# from .stats_args import register_stats_args

# Optionally, if you use shared argument groups or base flags
try:
    from .common_args import register_common_args
except ImportError:
    register_common_args = None  # Optional if not using common_args yet

__all__ = [
    "register_run_sim_args",
    # "register_plot_args",
    # "register_routing_args",
    # "register_spectrum_args",
    # "register_snr_args",
    # "register_sdn_args",
    # "register_stats_args",
    # "register_common_args"
]
