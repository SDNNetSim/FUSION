"""
CLI arguments for simulation configuration.
Consolidates routing, spectrum, SNR, and SDN related arguments.
"""

import argparse
from .common_args import add_config_args, add_debug_args, add_output_args
from .training_args import add_machine_learning_args


def add_simulation_args(parser: argparse.ArgumentParser) -> None:
    """
    Add comprehensive simulation arguments to the parser.
    Includes routing, spectrum assignment, SNR, and network configuration.
    """
    # Routing arguments
    routing_group = parser.add_argument_group('Routing Configuration')
    routing_group.add_argument(
        "--route_method",
        type=str,
        choices=["shortest_path", "k_shortest_path", "yen", "congestion_aware", "fragmentation_aware"],
        help="Routing algorithm method"
    )
    routing_group.add_argument(
        "--k_paths",
        type=int,
        default=3,
        help="Number of candidate paths for k-shortest path routing"
    )

    # Spectrum assignment arguments
    spectrum_group = parser.add_argument_group('Spectrum Assignment Configuration')
    spectrum_group.add_argument(
        "--allocation_method",
        type=str,
        choices=["first_fit", "best_fit", "last_fit", "random_fit", "multi_band_priority"],
        help="Spectrum allocation method"
    )
    spectrum_group.add_argument(
        "--guard_slots",
        type=int,
        default=1,
        help="Number of guard slots between allocations"
    )
    spectrum_group.add_argument(
        "--spectrum_priority",
        type=str,
        choices=["c_band_first", "l_band_first", "balanced"],
        help="Priority order for multi-band allocation"
    )

    # SNR and modulation arguments
    snr_group = parser.add_argument_group('SNR and Modulation Configuration')
    snr_group.add_argument(
        "--mod_assumption",
        type=str,
        choices=["fixed", "adaptive", "precalculated"],
        help="Modulation format selection strategy"
    )
    snr_group.add_argument(
        "--mod_assumption_path",
        type=str,
        help="Path to modulation format configuration file"
    )
    snr_group.add_argument(
        "--snr_type",
        type=str,
        choices=["linear", "nonlinear", "egn"],
        help="SNR calculation method"
    )
    snr_group.add_argument(
        "--input_power",
        type=float,
        default=1e-3,
        help="Input power in Watts"
    )
    snr_group.add_argument(
        "--egn_model",
        action="store_true",
        help="Enable Enhanced Gaussian Noise model"
    )

    # SDN and dynamic switching
    sdn_group = parser.add_argument_group('SDN Configuration')
    sdn_group.add_argument(
        "--dynamic_lps",
        action="store_true",
        help="Enable SDN dynamic lightpath switching"
    )
    sdn_group.add_argument(
        "--single_core",
        action="store_true",
        help="Force single-core allocation per request"
    )


def add_network_args(parser: argparse.ArgumentParser) -> None:
    """
    Add network topology and physical layer arguments.
    """
    network_group = parser.add_argument_group('Network Configuration')
    network_group.add_argument(
        "--network",
        type=str,
        help="Network topology name (e.g., 'NSFNet', 'USbackbone60')"
    )
    network_group.add_argument(
        "--cores_per_link",
        type=int,
        default=1,
        help="Number of cores per fiber link"
    )
    network_group.add_argument(
        "--bw_per_slot",
        type=float,
        default=12.5,
        help="Bandwidth per spectral slot in GHz"
    )
    network_group.add_argument(
        "--const_link_weight",
        action="store_true",
        help="Use constant link weights for routing"
    )
    network_group.add_argument(
        "--bi_directional",
        action="store_true",
        help="Enable bidirectional links"
    )
    network_group.add_argument(
        "--multi_fiber",
        action="store_true",
        help="Enable multi-fiber links"
    )
    network_group.add_argument(
        "--is_only_core_node",
        action="store_true",
        help="Only allow core nodes to send requests"
    )

    # Spectrum band configuration
    spectrum_group = parser.add_argument_group('Spectrum Band Configuration')
    spectrum_group.add_argument(
        "--c_band",
        type=int,
        default=96,
        help="Number of spectral slots in C-band"
    )
    spectrum_group.add_argument(
        "--l_band",
        type=int,
        default=0,
        help="Number of spectral slots in L-band"
    )
    spectrum_group.add_argument(
        "--s_band",
        type=int,
        default=0,
        help="Number of spectral slots in S-band"
    )
    spectrum_group.add_argument(
        "--e_band",
        type=int,
        default=0,
        help="Number of spectral slots in E-band"
    )
    spectrum_group.add_argument(
        "--o_band",
        type=int,
        default=0,
        help="Number of spectral slots in O-band"
    )


def add_traffic_args(parser: argparse.ArgumentParser) -> None:
    """
    Add traffic generation and request handling arguments.
    """
    traffic_group = parser.add_argument_group('Traffic Configuration')
    traffic_group.add_argument(
        "--erlang_start",
        type=float,
        default=100.0,
        help="Starting Erlang load"
    )
    traffic_group.add_argument(
        "--erlang_stop",
        type=float,
        default=1000.0,
        help="Ending Erlang load"
    )
    traffic_group.add_argument(
        "--erlang_step",
        type=float,
        default=100.0,
        help="Erlang load increment"
    )
    traffic_group.add_argument(
        "--holding_time",
        type=float,
        default=1.0,
        help="Average holding time for requests"
    )
    traffic_group.add_argument(
        "--num_requests",
        type=int,
        default=10000,
        help="Total number of requests to generate"
    )
    traffic_group.add_argument(
        "--max_iters",
        type=int,
        default=3,
        help="Maximum number of simulation iterations"
    )
    traffic_group.add_argument(
        "--thread_erlangs",
        action="store_true",
        help="Enable multi-threaded Erlang processing"
    )


def add_run_sim_args(parser: argparse.ArgumentParser) -> None:
    """
    Add run_sim specific arguments. Legacy compatibility function.
    This consolidates arguments from simulation, network, and traffic groups.
    """
    add_simulation_args(parser)
    add_network_args(parser)
    add_traffic_args(parser)


def register_run_sim_args(subparsers) -> None:
    """
    Register run_sim subcommand parser. Legacy compatibility function.
    """
    parser = subparsers.add_parser('run_sim', help='Run network simulation')
    add_config_args(parser)
    add_debug_args(parser)
    add_output_args(parser)
    add_run_sim_args(parser)
    add_machine_learning_args(parser)
