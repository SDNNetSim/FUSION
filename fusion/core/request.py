"""Request generation module for FUSION simulations.

This module handles the generation of arrival and departure requests for network simulations.
It creates requests based on specified distributions and traffic patterns.
"""

from typing import Dict, Any, List, Tuple
from fusion.utils.random import set_random_seed, generate_uniform_random_variable, generate_exponential_random_variable
from fusion.utils.logging_config import get_logger

# Module-level constants
DEFAULT_REQUEST_TYPE_ARRIVAL = "arrival"
DEFAULT_REQUEST_TYPE_RELEASE = "release"

# Set up module logger
logger = get_logger(__name__)

__all__ = ["generate_simulation_requests", "validate_request_distribution"]


def validate_request_distribution(
        request_distribution: Dict[str, float],
        number_of_requests: int
) -> bool:
    """Validate that request distribution can be properly allocated.
    
    Checks if the request distribution percentages can be evenly distributed
    across the specified number of requests without remainder.
    
    :param request_distribution: Bandwidth distribution percentages
    :type request_distribution: Dict[str, float]
    :param number_of_requests: Total number of requests to generate
    :type number_of_requests: int
    :return: True if distribution is valid, False otherwise
    :rtype: bool
    
    Example:
        >>> distribution = {"50GHz": 0.5, "100GHz": 0.5}
        >>> validate_request_distribution(distribution, 100)
        True
    """
    bandwidth_count_dict = {
        bandwidth: int(percentage * number_of_requests)
        for bandwidth, percentage in request_distribution.items()
    }

    total_allocated = sum(bandwidth_count_dict.values())
    return total_allocated == number_of_requests


def _select_random_node_pair(
        nodes_list: List[str]
) -> Tuple[str, str]:
    """Select a random source-destination pair from available nodes.
    
    Ensures that source and destination are different nodes.
    
    :param nodes_list: List of available node identifiers
    :type nodes_list: List[str]
    :return: Tuple of (source_node, destination_node)
    :rtype: Tuple[str, str]
    """
    source = nodes_list[generate_uniform_random_variable(scale_parameter=len(nodes_list))]
    destination = nodes_list[generate_uniform_random_variable(scale_parameter=len(nodes_list))]

    while destination == source:
        destination = nodes_list[generate_uniform_random_variable(scale_parameter=len(nodes_list))]

    return source, destination


def _generate_request_times(
        arrival_rate: float,
        holding_time: float,
        current_time: float
) -> Tuple[float, float]:
    """Generate arrival and departure times for a request.
    
    Uses exponential distribution to model arrival and holding times.
    
    :param arrival_rate: Rate parameter for arrival distribution
    :type arrival_rate: float
    :param holding_time: Mean holding time for requests
    :type holding_time: float
    :param current_time: Current simulation time
    :type current_time: float
    :return: Tuple of (arrival_time, departure_time)
    :rtype: Tuple[float, float]
    """
    arrival_time = current_time + generate_exponential_random_variable(scale_parameter=arrival_rate)
    departure_time = arrival_time + generate_exponential_random_variable(scale_parameter=1 / holding_time)

    return arrival_time, departure_time


def _create_request_entry(
        request_id: int,
        source: str,
        destination: str,
        arrival_time: float,
        departure_time: float,
        request_type: str,
        bandwidth: str,
        modulation_formats: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a single request dictionary entry.
    
    Note: Dictionary keys use abbreviated forms for backward compatibility,
    while function parameters use descriptive names per coding standards.
    
    :param request_id: Unique identifier for the request
    :type request_id: int
    :param source: Source node identifier
    :type source: str
    :param destination: Destination node identifier
    :type destination: str
    :param arrival_time: Time when request arrives
    :type arrival_time: float
    :param departure_time: Time when request departs
    :type departure_time: float
    :param request_type: Type of request (arrival or release)
    :type request_type: str
    :param bandwidth: Bandwidth requirement for the request
    :type bandwidth: str
    :param modulation_formats: Available modulation formats for this bandwidth
    :type modulation_formats: Dict[str, Any]
    :return: Request dictionary with all parameters
    :rtype: Dict[str, Any]
    """
    return {
        "req_id": request_id,  # Keep original key for compatibility
        "source": source,
        "destination": destination,
        "arrive": arrival_time,
        "depart": departure_time,
        "request_type": request_type,
        "bandwidth": bandwidth,
        "mod_formats": modulation_formats,  # Keep original key for compatibility
    }


def generate_simulation_requests(
        seed: int,
        engine_properties: Dict[str, Any]
) -> Dict[float, Dict[str, Any]]:
    """Generate requests for a single simulation run.
    
    Creates both arrival and departure events for network requests based on
    the specified traffic parameters and distributions. Each request has a
    unique ID and is placed at specific simulation times.
    
    :param seed: Random seed for reproducible generation
    :type seed: int
    :param engine_properties: Simulation engine configuration containing:
        - is_only_core_node: Whether all nodes can originate traffic
        - topology_info: Network topology information
        - core_nodes: List of core nodes (if not all nodes are core)
        - arrival_rate: Rate parameter for exponential arrivals
        - holding_time: Mean request duration
        - num_requests: Total number of requests to generate
        - request_distribution: Bandwidth distribution percentages
        - mod_per_bw: Modulation formats per bandwidth
    :type engine_properties: Dict[str, Any]
    :return: Dictionary mapping simulation times to request events
    :rtype: Dict[float, Dict[str, Any]]
    :raises ValueError: If no nodes available or distribution invalid
    
    Example:
        >>> engine_props = {
        ...     "is_only_core_node": True,
        ...     "topology_info": {"nodes": {"A": {}, "B": {}, "C": {}}},
        ...     "arrival_rate": 1.0,
        ...     "holding_time": 10.0,
        ...     "num_requests": 100,
        ...     "request_distribution": {"50GHz": 0.5, "100GHz": 0.5},
        ...     "mod_per_bw": {"50GHz": {...}, "100GHz": {...}}
        ... }
        >>> requests = generate_simulation_requests(42, engine_props)
    """
    logger.debug("Generating requests with seed %s", seed)

    # Initialize variables
    requests_dict: Dict[float, Dict[str, Any]] = {}
    current_time = 0.0
    request_id = 1

    # Determine available nodes for traffic generation
    if engine_properties['is_only_core_node']:
        nodes_list = list(engine_properties['topology_info']['nodes'].keys())
    else:
        nodes_list = engine_properties['core_nodes']

    # Validate nodes list
    if not nodes_list:
        if engine_properties['is_only_core_node']:
            topology_keys = list(engine_properties.get('topology_info', {}).keys())
            node_keys = list(
                engine_properties.get('topology_info', {}).get('nodes', {}).keys()
            ) if 'topology_info' in engine_properties else []
            error_msg = (
                f"No nodes found in topology_info. "
                f"is_only_core_node={engine_properties['is_only_core_node']}, "
                f"topology_keys={topology_keys}, node_keys={node_keys}"
            )
        else:
            error_msg = (
                f"No core nodes found. "
                f"is_only_core_node={engine_properties['is_only_core_node']}, "
                f"core_nodes={engine_properties.get('core_nodes', [])}"
            )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Set random seed for reproducibility
    set_random_seed(seed_value=seed)

    # Calculate bandwidth allocation counts
    bandwidth_count_dict = {
        bandwidth: int(
            engine_properties['request_distribution'][bandwidth] *
            engine_properties['num_requests']
        )
        for bandwidth in engine_properties['mod_per_bw']
    }
    bandwidth_list = list(engine_properties['mod_per_bw'].keys())

    # Validate distribution
    if not validate_request_distribution(
            engine_properties['request_distribution'],
            engine_properties['num_requests']
    ):
        error_msg = (
            "The number of requests could not be distributed according to the "
            "specified percentages. Please adjust either the number of requests "
            "or the bandwidth distribution percentages."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Generate requests (both arrival and departure events)
    total_events_needed = engine_properties['num_requests'] * 2

    while len(requests_dict) < total_events_needed:
        # Generate timing for this request
        arrival_time, departure_time = _generate_request_times(
            engine_properties['arrival_rate'],
            engine_properties['holding_time'],
            current_time
        )
        current_time = arrival_time

        # Select source and destination nodes
        source, destination = _select_random_node_pair(nodes_list)

        # Select bandwidth for this request
        chosen_bandwidth = None
        while chosen_bandwidth is None:
            candidate_bandwidth = bandwidth_list[
                generate_uniform_random_variable(scale_parameter=len(bandwidth_list))
            ]
            if bandwidth_count_dict[candidate_bandwidth] > 0:
                bandwidth_count_dict[candidate_bandwidth] -= 1
                chosen_bandwidth = candidate_bandwidth

        # Check if times are available (no collision)
        if arrival_time not in requests_dict and departure_time not in requests_dict:
            # Create arrival event
            requests_dict[arrival_time] = _create_request_entry(
                request_id=request_id,
                source=source,
                destination=destination,
                arrival_time=arrival_time,
                departure_time=departure_time,
                request_type=DEFAULT_REQUEST_TYPE_ARRIVAL,
                bandwidth=chosen_bandwidth,
                modulation_formats=engine_properties['mod_per_bw'][chosen_bandwidth]
            )

            # Create departure event
            requests_dict[departure_time] = _create_request_entry(
                request_id=request_id,
                source=source,
                destination=destination,
                arrival_time=arrival_time,
                departure_time=departure_time,
                request_type=DEFAULT_REQUEST_TYPE_RELEASE,
                bandwidth=chosen_bandwidth,
                modulation_formats=engine_properties['mod_per_bw'][chosen_bandwidth]
            )

            request_id += 1
            logger.debug(
                "Generated request %s: %s -> %s (%s)",
                request_id - 1, source, destination, chosen_bandwidth
            )
        else:
            # Time collision - return bandwidth to pool
            bandwidth_count_dict[chosen_bandwidth] += 1
            logger.debug(
                "Time collision at %s or %s, retrying",
                arrival_time, departure_time
            )

    logger.info("Generated %s requests", engine_properties['num_requests'])
    return requests_dict


# Maintain backward compatibility with old function name
def get_requests(seed: int, engine_props: dict) -> Dict[float, Dict[str, Any]]:
    """Legacy function name for backward compatibility.
    
    .. deprecated:: 2.0.0
        Use :func:`generate_simulation_requests` instead.
    
    :param seed: Random seed for generation
    :type seed: int
    :param engine_props: Engine properties dictionary
    :type engine_props: dict
    :return: Generated requests dictionary
    :rtype: Dict[float, Dict[str, Any]]
    """
    logger.warning(
        "get_requests is deprecated. Use generate_simulation_requests instead."
    )
    return generate_simulation_requests(seed, engine_props)
