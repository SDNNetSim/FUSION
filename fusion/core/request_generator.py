from helper_scripts.random_helpers import set_seed, get_uniform_rv, get_exponential_rv


def get_requests(seed: int, engine_props: dict):
    """
    Generates requests for a single simulation.

    :param seed: Seed for random generation.
    :param engine_props: Properties from the engine class.
    :return: The generated requests and request information.
    :rtype: dict
    """
    requests_dict = {}
    current_time = 0
    request_id = 1
    # Means ALL nodes are core nodes
    if engine_props['is_only_core_node']:
        nodes_list = list(engine_props['topology_info']['nodes'].keys())
    # Means some nodes are nodes
    else:
        nodes_list = engine_props['core_nodes']
    set_seed(seed=seed)

    bw_counts_dict = {bandwidth: int(engine_props['request_distribution'][bandwidth] * engine_props['num_requests'])
                      for bandwidth in engine_props['mod_per_bw']}
    bandwidth_list = list(engine_props['mod_per_bw'].keys())

    # Check to see if the number of requests can be distributed
    difference = engine_props['num_requests'] - sum(bw_counts_dict.values())
    if difference != 0:
        raise ValueError('The number of requests could not be distributed in the percentage distributed input. Please'
                         'either change the number of requests, or change the percentages for the bandwidth values'
                         'selected.')

    # Generate requests, multiply the number of requests by two since we have arrival and departure types
    while len(requests_dict) < (engine_props['num_requests'] * 2):
        current_time += get_exponential_rv(scale_param=engine_props['arrival_rate'])

        depart_time = current_time + get_exponential_rv(scale_param=1 / engine_props['holding_time'])

        source = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]
        dest = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]

        while dest == source:
            dest = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]

        while True:
            chosen_bandwidth = bandwidth_list[get_uniform_rv(scale_param=len(bandwidth_list))]
            if bw_counts_dict[chosen_bandwidth] > 0:
                bw_counts_dict[chosen_bandwidth] -= 1
                break

        if current_time not in requests_dict and depart_time not in requests_dict:
            requests_dict.update({current_time: {
                "req_id": request_id,
                "source": source,
                "destination": dest,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "arrival",
                "bandwidth": chosen_bandwidth,
                "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
            }})
            requests_dict.update({depart_time: {
                "req_id": request_id,
                "source": source,
                "destination": dest,
                "arrive": current_time,
                "depart": depart_time,
                "request_type": "release",
                "bandwidth": chosen_bandwidth,
                "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],
            }})
            request_id += 1
        # Bandwidth was not chosen due to either arrival or depart time already existing, add back to distribution
        else:
            bw_counts_dict[chosen_bandwidth] += 1

    return requests_dict
