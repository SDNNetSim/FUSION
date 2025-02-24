from random import sample  
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

    # Determine the list of available nodes  
    if engine_props['is_only_core_node']:  
        nodes_list = list(engine_props['topology_info']['nodes'].keys())  
    else:  
        nodes_list = engine_props['core_nodes']  

    set_seed(seed=seed)  

    # Get number of sources and destinations  
    num_sources = engine_props.get('num_sources', 2)  
    num_destinations = engine_props.get('num_destinations', 2)  

    # Validate that the number of sources/destinations does not exceed available nodes  
    if engine_props['multi_source_multi_destination']:  
        if num_sources > len(nodes_list) or num_destinations > len(nodes_list):  
            raise ValueError("Number of sources or destinations specified exceeds the number of available nodes.")  

    # Calculate bandwidth distribution  
    bw_counts_dict = {bandwidth: int(engine_props['request_distribution'][bandwidth] * engine_props['num_requests'])  
                      for bandwidth in engine_props['mod_per_bw']}  
    bandwidth_list = list(engine_props['mod_per_bw'].keys())  

    # Validate bandwidth distribution  
    difference = engine_props['num_requests'] - sum(bw_counts_dict.values())  
    if difference != 0:  
        raise ValueError('The number of requests could not be distributed in the percentage distributed input. Please'  
                         'either change the number of requests, or change the percentages for the bandwidth values'  
                         'selected.')  

    # Generate requests  
    while len(requests_dict) < (engine_props['num_requests'] * 2):  
        current_time += get_exponential_rv(scale_param=engine_props['arrival_rate'])  
        depart_time = current_time + get_exponential_rv(scale_param=1 / engine_props['holding_time'])  

        # Select sources and destinations  
        if engine_props['multi_source_multi_destination']:  
            sources = sample(nodes_list, num_sources)  
            remaining_nodes = [node for node in nodes_list if node not in sources]  
            destinations = sample(remaining_nodes, num_destinations)  
        else:  
            source = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]  
            dest = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]  
            while dest == source:  
                dest = nodes_list[get_uniform_rv(scale_param=len(nodes_list))]  

        # Choose bandwidth  
        while True:  
            chosen_bandwidth = bandwidth_list[get_uniform_rv(scale_param=len(bandwidth_list))]  
            if bw_counts_dict[chosen_bandwidth] > 0:  
                bw_counts_dict[chosen_bandwidth] -= 1  
                break  

        # Add request to dictionary  
        if current_time not in requests_dict and depart_time not in requests_dict:  
            request_data = {  
                "req_id": request_id,  
                "source": sources if engine_props['multi_source_multi_destination'] else source,  
                "destination": destinations if engine_props['multi_source_multi_destination'] else dest,  
                "arrive": current_time,  
                "depart": depart_time,  
                "request_type": "arrival",  
                "bandwidth": chosen_bandwidth,  
                "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],  
            }  
            requests_dict[current_time] = request_data  

            release_data = {  
                "req_id": request_id,  
                "source": sources if engine_props['multi_source_multi_destination'] else source,  
                "destination": destinations if engine_props['multi_source_multi_destination'] else dest,  
                "arrive": current_time,  
                "depart": depart_time,  
                "request_type": "release",  
                "bandwidth": chosen_bandwidth,  
                "mod_formats": engine_props['mod_per_bw'][chosen_bandwidth],  
            }  
            requests_dict[depart_time] = release_data  

            request_id += 1  
        else:  
            bw_counts_dict[chosen_bandwidth] += 1  

    return requests_dict
