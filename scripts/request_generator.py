import numpy as np

# TODO: Update docs


def generate(seed_no, nodes, holding_time_mean, inter_arrival_time_mean, req_no,
             slot_dict):
    """
    Generates every request with the necessary information inside each request.

    :param seed_no: The simulation seed number
    :type seed_no: int
    :param nodes: A list of nodes
    :type nodes: list
    :param holding_time_mean: The holding time mean
    :type holding_time_mean: int
    :param inter_arrival_time_mean: The arrival time mean
    :type inter_arrival_time_mean: int
    :param req_no: Number of requests to be created
    :type req_no: int
    :param slot_dict: A dictionary mapping bandwidths to slot numbers
    :return: Every request generated
    :rtype: dict
    """
    np.random.seed(seed_no)
    requests = {}
    current = 0
    counter_id = 0

    # TODO: What to do when these don't evenly distribute?
    # We want to create requests in a 3:5:2 fashion
    bw_ratio_one = 3
    bw_ratio_two = 5
    bw_ratio_three = 2
    req_per_ratio = req_no / (bw_ratio_one + bw_ratio_two + bw_ratio_three)

    # Number of requests allocated for each bandwidth
    bw_one_req = bw_ratio_one * req_per_ratio
    bw_two_req = bw_ratio_two * req_per_ratio
    bw_three_req = bw_ratio_three * req_per_ratio

    # Monitor the number of requests allocated for each bandwidth
    bands_dict = {'50': bw_one_req, '100': bw_two_req, '400': bw_three_req}
    # List of all possible bandwidths
    bands_list = list(slot_dict.keys())
    while len(requests) < (req_no * 2):
        current = current + np.random.exponential(inter_arrival_time_mean)
        new_hold = current + np.random.exponential(holding_time_mean)

        src = nodes[np.random.randint(0, len(nodes))]
        des = nodes[np.random.randint(0, len(nodes))]

        while src == des:
            des = nodes[np.random.randint(0, len(nodes))]

        while True:
            chosen_band = bands_list[np.random.randint(0, len(bands_list))]  # pylint: disable=invalid-sequence-index
            if bands_dict[chosen_band] != 0:
                bands_dict[chosen_band] -= 1
                break
            else:
                continue

        if current not in requests and new_hold not in requests:
            requests.update({current: {
                "id": counter_id,
                "source": src,
                "destination": des,
                "arrive": current,
                "depart": new_hold,
                "request_type": "Arrival",
                "bandwidth": chosen_band,
                "mod_formats": slot_dict[chosen_band],
                "start_slot_NO": None,
                "working_path": None,
                "protection_path": None

            }})
            requests.update({new_hold: {
                "id": counter_id,
                "source": src,
                "destination": des,
                "arrive": current,
                "depart": new_hold,
                "request_type": "Release",
                "bandwidth": chosen_band,
                "mod_formats": slot_dict[chosen_band],
                "start_slot_NO": None,
                "working_path": None,
                "protection_path": None

            }})
            counter_id += 1
        else:
            raise NotImplementedError('This line of code should not be reached.')

    return requests
