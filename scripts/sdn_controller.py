from scripts.routing import Routing
from scripts.spectrum_assignment import SpectrumAssignment


def handle_arrive_rel(req_id, network_spec_db, path, start_slot, num_slots, core_num=0, req_type=None):
    """
    Releases or fills slots in the network spectrum database arrays.

    :param req_id: The request's id number
    :type req_id: int
    :param network_spec_db: The current network spectrum database
    :type network_spec_db: dict
    :param path: The shortest path computed
    :type path: list
    :param start_slot: The first slot number taken or desired
    :type start_slot: int
    :param num_slots: The number of slots occupied or to be occupied
    :type num_slots: int
    :param core_num: Index of the core to be released or taken
    :type core_num: int
    :param req_type: Indicates whether it's an arrival or release
    :type req_type: str
    :return: The updated network spectrum database
    :rtype: dict
    """
    # TODO: This will most likely change for slicing, different functions?
    # TODO: Request number can be zero! (Free slot!) Uh oh...
    for i in range(len(path) - 1):
        if req_type == 'arrival':
            # Remember, Python list indexing is up to and NOT including!
            network_spec_db[(path[i], path[i + 1])]['cores_matrix'][core_num][start_slot:start_slot + num_slots - 1] \
                = req_id
            # A guard band for us is a -1, as it's important to differentiate the rest of the request from it
            network_spec_db[(path[i], path[i + 1])]['cores_matrix'][core_num][start_slot + num_slots - 1] = -req_id
        elif req_type == 'release':
            network_spec_db[(path[i], path[i + 1])]['cores_matrix'][core_num][start_slot:start_slot + num_slots] = 0
            network_spec_db[(path[i + 1], path[i])]['cores_matrix'][core_num][start_slot:start_slot + num_slots] = 0
        else:
            raise ValueError(f'Expected release or arrival, got {req_type}.')

    return network_spec_db


def controller_main(req_id, src, dest, request_type, physical_topology, network_spec_db, mod_formats,
                    slot_num=None, path=None, chosen_mod=None, chosen_bw=None):  # pylint: disable=unused-argument
    """
    Controls arrivals and departures for requests in the simulation. Return False if a request can't be allocated.

    :param req_id: The request's id number
    :type req_id: int
    :param src: Source node
    :type src: int
    :param dest: Destination node
    :type dest: int
    :param request_type: Determine if the request is an arrival or departure
    :type request_type: str
    :param physical_topology: The physical topology information
    :type physical_topology: graph
    :param network_spec_db: The network spectrum database, holding information about the network
    :type network_spec_db: dict
    :param mod_formats: Information relating to all modulation formats
    :type mod_formats: dict
    :param slot_num: The start slot number of the request to be allocated or released
    :type slot_num: int
    :param path: The chosen path
    :type path: list
    :param chosen_mod: The chosen modulation format
    :type chosen_mod: str
    :param chosen_bw: The chosen bandwidth
    :type chosen_bw: str
    :return: The modulation format, core, start slot, and end slot chosen along with the network DB and topology
    :rtype: (dict, dict, dict) or bool
    """
    # TODO: Modulation format hard coded for the moment, it's not important for temporary reasons
    if request_type == "release":
        network_spec_db = handle_arrive_rel(req_id=req_id,
                                            network_spec_db=network_spec_db,
                                            path=path,
                                            start_slot=slot_num,
                                            num_slots=mod_formats['QPSK']['slots_needed'],
                                            req_type='release'
                                            )
        return network_spec_db, physical_topology

    routing_obj = Routing(req_id=req_id, source=src, destination=dest, physical_topology=physical_topology,
                          network_spec_db=network_spec_db, mod_formats=mod_formats, bw=chosen_bw)

    # TODO: Whose routing function are we using?
    # TODO: Whose modulation format assumptions are we using?
    # This is used for Yue's assumptions in his dissertation
    # selected_path, path_mod, slots_needed = routing_obj.shortest_path()
    # Used for Arash's assumptions in previous research papers
    selected_path = routing_obj.least_congested_path()

    if selected_path is not False:
        # Hard coding QPSK for now, all slots needed are actually the same (Not actually QPSK)
        spectrum_assignment = SpectrumAssignment(selected_path, mod_formats['QPSK']['slots_needed'], network_spec_db)
        selected_sp = spectrum_assignment.find_free_spectrum()

        if selected_sp is not False:
            ras_output = {
                'path': selected_path,
                'mod_format': 'QPSK',
                'core_num': selected_sp['core_num'],
                'start_res_slot': selected_sp['start_slot'],
                'end_res_slot': selected_sp['end_slot'],
            }
            network_spec_db = handle_arrive_rel(req_id=req_id,
                                                network_spec_db=network_spec_db,
                                                path=selected_path,
                                                start_slot=selected_sp['start_slot'],
                                                num_slots=mod_formats['QPSK']['slots_needed'],
                                                req_type='arrival'
                                                )
            return ras_output, network_spec_db, physical_topology

        return False

    return False
