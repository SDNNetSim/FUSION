# Standard library imports
import copy
import os
import signal

# Third party library imports
import networkx as nx
import numpy as np

# Local application imports
from src.request_generator import get_requests
from src.sdn_controller import SDNController
from helper_scripts.stats_helpers import SimStats
from helper_scripts.ml_helpers import load_model


class Engine:
    """
    Controls a single simulation.
    """

    def __init__(self, engine_props: dict):
        self.engine_props = engine_props
        self.net_spec_dict = dict()
        self.reqs_dict = None
        self.reqs_status_dict = dict()
        self.lightpath_status_dict = dict()
        self.transponder_usage_dict = dict()

        self.iteration = 0
        self.topology = nx.Graph()
        self.sim_info = os.path.join(self.engine_props['network'], self.engine_props['date'],
                                     self.engine_props['sim_start'])

        self.sdn_obj = SDNController(engine_props=self.engine_props)
        self.stats_obj = SimStats(engine_props=self.engine_props, sim_info=self.sim_info)

        self.ml_model = None

    def update_arrival_params(self, curr_time: float):
        """
        Updates parameters for a request after attempted allocation.

        :param curr_time: The current simulated time.
        """
        sdn_props = self.sdn_obj.sdn_props
        self.stats_obj.iter_update(req_data=self.reqs_dict[curr_time], sdn_data=sdn_props, net_spec_dict = self.net_spec_dict)
        if sdn_props.was_routed or (sdn_props.was_partially_routed and self.engine_props['can_partially_serve']):
            self.stats_obj.curr_trans = sdn_props.num_trans

            self.reqs_status_dict.update({self.reqs_dict[curr_time]['req_id']: {
                "mod_format": sdn_props.modulation_list,
                "path": sdn_props.path_list,
                "is_sliced": sdn_props.is_sliced,
                "was_routed": sdn_props.was_routed,
                "was_groomed": sdn_props.was_groomed,
                "was_new_lp_established": sdn_props.was_new_lp_established,
                "core_list": sdn_props.core_list,
                "start_slot_list":sdn_props.start_slot_list,
                "end_slot_list":sdn_props.end_slot_list,
                "bandwidth_list": sdn_props.bandwidth_list,
                "lightpath_id_list": sdn_props.lightpath_id_list,
                "lightpath_bandwidth_list": sdn_props.lightpath_bandwidth_list,
                "snr_cost": sdn_props.xt_list,
                "was_partially_routed": sdn_props.was_partially_routed,
                "was_partially_groomed": sdn_props.was_partially_groomed,
                "remaining_bw": sdn_props.remaining_bw,
                # TODO: Update
                "band": sdn_props.band_list,
            }})
        if sdn_props.was_new_lp_established:
            light_id = tuple(sorted([sdn_props.path_list[0], sdn_props.path_list[-1]]))
            for lp_cnt in range(0, len(sdn_props.lightpath_id_list)):
                if sdn_props.lightpath_id_list[lp_cnt] not in sdn_props.was_new_lp_established:
                    continue
                if light_id not in self.lightpath_status_dict:
                    self.lightpath_status_dict[light_id] = {}
                self.lightpath_status_dict[light_id][sdn_props.lightpath_id_list[lp_cnt]] = {
                    "path": sdn_props.path_list,
                    "core": sdn_props.core_list[lp_cnt],
                    "start_slot":sdn_props.start_slot_list[lp_cnt],
                    "end_slot":sdn_props.end_slot_list[lp_cnt],
                    "band": sdn_props.band_list[lp_cnt],
                    "mod_format": sdn_props.modulation_list[lp_cnt],
                    "snr_cost": sdn_props.xt_list[lp_cnt],
                    "lightpath_bandwidth": sdn_props.lightpath_bandwidth_list[lp_cnt], # TODO: check
                    "remaining_bandwidth": sdn_props.lightpath_bandwidth_list[lp_cnt] - int(sdn_props.bandwidth_list[lp_cnt]),
                    'requests_dict':{self.reqs_dict[curr_time]['req_id']: int(sdn_props.bandwidth_list[lp_cnt])},
                    'time_bw_usage':{sdn_props.arrive: (int(sdn_props.bandwidth_list[lp_cnt])/sdn_props.lightpath_bandwidth_list[lp_cnt])*100},
                    "path_weight": sdn_props.path_weight,
                }
            for node in [sdn_props.source, sdn_props.destination]:

                if node not in self.transponder_usage_dict:
                    raise KeyError(f"Node '{node}' not found in transponder usage dictionary.")
                
                required_transponders = sdn_props.num_trans
                available = self.transponder_usage_dict[node]["available_transponder"]
                if available >= required_transponders:
                    self.transponder_usage_dict[node]["available_transponder"] -= required_transponders
                else:
                    self.transponder_usage_dict[node]["available_transponder"] = 0
                    self.transponder_usage_dict[node]["total_transponder"] += required_transponders - available


    def update_release_param(self, curr_time: float):
        for req_key, req_value in self.reqs_dict[curr_time].items():
            # TODO: This should be changed in reqs_dict eventually
            if req_key == 'mod_formats':
                req_key = 'mod_formats_dict'
            self.sdn_obj.sdn_props.update_params(key=req_key, spectrum_key=None, spectrum_obj=None, value=req_value)
        for req_key, req_value in self.reqs_status_dict[self.reqs_dict[curr_time]['req_id']].items():
            props_key = req_key.split('_')[0]  # pylint: disable=use-maxsplit-arg
            if req_key == 'mod_format':
                props_key = 'modulation_list'
            elif props_key == 'path':
                props_key = 'path_list'
            elif props_key == 'band':
                props_key = 'band_list'
            elif props_key == 'snr_cost':
                props_key = 'xt_list'
            else:
                props_key = req_key
            self.sdn_obj.sdn_props.update_params(key=props_key, spectrum_key=None, spectrum_obj=None, value=req_value)


    def handle_arrival(self, curr_time: float, force_route_matrix: list = None, force_core: int = None,
                       force_slicing: bool = False, forced_index: int = None, force_mod_format: str = None):
        """
        Updates the SDN controller to handle an arrival request and retrieves relevant request statistics.

        :param curr_time: The arrival time of the request.
        :param force_route_matrix: Passes forced routes to the SDN controller.
        :param force_slicing: Forces slicing in the SDN controller.
        :param forced_index: Forces an index in the SDN controller.
        :param force_mod_format: Forces a modulation format.
        :param force_core: Force a certain core for allocation.
        """
        for req_key, req_value in self.reqs_dict[curr_time].items():
            # TODO: This should be changed in reqs_dict eventually
            if req_key == 'mod_formats':
                req_key = 'mod_formats_dict'
            if req_key in ['lightpath_id_list']:
                continue
            self.sdn_obj.sdn_props.update_params(key=req_key, spectrum_key=None, spectrum_obj=None, value=req_value)
        self.sdn_obj.sdn_props.remaining_bw = self.sdn_obj.sdn_props.bandwidth
        
        self.sdn_obj.handle_event(request_type='arrival', force_route_matrix=force_route_matrix,
                                  force_slicing=force_slicing, forced_index=forced_index, force_core=force_core,
                                  ml_model=self.ml_model, req_dict=self.reqs_dict[curr_time],
                                  force_mod_format=force_mod_format)
        self.net_spec_dict = self.sdn_obj.sdn_props.net_spec_dict
        self.lightpath_status_dict = self.sdn_obj.sdn_props.lightpath_status_dict 
        self.transponder_usage_dict = self.sdn_obj.sdn_props.transponder_usage_dict
        self.update_arrival_params(curr_time=curr_time)

    def handle_release(self, curr_time: float):
        """
        Updates the SDN controller to handle the release of a request.

        :param curr_time: The arrival time of the request.
        """
        if self.reqs_dict[curr_time]['req_id'] in self.reqs_status_dict:
            self.sdn_obj._init_req_stats()
            self.update_release_param(curr_time)
            self.sdn_obj.sdn_props.path_list = self.reqs_status_dict[self.reqs_dict[curr_time]['req_id']]['path']
            self.sdn_obj.handle_event(req_dict=self.reqs_dict[curr_time], request_type='release')
            self.stats_obj.update_utilization_dict(self.sdn_obj.sdn_props.lp_bw_utilization_dict)
            self.net_spec_dict = self.sdn_obj.sdn_props.net_spec_dict
            self.lightpath_status_dict = self.sdn_obj.sdn_props.lightpath_status_dict
            self.transponder_usage_dict = self.sdn_obj.sdn_props.transponder_usage_dict
        # Request was blocked, nothing to release
        else:
            pass

    def create_topology(self):
        """
        Create the physical topology of the simulation.
        """
        self.net_spec_dict = {}
        self.topology.add_nodes_from(self.engine_props['topology_info']['nodes'])

        # TODO: Improve this
        for band in ['c', 'l', 's', 'o', 'e']:
            try:
                if self.engine_props[f'{band}_band']:
                    self.engine_props['band_list'].append(band)
            except KeyError:
                continue

        for link_num, link_data in self.engine_props['topology_info']['links'].items():
            source = link_data['source']
            dest = link_data['destination']

            cores_matrix = dict()
            for band in self.engine_props['band_list']:
                # TODO: We might want to name it the same thing
                band_slots = self.engine_props[f'{band}_band']
                cores_matrix[band] = np.zeros((link_data['fiber']['num_cores'], band_slots))

            self.net_spec_dict[(source, dest)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.net_spec_dict[(dest, source)] = {'cores_matrix': cores_matrix, 'link_num': int(link_num)}
            self.topology.add_edge(source, dest, length=link_data['length'], nli_cost=None)

        self.engine_props['topology'] = self.topology
        self.stats_obj.topology = self.topology
        self.sdn_obj.sdn_props.net_spec_dict = self.net_spec_dict
        self.sdn_obj.sdn_props.topology = self.topology
        self.sdn_obj.sdn_props.lightpath_status_dict = self.lightpath_status_dict 
        self.sdn_obj.sdn_props.transponder_usage_dict = self.transponder_usage_dict

    def generate_requests(self, seed: int):
        """
        Calls the request generator to generate requests.

        :param seed: The seed to use for the random generation.
        """
        # TODO: Needs to be a flag for artificial intelligence (especially RL) simulations
        # seed = 0
        self.reqs_dict = get_requests(seed=seed, engine_props=self.engine_props)
        self.reqs_dict = dict(sorted(self.reqs_dict.items(), key=lambda x: x[0][1]))

    def initialize_transponder_usage_dict(self):
        """
        Initializes the transponder usage dictionary for all core nodes in the network.

        This method creates a dictionary where each node is assigned an entry
        with `available_transponder` and `total_transponder` both set to 0.

        Updates:
            self.transponder_usage_dict (dict): A dictionary with the node names as keys 
                                                and their transponder usage details as values.
        """
        nodes_list = (
            list(self.engine_props['topology_info']['nodes'].keys())
            if self.engine_props['is_only_core_node']
            else self.engine_props['core_nodes']
        )

        self.transponder_usage_dict = {
            node: {"available_transponder": 0, "total_transponder": 0}
            for node in nodes_list
        }
        self.sdn_obj.sdn_props.transponder_usage_dict  = self.transponder_usage_dict 


    def handle_request(self, curr_time: float, req_num: int):
        """
        Carries out arrival or departure functions for a given request.

        :param curr_time: The current simulated time.
        :param req_num: The request number.
        """
        req_type = self.reqs_dict[curr_time]["request_type"]
        if req_type == "arrival":
            old_net_spec_dict = copy.deepcopy(self.net_spec_dict)
            old_req_info_dict = copy.deepcopy(self.reqs_dict[curr_time])
            old_lightpath_dict = copy.deepcopy(self.lightpath_status_dict)  # TODO: check for future AI based methods
            self.handle_arrival(curr_time=curr_time)

            if self.engine_props['save_snapshots'] and req_num % self.engine_props['snapshot_step'] == 0:
                self.stats_obj.update_snapshot(net_spec_dict=self.net_spec_dict, req_num=req_num)

            if self.engine_props['output_train_data']:
                was_routed = self.sdn_obj.sdn_props.was_routed
                if was_routed:
                    req_info_dict = self.reqs_status_dict[self.reqs_dict[curr_time]['req_id']]
                    self.stats_obj.update_train_data(old_req_info_dict=old_req_info_dict, req_info_dict=req_info_dict,
                                                     net_spec_dict=old_net_spec_dict)

        elif req_type == "release":
            self.handle_release(curr_time=curr_time)
        else:
            raise NotImplementedError(f'Request type unrecognized. Expected arrival or release, '
                                      f'got: {req_type}')
        if self.engine_props['fragmentation_metrics']:
            self.stats_obj.update_frag_metric_iter(req_id = self.reqs_dict[curr_time]['req_id'], net_spec_dict= self.net_spec_dict, req_type =  req_type)

    def end_iter(self, iteration: int, print_flag: bool = True, base_fp: str = None):
        """
        Updates iteration statistics.

        :param iteration: The current iteration.
        :param print_flag: Whether to print or not.
        :param base_fp: The base file path to save output statistics.
        """
        self.stats_obj.get_blocking()
        if self.engine_props["transponder_usage_per_node"]:
            total_transponders = sum(
            node_info["total_transponder"]
                for node_info in self.transponder_usage_dict.values()
            )
        else:
            total_transponders = None
        self.stats_obj.end_iter_update(total_transponders = total_transponders)
        # Some form of ML/RL is being used, ignore confidence intervals for training and testing
        if not self.engine_props['is_training']:
            resp = bool(self.stats_obj.get_conf_inter())
        else:
            resp = False
        if (iteration + 1) % self.engine_props['print_step'] == 0 or iteration == 0:
            self.stats_obj.print_iter_stats(max_iters=self.engine_props['max_iters'], print_flag=print_flag)

        if (iteration + 1) % self.engine_props['save_step'] == 0 or iteration == 0 or (iteration + 1) == self.engine_props['max_iters']:
            self.stats_obj.save_stats(base_fp=base_fp)

        
        return resp
    def reset(self):
        "reset Parameters for intiating new iteration"
        self.sdn_obj.sdn_props.reset_lightpath_id_couter()
        self.lightpath_status_dict = dict()
        self.sdn_obj.sdn_props.lightpath_status_dict = dict()
        self.sdn_obj.grooming_obj.grooming_props.lightpath_status_dict = dict()
        self.sdn_obj.sdn_props.transponder_usage_dict = dict()
        self.reqs_status_dict = dict()
        self.transponder_usage_dict = dict()


    def init_iter(self, iteration: int):
        """
        Initializes an iteration.

        :param iteration: The current iteration number.
        """
        self.iteration = iteration

        self.stats_obj.iteration = iteration
        self.stats_obj.init_iter_stats()
        # To prevent incomplete saves
        try:
            signal.signal(signal.SIGINT, self.stats_obj.save_stats)
            signal.signal(signal.SIGTERM, self.stats_obj.save_stats)
        # Signal only works in the main thread...
        except ValueError:
            pass

        if iteration == 0:
            print(f"Simulation started for Erlang: {self.engine_props['erlang']} "
                  f"simulation number: {self.engine_props['thread_num']}.")

            if self.engine_props['deploy_model']:
                self.ml_model = load_model(engine_props=self.engine_props)

        seed = self.engine_props["seeds"][iteration] if self.engine_props["seeds"] else iteration + 1
        
        self.reset()
        self.generate_requests(seed)
        if self.engine_props["transponder_usage_per_node"]:
            self.initialize_transponder_usage_dict()

    def run(self):
        """
        Controls the Engine class methods.
        """
        self.create_topology()
        for iteration in range(self.engine_props["max_iters"]):
            self.init_iter(iteration=iteration)
            req_num = 1
            for curr_time in self.reqs_dict:
                self.handle_request(curr_time=curr_time, req_num=req_num)

                if self.reqs_dict[curr_time]['request_type'] == 'arrival':
                    req_num += 1

            end_iter = self.end_iter(iteration=iteration)
            if end_iter:
                break

        print(f"Erlang: {self.engine_props['erlang']} finished for "
              f"simulation number: {self.engine_props['thread_num']}.")
