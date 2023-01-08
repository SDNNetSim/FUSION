import math

import networkx as nx
import numpy as np


class Routing:
    """
    Contains the routing methods for the simulation.
    """

    def __init__(self, req_id, source, destination, physical_topology, network_spec_db, mod_formats,
                 slots_needed=None, bw=None):  # pylint: disable=invalid-name
        self.path = None

        self.req_id = req_id
        self.source = source
        self.destination = destination
        self.physical_topology = physical_topology
        self.network_spec_db = network_spec_db
        self.slots_needed = slots_needed
        self.bw = bw  # pylint: disable=invalid-name

        self.mod_formats = mod_formats

        self.paths_list = list()

    def find_least_cong_route(self):
        """
        Given a list of dictionaries containing the most congested routes for each path,
        find the least congested route.

        :return: The least congested route
        :rtype: list
        """
        # Sort dictionary by number of free slots, descending (least congested)
        sorted_paths_list = sorted(self.paths_list, key=lambda d: d['link_info']['free_slots'], reverse=True)

        return sorted_paths_list[0]['path']

    def find_most_cong_link(self, path):
        """
        Given a list of nodes, or a path, find the most congested link between all nodes. Count how many
        slots are taken. For multiple cores, the spectrum slots occupied is added for each link.

        :param path: A given path
        :type path: list
        """
        res_dict = {'link': None, 'free_slots': None}

        for i in range(len(path) - 1):
            cores_matrix = self.network_spec_db[(path[i]), path[i + 1]]['cores_matrix']
            link_num = self.network_spec_db[(path[i]), path[i + 1]]['link_num']
            # The total amount of free spectral slots
            free_slots = 0

            for core_num, core_arr in enumerate(cores_matrix):  # pylint: disable=unused-variable
                free_slots += len(np.where(core_arr == 0)[0])
                # We want to find the least amount of free slots
            if res_dict['free_slots'] is None or free_slots < res_dict['free_slots']:
                res_dict['free_slots'] = free_slots
                res_dict['link'] = link_num

        # Link info is information about the most congested link found
        self.paths_list.append({'path': path, 'link_info': res_dict})

    def least_congested_path(self):
        """
        Given a graph with a desired source and destination, implement the least congested pathway algorithm. (Based on
        Arash Rezaee's research paper's assumptions)

        :return: The least congested path
        :rtype: list
        """
        paths_obj = nx.shortest_simple_paths(G=self.physical_topology, source=self.source, target=self.destination)
        min_hops = None

        for i, path in enumerate(paths_obj):
            num_hops = len(path)
            if i == 0:
                min_hops = num_hops
                self.find_most_cong_link(path)
            else:
                if num_hops <= min_hops + 1:
                    self.find_most_cong_link(path)
                else:
                    path = self.find_least_cong_route()
                    mod_format = 'QPSK'

                    # TODO: Find a better way to do this it's sort of redundant
                    if self.bw == '100':
                        slots_needed = 3
                    elif self.bw == '400':
                        slots_needed = 10
                    else:
                        raise NotImplementedError

                    return path, mod_format, slots_needed

        return False, False, False

    def shortest_path(self):
        """
        Given a graph with a desired source and destination, find the shortest path with respect to link lengths.

        :return: The shortest path
        :rtype: list
        """
        paths_obj = nx.shortest_simple_paths(G=self.physical_topology, source=self.source, target=self.destination,
                                             weight='length')

        # Modulation format calculations based on Yue Wang's dissertation
        for path in paths_obj:
            mod_format, slots_needed = self.assign_mod_format(path)
            return path, mod_format, slots_needed

    def spectral_slot_comp(self, bits_per_symbol, bw_slot=12.5):
        """
        Compute the amount of spectral slots needed.

        :param bits_per_symbol:  The number of bits per symbol
        :type bits_per_symbol: int
        :param bw_slot: The frequency for one spectral slot
        :type bw_slot: float
        :return: Amount of spectral slots needed to allocate a request
        :rtype: int
        """
        return math.ceil(float(self.bw) / float(bits_per_symbol) / bw_slot)

    def assign_mod_format(self, path):
        """
        Given a path, assign an appropriate modulation format to the request. Return False if the length of the path
        exceeds the maximum lengths for modulation formats used here.

        :param path: The path chosen to allocate a request
        :type path: list
        :return: The modulation format chosen and the number of slots needed to allocate the request
        :rtype: (str, int) or bool
        """
        path_len = 0
        for i in range(0, len(path) - 1):
            path_len += self.physical_topology[path[i]][path[i + 1]]['length']

        # It's important to check modulation formats in this order
        if self.mod_formats['QPSK']['max_length'] >= path_len > self.mod_formats['16-QAM']['max_length']:
            mod_format = 'QPSK'
            bits_per_symbol = 2
        elif self.mod_formats['16-QAM']['max_length'] >= path_len > self.mod_formats['64-QAM']['max_length']:
            mod_format = '16-QAM'
            bits_per_symbol = 4
        elif self.mod_formats['64-QAM']['max_length'] >= path_len:
            mod_format = '64-QAM'
            bits_per_symbol = 6
        # Failure to assign modulation format
        else:
            return False, False

        slots_needed = self.spectral_slot_comp(bits_per_symbol)
        return mod_format, slots_needed