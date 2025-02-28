# pylint: disable=too-few-public-methods

class StatsProps:
    """
    Main properties used for the stats_helpers.py script.
    """

    def __init__(self):
        self.snapshots_dict = dict()  # Keeps track of statistics at different request snapshots
        self.cores_dict = dict()  # Cores used in simulation(s)
        self.weights_dict = dict()  # Weights of paths
        self.mods_used_dict = dict()  # Modulations used in simulation(s)
        self.block_bw_dict = dict()  # Block per bandwidth
        self.block_reasons_dict = {'distance': None, 'congestion': None, 'xt_threshold': None}  # Block reasons
        self.sim_block_list = list()  # List of blocking probabilities per simulation
        self.sim_br_block_list = list()  # List of blocking probabilities per simulation
        self.trans_list = list()  # List of transponders used per simulation
        self.hops_list = list()  # Average hops per simulation
        self.lengths_list = list()  # Average lengths per simulation
        self.route_times_list = list()  # Average route times per simulation
        self.xt_list = list()  # Average cross-talk per simulation
        self.bands_list = list()  # Tracks the band allocated in a simulation
        self.start_slot_list = list() # Tracks the end slot allocated in a simulation
        self.end_slot_list = list() # # Tracks the end slot allocated in a simulation
        self.modulation_list = list() # Tracks the modulation
        self.bandwidth_list = list() # # Tracks the bandwidth

    def __repr__(self):
        return f"StatsProps({self.__dict__})"


SNAP_KEYS_LIST = ['occupied_slots', 'guard_slots', 'active_requests', 'blocking_prob', 'num_segments']
