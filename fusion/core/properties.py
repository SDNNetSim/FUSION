"""
Core simulation properties and data classes.
Consolidated from various CLI argument files for better organization.
"""

# pylint: disable=too-few-public-methods

class RoutingProps:
    """
    Main properties used for routing operations.
    """

    def __init__(self):
        self.paths_matrix = []  # Matrix of potential paths for a single request
        self.mod_formats_matrix = []  # Modulation formats corresponding to each path in paths_matrix
        self.weights_list = []  # Keeping track of one weight of the path (Length, XT, Etc.)
        self.path_index_list = []  # Keeping track of path index in precalculated routing

        self.input_power = 1e-3  # Power in Watts
        self.freq_spacing = 12.5e9  # Frequency spacing in Hz
        self.mci_worst = 6.3349755556585961e-027  # Worst-case mutual coupling interference value
        self.max_link_length = None  # Maximum link length in km
        self.span_len = 100.0  # Length of a span in km
        self.max_span = None  # Maximum number of spans in the network
        self.connection_index = None  # Keeping track of source destination index in precalculated routing
        self.path_index = None  # Keeps track of selected path indexes during spectrum assignment

    def __repr__(self):
        return f"RoutingProps({self.__dict__})"


class SpectrumProps:
    """
    Main properties used for spectrum assignment operations.
    """

    def __init__(self):
        self.path_list = None  # List of nodes for the current request
        self.slots_needed = None  # Slots needed for current request
        self.forced_core = None  # Flag to force a certain core
        self.is_free = False  # Flag to determine if spectrum is free
        self.modulation = None  # Modulation format for current request
        self.xt_cost = None  # XT cost (if considered) for current request
        self.cores_matrix = None  # The current matrix of cores being evaluated for a single link
        self.rev_cores_matrix = None  # The reverse of cores matrix e.g., if looking at 3-->4 check 4-->3
        self.core_num = None  # Core number selected for current request
        self.forced_index = None  # Flag to determine forced spectral start slot (Usually for AI algorithms)
        self.forced_band = None  # Forces a specific band in multi-band scenarios
        self.curr_band = None  # The chosen band to allocate
        self.start_slot = None  # Start slot assigned for current request
        self.end_slot = None  # End slot assigned for current request

    def __repr__(self):
        return f"SpectrumProps({self.__dict__})"


class SNRProps:
    """
    Main properties used for SNR measurements.
    """

    def __init__(self):
        self.light_frequency = 1.9341 * 10 ** 14  # Center light frequency
        self.plank = 6.62607004e-34  # Plank's constant
        self.req_bit_rate = 12.5  # Request bit rate
        self.req_snr = 8.5  # Request signal to noise ratio value
        self.nsp = 1.8  # Noise spectral density

        self.center_freq = None  # Center frequency for current request
        self.bandwidth = None  # Bandwidth for current request
        self.center_psd = None  # Center power spectral density for current request
        self.mu_param = None  # Mu parameter for calculating PSD
        self.sci_psd = None  # Self-channel interference PSD
        self.xci_psd = None  # Cross-channel interference PSD
        self.length = None  # Length of a current span
        self.num_span = None  # Number of span

        self.link_dict = None  # Dictionary of links for calculating various metrics
        self.mod_format_mapping_dict = {6: "64-QAM", 5: "32-QAM", 4: "16-QAM", 3: "8-QAM", 2: "QPSK",
                                        1: "BPSK"}  # Dictionary of Modulation formats for precalculated SNR
        self.bw_mapping_dict = {"64-QAM": 600, "32-QAM": 500, "16-QAM": 400, "8-QAM": 300, "QPSK": 200,
                                "BPSK": 100}  # Dictionary of Modulation formats to calculate the supported bit rate for fixed grid
        self.file_mapping_dict = {
            'USbackbone60': {
                'multi_fiber': {'mf': 'MF-USB6014-MF.npy', 'gsnr': 'GSNR-USB6014-MF.npy'},
                (2, 4): {'mf': 'MF-USB6014-MCF4-C2.npy', 'gsnr': 'GSNR-USB6014-MCF4-C2.npy'},
                (6, 7): {'mf': 'MF-USB6014-MCF7-C6.npy', 'gsnr': 'GSNR-USB6014-MCF7-C6.npy'},
                (3, 7): {'mf': 'MF-USB6014-MCF7-C3.npy', 'gsnr': 'GSNR-USB6014-MCF7-C3.npy'},
                (2, 13): {'mf': 'MF-USB6014-MCF13-C2.npy', 'gsnr': 'GSNR-USB6014-MCF13-C2.npy'},
                (5, 13): {'mf': 'MF-USB6014-MCF13-C5.npy', 'gsnr': 'GSNR-USB6014-MCF13-C5.npy'},
                (6, 13): {'mf': 'MF-USB6014-MCF13-C6.npy', 'gsnr': 'GSNR-USB6014-MCF13-C6.npy'},
                (6, 19): {'mf': 'MF-USB6014-MCF19-C6.npy', 'gsnr': 'GSNR-USB6014-MCF19-C6.npy'},
                (3, 19): {'mf': 'MF-USB6014-MCF19-C3.npy', 'gsnr': 'GSNR-USB6014-MCF19-C3.npy'},
                (4, 19): {'mf': 'MF-USB6014-MCF19-C4.npy', 'gsnr': 'GSNR-USB6014-MCF19-C4.npy'},
            },
            'Spainbackbone30': {
                'multi_fiber': {'mf': 'MF-SPNB3014-MF.npy', 'gsnr': 'GSNR-SPNB3014-MF.npy'},
                (2, 4): {'mf': 'MF-SPNB3014-MCF4-C2.npy', 'gsnr': 'GSNR-SPNB3014-MCF4-C2.npy'},
                (6, 7): {'mf': 'MF-SPNB3014-MCF7-C6.npy', 'gsnr': 'GSNR-SPNB3014-MCF7-C6.npy'},
                (3, 7): {'mf': 'MF-SPNB3014-MCF7-C3.npy', 'gsnr': 'GSNR-SPNB3014-MCF7-C3.npy'},
                (2, 13): {'mf': 'MF-SPNB3014-MCF13-C2.npy', 'gsnr': 'GSNR-SPNB3014-MCF13-C2.npy'},
                (5, 13): {'mf': 'MF-SPNB3014-MCF13-C5.npy', 'gsnr': 'GSNR-SPNB3014-MCF13-C5.npy'},
                (6, 13): {'mf': 'MF-SPNB3014-MCF13-C6.npy', 'gsnr': 'GSNR-SPNB3014-MCF13-C6.npy'},
                (6, 19): {'mf': 'MF-SPNB3014-MCF19-C6.npy', 'gsnr': 'GSNR-SPNB3014-MCF19-C6.npy'},
                (3, 19): {'mf': 'MF-SPNB3014-MCF19-C3.npy', 'gsnr': 'GSNR-SPNB3014-MCF19-C3.npy'},
                (4, 19): {'mf': 'MF-SPNB3014-MCF19-C4.npy', 'gsnr': 'GSNR-SPNB3014-MCF19-C4.npy'},
            }
        }

    def __repr__(self):
        return f"SNRProps({self.__dict__})"


class SDNProps:
    """
    Main properties used for SDN controller operations.
    """

    def __init__(self):
        self.path_list = None  # List of nodes for the current request
        self.path_index = None  # Index of selected path in the computed path set
        self.was_routed = None  # Flag to determine successful route
        self.topology = None  # Networkx topology
        self.net_spec_dict = None  # Current network spectrum database

        self.req_id = None  # Current request ID number
        self.source = None  # Source node
        self.destination = None  # Destination node
        self.bandwidth = None  # Current bandwidth
        self.bandwidth_list = []  # Multiple bandwidths used (typically for light-segment slicing)
        self.modulation_list = []  # List of modulation formats used by a single request
        self.core_list = []  # List of cores used (typically for light-segment slicing)
        self.band_list = []  # List of bands used (typically for light-segment slicing)
        self.xt_list = []  # List of crosstalk calculations for a single request
        self.start_slot_list = []  # List of allocated start slot  index for slicing approach
        self.end_slot_list = []  # List of allocated end slot  index for slicing approach
        self.num_trans = None  # Number of transponders a single request has used
        self.arrive = None  # Arrival time for a single request
        self.depart = None  # Departure time for a single request
        self.request_type = None  # Determines arrival or departure
        self.slots_needed = None  # Slots needed for the current request
        self.single_core = False  # Whether to force single-core
        self.block_reason = None  # Reason for blocking a request
        self.mod_formats_dict = None  # List of valid modulation formats for this bandwidth

        self.stat_key_list = ['modulation_list', 'xt_list', 'core_list', 'band_list', 'start_slot_list',
                              'end_slot_list']  # Statistical keys used to save results

    def update_params(self, key: str, spectrum_key: str, spectrum_obj: object, value: int = None):
        """
        Update lists to track statistics of routed requests or general network metrics.

        :param key: Key to update.
        :param spectrum_key: Spectrum key to get a spectrum object value.
        :param spectrum_obj: Spectrum assignment main object.
        :param value: Value related to the key, it may vary widely.
        """
        if hasattr(self, key):
            if spectrum_key:
                spectrum_value = getattr(spectrum_obj.spectrum_props, spectrum_key)
                current_value = getattr(self, key)
                if isinstance(current_value, list):
                    current_value.append(spectrum_value)
                else:
                    setattr(self, key, spectrum_value)
            else:
                setattr(self, key, value)

    def reset_params(self):
        """
        Reset select lists used to track statistics.
        """
        self.modulation_list = list()
        self.xt_list = list()
        self.core_list = list()
        self.band_list = list()
        self.start_slot_list = list()
        self.end_slot_list = list()

    def get_data(self, key: str):
        """
        Retrieve a property of the object.

        :param key: The property name.
        :return: The value of the property.
        """
        if hasattr(self, key):
            return getattr(self, key)

        raise AttributeError(f"'SDNProps' object has no attribute '{key}'")

    def __repr__(self):
        return f"SDNProps({self.__dict__})"


class StatsProps:
    """
    Main properties used for statistics collection and tracking.
    """

    def __init__(self):
        self.snapshots_dict = dict()  # Keeps track of statistics at different request snapshots
        self.cores_dict = dict()  # Cores used in simulation(s)
        self.weights_dict = dict()  # Weights of paths
        self.mods_used_dict = dict()  # Modulations used in simulation(s)
        self.block_bw_dict = dict()  # Block per bandwidth
        self.block_reasons_dict = {'distance': None, 'congestion': None, 'xt_threshold': None}  # Block reasons
        self.link_usage_dict = dict()  # Tracks per link usage data
        self.sim_block_list = list()  # List of blocking probabilities per simulation
        self.sim_br_block_list = list()  # List of blocking probabilities per simulation
        self.trans_list = list()  # List of transponders used per simulation
        self.hops_list = list()  # Average hops per simulation
        self.lengths_list = list()  # Average lengths per simulation
        self.route_times_list = list()  # Average route times per simulation
        self.xt_list = list()  # Average cross-talk per simulation
        self.bands_list = list()  # Tracks the band allocated in a simulation
        self.start_slot_list = list()  # Tracks the end slot allocated in a simulation
        self.end_slot_list = list()  # # Tracks the end slot allocated in a simulation
        self.modulation_list = list()  # Tracks the modulation
        self.bandwidth_list = list()  # # Tracks the bandwidth
        self.path_index_list = list()  # Tracks the number of times a path was selected

    def __repr__(self):
        return f"StatsProps({self.__dict__})"


# Constants for statistics tracking
SNAP_KEYS_LIST = ['occupied_slots', 'guard_slots', 'active_requests', 'blocking_prob', 'num_segments']
