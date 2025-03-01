# pylint: disable=too-few-public-methods

class SDNProps:
    """
    Main properties used for the sdn_controller.py script.
    """

    def __init__(self):
        self.path_list = None  # List of nodes for the current request
        self.was_routed = None  # Flag to determine successful route
        self.was_groomed = None # Flag to determine successful fully groomed without new lightpath
        self.was_partially_groomed = False # Flag to determine a portion of service successfully groomed 
        self.was_partially_routed = False # Flag to determine a portion of service successfully routed 
        self.was_new_lp_established = [] # Falg to now a new lightpath is established for request or not
        self.topology = None  # Networkx topology
        self.net_spec_dict = None  # Current network spectrum database
        self.lightpath_status_dict = None # Current lightpath status
        self.transponder_usage_dict = None # Transponder usage status per node
        self.lp_bw_utilization_dict = None # average lightpath utilization over time

        self.req_id = None  # Current request ID number
        self.source = None  # Source node
        self.destination = None  # Destination node
        self.bandwidth = None  # Current bandwidth
        self.remaining_bw = None 
        self.bandwidth_list = []  # Multiple bandwidths used (typically for light-segment slicing)
        self.modulation_list = []  # List of modulation formats used by a single request
        self.core_list = []  # List of cores used (typically for light-segment slicing)
        self.band_list = []  # List of bands used (typically for light-segment slicing)
        self.lightpath_bandwidth_list = []  # List of lightpath bandwidth
        self.lightpath_id_list = []  # List of lightpath id
        self.xt_list = []  # List of crosstalk calculations for a single request
        self.start_slot_list = [] # List of allocated start slot  index for slicing approach
        self.end_slot_list = [] # List of allocated end slot  index for slicing approach
        self.num_trans = None  # Number of transponders a single request has used
        self.arrive = None  # Arrival time for a single request
        self.depart = None  # Departure time for a single request
        self.request_type = None  # Determines arrival or departure
        self.slots_needed = None  # Slots needed for the current request
        self.single_core = False  # Whether to force single-core
        self.block_reason = None  # Reason for blocking a request
        self.mod_formats_dict = None  # List of valid modulation formats for this bandwidth
        self.lightpath_counter = 0 # counter to set id t lightpaths
        

        self.stat_key_list = ['modulation_list', 'xt_list', 'core_list', 'band_list', 'start_slot_list', 'end_slot_list', 'lightpath_bandwidth_list', 'lightpath_id_list', 'remaining_bw']  # Statistical keys used to save results

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
        self.lp_bw_utilization_dict = dict()
        self.modulation_list = list()
        self.xt_list = list()
        self.core_list = list()
        self.band_list = list()
        self.start_slot_list = list()
        self.end_slot_list = list()
        self.bandwidth_list = list()
        self.lightpath_bandwidth_list = list()
        self.lightpath_id_list = list()
        self.path_list = list()
        self.was_new_lp_established = list()
        self.was_routed = None
        self.was_groomed = None
        self.was_partially_groomed = False
        self.was_partially_routed = False
        self.remaining_bw = None
        self.is_sliced = None

    def get_lightpath_id(self):
        self.lightpath_counter += 1
        return self.lightpath_counter
    
    def reset_lightpath_id_couter(self):
        self.lightpath_counter = 0
        
                
    # TODO: Update standards and guidelines, this should be a standardized function name.
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
