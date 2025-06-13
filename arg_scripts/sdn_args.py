# pylint: disable=too-few-public-methods

class SDNProps:
    """
    Main properties used for the sdn_controller.py script.
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
