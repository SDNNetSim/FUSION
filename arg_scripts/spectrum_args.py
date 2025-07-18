# pylint: disable=too-few-public-methods

class SpectrumProps:
    """
    Main properties used for the spectrum_assignment.py script.
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
        self.lightpath_id = None # Unique ID for lightpath
        self.lightpath_bandwidth = None # BW of lightpath
        self.slicing_flag = False # Falg to detect is sliced or not

    def __repr__(self):
        return f"SpectrumProps({self.__dict__})"
