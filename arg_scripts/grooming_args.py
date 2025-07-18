# pylint: disable=too-few-public-methods

class GroomingProps:
    """
    Main properties used for the grooming.py script.
    """

    def __init__(self):
        self.gooming_type = None  # select grooming method
        self.lightpath_status_dict = None # Established lightpath status


    def __repr__(self):
        return f"GroomingProps({self.__dict__})"
