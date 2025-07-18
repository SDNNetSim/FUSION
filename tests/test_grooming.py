import unittest

class MockSDNProps:
    def __init__(self):
        self.source = 'A'
        self.destination = 'B'
        self.req_id = 1
        self.bandwidth = 30
        self.arrive = 10
        self.depart = 20
        self.bandwidth_list = []
        self.core_list = []
        self.band_list = []
        self.start_slot_list = []
        self.end_slot_list = []
        self.modulation_list = []
        self.path_list = []
        self.xt_list = []
        self.lightpath_bandwidth_list = []
        self.lightpath_id_list = []
        self.was_routed = False
        self.was_groomed = False
        self.was_partially_groomed = False
        self.was_new_lp_established = []
        self.remaining_bw = 0
        self.path_weight = 0
        self.is_sliced = False
        self.num_trans = 0
        self.lightpath_status_dict = {
            ('A', 'B'): {
                10: {
                    'remaining_bandwidth': 40,
                    'lightpath_bandwidth': 40,
                    'path': ['A', 'B'],
                    'core': 0,
                    'band': 1,
                    'start_slot': 0,
                    'end_slot': 9,
                    'mod_format': 'PM-QPSK',
                    'snr_cost': 20,
                    'path_weight': 1.0,
                    'requests_dict': {},
                    'time_bw_usage': {}
                }
            }
        }

from src.grooming import Grooming

class TestGrooming(unittest.TestCase):
    def test_end_to_end_grooming_success(self):
        engine_props = {}
        sdn_props = MockSDNProps()
        grooming = Grooming(engine_props, sdn_props)

        result = grooming.handle_grooming(request_type="arrival")

        self.assertTrue(result)
        self.assertEqual(sdn_props.was_groomed, True)
        self.assertEqual(sdn_props.remaining_bw, "0")
        self.assertEqual(len(sdn_props.lightpath_id_list), 1)
        self.assertEqual(sdn_props.lightpath_id_list[0], 10)

    def test_release_service(self):
        engine_props = {}
        sdn_props = MockSDNProps()
        grooming = Grooming(engine_props, sdn_props)

        grooming.handle_grooming(request_type="arrival")
        released = grooming.handle_grooming(request_type="release")
        
        self.assertEqual([10], released)
        self.assertEqual(sdn_props.lightpath_status_dict[('A', 'B')][10]['remaining_bandwidth'], 40)

unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestGrooming))
