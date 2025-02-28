import unittest

from src.request_generator import get_requests


class TestGetRequests(unittest.TestCase):
    """
    Test request_generator.py
    """

    def setUp(self):
        self.seed = 12345

        # TODO: Write a test for 'core nodes', we should only use these nodes
        self.engine_props = {
            'core_nodes': ['A', 'B'],
            'topology_info': {
                'nodes': {'A': {}, 'B': {}, 'C': {}},  # Example nodes
            },
            'mod_per_bw': {
                '50GHz': ['QPSK', '16QAM'],  # Example modulation formats per bandwidth
                '100GHz': ['QPSK']
            },
            'request_distribution': {
                '50GHz': 0.5,
                '100GHz': 0.5
            },
            'num_requests': 10,  # Example number of requests
            'arrival_rate': 1.0,
            'holding_time': 2.0,
            'sim_type': 'default',  # Adjust as needed
            'is_only_core_node': True
        }

    def test_requests_length(self):
        """
        Test the number of requests generated.
        """
        requests = get_requests(seed=self.seed, engine_props=self.engine_props)
        self.assertEqual(len(requests), self.engine_props['num_requests'] * 2)

    def test_no_duplicate_source_destination(self):
        """
        Test source and destination pairs.
        """
        requests = get_requests(seed=self.seed, engine_props=self.engine_props)
        for _, value in requests.items():
            self.assertNotEqual(value['source'], value['destination'])

    def test_correct_bandwidth_distribution(self):
        """
        Tests the bandwidth distribution.
        """
        requests = get_requests(seed=self.seed, engine_props=self.engine_props)
        bw_distribution = {bw: 0 for bw in self.engine_props['mod_per_bw']}
        for _, value in requests.items():
            if value['request_type'] == 'arrival':
                bw_distribution[value['bandwidth']] += 1
        for bandwidth, count in self.engine_props['request_distribution'].items():
            expected_count = int(count * self.engine_props['num_requests'])
            self.assertEqual(bw_distribution[bandwidth], expected_count)

    def test_arrival_departure_times(self):
        """
        Test arrival and departure times.
        """
        requests = get_requests(seed=self.seed, engine_props=self.engine_props)
        for _, value in requests.items():
            if value['request_type'] == 'arrival':
                self.assertLess(value['arrive'], value['depart'])
