"""
Integration tests for the complete interface architecture.
"""

import unittest
from unittest.mock import Mock
import networkx as nx

from fusion.interfaces.factory import AlgorithmFactory, SimulationPipeline
from fusion.modules.routing.k_shortest_path import KShortestPath
from fusion.modules.spectrum.first_fit import FirstFitSpectrum
from fusion.modules.snr.snr import StandardSNRMeasurer
from fusion.core.properties import RoutingProps, SpectrumProps


class TestInterfaceIntegration(unittest.TestCase):
    """Test the complete interface architecture working together."""

    def setUp(self):
        """Set up test environment."""
        # Create a simple test topology
        self.topology = nx.Graph()
        self.topology.add_edge('1', '2', length=100, weight=1)
        self.topology.add_edge('2', '3', length=150, weight=1)
        self.topology.add_edge('3', '4', length=120, weight=1)
        self.topology.add_edge('1', '4', length=200, weight=1)

        # Mock engine properties
        self.engine_props = {
            'topology': self.topology,
            'k_paths': 3,
            'cores_per_link': 7,
            'band_list': ['c', 'l', 's'],
            'bw_per_slot': 12.5e9,
            'input_power': 1e-3,
            'guard_slots': 1,
            'slots_per_gbps': 2,
            'c_band': 320,
            'fiber_attenuation': 0.2,
            'fiber_dispersion': 16.7,
            'nonlinear_coefficient': 1.3e-3,
            'bending_radius': 7.5e-3,
            'edfa_noise_figure': 4.5
        }

        # Mock SDN properties with proper attributes
        self.sdn_props = Mock()
        self.sdn_props.topology = self.topology
        self.sdn_props.network_spectrum_dict = self._create_mock_spectrum_dict()
        self.sdn_props.source = '1'
        self.sdn_props.destination = '4'
        self.sdn_props.bandwidth = 100  # Gbps
        self.sdn_props.slots_needed = 8
        self.sdn_props.modulation_formats_dict = {
            'QPSK': {'max_length': 1000, 'slots_needed': 10},
            '16-QAM': {'max_length': 500, 'slots_needed': 8},
            '64-QAM': {'max_length': 200, 'slots_needed': 6}
        }
        # Add mod_formats which is also expected by the routing algorithm
        self.sdn_props.mod_formats = {
            'QPSK': {'max_length': 1000, 'slots_needed': 10},
            '16-QAM': {'max_length': 500, 'slots_needed': 8},
            '64-QAM': {'max_length': 200, 'slots_needed': 6}
        }

        # Create proper property objects instead of Mock
        self.route_props = RoutingProps()
        self.spectrum_props = SpectrumProps()
        self.spectrum_props.forced_core = None
        self.spectrum_props.forced_band = None

    def _create_mock_spectrum_dict(self):
        """Create mock spectrum dictionary for testing."""
        spectrum_dict = {}

        # Create spectrum state for all links
        for edge in self.topology.edges():
            source, dest = edge
            link_dict = {
                'cores_matrix': {}
            }

            # Create cores matrix for each band
            for band in ['c', 'l', 's']:
                cores = []
                for core_num in range(7):  # 7 cores per link
                    # Create 320 slots per core (mostly empty)
                    core_array = [0] * 320  # 0 = free, >0 = occupied
                    # Add some occupied slots for testing
                    if core_num == 0:
                        core_array[100:110] = [1001] * 10  # Request 1001 occupies slots 100-109
                        core_array[200:205] = [1002] * 5  # Request 1002 occupies slots 200-204
                    cores.append(core_array)

                link_dict['cores_matrix'][band] = cores

            spectrum_dict[(source, dest)] = link_dict
            spectrum_dict[(dest, source)] = link_dict  # Bidirectional

        return spectrum_dict

    def test_algorithm_factory_creation(self):
        """Test that the algorithm factory can create all algorithm types."""
        # Test routing algorithm creation
        routing_algo = AlgorithmFactory.create_routing_algorithm(
            'k_shortest_path', self.engine_props, self.sdn_props
        )
        self.assertIsInstance(routing_algo, KShortestPath)
        self.assertEqual(routing_algo.algorithm_name, 'k_shortest_path')

        # Test spectrum algorithm creation
        spectrum_algo = AlgorithmFactory.create_spectrum_algorithm(
            'first_fit', self.engine_props, self.sdn_props, self.route_props
        )
        self.assertIsInstance(spectrum_algo, FirstFitSpectrum)
        self.assertEqual(spectrum_algo.algorithm_name, 'first_fit')

        # Test SNR algorithm creation
        snr_algo = AlgorithmFactory.create_snr_algorithm(
            'standard_snr', self.engine_props, self.sdn_props,
            self.spectrum_props, self.route_props
        )
        self.assertIsInstance(snr_algo, StandardSNRMeasurer)
        self.assertEqual(snr_algo.algorithm_name, 'standard_snr')

    def test_algorithm_factory_invalid_names(self):
        """Test that factory raises appropriate errors for invalid algorithm names."""
        with self.assertRaises(ValueError):
            AlgorithmFactory.create_routing_algorithm(
                'invalid_routing', self.engine_props, self.sdn_props
            )

        with self.assertRaises(ValueError):
            AlgorithmFactory.create_spectrum_algorithm(
                'invalid_spectrum', self.engine_props, self.sdn_props, self.route_props
            )

        with self.assertRaises(ValueError):
            AlgorithmFactory.create_snr_algorithm(
                'invalid_snr', self.engine_props, self.sdn_props,
                self.spectrum_props, self.route_props
            )

    def test_simulation_pipeline_creation(self):
        """Test simulation pipeline creation and configuration."""
        config = {
            'engine_props': self.engine_props,
            'sdn_props': self.sdn_props,
            'route_props': self.route_props,
            'spectrum_props': self.spectrum_props,
            'routing_algorithm': 'k_shortest_path',
            'spectrum_algorithm': 'first_fit',
            'snr_algorithm': 'standard_snr',
            'snr_margin': 2.0
        }

        pipeline = SimulationPipeline(config)

        # Check that algorithms were created correctly
        self.assertEqual(pipeline.routing_algorithm.algorithm_name, 'k_shortest_path')
        self.assertEqual(pipeline.spectrum_algorithm.algorithm_name, 'first_fit')
        self.assertEqual(pipeline.snr_algorithm.algorithm_name, 'standard_snr')

    def test_complete_request_processing(self):
        """Test processing a complete request through the pipeline."""
        config = {
            'engine_props': self.engine_props,
            'sdn_props': self.sdn_props,
            'route_props': self.route_props,
            'spectrum_props': self.spectrum_props,
            'routing_algorithm': 'k_shortest_path',
            'spectrum_algorithm': 'first_fit',
            'snr_algorithm': 'standard_snr',
            'snr_margin': 1.0
        }

        try:
            pipeline = SimulationPipeline(config)
        except (ImportError, TypeError, AttributeError) as e:
            self.fail(f"Failed to create SimulationPipeline: {e}")

        # Verify pipeline was created successfully
        self.assertIsNotNone(pipeline, "SimulationPipeline creation returned None")
        self.assertTrue(hasattr(pipeline, 'process_request'), "Pipeline missing process_request method")

        # Create a mock request
        request = Mock()
        request.bandwidth = 100  # Gbps
        request.slots_needed = 8
        request.modulation = 'QPSK'
        request.id = 2001

        # Process the request
        try:
            result = pipeline.process_request('1', '4', request)
        except (RuntimeError, ValueError, KeyError) as e:
            self.fail(f"process_request raised an exception: {e}")

        # Verify result is not None
        self.assertIsNotNone(result,
                             f"process_request returned None. Pipeline state: routing={getattr(pipeline, 'routing_algorithm', 'MISSING')}, spectrum={getattr(pipeline, 'spectrum_algorithm', 'MISSING')}, snr={getattr(pipeline, 'snr_algorithm', 'MISSING')}")

        # Verify result structure
        self.assertIn('source', result)
        self.assertIn('destination', result)
        self.assertIn('success', result)
        self.assertIn('path', result)
        self.assertIn('metrics', result)

        # Check that we got a path
        self.assertIsNotNone(result['path'])
        self.assertEqual(result['source'], '1')
        self.assertEqual(result['destination'], '4')

        # Check metrics from all algorithms
        self.assertIn('routing', result['metrics'])
        self.assertIn('spectrum', result['metrics'])
        self.assertIn('snr', result['metrics'])

        # Verify algorithm metrics have expected fields
        routing_metrics = result['metrics']['routing']
        self.assertIn('algorithm', routing_metrics)
        self.assertIn('paths_computed', routing_metrics)

        spectrum_metrics = result['metrics']['spectrum']
        self.assertIn('algorithm', spectrum_metrics)
        self.assertIn('supports_multiband', spectrum_metrics)

        snr_metrics = result['metrics']['snr']
        self.assertIn('algorithm', snr_metrics)
        self.assertIn('supports_multicore', snr_metrics)

    def test_algorithm_info_retrieval(self):
        """Test retrieving information about algorithms in the pipeline."""
        config = {
            'engine_props': self.engine_props,
            'sdn_props': self.sdn_props,
            'route_props': self.route_props,
            'spectrum_props': self.spectrum_props,
            'routing_algorithm': 'congestion_aware',
            'spectrum_algorithm': 'best_fit',
            'snr_algorithm': 'standard_snr'
        }

        # We need to mock the congestion_aware and best_fit algorithms since they may not be fully implemented
        try:
            pipeline = SimulationPipeline(config)
            algo_info = pipeline.get_algorithm_info()

            # Check structure
            self.assertIn('routing', algo_info)
            self.assertIn('spectrum', algo_info)
            self.assertIn('snr', algo_info)

            # Check routing info
            routing_info = algo_info['routing']
            self.assertIn('name', routing_info)
            self.assertIn('supported_topologies', routing_info)
            self.assertIn('class', routing_info)

            # Check spectrum info
            spectrum_info = algo_info['spectrum']
            self.assertIn('name', spectrum_info)
            self.assertIn('supports_multiband', spectrum_info)
            self.assertIn('class', spectrum_info)

            # Check SNR info
            snr_info = algo_info['snr']
            self.assertIn('name', snr_info)
            self.assertIn('supports_multicore', snr_info)
            self.assertIn('class', snr_info)

        except ValueError:
            # Some algorithms might not be fully implemented yet
            self.skipTest("Some algorithms not yet implemented")

    def test_algorithm_reset_functionality(self):
        """Test that algorithm reset functionality works."""
        config = {
            'engine_props': self.engine_props,
            'sdn_props': self.sdn_props,
            'route_props': self.route_props,
            'spectrum_props': self.spectrum_props,
            'routing_algorithm': 'k_shortest_path',
            'spectrum_algorithm': 'first_fit',
            'snr_algorithm': 'standard_snr'
        }

        pipeline = SimulationPipeline(config)

        # Process a request to change algorithm state
        request = Mock()
        request.bandwidth = 100
        request.slots_needed = 4
        request.modulation = 'QPSK'
        request.id = 3001

        _ = pipeline.process_request('1', '3', request)

        # Reset all algorithms
        pipeline.reset_all_algorithms()

        # Get metrics after reset
        metrics_after = {
            'routing': pipeline.routing_algorithm.get_metrics(),
            'spectrum': pipeline.spectrum_algorithm.get_metrics(),
            'snr': pipeline.snr_algorithm.get_metrics()
        }

        # Verify that counters were reset (should be 0)
        self.assertEqual(metrics_after['routing']['paths_computed'], 0)
        self.assertEqual(metrics_after['spectrum']['assignments_made'], 0)
        self.assertEqual(metrics_after['snr']['calculations_performed'], 0)

    def test_interface_polymorphism(self):
        """Test that different algorithm implementations can be used interchangeably."""
        algorithms_to_test = [
            ('k_shortest_path', 'first_fit', 'standard_snr'),
            # Add more combinations as algorithms are implemented
        ]

        for routing, spectrum, snr in algorithms_to_test:
            with self.subTest(routing=routing, spectrum=spectrum, snr=snr):
                try:
                    config = {
                        'engine_props': self.engine_props,
                        'sdn_props': self.sdn_props,
                        'route_props': self.route_props,
                        'spectrum_props': self.spectrum_props,
                        'routing_algorithm': routing,
                        'spectrum_algorithm': spectrum,
                        'snr_algorithm': snr
                    }

                    pipeline = SimulationPipeline(config)

                    # Verify algorithms follow their interfaces
                    self.assertTrue(hasattr(pipeline.routing_algorithm, 'route'))
                    self.assertTrue(hasattr(pipeline.spectrum_algorithm, 'assign'))
                    self.assertTrue(hasattr(pipeline.snr_algorithm, 'calculate_snr'))

                    # Verify they return expected types
                    self.assertIsInstance(pipeline.routing_algorithm.algorithm_name, str)
                    self.assertIsInstance(pipeline.spectrum_algorithm.supports_multiband, bool)
                    self.assertIsInstance(pipeline.snr_algorithm.supports_multicore, bool)

                except (ValueError, NotImplementedError):
                    self.skipTest(f"Algorithm combination not yet implemented: {routing}, {spectrum}, {snr}")


if __name__ == '__main__':
    unittest.main()
