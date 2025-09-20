"""
Basic interface tests without external dependencies.
"""

import unittest
from unittest.mock import Mock

from fusion.interfaces.agent import AgentInterface
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.interfaces.snr import AbstractSNRMeasurer
from fusion.interfaces.spectrum import AbstractSpectrumAssigner


class MockRoutingAlgorithm(AbstractRoutingAlgorithm):
    """Mock routing algorithm for testing."""

    @property
    def algorithm_name(self) -> str:
        return "mock_routing"

    @property
    def supported_topologies(self):
        return ["Test"]

    def validate_environment(self, topology):
        return True

    def route(self, source, destination, request):
        return [source, destination]

    def get_paths(self, source, destination, k=1):
        return [[source, destination]]

    def update_weights(self, topology):
        pass

    def get_metrics(self):
        return {"algorithm": self.algorithm_name}


class MockSpectrumAssigner(AbstractSpectrumAssigner):
    """Mock spectrum assigner for testing."""

    @property
    def algorithm_name(self) -> str:
        return "mock_spectrum"

    @property
    def supports_multiband(self) -> bool:
        return True

    def assign(self, path, request):
        return {"start_slot": 0, "end_slot": 10, "core_num": 0, "band": "c"}

    def check_spectrum_availability(self, path, start_slot, end_slot, core_num, band):
        return True

    def allocate_spectrum(self, path, start_slot, end_slot, core_num, band, request_id):
        return True

    def deallocate_spectrum(self, path, start_slot, end_slot, core_num, band):
        return True

    def get_fragmentation_metric(self, path):
        return 0.1

    def get_metrics(self):
        return {"algorithm": self.algorithm_name}


class MockSNRMeasurer(AbstractSNRMeasurer):
    """Mock SNR measurer for testing."""

    @property
    def algorithm_name(self) -> str:
        return "mock_snr"

    @property
    def supports_multicore(self) -> bool:
        return True

    def calculate_snr(self, path, spectrum_info):
        return 20.0  # 20 dB

    def calculate_link_snr(self, source, destination, spectrum_info):
        return 25.0  # 25 dB

    def calculate_crosstalk(self, path, core_num, spectrum_info):
        return 0.01  # Linear units

    def calculate_nonlinear_noise(self, path, spectrum_info):
        return {"sci": 0.001, "xci": 0.002, "xpm": 0.0, "fwm": 0.0}

    def get_required_snr_threshold(self, modulation, reach):
        return 15.0  # 15 dB

    def is_snr_acceptable(self, calculated_snr, required_snr, margin=0.0):
        return calculated_snr >= (required_snr + margin)

    def update_link_state(self, source, destination, spectrum_info):
        pass

    def get_metrics(self):
        return {"algorithm": self.algorithm_name}


class MockAgent(AgentInterface):
    """Mock RL agent for testing."""

    @property
    def algorithm_name(self) -> str:
        return "mock_agent"

    @property
    def action_space_type(self) -> str:
        return "discrete"

    @property
    def observation_space_shape(self):
        return (10,)

    def act(self, observation, deterministic=False):
        return 0  # Always select action 0

    def train(self, env, total_timesteps, **kwargs):
        return {"total_timesteps": total_timesteps}

    def learn_from_experience(
        self, observation, action, reward, next_observation, done
    ):
        return {"loss": 0.01}

    def save(self, path):
        pass

    def load(self, path):
        pass

    def get_reward(self, state, action, next_state, info):
        return 1.0

    def update_exploration_params(self, timestep, total_timesteps):
        pass

    def get_config(self):
        return {"algorithm": self.algorithm_name}

    def set_config(self, config):
        pass

    def get_metrics(self):
        return {"algorithm": self.algorithm_name}


class TestInterfaceImplementations(unittest.TestCase):
    """Test that mock implementations follow the interfaces correctly."""

    def setUp(self):
        """Set up test instances."""
        self.engine_props = {"test": True}
        self.sdn_props = Mock()
        self.route_props = Mock()
        self.spectrum_props = Mock()

    def test_mock_routing_algorithm(self):
        """Test mock routing algorithm implementation."""
        router = MockRoutingAlgorithm(self.engine_props, self.sdn_props)

        # Test interface compliance
        self.assertIsInstance(router, AbstractRoutingAlgorithm)
        self.assertEqual(router.algorithm_name, "mock_routing")
        self.assertIn("Test", router.supported_topologies)

        # Test functionality
        self.assertTrue(router.validate_environment(None))
        path = router.route(1, 4, None)
        self.assertEqual(path, [1, 4])

        paths = router.get_paths(1, 4, k=2)
        self.assertEqual(len(paths), 1)

        metrics = router.get_metrics()
        self.assertEqual(metrics["algorithm"], "mock_routing")

    def test_mock_spectrum_assigner(self):
        """Test mock spectrum assigner implementation."""
        assigner = MockSpectrumAssigner(
            self.engine_props, self.sdn_props, self.route_props
        )

        # Test interface compliance
        self.assertIsInstance(assigner, AbstractSpectrumAssigner)
        self.assertEqual(assigner.algorithm_name, "mock_spectrum")
        self.assertTrue(assigner.supports_multiband)

        # Test functionality
        assignment = assigner.assign([1, 2], Mock())
        self.assertIn("start_slot", assignment)
        self.assertIn("end_slot", assignment)

        available = assigner.check_spectrum_availability([1, 2], 0, 10, 0, "c")
        self.assertTrue(available)

        success = assigner.allocate_spectrum([1, 2], 0, 10, 0, "c", 123)
        self.assertTrue(success)

        fragmentation = assigner.get_fragmentation_metric([1, 2])
        self.assertEqual(fragmentation, 0.1)

    def test_mock_snr_measurer(self):
        """Test mock SNR measurer implementation."""
        measurer = MockSNRMeasurer(
            self.engine_props, self.sdn_props, self.spectrum_props, self.route_props
        )

        # Test interface compliance
        self.assertIsInstance(measurer, AbstractSNRMeasurer)
        self.assertEqual(measurer.algorithm_name, "mock_snr")
        self.assertTrue(measurer.supports_multicore)

        # Test functionality
        snr = measurer.calculate_snr([1, 2], {"band": "c"})
        self.assertEqual(snr, 20.0)

        link_snr = measurer.calculate_link_snr(1, 2, {"band": "c"})
        self.assertEqual(link_snr, 25.0)

        xt = measurer.calculate_crosstalk([1, 2], 0, {"band": "c"})
        self.assertEqual(xt, 0.01)

        noise = measurer.calculate_nonlinear_noise([1, 2], {"band": "c"})
        self.assertIn("sci", noise)
        self.assertIn("xci", noise)

        threshold = measurer.get_required_snr_threshold("QPSK", 100)
        self.assertEqual(threshold, 15.0)

        acceptable = measurer.is_snr_acceptable(20.0, 15.0, 1.0)
        self.assertTrue(acceptable)

    def test_mock_agent(self):
        """Test mock RL agent implementation."""
        agent = MockAgent()

        # Test interface compliance
        self.assertIsInstance(agent, AgentInterface)
        self.assertEqual(agent.algorithm_name, "mock_agent")
        self.assertEqual(agent.action_space_type, "discrete")
        self.assertEqual(agent.observation_space_shape, (10,))

        # Test functionality
        action = agent.act([0.1, 0.2, 0.3])
        self.assertEqual(action, 0)

        result = agent.train(Mock(), 1000)
        self.assertEqual(result["total_timesteps"], 1000)

        loss = agent.learn_from_experience([0.1], 0, 1.0, [0.2], False)
        self.assertEqual(loss["loss"], 0.01)

        reward = agent.get_reward({}, 0, {}, {})
        self.assertEqual(reward, 1.0)

        config = agent.get_config()
        self.assertEqual(config["algorithm"], "mock_agent")


class TestInterfacePolymorphism(unittest.TestCase):
    """Test that interfaces enable polymorphism."""

    def test_algorithm_list_polymorphism(self):
        """Test that different implementations can be used polymorphically."""
        # Create instances
        router = MockRoutingAlgorithm({}, Mock())
        spectrum = MockSpectrumAssigner({}, Mock(), Mock())
        snr = MockSNRMeasurer({}, Mock(), Mock(), Mock())
        agent = MockAgent()

        # Test that they can be used as their interface types
        algorithms = [router, spectrum, snr, agent]

        # All should have algorithm_name property
        names = [alg.algorithm_name for alg in algorithms]
        expected_names = ["mock_routing", "mock_spectrum", "mock_snr", "mock_agent"]
        self.assertEqual(names, expected_names)

        # All should have get_metrics method
        metrics_list = [alg.get_metrics() for alg in algorithms]
        self.assertEqual(len(metrics_list), 4)

        # Each should return their algorithm name in metrics
        for i, metrics in enumerate(metrics_list):
            self.assertEqual(metrics["algorithm"], expected_names[i])


if __name__ == "__main__":
    unittest.main()
