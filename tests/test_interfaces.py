"""
Unit tests for interface compliance.
"""

import inspect
import unittest

from fusion.interfaces.agent import AgentInterface
from fusion.interfaces.router import AbstractRoutingAlgorithm
from fusion.interfaces.snr import AbstractSNRMeasurer
from fusion.interfaces.spectrum import AbstractSpectrumAssigner


class TestInterfaceCompliance(unittest.TestCase):
    """Test that implementations properly follow the abstract interfaces."""

    def test_abstract_interface_cannot_be_instantiated(self):
        """Test that abstract interfaces cannot be directly instantiated."""
        with self.assertRaises(TypeError):
            AbstractRoutingAlgorithm.__new__(AbstractRoutingAlgorithm, {}, None)

        with self.assertRaises(TypeError):
            AbstractSpectrumAssigner.__new__(AbstractSpectrumAssigner, {}, None, None)

        with self.assertRaises(TypeError):
            AbstractSNRMeasurer.__new__(AbstractSNRMeasurer, {}, None, None, None)

        with self.assertRaises(TypeError):
            AgentInterface.__new__(AgentInterface)

    def test_interface_methods_are_abstract(self):
        """Test that all required methods are marked as abstract."""
        # Check AbstractRoutingAlgorithm
        abstract_methods = {
            method
            for method in dir(AbstractRoutingAlgorithm)
            if hasattr(
                getattr(AbstractRoutingAlgorithm, method), "__isabstractmethod__"
            )
            and getattr(AbstractRoutingAlgorithm, method).__isabstractmethod__
        }

        expected_methods = {
            "algorithm_name",
            "supported_topologies",
            "validate_environment",
            "route",
            "get_paths",
            "update_weights",
            "get_metrics",
        }

        self.assertEqual(abstract_methods, expected_methods)

    def test_interface_has_required_methods(self):
        """Test that interfaces have all required methods."""
        # Test routing interface methods
        expected_routing = [
            "algorithm_name",
            "supported_topologies",
            "validate_environment",
            "route",
            "get_paths",
            "update_weights",
            "get_metrics",
            "reset",
        ]

        for method in expected_routing:
            self.assertTrue(hasattr(AbstractRoutingAlgorithm, method))

        # Test spectrum interface methods
        expected_spectrum = [
            "algorithm_name",
            "supports_multiband",
            "assign",
            "check_spectrum_availability",
            "allocate_spectrum",
            "deallocate_spectrum",
            "get_fragmentation_metric",
            "get_metrics",
            "reset",
        ]

        for method in expected_spectrum:
            self.assertTrue(hasattr(AbstractSpectrumAssigner, method))

        # Test agent interface methods
        expected_agent = [
            "algorithm_name",
            "action_space_type",
            "observation_space_shape",
            "act",
            "train",
            "learn_from_experience",
            "save",
            "load",
            "get_reward",
            "update_exploration_params",
            "get_config",
            "set_config",
            "get_metrics",
            "reset",
            "on_episode_start",
            "on_episode_end",
        ]

        for method in expected_agent:
            self.assertTrue(hasattr(AgentInterface, method))


class TestInterfaceMethodSignatures(unittest.TestCase):
    """Test that interface method signatures are consistent."""

    def test_routing_interface_signatures(self):
        """Test AbstractRoutingAlgorithm method signatures."""
        # Get the route method signature
        route_sig = inspect.signature(AbstractRoutingAlgorithm.route)
        params = list(route_sig.parameters.keys())

        # Should have self, source, destination, request
        self.assertEqual(params, ["self", "source", "destination", "request"])

        # Check return annotation (handle both typing.Optional and types.UnionType)
        annotation = route_sig.return_annotation
        if hasattr(annotation, "__name__"):
            self.assertEqual(annotation.__name__, "Optional")
        else:
            # Handle UnionType or other annotation types - check for None union
            annotation_str = str(annotation)
            self.assertTrue(
                "None" in annotation_str or "Optional" in annotation_str,
                f"Expected Optional or None in annotation, got: {annotation_str}",
            )

    def test_agent_interface_signatures(self):
        """Test AgentInterface method signatures."""
        # Check act method
        act_sig = inspect.signature(AgentInterface.act)
        params = list(act_sig.parameters.keys())

        self.assertEqual(params, ["self", "observation", "deterministic"])

        train_sig = inspect.signature(AgentInterface.train)
        self.assertIn("env", train_sig.parameters)
        self.assertIn("total_timesteps", train_sig.parameters)


if __name__ == "__main__":
    unittest.main()
