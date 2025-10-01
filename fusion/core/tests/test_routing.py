"""
Unit tests for fusion.core.routing module.

This module provides comprehensive testing for the Routing class which serves as
a dispatcher to modular routing algorithms and maintains backward compatibility.
"""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np

from fusion.core.properties import RoutingProps
from fusion.core.routing import Routing


class TestRouting(unittest.TestCase):
    """Unit tests for Routing functionality."""

    def setUp(self) -> None:
        """Set up test fixtures with proper isolation."""
        self.engine_props = {
            "topology": nx.Graph(),
            "mod_per_bw": {
                "50GHz": {"QPSK": {"max_length": 10}},
                "100GHz": {"QPSK": {"max_length": 20}},
            },
            "k_paths": 2,
            "xt_type": "with_length",
            "beta": 0.5,
            "route_method": "k_shortest_path",
            "pre_calc_mod_selection": False,
            "network": "USbackbone60",
            "spectral_slots": 320,
            "guard_slots": 1,
        }
        self.engine_props["topology"].add_edge("A", "B", weight=1, xt_cost=10, length=1)
        self.engine_props["topology"].add_edge("B", "C", weight=1, xt_cost=5, length=1)
        self.engine_props["topology"].add_edge(
            "A", "C", weight=3, xt_cost=100, length=2
        )

        self.sdn_props = MagicMock()
        self.sdn_props.network_spectrum_dict = {
            ("A", "B"): {"cores_matrix": {"c": np.zeros((1, 10))}},
            ("B", "C"): {"cores_matrix": {"c": np.ones((1, 10))}},
            ("A", "C"): {"cores_matrix": {"c": np.zeros((1, 10))}},
        }
        self.sdn_props.source = "A"
        self.sdn_props.destination = "C"
        self.sdn_props.topology = self.engine_props["topology"]

    def test_init_with_valid_parameters_creates_instance(self) -> None:
        """Test Routing initialization with valid parameters."""
        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )

        self.assertEqual(routing.engine_props, self.engine_props)
        self.assertEqual(routing.sdn_props, self.sdn_props)
        self.assertIsInstance(routing.route_props, RoutingProps)
        self.assertIsNotNone(routing.routing_registry)
        self.assertIsNotNone(routing.route_helper)
        # Check backward compatibility alias
        self.assertEqual(routing.route_help_obj, routing.route_helper)

    def test_init_sets_current_algorithm_to_none(self) -> None:
        """Test Routing initialization sets current algorithm to None."""
        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )

        self.assertIsNone(routing._current_algorithm)

    @patch("fusion.modules.routing.registry.RoutingRegistry.get")
    def test_get_algorithm_for_method_with_valid_method_returns_algorithm(
        self, mock_get: Any
    ) -> None:
        """Test algorithm retrieval for valid routing method."""
        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )
        mock_algorithm_class = MagicMock()
        mock_algorithm_instance = MagicMock()
        mock_algorithm_class.return_value = mock_algorithm_instance
        mock_get.return_value = mock_algorithm_class

        algorithm = routing._get_algorithm_for_method("k_shortest_path")

        self.assertEqual(algorithm, mock_algorithm_instance)
        mock_get.assert_called_once_with("k_shortest_path")
        mock_algorithm_class.assert_called_once_with(self.engine_props, self.sdn_props)

    @patch("fusion.modules.routing.registry.RoutingRegistry.get")
    def test_get_algorithm_for_method_with_legacy_method_maps_correctly(
        self, mock_get: Any
    ) -> None:
        """Test legacy method name mapping works correctly."""
        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )
        mock_algorithm_class = MagicMock()
        mock_algorithm_instance = MagicMock()
        mock_algorithm_class.return_value = mock_algorithm_instance
        mock_get.return_value = mock_algorithm_class

        # Test legacy mapping
        routing._get_algorithm_for_method("shortest_path")

        mock_get.assert_called_once_with("k_shortest_path")

    @patch("fusion.modules.routing.registry.RoutingRegistry.get")
    def test_get_algorithm_for_method_with_unknown_method_uses_as_is(
        self, mock_get: Any
    ) -> None:
        """Test unknown method name is passed through as-is."""
        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )
        mock_algorithm_class = MagicMock()
        mock_algorithm_instance = MagicMock()
        mock_algorithm_class.return_value = mock_algorithm_instance
        mock_get.return_value = mock_algorithm_class

        routing._get_algorithm_for_method("custom_algorithm")

        mock_get.assert_called_once_with("custom_algorithm")

    def test_legacy_method_mapping_contains_expected_mappings(self) -> None:
        """Test that legacy method mapping contains expected transformations."""
        from fusion.core.routing import LEGACY_METHOD_MAPPING

        expected_mappings = {
            "shortest_path": "k_shortest_path",
            "k_shortest_path": "k_shortest_path",
            "least_congested": "least_congested",
            "cong_aware": "congestion_aware",
            "frag_aware": "fragmentation_aware",
            "nli_aware": "nli_aware",
            "xt_aware": "xt_aware",
            "external_ksp": "k_shortest_path",
        }

        for legacy_name, expected_name in expected_mappings.items():
            self.assertEqual(LEGACY_METHOD_MAPPING[legacy_name], expected_name)

    @patch("fusion.core.routing.RoutingRegistry")
    def test_routing_registry_initialization(self, mock_registry_class: Any) -> None:
        """Test that routing registry is properly initialized."""
        mock_registry_instance = MagicMock()
        mock_registry_class.return_value = mock_registry_instance

        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )

        mock_registry_class.assert_called_once()
        self.assertEqual(routing.routing_registry, mock_registry_instance)

    @patch("fusion.core.routing.RoutingHelpers")
    def test_routing_helpers_initialization(self, mock_helpers_class: Any) -> None:
        """Test that routing helpers are properly initialized."""
        mock_helpers_instance = MagicMock()
        mock_helpers_class.return_value = mock_helpers_instance

        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )

        mock_helpers_class.assert_called_once_with(
            route_props=routing.route_props,
            engine_props=self.engine_props,
            sdn_props=self.sdn_props,
        )
        self.assertEqual(routing.route_helper, mock_helpers_instance)

    def test_route_props_initialization(self) -> None:
        """Test that route props are properly initialized."""
        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )

        self.assertIsInstance(routing.route_props, RoutingProps)
        # Check default values are set
        self.assertEqual(routing.route_props.paths_matrix, [])
        self.assertEqual(routing.route_props.modulation_formats_matrix, [])
        self.assertEqual(routing.route_props.weights_list, [])

    def test_backward_compatibility_imports_exist(self) -> None:
        """Test backward compatibility import aliases exist."""
        from fusion.core.routing import (
            find_path_len,
            get_path_mod,
            sort_nested_dict_vals,
        )

        # These should be aliases to the actual functions
        self.assertIsNotNone(find_path_len)
        self.assertIsNotNone(get_path_mod)
        self.assertIsNotNone(sort_nested_dict_vals)

    def test_logger_initialization(self) -> None:
        """Test that logger is properly initialized."""
        # The logger should be initialized when the module is imported
        from fusion.core.routing import logger

        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "fusion.core.routing")

    def test_engine_props_storage(self) -> None:
        """Test that engine properties are correctly stored."""
        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )

        self.assertEqual(routing.engine_props, self.engine_props)
        self.assertIn("k_paths", routing.engine_props)
        self.assertIn("route_method", routing.engine_props)
        self.assertIn("topology", routing.engine_props)

    def test_sdn_props_storage(self) -> None:
        """Test that SDN properties are correctly stored."""
        routing = Routing(
            engine_props=self.engine_props, sdn_props=self.sdn_props
        )

        self.assertEqual(routing.sdn_props, self.sdn_props)
        self.assertEqual(routing.sdn_props.source, "A")
        self.assertEqual(routing.sdn_props.destination, "C")


if __name__ == "__main__":
    unittest.main()
