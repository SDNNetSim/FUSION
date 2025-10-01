"""
Unit tests for fusion.core.request module.

Tests request generation functionality including validation, node selection,
timing generation, and complete request creation workflows.
"""

from typing import Any
from unittest.mock import patch

import pytest

from ..request import (
    DEFAULT_REQUEST_TYPE_ARRIVAL,
    DEFAULT_REQUEST_TYPE_RELEASE,
    generate_simulation_requests,
    get_requests,
    validate_request_distribution,
)


class TestRequestDistributionValidation:
    """Tests for request distribution validation functionality."""

    def test_validate_request_distribution_with_valid_even_distribution_returns_true(
        self,
    ) -> None:
        """Test validation passes with evenly distributable percentages."""
        # Arrange
        distribution = {"50GHz": 0.5, "100GHz": 0.5}
        num_requests = 100

        # Act
        result = validate_request_distribution(distribution, num_requests)

        # Assert
        assert result is True

    def test_validate_request_distribution_with_uneven_distribution_returns_false(
        self,
    ) -> None:
        """Test validation fails with non-distributable percentages."""
        # Arrange
        distribution = {"50GHz": 0.33, "100GHz": 0.33, "150GHz": 0.33}
        num_requests = 100  # 33 + 33 + 33 = 99, not 100

        # Act
        result = validate_request_distribution(distribution, num_requests)

        # Assert
        assert result is False

    def test_validate_request_distribution_with_single_bandwidth_returns_true(
        self,
    ) -> None:
        """Test validation with single bandwidth distribution."""
        # Arrange
        distribution = {"100GHz": 1.0}
        num_requests = 50

        # Act
        result = validate_request_distribution(distribution, num_requests)

        # Assert
        assert result is True

    def test_validate_request_distribution_with_zero_requests_returns_true(
        self,
    ) -> None:
        """Test validation with zero requests."""
        # Arrange
        distribution = {"50GHz": 0.6, "100GHz": 0.4}
        num_requests = 0

        # Act
        result = validate_request_distribution(distribution, num_requests)

        # Assert
        assert result is True

    @pytest.mark.parametrize("distribution,num_requests,expected", [
        ({"50GHz": 0.25, "100GHz": 0.75}, 4, True),
        ({"50GHz": 0.2, "100GHz": 0.8}, 5, True),
        ({"50GHz": 0.3, "100GHz": 0.7}, 10, True),
        ({"50GHz": 0.33, "100GHz": 0.67}, 3, False),  # 0.99 + 2.01 ≠ 3
    ])
    def test_validate_request_distribution_parametrized_cases(
        self, distribution: Any, num_requests: Any, expected: Any
    ) -> None:
        """Test validation with various distribution scenarios."""
        result = validate_request_distribution(distribution, num_requests)
        assert result == expected


class TestRequestGenerationCore:
    """Tests for core request generation functionality."""

    @pytest.fixture
    def basic_engine_props(self) -> Any:
        """Provide basic engine properties for testing."""
        return {
            "is_only_core_node": True,
            "topology_info": {
                "nodes": {"A": {}, "B": {}, "C": {}}
            },
            "arrival_rate": 1.0,
            "holding_time": 10.0,
            "num_requests": 10,
            "request_distribution": {"50GHz": 0.5, "100GHz": 0.5},
            "mod_per_bw": {
                "50GHz": {"QPSK": {"max_length": [100]}},
                "100GHz": {"QPSK": {"max_length": [200]}}
            }
        }

    @patch('fusion.core.request.set_random_seed')
    @patch('fusion.core.request.generate_exponential_random_variable')
    @patch('fusion.core.request.generate_uniform_random_variable')
    def test_generate_simulation_requests_creates_correct_number_of_events(
        self,
        mock_uniform: Any,
        mock_exponential: Any,
        mock_seed: Any,
        basic_engine_props: Any,
    ) -> None:
        """Test that correct number of arrival and departure events are created."""
        # Arrange
        # Create a cycle of values to avoid StopIteration
        arrival_times = [float(i) for i in range(1, 50)]
        holding_times = [5.0] * 50
        mock_exponential.side_effect = [
            val
            for pair in zip(arrival_times, holding_times, strict=False)
            for val in pair
        ]

        # Create enough uniform values for node and bandwidth selection
        mock_uniform.side_effect = ([0, 1, 0] * 50)
        seed = 42

        # Act
        requests = generate_simulation_requests(seed, basic_engine_props)

        # Assert
        assert len(requests) == 20  # 10 requests * 2 events each
        mock_seed.assert_called_once_with(seed_value=42)

    @patch('fusion.core.request.set_random_seed')
    @patch('fusion.core.request.generate_exponential_random_variable')
    @patch('fusion.core.request.generate_uniform_random_variable')
    def test_generate_simulation_requests_creates_arrival_and_departure_pairs(
        self,
        mock_uniform: Any,
        mock_exponential: Any,
        mock_seed: Any,
        basic_engine_props: Any,
    ) -> None:
        """Test that each request creates both arrival and departure events."""
        # Arrange
        basic_engine_props["num_requests"] = 2
        mock_exponential.side_effect = [1.0, 5.0, 2.0, 5.0]  # Times for 2 requests
        mock_uniform.side_effect = [0, 1, 0, 0, 2, 1]  # Node and bandwidth selections
        seed = 123

        # Act
        requests = generate_simulation_requests(seed, basic_engine_props)

        # Assert
        assert len(requests) == 4  # 2 requests * 2 events each

        # Check that we have both arrival and departure events
        request_types = [req["request_type"] for req in requests.values()]
        assert request_types.count(DEFAULT_REQUEST_TYPE_ARRIVAL) == 2
        assert request_types.count(DEFAULT_REQUEST_TYPE_RELEASE) == 2

    def test_generate_simulation_requests_with_core_nodes_only_uses_core_nodes(
        self, basic_engine_props: Any
    ) -> None:
        """Test request generation when is_only_core_node is False."""
        # Arrange
        basic_engine_props["is_only_core_node"] = False
        basic_engine_props["core_nodes"] = ["A", "B"]
        basic_engine_props["num_requests"] = 2
        # Ensure valid distribution
        basic_engine_props["request_distribution"] = {"50GHz": 1.0}
        basic_engine_props["mod_per_bw"] = {
            "50GHz": {"QPSK": {"max_length": [100]}}
        }

        # Generate enough mock values to avoid StopIteration
        exponential_values = [float(i) for i in range(1, 20)] + [5.0] * 20
        uniform_values = ([0, 1, 0] * 20)

        # Act & Assert - should use core_nodes list instead of all nodes
        with (
            patch('fusion.core.request.set_random_seed'),
            patch(
                'fusion.core.request.generate_exponential_random_variable',
                side_effect=exponential_values,
            ),
            patch(
                'fusion.core.request.generate_uniform_random_variable',
                side_effect=uniform_values,
            ),
        ):
            requests = generate_simulation_requests(42, basic_engine_props)
            assert len(requests) == 4  # 2 requests * 2 events each

    def test_generate_simulation_requests_with_empty_nodes_raises_error(self) -> None:
        """Test error handling when no nodes are available."""
        # Arrange
        engine_props = {
            "is_only_core_node": True,
            "topology_info": {"nodes": {}},
            "num_requests": 5,
            "request_distribution": {"50GHz": 1.0},
            "mod_per_bw": {"50GHz": {"QPSK": {"max_length": [100]}}}
        }

        # Act & Assert
        with pytest.raises(ValueError, match="No nodes found in topology_info"):
            generate_simulation_requests(42, engine_props)

    def test_generate_simulation_requests_with_empty_core_nodes_raises_error(
        self,
    ) -> None:
        """Test error handling when core_nodes list is empty."""
        # Arrange
        engine_props = {
            "is_only_core_node": False,
            "core_nodes": [],
            "num_requests": 5,
            "request_distribution": {"50GHz": 1.0},
            "mod_per_bw": {"50GHz": {"QPSK": {"max_length": [100]}}}
        }

        # Act & Assert
        with pytest.raises(ValueError, match="No core nodes found"):
            generate_simulation_requests(42, engine_props)

    def test_generate_simulation_requests_with_invalid_distribution_raises_error(
        self, basic_engine_props: Any
    ) -> None:
        """Test error handling with invalid request distribution."""
        # Arrange
        basic_engine_props["request_distribution"] = {"50GHz": 0.33, "100GHz": 0.33}
        basic_engine_props["num_requests"] = 100  # Cannot distribute evenly

        # Act & Assert
        with pytest.raises(ValueError, match="could not be distributed according"):
            generate_simulation_requests(42, basic_engine_props)


class TestRequestStructure:
    """Tests for request data structure and content validation."""

    @pytest.fixture
    def sample_engine_props(self) -> Any:
        """Provide sample engine properties with deterministic behavior."""
        return {
            "is_only_core_node": True,
            "topology_info": {"nodes": {"A": {}, "B": {}}},
            "arrival_rate": 1.0,
            "holding_time": 10.0,
            "num_requests": 1,
            "request_distribution": {"100GHz": 1.0},
            "mod_per_bw": {
                "100GHz": {"QPSK": {"max_length": [200]}}
            }
        }

    @patch('fusion.core.request.set_random_seed')
    @patch('fusion.core.request.generate_exponential_random_variable')
    @patch('fusion.core.request.generate_uniform_random_variable')
    def test_generated_request_contains_required_fields(
        self,
        mock_uniform: Any,
        mock_exponential: Any,
        mock_seed: Any,
        sample_engine_props: Any,
    ) -> None:
        """Test that generated requests contain all required fields."""
        # Arrange
        mock_exponential.side_effect = [1.0, 5.0]  # arrival time, holding time
        mock_uniform.side_effect = [0, 1, 0]  # source, dest, bandwidth selection

        # Act
        requests = generate_simulation_requests(42, sample_engine_props)

        # Assert
        request = list(requests.values())[0]  # Get first request
        assert "req_id" in request
        assert "source" in request
        assert "destination" in request
        assert "arrive" in request
        assert "depart" in request
        assert "request_type" in request
        assert "bandwidth" in request
        assert "mod_formats" in request

    @patch('fusion.core.request.set_random_seed')
    @patch('fusion.core.request.generate_exponential_random_variable')
    @patch('fusion.core.request.generate_uniform_random_variable')
    def test_generated_request_has_different_source_and_destination(
        self,
        mock_uniform: Any,
        mock_exponential: Any,
        mock_seed: Any,
        sample_engine_props: Any,
    ) -> None:
        """Test that requests have different source and destination nodes."""
        # Arrange
        mock_exponential.side_effect = [1.0, 5.0]
        # First node selection: source=0 (A), dest=0 (A) -> retry -> dest=1 (B)
        # source, dest(retry), dest(final), bandwidth
        mock_uniform.side_effect = [0, 0, 1, 0]

        # Act
        requests = generate_simulation_requests(42, sample_engine_props)

        # Assert
        request = list(requests.values())[0]
        assert request["source"] != request["destination"]

    @patch('fusion.core.request.set_random_seed')
    @patch('fusion.core.request.generate_exponential_random_variable')
    @patch('fusion.core.request.generate_uniform_random_variable')
    def test_generated_request_timing_is_consistent(
        self,
        mock_uniform: Any,
        mock_exponential: Any,
        mock_seed: Any,
        sample_engine_props: Any,
    ) -> None:
        """Test that arrival and departure times are consistent."""
        # Arrange
        mock_exponential.side_effect = [2.0, 3.0]  # arrival offset, holding time
        mock_uniform.side_effect = [0, 1, 0]

        # Act
        requests = generate_simulation_requests(42, sample_engine_props)

        # Assert
        arrival_req = None
        departure_req = None

        for request in requests.values():
            if request["request_type"] == DEFAULT_REQUEST_TYPE_ARRIVAL:
                arrival_req = request
            elif request["request_type"] == DEFAULT_REQUEST_TYPE_RELEASE:
                departure_req = request

        assert arrival_req is not None
        assert departure_req is not None
        assert arrival_req["arrive"] == departure_req["arrive"]
        assert arrival_req["depart"] == departure_req["depart"]
        assert departure_req["depart"] > departure_req["arrive"]

    @patch('fusion.core.request.set_random_seed')
    @patch('fusion.core.request.generate_exponential_random_variable')
    @patch('fusion.core.request.generate_uniform_random_variable')
    def test_generated_requests_respect_bandwidth_distribution(
        self, mock_uniform: Any, mock_exponential: Any, mock_seed: Any
    ) -> None:
        """Test that bandwidth distribution is respected."""
        # Arrange
        engine_props = {
            "is_only_core_node": True,
            "topology_info": {"nodes": {"A": {}, "B": {}}},
            "arrival_rate": 1.0,
            "holding_time": 10.0,
            "num_requests": 4,
            "request_distribution": {"50GHz": 0.25, "100GHz": 0.75},
            "mod_per_bw": {
                "50GHz": {"QPSK": {"max_length": [100]}},
                "100GHz": {"QPSK": {"max_length": [200]}}
            }
        }
        mock_exponential.side_effect = [1.0, 5.0] * 10  # Times for all requests
        # Multiple selections
        mock_uniform.side_effect = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0] * 5

        # Act
        requests = generate_simulation_requests(42, engine_props)

        # Assert
        bandwidths = [
            req["bandwidth"]
            for req in requests.values()
            if req["request_type"] == DEFAULT_REQUEST_TYPE_ARRIVAL
        ]
        assert bandwidths.count("50GHz") == 1  # 25% of 4
        assert bandwidths.count("100GHz") == 3  # 75% of 4


class TestLegacyCompatibility:
    """Tests for backward compatibility functionality."""

    @patch('fusion.core.request.generate_simulation_requests')
    @patch('fusion.core.request.logger')
    def test_get_requests_calls_new_function_and_logs_warning(
        self, mock_logger: Any, mock_generate_requests: Any
    ) -> None:
        """Test that legacy function calls new function and logs deprecation warning."""
        # Arrange
        seed = 42
        engine_props = {"test": "data"}
        expected_result = {"time": {"request": "data"}}
        mock_generate_requests.return_value = expected_result

        # Act
        result = get_requests(seed, engine_props)

        # Assert
        mock_generate_requests.assert_called_once_with(seed, engine_props)
        mock_logger.warning.assert_called_once_with(
            "get_requests is deprecated. Use generate_simulation_requests instead."
        )
        assert result == expected_result


class TestRequestGenerationEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def edge_case_props(self) -> Any:
        """Provide properties for edge case testing."""
        return {
            "is_only_core_node": True,
            "topology_info": {"nodes": {"A": {}, "B": {}}},
            "arrival_rate": 1.0,
            "holding_time": 10.0,
            "num_requests": 2,
            "request_distribution": {"100GHz": 1.0},
            "mod_per_bw": {"100GHz": {"QPSK": {"max_length": [200]}}}
        }

    @patch('fusion.core.request.set_random_seed')
    @patch('fusion.core.request.generate_exponential_random_variable')
    @patch('fusion.core.request.generate_uniform_random_variable')
    def test_generate_requests_handles_time_collisions(
        self,
        mock_uniform: Any,
        mock_exponential: Any,
        mock_seed: Any,
        edge_case_props: Any,
    ) -> None:
        """Test handling of time collisions during request generation."""
        # Arrange - create scenario where times collide
        mock_exponential.side_effect = [
            1.0, 5.0,  # First request times
            0.0, 5.0,  # Second request - collision on arrival (1.0)
            2.0, 5.0   # Second request retry - different arrival time
        ]
        mock_uniform.side_effect = [0, 1, 0] * 3  # Multiple attempts

        # Act
        requests = generate_simulation_requests(42, edge_case_props)

        # Assert
        # Should eventually complete despite collision
        assert len(requests) == 4  # 2 requests * 2 events each

    def test_generate_requests_with_missing_topology_info_raises_error(self) -> None:
        """Test error handling when topology_info is missing."""
        # Arrange
        engine_props = {
            "is_only_core_node": True,
            "num_requests": 1,
            "request_distribution": {"50GHz": 1.0},
            "mod_per_bw": {"50GHz": {"QPSK": {"max_length": [100]}}},
            # topology_info is intentionally missing to trigger the KeyError
        }

        # Act & Assert - this will raise KeyError, not ValueError
        with pytest.raises(KeyError, match="topology_info"):
            generate_simulation_requests(42, engine_props)

    def test_generate_requests_with_missing_core_nodes_raises_error(self) -> None:
        """Test error handling when core_nodes is missing for non-core-only mode."""
        # Arrange
        engine_props = {
            "is_only_core_node": False,
            "num_requests": 1,
            "request_distribution": {"50GHz": 1.0},
            "mod_per_bw": {"50GHz": {"QPSK": {"max_length": [100]}}},
            # core_nodes is intentionally missing to trigger the KeyError
        }

        # Act & Assert - this will raise KeyError, not ValueError
        with pytest.raises(KeyError, match="core_nodes"):
            generate_simulation_requests(42, engine_props)

    @patch('fusion.core.request.set_random_seed')
    def test_generate_requests_with_zero_requests_returns_empty_dict(
        self, mock_seed: Any
    ) -> None:
        """Test generation with zero requests returns empty dictionary."""
        # Arrange
        engine_props = {
            "is_only_core_node": True,
            "topology_info": {"nodes": {"A": {}, "B": {}}},
            "num_requests": 0,
            "request_distribution": {"100GHz": 1.0},
            "mod_per_bw": {"100GHz": {"QPSK": {"max_length": [200]}}}
        }

        # Act
        requests = generate_simulation_requests(42, engine_props)

        # Assert
        assert requests == {}
        mock_seed.assert_called_once_with(seed_value=42)
