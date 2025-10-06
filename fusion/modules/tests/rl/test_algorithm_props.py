"""Unit tests for fusion.modules.rl.algorithms.algorithm_props module."""

import numpy as np
import pytest

from fusion.modules.rl.algorithms.algorithm_props import (
    BanditProps,
    PPOProps,
    QProps,
    RLProps,
)


class TestRLProps:
    """Tests for RLProps class."""

    def test_init_creates_instance_with_none_values(self) -> None:
        """Test that RLProps initializes with None values for basic properties."""
        # Act
        props = RLProps()

        # Assert
        assert props.k_paths is None
        assert props.cores_per_link is None
        assert props.spectral_slots is None
        assert props.num_nodes is None
        assert props.source is None
        assert props.destination is None
        assert props.path_index is None
        assert props.core_index is None

    def test_init_creates_empty_lists(self) -> None:
        """Test that RLProps initializes with empty lists."""
        # Act
        props = RLProps()

        # Assert
        assert props.arrival_list == []
        assert props.depart_list == []
        assert props.paths_list == []
        assert props.chosen_path_list == []

    def test_init_creates_empty_dict(self) -> None:
        """Test that RLProps initializes with empty mock_sdn_dict."""
        # Act
        props = RLProps()

        # Assert
        assert props.mock_sdn_dict == {}
        assert isinstance(props.mock_sdn_dict, dict)

    def test_repr_contains_dict_representation(self) -> None:
        """Test that __repr__ returns string with RLProps and dict."""
        # Arrange
        props = RLProps()
        props.k_paths = 3
        props.num_nodes = 10

        # Act
        result = repr(props)

        # Assert
        assert result.startswith("RLProps(")
        assert "k_paths" in result
        assert "num_nodes" in result

    def test_properties_are_mutable(self) -> None:
        """Test that RLProps properties can be modified."""
        # Arrange
        props = RLProps()

        # Act
        props.k_paths = 5
        props.cores_per_link = 7
        props.spectral_slots = 320
        props.num_nodes = 14

        # Assert
        assert props.k_paths == 5
        assert props.cores_per_link == 7
        assert props.spectral_slots == 320
        assert props.num_nodes == 14


class TestQProps:
    """Tests for QProps class."""

    def test_init_creates_none_values(self) -> None:
        """Test that QProps initializes with None values."""
        # Act
        props = QProps()

        # Assert
        assert props.epsilon is None
        assert props.epsilon_start is None
        assert props.epsilon_end is None
        assert props.is_training is None
        assert props.routes_matrix is None
        assert props.cores_matrix is None
        assert props.num_nodes is None

    def test_init_creates_empty_lists(self) -> None:
        """Test that QProps initializes with empty lists."""
        # Act
        props = QProps()

        # Assert
        assert props.epsilon_list == []

    def test_init_creates_reward_error_dicts(self) -> None:
        """Test that QProps initializes rewards_dict and errors_dict correctly."""
        # Act
        props = QProps()

        # Assert
        assert "routes_dict" in props.rewards_dict
        assert "cores_dict" in props.rewards_dict
        assert props.rewards_dict["routes_dict"] == {
            "average": [],
            "min": [],
            "max": [],
            "rewards": {},
        }
        assert "routes_dict" in props.errors_dict
        assert "cores_dict" in props.errors_dict

    def test_init_creates_sum_dicts(self) -> None:
        """Test that QProps initializes sum dictionaries."""
        # Act
        props = QProps()

        # Assert
        assert props.sum_rewards_dict == {}
        assert props.sum_errors_dict == {}

    def test_init_creates_save_params_dict(self) -> None:
        """Test that QProps initializes save_params_dict with expected keys."""
        # Act
        props = QProps()

        # Assert
        assert "q_params_list" in props.save_params_dict
        assert "engine_params_list" in props.save_params_dict
        assert "rewards_dict" in props.save_params_dict["q_params_list"]
        assert "epsilon_start" in props.save_params_dict["engine_params_list"]

    def test_get_data_returns_existing_attribute(self) -> None:
        """Test that get_data returns value of existing attribute."""
        # Arrange
        props = QProps()
        props.epsilon_start = 0.9
        props.num_nodes = 10

        # Act
        epsilon = props.get_data("epsilon_start")
        nodes = props.get_data("num_nodes")

        # Assert
        assert epsilon == 0.9
        assert nodes == 10

    def test_get_data_raises_for_missing_attribute(self) -> None:
        """Test that get_data raises AttributeError for missing attribute."""
        # Arrange
        props = QProps()

        # Act & Assert
        with pytest.raises(AttributeError, match="'RLProps' object has no attribute"):
            props.get_data("nonexistent_key")

    def test_repr_contains_dict_representation(self) -> None:
        """Test that __repr__ returns string with QProps and dict."""
        # Arrange
        props = QProps()
        props.epsilon_start = 1.0

        # Act
        result = repr(props)

        # Assert
        assert result.startswith("QProps(")
        assert "epsilon_start" in result

    def test_matrices_can_be_set_to_numpy_arrays(self) -> None:
        """Test that matrix properties accept numpy arrays."""
        # Arrange
        props = QProps()
        routes_matrix = np.zeros((5, 5, 3))
        cores_matrix = np.ones((5, 5, 4, 2))

        # Act
        props.routes_matrix = routes_matrix
        props.cores_matrix = cores_matrix

        # Assert
        np.testing.assert_array_equal(props.routes_matrix, routes_matrix)
        np.testing.assert_array_equal(props.cores_matrix, cores_matrix)


class TestBanditProps:
    """Tests for BanditProps class."""

    def test_init_creates_empty_rewards_matrix(self) -> None:
        """Test that BanditProps initializes with empty rewards_matrix."""
        # Act
        props = BanditProps()

        # Assert
        assert props.rewards_matrix == []
        assert isinstance(props.rewards_matrix, list)

    def test_init_creates_empty_counts_list(self) -> None:
        """Test that BanditProps initializes with empty counts_list."""
        # Act
        props = BanditProps()

        # Assert
        assert props.counts_list == []
        assert isinstance(props.counts_list, list)

    def test_init_creates_empty_state_values_list(self) -> None:
        """Test that BanditProps initializes with empty state_values_list."""
        # Act
        props = BanditProps()

        # Assert
        assert props.state_values_list == []
        assert isinstance(props.state_values_list, list)

    def test_repr_contains_dict_representation(self) -> None:
        """Test that __repr__ returns string with BanditProps and dict."""
        # Arrange
        props = BanditProps()

        # Act
        result = repr(props)

        # Assert
        assert result.startswith("BanditProps(")
        assert "rewards_matrix" in result
        assert "counts_list" in result
        assert "state_values_list" in result

    def test_lists_are_mutable(self) -> None:
        """Test that BanditProps lists can be modified."""
        # Arrange
        props = BanditProps()

        # Act
        props.rewards_matrix.append([1.0, 2.0, 3.0])
        props.counts_list.append(np.array([5, 10, 15]))
        props.state_values_list.append({"state1": 0.5})

        # Assert
        assert len(props.rewards_matrix) == 1
        assert props.rewards_matrix[0] == [1.0, 2.0, 3.0]
        assert len(props.counts_list) == 1
        assert len(props.state_values_list) == 1


class TestPPOProps:
    """Tests for PPOProps class."""

    def test_init_creates_instance(self) -> None:
        """Test that PPOProps can be instantiated."""
        # Act
        props = PPOProps()

        # Assert
        assert props is not None
        assert isinstance(props, PPOProps)

    def test_is_placeholder_class(self) -> None:
        """Test that PPOProps is currently a placeholder with no attributes."""
        # Arrange
        props = PPOProps()

        # Act
        attrs = [attr for attr in dir(props) if not attr.startswith("_")]

        # Assert
        # Should have no public attributes (placeholder class)
        assert len(attrs) == 0
