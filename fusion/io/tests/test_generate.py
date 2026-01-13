"""Unit tests for fusion.io.generate module."""

import json
from typing import Any
from unittest.mock import Mock, mock_open, patch

import pytest

from fusion.io.generate import create_bw_info, create_pt


class TestCreatePt:
    """Tests for create_pt function."""

    def test_with_valid_network_dict_returns_topology_dict(self) -> None:
        """Test creating physical topology from valid network spectrum dict."""
        # Arrange
        cores_per_link = 4
        network_spectrum_dict = {
            ("A", "B"): 100.0,
            ("B", "C"): 200.0,
        }

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert "nodes" in result
        assert "links" in result
        assert "A" in result["nodes"]
        assert "B" in result["nodes"]
        assert "C" in result["nodes"]

    def test_nodes_have_cdc_type(self) -> None:
        """Test that all nodes have type 'CDC'."""
        # Arrange
        cores_per_link = 2
        network_spectrum_dict = {("X", "Y"): 50.0, ("Y", "Z"): 75.0}

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert result["nodes"]["X"]["type"] == "CDC"
        assert result["nodes"]["Y"]["type"] == "CDC"
        assert result["nodes"]["Z"]["type"] == "CDC"

    def test_links_have_correct_properties(self) -> None:
        """Test that links have correct properties including source and destination."""
        # Arrange
        cores_per_link = 3
        network_spectrum_dict = {("A", "B"): 100.0}

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        link = result["links"][1]
        assert link["source"] == "A"
        assert link["destination"] == "B"
        assert link["length"] == 100.0
        assert "fiber" in link
        assert link["span_length"] == 100

    def test_fiber_properties_include_num_cores(self) -> None:
        """Test that fiber properties include correct number of cores."""
        # Arrange
        cores_per_link = 7
        network_spectrum_dict = {("A", "B"): 100.0}

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        fiber = result["links"][1]["fiber"]
        assert fiber["num_cores"] == 7

    def test_fiber_properties_have_expected_values(self) -> None:
        """Test that fiber properties have physically meaningful values."""
        # Arrange
        cores_per_link = 1
        network_spectrum_dict = {("A", "B"): 100.0}

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        fiber = result["links"][1]["fiber"]
        assert "attenuation" in fiber
        assert "non_linearity" in fiber
        assert "dispersion" in fiber
        assert "fiber_type" in fiber
        assert "bending_radius" in fiber
        assert "mode_coupling_co" in fiber
        assert "propagation_const" in fiber
        assert "core_pitch" in fiber

    def test_with_multiple_links_creates_numbered_links(self) -> None:
        """Test that multiple links are created with sequential numbering."""
        # Arrange
        cores_per_link = 2
        network_spectrum_dict = {
            ("A", "B"): 100.0,
            ("B", "C"): 150.0,
            ("C", "D"): 200.0,
        }

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert len(result["links"]) == 3
        assert 1 in result["links"]
        assert 2 in result["links"]
        assert 3 in result["links"]

    def test_with_single_link_creates_one_link(self) -> None:
        """Test creating topology with single link."""
        # Arrange
        cores_per_link = 1
        network_spectrum_dict = {("Source", "Dest"): 250.5}

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert len(result["links"]) == 1
        assert result["links"][1]["source"] == "Source"
        assert result["links"][1]["destination"] == "Dest"
        assert result["links"][1]["length"] == 250.5

    def test_link_lengths_match_input_dict(self) -> None:
        """Test that link lengths match input network spectrum dict."""
        # Arrange
        cores_per_link = 4
        network_spectrum_dict = {
            ("A", "B"): 123.45,
            ("C", "D"): 678.90,
        }

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        link1 = result["links"][1]
        link2 = result["links"][2]
        lengths = {link1["length"], link2["length"]}
        assert 123.45 in lengths
        assert 678.90 in lengths

    def test_nodes_created_from_all_unique_endpoints(self) -> None:
        """Test that nodes are created from all unique node names in links."""
        # Arrange
        cores_per_link = 2
        network_spectrum_dict = {
            ("A", "B"): 100.0,
            ("B", "C"): 150.0,
            ("C", "A"): 125.0,
        }

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert len(result["nodes"]) == 3
        assert set(result["nodes"].keys()) == {"A", "B", "C"}

    def test_with_empty_network_dict_raises_value_error(self) -> None:
        """Test that empty network dict raises ValueError."""
        # Arrange
        cores_per_link = 1
        network_spectrum_dict: dict[tuple[str, str], float] = {}

        # Act & Assert
        with pytest.raises(ValueError, match="empty nodes dictionary"):
            create_pt(cores_per_link, network_spectrum_dict)

    def test_with_zero_cores_creates_topology(self) -> None:
        """Test creating topology with zero cores per link."""
        # Arrange
        cores_per_link = 0
        network_spectrum_dict = {("A", "B"): 100.0}

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert result["links"][1]["fiber"]["num_cores"] == 0

    def test_fiber_type_is_zero(self) -> None:
        """Test that fiber_type is set to 0."""
        # Arrange
        cores_per_link = 1
        network_spectrum_dict = {("A", "B"): 100.0}

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert result["links"][1]["fiber"]["fiber_type"] == 0

    def test_span_length_is_100(self) -> None:
        """Test that span_length is set to 100."""
        # Arrange
        cores_per_link = 1
        network_spectrum_dict = {("A", "B"): 100.0}

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert result["links"][1]["span_length"] == 100

    def test_with_large_network_creates_all_nodes_and_links(self) -> None:
        """Test creating topology with large network."""
        # Arrange
        cores_per_link = 2
        network_spectrum_dict = {
            (f"Node{i}", f"Node{i + 1}"): float(i * 10) for i in range(20)
        }

        # Act
        result = create_pt(cores_per_link, network_spectrum_dict)

        # Assert
        assert len(result["links"]) == 20
        assert len(result["nodes"]) == 21


class TestCreateBwInfo:
    """Tests for create_bw_info function."""

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_valid_mod_assumption_returns_dict(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading valid modulation assumption returns dict."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mod_data = {
            "assumption1": {"100G": {"slots": 4, "reach": 1000}},
            "assumption2": {"200G": {"slots": 8, "reach": 500}},
        }
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info("assumption1")

        # Assert
        assert result == {"100G": {"slots": 4, "reach": 1000}}

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    @patch("fusion.io.generate.Path.is_absolute")
    def test_with_custom_path_uses_provided_path(
        self,
        mock_is_absolute: Mock,
        mock_path_open: Mock,
        mock_find_project_root: Mock,
    ) -> None:
        """Test custom modulation assumptions path is used."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_is_absolute.return_value = True
        mod_data = {"test_assumption": {"50G": {"slots": 2, "reach": 2000}}}
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info(
            "test_assumption", mod_assumptions_path="/custom/path.json"
        )

        # Assert
        assert result == {"50G": {"slots": 2, "reach": 2000}}

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_none_path_uses_default(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test None path uses default modulation formats file."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mod_data = {"default": {"100G": {"slots": 4, "reach": 1000}}}
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info("default", mod_assumptions_path=None)

        # Assert
        assert result == {"100G": {"slots": 4, "reach": 1000}}

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_none_string_path_uses_default(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test "None" string path uses default modulation formats file."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mod_data = {"test": {"100G": {"slots": 4, "reach": 1000}}}
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info("test", mod_assumptions_path="None")

        # Assert
        assert result == {"100G": {"slots": 4, "reach": 1000}}

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_unknown_assumption_raises_not_implemented_error(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test unknown modulation assumption raises NotImplementedError."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mod_data = {"known_assumption": {"100G": {"slots": 4, "reach": 1000}}}
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act & Assert
        with pytest.raises(NotImplementedError, match="Unknown modulation assumption"):
            create_bw_info("unknown_assumption")

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_file_not_found_raises_file_not_found_error(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test missing modulation file raises FileNotFoundError."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_path_open.side_effect = FileNotFoundError("File not found")

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="File not found"):
            create_bw_info("test_assumption")

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_invalid_json_raises_file_not_found_error(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test invalid JSON raises FileNotFoundError with parse message."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data="invalid json {"
        ).return_value

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Could not parse JSON"):
            create_bw_info("test")

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    @patch("fusion.io.generate.Path.is_absolute")
    def test_with_relative_path_resolves_to_absolute(
        self,
        mock_is_absolute: Mock,
        mock_path_open: Mock,
        mock_find_project_root: Mock,
    ) -> None:
        """Test relative path is resolved to absolute path."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_is_absolute.return_value = False
        mod_data = {"test": {"100G": {"slots": 4, "reach": 1000}}}
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info("test", mod_assumptions_path="relative/path.json")

        # Assert
        assert result == {"100G": {"slots": 4, "reach": 1000}}

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_returns_dict_of_dicts(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test function returns dict of dicts structure."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mod_data = {
            "test": {
                "100G": {"slots": 4, "reach": 1000},
                "200G": {"slots": 8, "reach": 500},
            }
        }
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info("test")

        # Assert
        assert isinstance(result, dict)
        assert "100G" in result
        assert "200G" in result
        assert isinstance(result["100G"], dict)

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_multiple_assumptions_returns_correct_one(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test selecting specific assumption from multiple options."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mod_data = {
            "assumption_a": {"100G": {"slots": 4, "reach": 1000}},
            "assumption_b": {"100G": {"slots": 6, "reach": 800}},
            "assumption_c": {"100G": {"slots": 8, "reach": 600}},
        }
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info("assumption_b")

        # Assert
        assert result == {"100G": {"slots": 6, "reach": 800}}

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_empty_assumption_dict_returns_empty_dict(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test modulation assumption with empty dict returns empty dict."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mod_data: dict[str, dict[str, Any]] = {"empty_assumption": {}}
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info("empty_assumption")

        # Assert
        assert result == {}

    @patch("fusion.io.generate.find_project_root")
    @patch("fusion.io.generate.Path.open")
    def test_with_nested_dict_structure_returns_correctly(
        self, mock_path_open: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test nested dict structure is returned correctly."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mod_data = {
            "complex": {
                "100G": {
                    "slots": 4,
                    "reach": 1000,
                    "modulation": {"type": "QPSK", "efficiency": 2.0},
                }
            }
        }
        mock_path_open.return_value.__enter__.return_value = mock_open(
            read_data=json.dumps(mod_data)
        ).return_value

        # Act
        result = create_bw_info("complex")

        # Assert
        assert result["100G"]["modulation"]["type"] == "QPSK"
        assert result["100G"]["modulation"]["efficiency"] == 2.0
