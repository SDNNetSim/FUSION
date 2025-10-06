"""Unit tests for fusion.io.structure module."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import Mock, patch

import pytest

from fusion.io.structure import assign_core_nodes, assign_link_lengths, create_network


class TestAssignLinkLengths:
    """Tests for assign_link_lengths function."""

    def test_with_valid_network_file_returns_link_lengths_dict(self) -> None:
        """Test assigning link lengths from valid network file."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("A\tB\t100.5\n")
            f.write("B\tC\t200.75\n")
            f.write("C\tD\t150.25\n")
            network_fp = Path(f.name)

        try:
            node_pairs_dict: dict[str, str] = {}

            # Act
            result = assign_link_lengths(network_fp, node_pairs_dict)

            # Assert
            assert len(result) == 3
            assert result[("A", "B")] == 100.5
            assert result[("B", "C")] == 200.75
            assert result[("C", "D")] == 150.25
        finally:
            network_fp.unlink()

    def test_with_constant_weight_returns_all_ones(self) -> None:
        """Test assigning constant weight of 1.0 to all links."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("A\tB\t100.5\n")
            f.write("B\tC\t200.75\n")
            network_fp = Path(f.name)

        try:
            node_pairs_dict: dict[str, str] = {}

            # Act
            result = assign_link_lengths(
                network_fp, node_pairs_dict, constant_weight=True
            )

            # Assert
            assert result[("A", "B")] == 1.0
            assert result[("B", "C")] == 1.0
        finally:
            network_fp.unlink()

    def test_with_node_pairs_dict_maps_node_names(self) -> None:
        """Test node name mapping using node_pairs_dict."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Node1\tNode2\t100.0\n")
            f.write("Node2\tNode3\t200.0\n")
            network_fp = Path(f.name)

        try:
            node_pairs_dict = {"Node1": "A", "Node2": "B", "Node3": "C"}

            # Act
            result = assign_link_lengths(network_fp, node_pairs_dict)

            # Assert
            assert ("A", "B") in result
            assert ("B", "C") in result
            assert result[("A", "B")] == 100.0
            assert result[("B", "C")] == 200.0
        finally:
            network_fp.unlink()

    def test_with_partial_node_pairs_dict_maps_only_specified_nodes(self) -> None:
        """Test partial node name mapping preserves unmapped node names."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Node1\tNode2\t100.0\n")
            network_fp = Path(f.name)

        try:
            node_pairs_dict = {"Node1": "A"}

            # Act
            result = assign_link_lengths(network_fp, node_pairs_dict)

            # Assert
            assert ("A", "Node2") in result
            assert result[("A", "Node2")] == 100.0
        finally:
            network_fp.unlink()

    def test_with_duplicate_links_keeps_first_occurrence(self) -> None:
        """Test duplicate links only stores first occurrence."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("A\tB\t100.0\n")
            f.write("A\tB\t200.0\n")
            network_fp = Path(f.name)

        try:
            node_pairs_dict: dict[str, str] = {}

            # Act
            result = assign_link_lengths(network_fp, node_pairs_dict)

            # Assert
            assert len(result) == 1
            assert result[("A", "B")] == 100.0
        finally:
            network_fp.unlink()

    def test_with_empty_file_returns_empty_dict(self) -> None:
        """Test assigning link lengths from empty file returns empty dict."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            network_fp = Path(f.name)

        try:
            node_pairs_dict: dict[str, str] = {}

            # Act
            result = assign_link_lengths(network_fp, node_pairs_dict)

            # Assert
            assert result == {}
        finally:
            network_fp.unlink()

    def test_with_whitespace_in_node_names_preserved(self) -> None:
        """Test that whitespace in node names is preserved after tab split."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("A  \t  B\t100.0\n")
            network_fp = Path(f.name)

        try:
            node_pairs_dict: dict[str, str] = {}

            # Act
            result = assign_link_lengths(network_fp, node_pairs_dict)

            # Assert
            assert ("A  ", "  B") in result
            assert result[("A  ", "  B")] == 100.0
        finally:
            network_fp.unlink()

    def test_with_empty_node_pairs_dict_preserves_original_names(self) -> None:
        """Test with empty node_pairs_dict preserves original node names."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("NodeA\tNodeB\t50.0\n")
            network_fp = Path(f.name)

        try:
            node_pairs_dict: dict[str, str] = {}

            # Act
            result = assign_link_lengths(network_fp, node_pairs_dict)

            # Assert
            assert ("NodeA", "NodeB") in result
        finally:
            network_fp.unlink()


class TestAssignCoreNodes:
    """Tests for assign_core_nodes function."""

    def test_with_valid_core_nodes_file_returns_list(self) -> None:
        """Test reading core nodes from valid file."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Node1\tCore\n")
            f.write("Node2\tCore\n")
            f.write("Node3\tCore\n")
            core_nodes_fp = Path(f.name)

        try:
            # Act
            result = assign_core_nodes(core_nodes_fp)

            # Assert
            assert len(result) == 3
            assert "Node1" in result
            assert "Node2" in result
            assert "Node3" in result
        finally:
            core_nodes_fp.unlink()

    def test_with_single_core_node_returns_single_element_list(self) -> None:
        """Test reading single core node returns list with one element."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("CoreNode\tdata\n")
            core_nodes_fp = Path(f.name)

        try:
            # Act
            result = assign_core_nodes(core_nodes_fp)

            # Assert
            assert len(result) == 1
            assert result[0] == "CoreNode"
        finally:
            core_nodes_fp.unlink()

    def test_with_empty_file_returns_empty_list(self) -> None:
        """Test reading empty file returns empty list."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            core_nodes_fp = Path(f.name)

        try:
            # Act
            result = assign_core_nodes(core_nodes_fp)

            # Assert
            assert result == []
        finally:
            core_nodes_fp.unlink()

    def test_extracts_only_first_column(self) -> None:
        """Test that only first column is extracted from each line."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Node1\textra\tdata\there\n")
            f.write("Node2\tmore\tcolumns\n")
            core_nodes_fp = Path(f.name)

        try:
            # Act
            result = assign_core_nodes(core_nodes_fp)

            # Assert
            assert len(result) == 2
            assert result[0] == "Node1"
            assert result[1] == "Node2"
        finally:
            core_nodes_fp.unlink()

    def test_with_whitespace_in_node_names_preserved(self) -> None:
        """Test that trailing whitespace in node names is preserved."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Node1  \tdata\n")
            f.write("Node2\tdata\n")
            core_nodes_fp = Path(f.name)

        try:
            # Act
            result = assign_core_nodes(core_nodes_fp)

            # Assert
            assert result[0] == "Node1  "
            assert result[1] == "Node2"
        finally:
            core_nodes_fp.unlink()

    def test_preserves_order_of_nodes(self) -> None:
        """Test that order of nodes in file is preserved."""
        # Arrange
        with NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Z\tdata\n")
            f.write("A\tdata\n")
            f.write("M\tdata\n")
            core_nodes_fp = Path(f.name)

        try:
            # Act
            result = assign_core_nodes(core_nodes_fp)

            # Assert
            assert result == ["Z", "A", "M"]
        finally:
            core_nodes_fp.unlink()


class TestCreateNetwork:
    """Tests for create_network function."""

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_usnet_loads_network_successfully(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading USNet network successfully."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("A", "B"): 100.0}

        # Act
        result = create_network("USNet")

        # Assert
        network_dict, core_nodes = result
        assert network_dict == {("A", "B"): 100.0}
        assert core_nodes == []
        mock_assign_link_lengths.assert_called_once()

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_nsfnet_loads_network_successfully(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading NSFNet network successfully."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("X", "Y"): 200.0}

        # Act
        result = create_network("NSFNet")

        # Assert
        network_dict, core_nodes = result
        assert network_dict == {("X", "Y"): 200.0}
        assert core_nodes == []

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_const_weight_passes_constant_weight_flag(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test const_weight parameter is passed to assign_link_lengths."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("A", "B"): 1.0}

        # Act
        create_network("USNet", const_weight=True)

        # Assert
        call_args = mock_assign_link_lengths.call_args
        assert call_args.kwargs["constant_weight"] is True

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    @patch("fusion.io.structure.assign_core_nodes")
    def test_with_usbackbone60_loads_core_nodes(
        self,
        mock_assign_core_nodes: Mock,
        mock_assign_link_lengths: Mock,
        mock_find_project_root: Mock,
    ) -> None:
        """Test loading USbackbone60 loads core nodes file."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("A", "B"): 100.0}
        mock_assign_core_nodes.return_value = ["Node1", "Node2"]

        # Act
        result = create_network("USbackbone60")

        # Assert
        network_dict, core_nodes = result
        assert core_nodes == ["Node1", "Node2"]
        mock_assign_core_nodes.assert_called_once()

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_usbackbone60_and_only_core_node_skips_core_nodes(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test USbackbone60 with is_only_core_node=True skips core nodes."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("A", "B"): 100.0}

        # Act
        result = create_network("USbackbone60", is_only_core_node=True)

        # Assert
        network_dict, core_nodes = result
        assert core_nodes == []

    @patch("fusion.io.structure.find_project_root")
    def test_with_unknown_network_raises_not_implemented_error(
        self, mock_find_project_root: Mock
    ) -> None:
        """Test unknown network name raises NotImplementedError."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"

        # Act & Assert
        with pytest.raises(NotImplementedError, match="Unknown network name"):
            create_network("UnknownNetwork")

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_base_fp_uses_custom_path(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test custom base_fp is used for network files."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("A", "B"): 100.0}

        # Act
        create_network("USNet", base_fp="/custom/path")

        # Assert
        call_args = mock_assign_link_lengths.call_args
        network_fp = call_args.kwargs["network_fp"]
        assert "/custom/path/raw" in str(network_fp)

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_none_base_fp_uses_default_path(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test None base_fp uses default data/raw path."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("A", "B"): 100.0}

        # Act
        create_network("USNet", base_fp=None)

        # Assert
        call_args = mock_assign_link_lengths.call_args
        network_fp = call_args.kwargs["network_fp"]
        assert "data/raw" in str(network_fp)

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_returns_tuple_with_dict_and_list(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test function returns tuple of (dict, list)."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("A", "B"): 100.0}

        # Act
        result = create_network("USNet")

        # Assert
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], list)

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_pan_european_network_loads_successfully(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading Pan-European network successfully."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("E1", "E2"): 300.0}

        # Act
        result = create_network("Pan-European")

        # Assert
        network_dict, core_nodes = result
        assert network_dict == {("E1", "E2"): 300.0}

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_spainbackbone30_loads_successfully(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading Spainbackbone30 network successfully."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("S1", "S2"): 150.0}

        # Act
        result = create_network("Spainbackbone30")

        # Assert
        network_dict, core_nodes = result
        assert network_dict == {("S1", "S2"): 150.0}

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_geant_network_loads_successfully(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading geant network successfully."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("G1", "G2"): 250.0}

        # Act
        result = create_network("geant")

        # Assert
        network_dict, core_nodes = result
        assert network_dict == {("G1", "G2"): 250.0}

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_toy_network_loads_successfully(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading toy_network successfully."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("T1", "T2"): 50.0}

        # Act
        result = create_network("toy_network")

        # Assert
        network_dict, core_nodes = result
        assert network_dict == {("T1", "T2"): 50.0}

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_metro_net_loads_successfully(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading metro_net successfully."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("M1", "M2"): 75.0}

        # Act
        result = create_network("metro_net")

        # Assert
        network_dict, core_nodes = result
        assert network_dict == {("M1", "M2"): 75.0}

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_with_dt_network_loads_successfully(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test loading dt_network successfully."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("D1", "D2"): 125.0}

        # Act
        result = create_network("dt_network")

        # Assert
        network_dict, core_nodes = result
        assert network_dict == {("D1", "D2"): 125.0}

    @patch("fusion.io.structure.find_project_root")
    @patch("fusion.io.structure.assign_link_lengths")
    def test_passes_empty_node_pairs_dict_to_assign_link_lengths(
        self, mock_assign_link_lengths: Mock, mock_find_project_root: Mock
    ) -> None:
        """Test that empty node_pairs_dict is passed to assign_link_lengths."""
        # Arrange
        mock_find_project_root.return_value = "/fake/root"
        mock_assign_link_lengths.return_value = {("A", "B"): 100.0}

        # Act
        create_network("USNet")

        # Assert
        call_args = mock_assign_link_lengths.call_args
        assert call_args.kwargs["node_pairs_dict"] == {}
