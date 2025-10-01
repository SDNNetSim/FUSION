"""Unit tests for fusion.io.exporter module."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock

import pytest

from fusion.io.exporter import (
    BaseExporter,
    CSVExporter,
    ExporterRegistry,
    JSONExporter,
    SimulationDataExporter,
)


class TestBaseExporter:
    """Tests for BaseExporter abstract base class."""

    def test_str_returns_class_name(self) -> None:
        """Test that __str__ returns the class name."""
        # Arrange
        class ConcreteExporter(BaseExporter):
            def export(self, data: Any, output_path: Path) -> None:
                pass

        exporter = ConcreteExporter()

        # Act
        result = str(exporter)

        # Assert
        assert result == "ConcreteExporter"

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseExporter cannot be instantiated directly."""
        # Act & Assert
        with pytest.raises(TypeError):
            BaseExporter()  # type: ignore


class TestJSONExporter:
    """Tests for JSONExporter class."""

    def test_str_returns_json_exporter(self) -> None:
        """Test that __str__ returns 'JSONExporter'."""
        # Arrange
        exporter = JSONExporter()

        # Act
        result = str(exporter)

        # Assert
        assert result == "JSONExporter"

    def test_export_with_dict_creates_json_file(self) -> None:
        """Test exporting dictionary data to JSON file."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = JSONExporter()
            data = {"key": "value", "number": 42}
            output_path = Path(temp_dir) / "test.json"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                loaded = json.load(f)
            assert loaded == data

    def test_export_with_list_creates_json_file(self) -> None:
        """Test exporting list data to JSON file."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = JSONExporter()
            data = [{"id": 1}, {"id": 2}]
            output_path = Path(temp_dir) / "test.json"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                loaded = json.load(f)
            assert loaded == data

    def test_export_with_nested_data_creates_json_file(self) -> None:
        """Test exporting nested data structures to JSON file."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = JSONExporter()
            data = {
                "nodes": {"A": {"type": "CDC"}, "B": {"type": "CDC"}},
                "links": [{"source": "A", "destination": "B"}],
            }
            output_path = Path(temp_dir) / "test.json"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                loaded = json.load(f)
            assert loaded == data

    def test_export_creates_parent_directories(self) -> None:
        """Test that export creates parent directories if they don't exist."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = JSONExporter()
            data = {"key": "value"}
            output_path = Path(temp_dir) / "subdir1" / "subdir2" / "test.json"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_export_with_empty_dict_creates_json_file(self) -> None:
        """Test exporting empty dictionary to JSON file."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = JSONExporter()
            data: dict[str, Any] = {}
            output_path = Path(temp_dir) / "test.json"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                loaded = json.load(f)
            assert loaded == {}

    def test_export_with_special_types_uses_default_str(self) -> None:
        """Test exporting data with non-serializable types uses str conversion."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = JSONExporter()
            data = {"path": Path("/test/path")}
            output_path = Path(temp_dir) / "test.json"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                loaded = json.load(f)
            assert isinstance(loaded["path"], str)

    def test_export_overwrites_existing_file(self) -> None:
        """Test that export overwrites existing file."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = JSONExporter()
            output_path = Path(temp_dir) / "test.json"
            output_path.write_text('{"old": "data"}')
            data = {"new": "data"}

            # Act
            exporter.export(data, output_path)

            # Assert
            with output_path.open() as f:
                loaded = json.load(f)
            assert loaded == data
            assert "old" not in loaded


class TestCSVExporter:
    """Tests for CSVExporter class."""

    def test_str_returns_csv_exporter(self) -> None:
        """Test that __str__ returns 'CSVExporter'."""
        # Arrange
        exporter = CSVExporter()

        # Act
        result = str(exporter)

        # Assert
        assert result == "CSVExporter"

    def test_export_with_list_of_dicts_creates_csv_file(self) -> None:
        """Test exporting list of dictionaries to CSV file."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = CSVExporter()
            data = [{"id": 1, "name": "Test1"}, {"id": 2, "name": "Test2"}]
            output_path = Path(temp_dir) / "test.csv"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["id"] == "1"
            assert rows[0]["name"] == "Test1"
            assert rows[1]["id"] == "2"
            assert rows[1]["name"] == "Test2"

    def test_export_with_empty_list_does_not_create_file(self) -> None:
        """Test exporting empty list does not create file."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = CSVExporter()
            data: list[dict[str, Any]] = []
            output_path = Path(temp_dir) / "test.csv"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert not output_path.exists()

    def test_export_with_inconsistent_keys_includes_all_keys(self) -> None:
        """Test exporting list with inconsistent keys includes all unique keys."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = CSVExporter()
            data: list[dict[Any, Any]] = [
                {"id": 1, "name": "Test1"},
                {"id": 2, "value": 100},
            ]
            output_path = Path(temp_dir) / "test.csv"

            # Act
            exporter.export(data, output_path)

            # Assert
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            # Check fieldnames
            fieldnames = list(rows[0].keys())
            assert set(fieldnames) == {"id", "name", "value"}

    def test_export_creates_parent_directories(self) -> None:
        """Test that export creates parent directories if they don't exist."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = CSVExporter()
            data = [{"id": 1, "name": "Test"}]
            output_path = Path(temp_dir) / "subdir1" / "subdir2" / "test.csv"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            assert output_path.parent.exists()

    def test_export_with_single_dict_creates_csv_with_one_row(self) -> None:
        """Test exporting single dictionary creates CSV with one row."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = CSVExporter()
            data = [{"metric": "test", "value": 42}]
            output_path = Path(temp_dir) / "test.csv"

            # Act
            exporter.export(data, output_path)

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1

    def test_export_overwrites_existing_file(self) -> None:
        """Test that export overwrites existing CSV file."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = CSVExporter()
            output_path = Path(temp_dir) / "test.csv"
            output_path.write_text("old,data\n1,2")
            data = [{"new": "data"}]

            # Act
            exporter.export(data, output_path)

            # Assert
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 1
            assert "new" in rows[0]
            assert "old" not in rows[0]

    def test_export_sorts_fieldnames_alphabetically(self) -> None:
        """Test that export sorts fieldnames alphabetically."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = CSVExporter()
            data = [{"z": 1, "a": 2, "m": 3}]
            output_path = Path(temp_dir) / "test.csv"

            # Act
            exporter.export(data, output_path)

            # Assert
            with output_path.open() as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
            assert fieldnames == ["a", "m", "z"]


class TestExporterRegistry:
    """Tests for ExporterRegistry class."""

    def test_default_exporters_includes_json(self) -> None:
        """Test registry includes JSON exporter by default."""
        # Arrange
        registry = ExporterRegistry()

        # Act & Assert
        assert "json" in registry.supported_formats

    def test_default_exporters_includes_csv(self) -> None:
        """Test registry includes CSV exporter by default."""
        # Arrange
        registry = ExporterRegistry()

        # Act & Assert
        assert "csv" in registry.supported_formats

    def test_get_exporter_with_json_returns_json_exporter(self) -> None:
        """Test getting JSON exporter returns JSONExporter instance."""
        # Arrange
        registry = ExporterRegistry()

        # Act
        exporter = registry.get_exporter("json")

        # Assert
        assert isinstance(exporter, JSONExporter)

    def test_get_exporter_with_csv_returns_csv_exporter(self) -> None:
        """Test getting CSV exporter returns CSVExporter instance."""
        # Arrange
        registry = ExporterRegistry()

        # Act
        exporter = registry.get_exporter("csv")

        # Assert
        assert isinstance(exporter, CSVExporter)

    def test_get_exporter_with_unknown_format_raises_value_error(self) -> None:
        """Test getting unknown format raises ValueError."""
        # Arrange
        registry = ExporterRegistry()

        # Act & Assert
        with pytest.raises(ValueError, match="Unknown export format: unknown"):
            registry.get_exporter("unknown")

    def test_register_exporter_adds_new_exporter(self) -> None:
        """Test registering new exporter adds it to supported formats."""
        # Arrange
        registry = ExporterRegistry()
        custom_exporter = Mock(spec=BaseExporter)

        # Act
        registry.register_exporter("custom", custom_exporter)

        # Assert
        assert "custom" in registry.supported_formats
        assert registry.get_exporter("custom") == custom_exporter

    def test_register_exporter_overwrites_existing_exporter(self) -> None:
        """Test registering exporter with existing name overwrites it."""
        # Arrange
        registry = ExporterRegistry()
        custom_exporter = Mock(spec=BaseExporter)

        # Act
        registry.register_exporter("json", custom_exporter)

        # Assert
        assert registry.get_exporter("json") == custom_exporter

    def test_supported_formats_returns_list_of_strings(self) -> None:
        """Test supported_formats property returns list of strings."""
        # Arrange
        registry = ExporterRegistry()

        # Act
        formats = registry.supported_formats

        # Assert
        assert isinstance(formats, list)
        assert all(isinstance(fmt, str) for fmt in formats)


class TestSimulationDataExporter:
    """Tests for SimulationDataExporter class."""

    def test_export_topology_with_json_format_creates_json_file(self) -> None:
        """Test exporting topology data to JSON format."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            topology = {
                "nodes": {"A": {"type": "CDC"}, "B": {"type": "CDC"}},
                "links": {"1": {"source": "A", "destination": "B"}},
            }
            output_path = Path(temp_dir) / "topology.json"

            # Act
            exporter.export_topology(topology, output_path, output_format="json")

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                loaded = json.load(f)
            assert loaded == topology

    def test_export_topology_with_string_path_converts_to_path(self) -> None:
        """Test exporting topology with string path converts to Path."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            topology = {"nodes": {"A": {"type": "CDC"}}}
            output_path_str = str(Path(temp_dir) / "topology.json")

            # Act
            exporter.export_topology(topology, output_path_str, output_format="json")

            # Assert
            assert Path(output_path_str).exists()

    def test_export_results_with_json_format_creates_json_file(self) -> None:
        """Test exporting results data to JSON format."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            results = {"blocking_probability": 0.05, "utilization": 0.8}
            output_path = Path(temp_dir) / "results.json"

            # Act
            exporter.export_results(results, output_path, output_format="json")

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                loaded = json.load(f)
            assert loaded == results

    def test_export_results_with_string_path_converts_to_path(self) -> None:
        """Test exporting results with string path converts to Path."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            results = {"metric": "value"}
            output_path_str = str(Path(temp_dir) / "results.json")

            # Act
            exporter.export_results(results, output_path_str, output_format="json")

            # Assert
            assert Path(output_path_str).exists()

    def test_export_metrics_with_csv_format_creates_csv_file(self) -> None:
        """Test exporting metrics data to CSV format."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            metrics = {"blocking_prob": 0.05, "utilization": 0.8}
            output_path = Path(temp_dir) / "metrics.csv"

            # Act
            exporter.export_metrics(metrics, output_path, output_format="csv")

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            metrics_dict = {row["metric"]: row["value"] for row in rows}
            assert metrics_dict["blocking_prob"] == "0.05"
            assert metrics_dict["utilization"] == "0.8"

    def test_export_metrics_with_dict_converts_to_list_for_csv(self) -> None:
        """Test exporting metrics dict converts to list of dicts for CSV."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            metrics = {"metric1": 1.0, "metric2": 2.0}
            output_path = Path(temp_dir) / "metrics.csv"

            # Act
            exporter.export_metrics(metrics, output_path, output_format="csv")

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2

    def test_export_metrics_with_list_of_dicts_exports_directly(self) -> None:
        """Test exporting metrics as list of dicts exports directly."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            metrics = [
                {"name": "metric1", "value": 1.0},
                {"name": "metric2", "value": 2.0},
            ]
            output_path = Path(temp_dir) / "metrics.csv"

            # Act
            exporter.export_metrics(metrics, output_path, output_format="csv")

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["name"] == "metric1"

    def test_export_metrics_with_json_format_exports_as_is(self) -> None:
        """Test exporting metrics to JSON format exports without conversion."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            metrics = {"blocking_prob": 0.05}
            output_path = Path(temp_dir) / "metrics.json"

            # Act
            exporter.export_metrics(metrics, output_path, output_format="json")

            # Assert
            assert output_path.exists()
            with output_path.open() as f:
                loaded = json.load(f)
            assert loaded == metrics

    def test_export_metrics_with_string_path_converts_to_path(self) -> None:
        """Test exporting metrics with string path converts to Path."""
        with TemporaryDirectory() as temp_dir:
            # Arrange
            exporter = SimulationDataExporter()
            metrics = [{"metric": "test", "value": 1}]
            output_path_str = str(Path(temp_dir) / "metrics.csv")

            # Act
            exporter.export_metrics(metrics, output_path_str, output_format="csv")

            # Assert
            assert Path(output_path_str).exists()

    def test_has_registry_attribute(self) -> None:
        """Test that SimulationDataExporter has registry attribute."""
        # Arrange & Act
        exporter = SimulationDataExporter()

        # Assert
        assert hasattr(exporter, "registry")
        assert isinstance(exporter.registry, ExporterRegistry)
