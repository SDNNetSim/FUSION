"""Tests for the io.exporter module."""

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fusion.io.exporter import (CSVExporter, ExporterRegistry, JSONExporter,
                                SimulationDataExporter)


class TestJSONExporter(unittest.TestCase):
    """Test JSONExporter class."""

    def test_export_dict(self):
        """Test exporting dictionary to JSON."""
        with TemporaryDirectory() as temp_dir:
            exporter = JSONExporter()
            data = {"key": "value", "number": 42}
            output_path = Path(temp_dir) / "test.json"

            exporter.export(data, output_path)

            self.assertTrue(output_path.exists())
            with output_path.open() as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)

    def test_export_list(self):
        """Test exporting list to JSON."""
        with TemporaryDirectory() as temp_dir:
            exporter = JSONExporter()
            data = [{"id": 1}, {"id": 2}]
            output_path = Path(temp_dir) / "test.json"

            exporter.export(data, output_path)

            self.assertTrue(output_path.exists())
            with output_path.open() as f:
                loaded = json.load(f)
            self.assertEqual(loaded, data)


class TestCSVExporter(unittest.TestCase):
    """Test CSVExporter class."""

    def test_export_list_of_dicts(self):
        """Test exporting list of dictionaries to CSV."""
        with TemporaryDirectory() as temp_dir:
            exporter = CSVExporter()
            data = [{"id": 1, "name": "Test1"}, {"id": 2, "name": "Test2"}]
            output_path = Path(temp_dir) / "test.csv"

            exporter.export(data, output_path)

            self.assertTrue(output_path.exists())
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["id"], "1")
            self.assertEqual(rows[0]["name"], "Test1")

    def test_export_empty_list(self):
        """Test exporting empty list."""
        with TemporaryDirectory() as temp_dir:
            exporter = CSVExporter()
            data = []
            output_path = Path(temp_dir) / "test.csv"

            exporter.export(data, output_path)
            # Should not create file for empty data
            self.assertFalse(output_path.exists())


class TestExporterRegistry(unittest.TestCase):
    """Test ExporterRegistry class."""

    def test_default_exporters(self):
        """Test registry has default exporters."""
        registry = ExporterRegistry()
        self.assertIn("json", registry.supported_formats)
        self.assertIn("csv", registry.supported_formats)

    def test_get_exporter(self):
        """Test getting exporters by format name."""
        registry = ExporterRegistry()
        json_exporter = registry.get_exporter("json")
        self.assertIsInstance(json_exporter, JSONExporter)

        csv_exporter = registry.get_exporter("csv")
        self.assertIsInstance(csv_exporter, CSVExporter)

    def test_unknown_format(self):
        """Test error for unknown format."""
        registry = ExporterRegistry()
        with self.assertRaises(ValueError):
            registry.get_exporter("unknown")


class TestSimulationDataExporter(unittest.TestCase):
    """Test SimulationDataExporter class."""

    def test_export_topology(self):
        """Test exporting topology data."""
        with TemporaryDirectory() as temp_dir:
            exporter = SimulationDataExporter()
            topology = {
                "nodes": {"A": {"type": "CDC"}, "B": {"type": "CDC"}},
                "links": {"1": {"source": "A", "destination": "B"}},
            }
            output_path = Path(temp_dir) / "topology.json"

            exporter.export_topology(topology, output_path)

            self.assertTrue(output_path.exists())
            with output_path.open() as f:
                loaded = json.load(f)
            self.assertEqual(loaded, topology)

    def test_export_metrics_as_csv(self):
        """Test exporting metrics to CSV."""
        with TemporaryDirectory() as temp_dir:
            exporter = SimulationDataExporter()
            metrics = {"blocking_prob": 0.05, "utilization": 0.8}
            output_path = Path(temp_dir) / "metrics.csv"

            exporter.export_metrics(metrics, output_path, output_format="csv")

            self.assertTrue(output_path.exists())
            with output_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 2)
            # Check data was converted properly
            metrics_dict = {row["metric"]: row["value"] for row in rows}
            self.assertEqual(metrics_dict["blocking_prob"], "0.05")


if __name__ == "__main__":
    unittest.main()
