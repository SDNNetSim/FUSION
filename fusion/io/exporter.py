"""
Data export functionality for FUSION simulator.

This module provides unified data export capabilities supporting multiple formats
and export destinations.
"""

from pathlib import Path
from typing import Dict, Any, Union, List
import json
import csv
from abc import ABC, abstractmethod


class BaseExporter(ABC):
    """Abstract base class for data exporters."""

    def __str__(self) -> str:
        """String representation of exporter."""
        return self.__class__.__name__

    @abstractmethod
    def export(self, data: Any, output_path: Path) -> None:
        """Export data to the specified path."""


class JSONExporter(BaseExporter):
    """Export data to JSON format."""

    def __str__(self) -> str:
        """String representation of JSONExporter."""
        return "JSONExporter"

    def export(self, data: Any, output_path: Path) -> None:
        """Export data as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)


class CSVExporter(BaseExporter):
    """Export data to CSV format."""

    def __str__(self) -> str:
        """String representation of CSVExporter."""
        return "CSVExporter"

    def export(self, data: List[Dict], output_path: Path) -> None:
        """Export list of dictionaries as CSV."""
        if not data:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get all unique keys from all dictionaries
        fieldnames = set()
        for row in data:
            fieldnames.update(row.keys())
        fieldnames = sorted(fieldnames)

        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


class ExporterRegistry:
    """Registry for data exporters."""

    def __init__(self):
        self._exporters = {
            'json': JSONExporter(),
            'csv': CSVExporter(),
        }

    def register_exporter(self, format_name: str, exporter: BaseExporter) -> None:
        """Register a new exporter."""
        self._exporters[format_name] = exporter

    def get_exporter(self, format_name: str) -> BaseExporter:
        """Get an exporter by format name."""
        if format_name not in self._exporters:
            raise ValueError(f"Unknown export format: {format_name}")
        return self._exporters[format_name]

    @property
    def supported_formats(self) -> List[str]:
        """Get list of supported export formats."""
        return list(self._exporters.keys())


class SimulationDataExporter:
    """Main data export interface for simulation data."""

    def __init__(self):
        self.registry = ExporterRegistry()

    def export_topology(self, topology_data: Dict, output_path: Union[str, Path],
                        output_format: str = 'json') -> None:
        """Export network topology data."""
        output_path = Path(output_path)
        exporter = self.registry.get_exporter(output_format)
        exporter.export(topology_data, output_path)

    def export_results(self, results_data: Dict, output_path: Union[str, Path],
                       output_format: str = 'json') -> None:
        """Export simulation results data."""
        output_path = Path(output_path)
        exporter = self.registry.get_exporter(output_format)
        exporter.export(results_data, output_path)

    def export_metrics(self, metrics_data: Union[Dict, List[Dict]],
                       output_path: Union[str, Path], output_format: str = 'csv') -> None:
        """Export simulation metrics data."""
        output_path = Path(output_path)
        exporter = self.registry.get_exporter(output_format)

        # Convert dict to list of dicts for CSV export
        if output_format == 'csv' and isinstance(metrics_data, dict):
            metrics_data = [{'metric': k, 'value': v} for k, v in metrics_data.items()]

        exporter.export(metrics_data, output_path)
