# I/O Module

## Purpose
This module provides comprehensive data input/output functionality for the FUSION optical network simulator. It handles data generation, structuring, and export operations across multiple formats, enabling seamless data flow between simulation components and external systems.

## Key Components

### Core Files
- `exporter.py`: Multi-format data export system with registry pattern for extensibility
- `generate.py`: Physical topology and bandwidth information generation utilities
- `structure.py`: Network topology structuring and file parsing functionality

## Usage

### Basic Usage
```python
from fusion.io import SimulationDataExporter, create_network, create_pt

# Export simulation results
exporter = SimulationDataExporter()
exporter.export_results(results_data, "output/results.json")
exporter.export_metrics(metrics_data, "output/metrics.csv", "csv")

# Create network topology
network_dict, core_nodes = create_network("NSFNet")

# Generate physical topology
topology_dict = create_pt(cores_per_link=12, network_spectrum_dict=network_dict)
```

### Advanced Usage
```python
# Custom export formats
from fusion.io import ExporterRegistry
from fusion.io.exporter import BaseExporter

class CustomExporter(BaseExporter):
    def export(self, data, output_path):
        # Custom export logic
        pass

registry = ExporterRegistry()
registry.register_exporter("custom", CustomExporter())

# Network with custom base path and constant weights
network_dict, core_nodes = create_network(
    net_name="USNet",
    base_fp="/custom/topology/path",
    const_weight=True,
    is_only_core_node=False
)
```

## Dependencies

### Internal Dependencies
- `fusion.utils.os`: Project root finding utilities for path resolution
- Standard library: `json`, `csv`, `pathlib`, `typing`, `abc`

### External Dependencies
- `pathlib`: Modern path handling (Python standard library)
- `json`: JSON data serialization (Python standard library)
- `csv`: CSV file operations (Python standard library)

## Configuration

### File Paths
The module automatically resolves paths using the project root:
```python
# Default modulation formats file
DEFAULT_MOD_PATH = "data/json_input/run_mods/mod_formats.json"

# Default network topology files
DEFAULT_TOPOLOGY_PATH = "data/raw/"
```

### Supported Network Types
- `NSFNet`: NSF network topology
- `Pan-European`: European network topology
- `USNet`: US network topology
- `USbackbone60`: 60-node US backbone
- `Spainbackbone30`: 30-node Spain backbone
- `geant`: GEANT network topology
- `toy_network`: Simple test network
- `metro_net`: Metropolitan network
- `dt_network`: Deutsche Telekom network

## API Reference

### Main Classes
- `SimulationDataExporter`: Primary interface for exporting simulation data
- `ExporterRegistry`: Registry for managing export format handlers
- `BaseExporter`: Abstract base class for custom exporters (import from `fusion.io.exporter`)

### Key Functions
- `create_network()`: Build network structure from topology files
- `create_pt()`: Generate physical topology information
- `create_bw_info()`: Create bandwidth and modulation format mappings
- `assign_link_lengths()`: Assign lengths to network links
- `assign_core_nodes()`: Identify core nodes in topology

## Examples

### Example 1: Complete Simulation Data Export
```python
from fusion.io import SimulationDataExporter

exporter = SimulationDataExporter()

# Export different data types
topology_data = {"nodes": {...}, "links": {...}}
results_data = {"blocking_probability": 0.05, "utilization": 0.78}
metrics_data = [
    {"metric": "blocking", "value": 0.05},
    {"metric": "utilization", "value": 0.78}
]

exporter.export_topology(topology_data, "topology.json")
exporter.export_results(results_data, "results.json")
exporter.export_metrics(metrics_data, "metrics.csv")
```

### Example 2: Network Generation Pipeline
```python
from fusion.io import create_network, create_pt, create_bw_info

# Generate complete network configuration
network_dict, core_nodes = create_network("NSFNet")
physical_topology = create_pt(cores_per_link=7, network_spectrum_dict=network_dict)
bandwidth_info = create_bw_info("realistic")

# Use in simulation
simulation_config = {
    "network": network_dict,
    "core_nodes": core_nodes,
    "physical_topology": physical_topology,
    "bandwidth_formats": bandwidth_info
}
```

## Notes

### Design Decisions
- Registry pattern for exporters enables easy addition of new formats
- Path resolution uses project root detection for environment independence
- Type annotations follow modern Python 3.10+ union syntax
- Separate concerns: generation, structuring, and export are distinct modules

### Known Limitations
- Network topology files must follow tab-separated format
- Core node files currently only supported for USbackbone60 network
- Export formats limited to JSON and CSV (extensible via registry)

### Performance Considerations
- Large network topologies are loaded into memory entirely
- JSON export uses indent=2 for readability (impacts file size)
- CSV export builds complete fieldname set before writing (memory overhead)
