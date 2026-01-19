# Analysis Module

## Purpose

Provides utilities for analyzing network topology, link utilization, and spectrum usage patterns during FUSION simulations. Used to identify bottlenecks, measure congestion, and gather network-wide statistics.

## Key Components

### Core Files
- `network_analysis.py`: `NetworkAnalyzer` class for link usage and utilization analysis
- `__init__.py`: Module exports

## Usage

### Basic Usage
```python
from fusion.analysis import NetworkAnalyzer

analyzer = NetworkAnalyzer()

# Get per-link usage statistics
link_usage = analyzer.get_link_usage_summary(network_spectrum)

# Calculate network-wide utilization
utilization_stats = analyzer.get_network_utilization_stats(network_spectrum)
```

### Advanced Usage
```python
from fusion.analysis import NetworkAnalyzer

analyzer = NetworkAnalyzer()

# Analyze congestion on specific paths
congestion = analyzer.analyze_network_congestion(
    network_spectrum,
    specific_paths=[(0, 1), (1, 2)]
)

# Find links above 80% utilization
bottlenecks = analyzer.identify_bottleneck_links(network_spectrum, threshold=0.8)
```

## Dependencies

### Internal Dependencies
- `fusion.utils.logging_config`: Logging utilities

### External Dependencies
- `numpy`: Array operations for spectrum matrix analysis

## Testing

Unit tests are located in `tests/` directory:
```bash
# Run module tests
pytest fusion/analysis/tests/

# Run with coverage
pytest --cov=fusion.analysis fusion/analysis/tests/
```

## API Reference

### NetworkAnalyzer

- `get_link_usage_summary(network_spectrum)`: Returns per-link usage count, throughput, and link number
- `analyze_network_congestion(network_spectrum, specific_paths=None)`: Calculates occupied/guard slots and active requests
- `get_network_utilization_stats(network_spectrum)`: Returns overall, average, max, min utilization statistics
- `identify_bottleneck_links(network_spectrum, threshold=0.8)`: Finds links exceeding utilization threshold

## Notes

### Design Decisions
- All methods are static since no instance state is required
- Bidirectional links are processed once using `min(src, dst)-max(src, dst)` key normalization
- `get_link_usage_summary` processes each direction separately to maintain directional statistics

### Known Limitations
- Methods currently require the legacy `network_spectrum` dict format (use `NetworkState.network_spectrum_dict` for compatibility with v5.5.0)
- Will be migrated to use `NetworkState` and `LinkSpectrum` directly (v6.1.0)
- No caching of analysis results between calls
