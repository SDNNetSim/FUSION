# Analysis Module

## Purpose
Provides utilities for analyzing network topology, performance metrics, and simulation results in FUSION.

## Key Components
- `network_analysis.py`: Network topology and link utilization analysis
- `__init__.py`: Module exports

## Usage
```python
from fusion.analysis import NetworkAnalyzer

analyzer = NetworkAnalyzer()
link_usage = analyzer.get_link_usage_summary(network_spectrum)
congestion_stats = analyzer.analyze_network_congestion(network_spectrum)
utilization_stats = analyzer.get_network_utilization_stats(network_spectrum)
bottlenecks = analyzer.identify_bottleneck_links(network_spectrum, threshold=0.8)
```

## Dependencies
- numpy: For numerical computations
- fusion.utils.logging_config: For logging functionality
