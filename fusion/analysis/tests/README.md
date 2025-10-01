# Analysis Module Tests

## Test Coverage
- **network_analysis.py**: Network topology analysis and link utilization metrics

## Running Tests
```bash
# Run all module tests
pytest fusion/analysis/tests/

# Run with coverage
pytest --cov=fusion.analysis fusion/analysis/tests/

# Run specific test file
pytest fusion/analysis/tests/test_network_analysis.py

# Run specific test class
pytest fusion/analysis/tests/test_network_analysis.py::TestGetLinkUsageSummary

# Run with verbose output
pytest -v fusion/analysis/tests/
```

## Test Data
Test data is generated programmatically using NumPy arrays to simulate network spectrum data structures.

## Test Categories
- **Unit tests**: Test individual methods of NetworkAnalyzer in isolation
  - `get_link_usage_summary`: Tests for link usage aggregation
  - `analyze_network_congestion`: Tests for congestion analysis
  - `get_network_utilization_stats`: Tests for utilization calculations
  - `identify_bottleneck_links`: Tests for bottleneck identification

All external dependencies (logger) are mocked to ensure true unit test isolation.