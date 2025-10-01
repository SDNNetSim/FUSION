# Reporting Module Tests

## Test Coverage

- **simulation_reporter.py**: Simulation output formatting and reporting functionality

## Running Tests

```bash
# Run all reporting module tests
pytest fusion/reporting/tests/

# Run specific test file
pytest fusion/reporting/tests/test_simulation_reporter.py

# Run with coverage
pytest --cov=fusion.reporting fusion/reporting/tests/

# Run specific test class
pytest fusion/reporting/tests/test_simulation_reporter.py::TestReportIterationStats

# Run specific test method
pytest fusion/reporting/tests/test_simulation_reporter.py::TestReportIterationStats::test_report_iteration_stats_with_valid_data_logs_correctly
```

## Test Organization

Tests are organized by class method with clear naming patterns:
- `TestSimulationReporterInit`: Initialization tests
- `TestReportIterationStats`: Iteration statistics reporting tests
- `TestReportSimulationStart`: Simulation start reporting tests
- `TestReportSimulationComplete`: Simulation completion reporting tests
- `TestReportBlockingStatistics`: Blocking statistics reporting tests
- `TestReportSaveLocation`: Save location reporting tests
- `TestReportError`: Error reporting tests
- `TestReportWarning`: Warning reporting tests
- `TestCreateSummaryReport`: Summary report generation tests

## Test Categories

- **Unit tests**: All tests are unit tests with mocked dependencies
- **Logger mocking**: All logging operations are mocked to test output without side effects
- **Edge cases**: Tests cover empty inputs, zero values, None values, and boundary conditions

## Coverage Details

The test suite provides comprehensive coverage including:

### Initialization
- Default logger creation
- Custom logger usage
- Verbose flag configuration

### Iteration Stats Reporting
- Valid data logging
- Empty blocking lists
- Print flag control
- Verbose vs non-verbose modes
- Iteration number formatting (0-based to 1-based)

### Simulation Lifecycle
- Start reporting with various simulation info
- Completion with/without confidence intervals

### Blocking Statistics
- Zero request handling
- Probability calculations
- Bit rate calculations with edge cases
- Blocking reasons filtering

### Error and Warning Handling
- Error reporting with/without exceptions
- Various exception types
- Warning logging

### Summary Reports
- Complete statistics formatting
- Missing statistics (N/A handling)
- Partial statistics
- Report structure validation
