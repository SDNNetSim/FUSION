# Routing Module Tests

## Test Coverage
- **utils.py**: RoutingHelpers class - NLI calculations, XT cost calculations, spectrum utilities
- **registry.py**: RoutingRegistry - algorithm registration, retrieval, and validation
- **k_shortest_path.py**: KShortestPath routing algorithm
- **congestion_aware.py**: CongestionAwareRouting algorithm
- **least_congested.py**: LeastCongestedRouting algorithm
- **fragmentation_aware.py**: FragmentationAwareRouting algorithm
- **nli_aware.py**: NLIAwareRouting algorithm
- **xt_aware.py**: XTAwareRouting algorithm

## Running Tests
```bash
# Run all routing module tests
pytest fusion/modules/routing/tests/

# Run with coverage
pytest --cov=fusion.modules.routing fusion/modules/routing/tests/

# Run specific test file
pytest fusion/modules/routing/tests/test_utils.py

# Run specific test
pytest fusion/modules/routing/tests/test_utils.py::TestRoutingHelpers::test_get_indexes_with_even_slots_needed
```

## Test Categories
- **Unit tests**: Test individual functions and methods in isolation with mocked dependencies
- **Algorithm tests**: Test routing algorithm implementations against various network scenarios
- **Registry tests**: Test algorithm registration and discovery functionality

## Test Data
- Uses pytest fixtures for common test data (topologies, network properties, etc.)
- Mocks external dependencies (NetworkX operations, file I/O, etc.)
- Parametrized tests for edge cases and boundary conditions
