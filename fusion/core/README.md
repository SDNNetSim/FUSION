# FUSION Core Module

## Purpose
Core simulation engine and fundamental components for optical network simulations. Provides type-safe, production-ready simulation infrastructure with comprehensive error handling and modern Python practices.

This module supports two simulation architectures:
- **Legacy Engine**: The original `SimulationEngine` and `SDNController` for backward compatibility
- **Orchestrator (v6.0+)**: The new `SDNOrchestrator` with pipeline-based architecture for survivability experiments and policy-based routing

**Example**: *This module handles all core simulation logic including request processing, routing, spectrum assignment, SNR calculations, and metrics collection with full type safety and defensive programming.*

## Quick Start

### Legacy Simulation
```python
from fusion.core import SimulationEngine, SDNController
from fusion.core.properties import RoutingProps, SpectrumProps

# Initialize simulation engine
engine_props = {'max_iters': 10, 'num_requests': 1000}
sim_engine = SimulationEngine(engine_props)

# Run simulation
sim_engine.create_topology()
completed_iterations = sim_engine.run()
```

### Orchestrator-Based Simulation (v6.0+)
```python
from fusion.core import SDNOrchestrator, PipelineFactory
from fusion.domain.config import SimulationConfig

# Create configuration and orchestrator
config = SimulationConfig.from_engine_props(engine_props)
orchestrator = PipelineFactory.create_orchestrator(config)

# Handle requests through pipelines
result = orchestrator.handle_arrival(request, network_state)
```

### Metrics Collection
```python
from fusion.core import SimStats, MLMetricsCollector

# Initialize metrics collector
stats = SimStats(engine_props, sim_info='test_simulation')
stats.init_iter_stats()

# Collect ML training data (optional)
if engine_props.get('output_train_data'):
    ml_metrics = MLMetricsCollector(engine_props, sim_info)
```

## Architecture

```
core/
├── Legacy Engine
│   ├── simulation.py        # Main simulation orchestrator (legacy)
│   ├── sdn_controller.py    # SDN network controller (legacy)
│   └── request.py           # Request generation and management
├── Orchestrator (v6.0+)
│   ├── orchestrator.py      # SDNOrchestrator - thin coordination layer
│   ├── pipeline_factory.py  # PipelineFactory and PipelineSet
│   └── adapters/            # Legacy adapters for pipeline protocols
│       ├── routing_adapter.py
│       ├── spectrum_adapter.py
│       ├── snr_adapter.py
│       └── grooming_adapter.py
├── Network Components
│   ├── routing.py           # Path computation and routing
│   ├── spectrum_assignment.py # Spectrum allocation algorithms
│   ├── snr_measurements.py  # Signal quality calculations
│   └── grooming.py          # Traffic grooming for bandwidth packing
├── Data & Metrics
│   ├── properties.py        # Core data structures and properties
│   ├── metrics.py           # Statistics collection and analysis
│   ├── ml_metrics.py        # ML training data collection
│   └── persistence.py       # Data persistence and storage
└── Support
    ├── __init__.py          # Public API exports
    └── TODO.md              # Development roadmap
```

## Key Components

### SimulationEngine
Main simulation orchestrator that:
- Creates network topology from configuration
- Manages request generation and processing
- Coordinates routing and spectrum assignment
- Collects comprehensive performance metrics
- Supports multiprocessing and batch execution

### SDNController
Software-defined network controller that:
- Processes arrival and departure requests
- Manages spectrum allocation and deallocation
- Coordinates with routing and spectrum assignment modules
- Handles request lifecycle management
- Maintains network state consistency

### SDNOrchestrator (v6.0+)
Thin coordination layer for the new pipeline-based architecture:
- Routes requests through configurable pipelines without implementing algorithm logic
- Supports grooming, routing, spectrum assignment, SNR validation, and slicing stages
- Integrates with ControlPolicy for policy-driven path selection
- Handles protected (1+1) request allocation with disjoint paths
- Receives NetworkState per call (stateless design)
- Supports rollback on allocation failures

### PipelineFactory (v6.0+)
Factory for creating pipelines based on configuration:
- Stateless factory with static/class methods
- Config-driven pipeline selection
- Lazy imports to avoid circular dependencies
- Creates complete PipelineSet with routing, spectrum, grooming, SNR, and slicing pipelines

### Grooming
Traffic grooming handler for bandwidth packing:
- Grooms requests onto existing lightpaths
- Supports partial grooming with remaining bandwidth tracking
- Handles service release and bandwidth reclamation
- Groups lightpaths by physical path for optimal allocation

### Routing
Path computation module that:
- Supports multiple routing algorithms (k-shortest, NLI-aware, XT-aware)
- Provides modular algorithm architecture
- Maintains backward compatibility with legacy methods
- Integrates with external routing data sources
- Optimizes path selection based on various metrics

### SpectrumAssignment
Spectrum allocation engine that:
- Implements multiple allocation strategies (first-fit, best-fit, priority-based)
- Supports multi-core and multi-band spectrum management
- Handles cross-talk aware allocation
- Provides guard band management
- Integrates with SNR quality assessment

### SnrMeasurements
Signal quality assessment that:
- Calculates SNR, cross-talk, and non-linear interference
- Supports multiple core configurations (4, 7, 13, 19 cores)
- Integrates external SNR data sources
- Provides comprehensive noise modeling
- Validates signal quality thresholds

### SimStats & MLMetricsCollector
Comprehensive metrics collection that:
- Tracks blocking probabilities and resource utilization
- Calculates confidence intervals for statistical validity
- Supports snapshot-based performance monitoring
- Collects ML training data for model development
- Provides real-time performance reporting

## Usage Examples

### Example 1: Complete Simulation Setup
```python
from fusion.core import SimulationEngine, SimStats

# Configure simulation
engine_props = {
    'max_iters': 100,
    'num_requests': 5000,
    'network': 'NSFNet',
    'cores_per_link': 7,
    'allocation_method': 'first_fit',
    'route_method': 'k_shortest_path',
    'snr_type': 'xt_calculation'
}

# Initialize and run
engine = SimulationEngine(engine_props)
iterations_completed = engine.run()
print(f"Completed {iterations_completed} iterations")
```

### Example 2: Custom Routing and Spectrum Assignment
```python
from fusion.core import Routing, SpectrumAssignment
from fusion.core.properties import SDNProps, SpectrumProps

# Initialize properties
sdn_props = SDNProps()
spectrum_props = SpectrumProps()

# Configure request
sdn_props.source = '0'
sdn_props.destination = '5'
sdn_props.bandwidth = 100.0

# Find routes
routing = Routing(engine_props, sdn_props)
routing.get_route()

# Assign spectrum
spectrum_props.path_list = routing.route_props.paths_matrix[0]
spectrum_assignment = SpectrumAssignment(engine_props, sdn_props, routing.route_props)
spectrum_assignment.get_spectrum(['QPSK', '16-QAM'])
```

### Example 3: Metrics and Analysis
```python
from fusion.core import SimStats, StatsPersistence

# Initialize metrics collection
stats = SimStats(engine_props, 'simulation_001')

# Process requests and collect metrics
for request_data in request_stream:
    stats.iter_update(request_data, sdn_data, network_spectrum_dict)

# Calculate final statistics
stats.calculate_blocking_statistics()
stats.finalize_iteration_statistics()

# Check confidence interval
confidence_reached = stats.calculate_confidence_interval()
if confidence_reached:
    print("Statistically significant results achieved")

# Save results
persistence = StatsPersistence(engine_props, 'simulation_001')
persistence.save_stats(stats_dict, stats.stats_props, blocking_stats)
```

## Data Structures

### Core Properties Classes
All properties classes provide type-safe data structures with comprehensive validation:

- **RoutingProps**: Path computation and routing parameters
- **SpectrumProps**: Spectrum assignment and allocation state
- **SNRProps**: Signal quality measurement parameters
- **SDNProps**: Network controller state and request data
- **StatsProps**: Statistics collection and performance metrics

### Usage Pattern
```python
from fusion.core.properties import SpectrumProps

# Initialize with defaults
spectrum_props = SpectrumProps()

# Set required parameters
spectrum_props.path_list = [0, 1, 2, 5]
spectrum_props.slots_needed = 4
spectrum_props.modulation = 'QPSK'

# Access allocation results
if spectrum_props.is_free:
    print(f"Allocated slots {spectrum_props.start_slot}-{spectrum_props.end_slot}")
    print(f"Core: {spectrum_props.core_number}, Band: {spectrum_props.current_band}")
```

## Configuration

### Required Engine Properties
```python
engine_props = {
    # Simulation parameters
    'max_iters': 100,           # Simulation iterations
    'num_requests': 1000,       # Requests per iteration
    'erlang': 300,              # Traffic load in Erlangs
    'holding_time': 1.0,        # Request holding time

    # Network topology
    'network': 'NSFNet',        # Network topology name
    'cores_per_link': 7,        # Fiber cores per link
    'c_band': 320,              # C-band spectrum slots
    'bw_per_slot': 12.5,        # Bandwidth per slot (GHz)

    # Routing and spectrum
    'route_method': 'k_shortest_path',    # Routing algorithm
    'allocation_method': 'first_fit',     # Spectrum allocation
    'k_paths': 3,                         # Number of paths to consider

    # Signal quality
    'snr_type': 'xt_calculation',         # SNR calculation method
    'xt_noise': True,                     # Enable crosstalk modeling
    'input_power': 0.001,                 # Input power (Watts)

    # Output and debugging
    'print_step': 10,           # Progress reporting interval
    'save_snapshots': False,    # Enable snapshot collection
    'output_train_data': False  # Generate ML training data
}
```

### Optional Advanced Settings
```python
advanced_props = {
    # Multi-band support
    'band_list': ['c', 'l'],         # Enabled spectral bands
    'spectrum_priority': 'BSC',      # Band switching criteria

    # Cross-talk modeling
    'xt_type': 'with_length',        # Crosstalk calculation type
    'requested_xt': {'QPSK': -15.0}, # XT thresholds by modulation

    # Machine learning
    'deploy_model': False,           # Use ML model for routing
    'ml_model': 'decision_tree',     # ML model type

    # Performance optimization
    'thread_erlangs': True,          # Enable multithreading
    'fixed_grid': False,             # Use flexible grid
    'guard_slots': 1                 # Guard band slots
}
```

## Testing

Unit tests are comprehensive and focus on type safety:

```bash
# Run core module tests
pytest fusion/core/tests/

# Run with coverage reporting
pytest --cov=fusion.core fusion/core/tests/

# Run specific component tests
pytest fusion/core/tests/test_simulation.py
pytest fusion/core/tests/test_routing.py
pytest fusion/core/tests/test_spectrum_assignment.py

# Run orchestrator tests (v6.0+)
pytest fusion/core/tests/test_orchestrator.py
pytest fusion/core/tests/test_orchestrator_policy.py
pytest fusion/core/tests/test_pipeline_factory.py
```

## Error Handling

The core module provides robust error handling with:

### Type Safety
- **Comprehensive null checking**: All optional values validated before use
- **Type annotations**: Full mypy compliance with modern Python typing
- **Defensive programming**: Input validation at all public method boundaries
- **Clear error messages**: Descriptive exceptions with context and suggestions

### Common Error Patterns
```python
# Path list validation
if self.spectrum_props.path_list is None:
    raise ValueError("Path list must be initialized")

# Network state validation
if self.sdn_props.network_spectrum_dict is None:
    raise ValueError("Network spectrum dict must be initialized")

# Resource availability checking
if self.spectrum_props.slots_needed is None:
    raise ValueError("Slots needed cannot be None")
```

## Performance Considerations

### Optimization Features
- **Lazy loading**: Heavy modules loaded only when needed
- **Efficient data structures**: NumPy arrays for spectrum matrices
- **Caching**: Repeated calculations cached when possible
- **Memory management**: Proper resource cleanup and disposal
- **Parallel processing**: Multiprocessing support for large simulations

### Best Practices
```python
# Use appropriate data types
spectrum_matrix = np.zeros((cores, slots), dtype=np.int32)

# Validate inputs early
def process_request(self, request_data: dict[str, Any]) -> None:
    if not isinstance(request_data, dict):
        raise TypeError("Request data must be dictionary")

# Handle resource cleanup
try:
    # Simulation operations
    pass
finally:
    # Cleanup resources
    self.cleanup_resources()
```

## Dependencies

### Internal Dependencies
- `fusion.modules.routing`: Modular routing algorithm implementations
- `fusion.modules.spectrum`: Spectrum assignment algorithm implementations
- `fusion.modules.snr`: SNR calculation utilities and external data integration
- `fusion.utils.logging_config`: Standardized logging configuration
- `fusion.configs`: Configuration management and validation
- `fusion.domain`: Domain objects (Request, NetworkState, SimulationConfig) - v6.0+
- `fusion.interfaces`: Pipeline protocols and control policy interfaces - v6.0+
- `fusion.pipelines`: Pipeline implementations (slicing, protection, routing) - v6.0+
- `fusion.policies`: Policy implementations for path selection - v6.0+

### External Dependencies
- `numpy`: Numerical computations and spectrum matrices
- `networkx`: Network topology and graph operations
- `pandas`: Data analysis and ML metrics collection (optional)
- **Standard Library**: `json`, `pickle`, `copy`, `time`, `signal`, `itertools`
