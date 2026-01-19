# Pipelines

## Purpose

This module provides pipeline implementations for the FUSION simulation framework. Pipelines encapsulate complex multi-step operations like routing with protection, spectrum slicing, and disjoint path finding. They coordinate between lower-level components (routing algorithms, spectrum assigners) to implement higher-level network provisioning strategies.

## Key Components

### Core Files

- `routing_pipeline.py`: Protected routing pipeline for 1+1 protection scenarios
- `routing_strategies.py`: Pluggable routing strategy implementations (k-shortest, load-balanced, protection-aware)
- `slicing_pipeline.py`: Request slicing across multiple lightpaths for large bandwidth requests
- `protection_pipeline.py`: 1+1 dedicated path protection allocation
- `disjoint_path_finder.py`: Algorithms for finding link-disjoint and node-disjoint path pairs

### Test Files

- `tests/test_protection_pipeline.py`: Tests for protection pipeline and disjoint path finder
- `tests/test_routing_strategies.py`: Tests for routing strategy implementations

## Usage

### Basic Routing with Protection

```python
from fusion.pipelines import ProtectedRoutingPipeline

pipeline = ProtectedRoutingPipeline(config)
result = pipeline.find_routes("A", "Z", 100, network_state)

if result.has_protection:
    print(f"Working path: {result.best_path}")
    print(f"Backup path: {result.backup_paths[0]}")
```

### Request Slicing

```python
from fusion.pipelines.slicing_pipeline import StandardSlicingPipeline

pipeline = StandardSlicingPipeline(config)
result = pipeline.try_slice(
    request, path, "QPSK", 400, network_state,
    spectrum_pipeline=spectrum_pipeline,
)

if result.success:
    print(f"Sliced into {result.num_slices} lightpaths")
```

### Disjoint Path Finding

```python
from fusion.pipelines import DisjointPathFinder, DisjointnessType

finder = DisjointPathFinder(DisjointnessType.LINK)
paths = finder.find_disjoint_pair(topology, "A", "D")

if paths:
    primary, backup = paths
    print(f"Primary: {primary}, Backup: {backup}")
```

### Routing Strategies

```python
from fusion.pipelines import KShortestPathStrategy, LoadBalancedStrategy

# Basic k-shortest paths
ksp = KShortestPathStrategy(k=3)
result = ksp.select_routes("A", "Z", 100, network_state)

# Load-balanced routing
lbs = LoadBalancedStrategy(k=5, utilization_weight=0.5)
result = lbs.select_routes("A", "Z", 100, network_state)
```

## Dependencies

### Internal Dependencies

- `fusion.domain.config`: SimulationConfig for pipeline configuration
- `fusion.domain.network_state`: NetworkState for topology and spectrum access
- `fusion.domain.results`: RouteResult, SlicingResult for return types
- `fusion.interfaces.pipelines`: Pipeline protocols (SpectrumPipeline, SNRPipeline)

### External Dependencies

- `networkx`: Graph algorithms for path finding
- `numpy`: Array operations for spectrum availability

## Configuration

### Slicing Configuration

```python
config = SimulationConfig(
    max_slices=4,              # Maximum slices per request
    dynamic_lps=False,         # Enable GSNR-based dynamic slicing
    fixed_grid=False,          # Fixed vs flex grid mode
    can_partially_serve=False, # Accept partial bandwidth allocation
    mod_per_bw={...},          # Modulation formats per bandwidth tier
)
```

### Protection Configuration

```python
config = SimulationConfig(
    node_disjoint_protection=False,  # Require node-disjoint paths
    k_paths=3,                       # Number of candidate paths
)
```

## Testing

```bash
# Run pipeline tests
pytest fusion/pipelines/tests/

# Run with coverage
pytest --cov=fusion.pipelines fusion/pipelines/tests/
```

## API Reference

### Main Classes

- `ProtectedRoutingPipeline`: Routing pipeline with 1+1 protection support
- `StandardSlicingPipeline`: Request slicing across multiple lightpaths
- `ProtectionPipeline`: 1+1 dedicated path protection allocation
- `DisjointPathFinder`: Link-disjoint and node-disjoint path algorithms

### Routing Strategies

- `RoutingStrategy`: Protocol defining routing strategy interface
- `KShortestPathStrategy`: Basic k-shortest paths routing
- `LoadBalancedStrategy`: Routing considering link utilization
- `ProtectionAwareStrategy`: Routing for disjoint path pairs

### Data Classes

- `RouteConstraints`: Constraints for route selection (exclusions, limits)
- `ProtectedAllocationResult`: Result of protected spectrum allocation

## Notes

### Design Decisions

- Pipelines use the Strategy pattern for pluggable routing algorithms
- Protection uses 1+1 dedicated protection (same spectrum on both paths)
- Slicing supports both tier-based and dynamic (GSNR-based) allocation
- All pipelines are stateless - configuration passed at initialization

### Known Limitations

- See `TODO.md` for current issues and planned improvements
- Feasibility estimates in slicing use hardcoded values (see inline TODO)

### Performance Considerations

- K-shortest path computation is O(k * E * log V) per request
- Disjoint path finding uses NetworkX edge_disjoint_paths (Suurballe's algorithm)
- Slicing may require multiple spectrum searches per request
