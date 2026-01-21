# Pipelines TODOs

## High Priority

### Hardcoded Slicing Feasibility Estimates

- **Issue**: In `slicing_pipeline.py:try_slice()`, when `spectrum_pipeline` is None (feasibility check only), the method returns hardcoded estimates: `num_slices=2` and `slice_bandwidth_gbps=bandwidth_gbps // 2`.
- **Files**: `slicing_pipeline.py:204-210` (see detailed inline TODO comment)
- **Impact**: Callers relying on these estimates for capacity planning or blocking probability predictions get inaccurate results. The actual allocation uses tier-based slicing which may produce different slice counts and sizes (e.g., 3 slices of 500+300+200 Gbps).
- **Solution**: Implement `_estimate_slicing_result()` method that performs a "dry run" of tier-based allocation:
  - Use `mod_per_bw` config to estimate actual slice count and sizes
  - Consider path length and available modulations
  - Return realistic estimates matching `_try_allocate_tier_based()` behavior

## Medium Priority

### Consolidate Routing Strategies with modules/routing (v6.X)

- **Issue**: `routing_strategies.py` contains path computation logic that duplicates functionality in `fusion/modules/routing/` algorithms. This was done to quickly deliver the orchestrator architecture.
- **Files**: `routing_strategies.py`, `fusion/modules/routing/`
- **Impact**: Modifying routing behavior may require changes in both locations. Maintenance burden increases over time.
- **Solution**: Refactor routing strategies to wrap/delegate to the registered algorithms in `fusion/modules/routing/`:
  - `KShortestPathStrategy` should use `fusion.modules.routing.k_shortest_path`
  - `LoadBalancedStrategy` should use `fusion.modules.routing.congestion_aware` or similar
  - Eliminate duplicate NetworkX calls in strategies
  - Ensure both legacy and orchestrator paths use the same underlying algorithms

### Legacy Bug Compatibility in SNR Recheck

- **Issue**: When SNR recheck fails during slicing, the code releases spectrum but does NOT restore bandwidth tracking (marked with `TODO: (v6)` comments).
- **Files**: `slicing_pipeline.py:295-302`, `slicing_pipeline.py:476-484`
- **Impact**: Bandwidth accounting may become inconsistent after failed SNR rechecks during slicing.
- **Solution**: Decide whether to fix the bug (breaking legacy compatibility) or document it as expected behavior.

### Spectrum Pipeline Protocol Parameter

- **Issue**: `excluded_modulations` parameter was added to `SpectrumPipeline.find_spectrum()` protocol but may not be implemented in all adapters.
- **Files**: `fusion/interfaces/pipelines.py`, `slicing_pipeline.py:543`
- **Impact**: Dynamic flex-grid slicing may fail if adapter doesn't support excluded_modulations.
- **Solution**: Verify all SpectrumPipeline implementations support the parameter or make it truly optional.

## Low Priority

### Simplify Tier-Based vs Dynamic Slicing

- **Issue**: `_try_allocate_tier_based()` and `_try_allocate_dynamic_flex_grid()` share significant code.
- **Files**: `slicing_pipeline.py:215-405`, `slicing_pipeline.py:498-680`
- **Impact**: Code duplication makes maintenance harder.
- **Solution**: Extract common allocation loop into shared helper method.

### Add Slicing Metrics Collection

- **Issue**: Slicing operations don't emit detailed metrics about slice sizes, modulations used, or partial allocation rates.
- **Files**: `slicing_pipeline.py`
- **Impact**: Limited visibility into slicing behavior for analysis.
- **Solution**: Add optional metrics callback or integrate with existing metrics collection.
