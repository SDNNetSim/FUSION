# Failures Module TODOs

## Status: Beta

This module is currently in **beta** status. It has not been fully tested or used in production simulations yet. Use with caution and expect potential API changes.

## Critical: Orchestrator Integration Gap

### SDNOrchestrator Does Not Check Path Feasibility (v6.x)
- **Issue**: The orchestrator path does NOT check `is_path_feasible()` during routing, meaning new allocations may go through failed links
- **Current behavior**:
  - FailureManager is initialized and works (activation, repair, impact handling)
  - SDNController (legacy) checks `is_path_feasible()` before routing
  - SDNOrchestrator does NOT have FailureManager reference
  - New allocations can be made through failed links, then immediately impacted
- **Files**: `fusion/core/orchestrator.py`, `fusion/core/adapters/routing_adapter.py`
- **Workaround**: Use `use_orchestrator=False` for survivability experiments
- **Options to fix**:
  1. Pass FailureManager to SDNOrchestrator, check feasibility during routing
  2. Add `failed_links` to NetworkState, routing pipelines read from it
  3. Create a FailuresPipeline that filters infeasible paths
- **Target version**: v6.x (no decision made yet)

## High Priority

### Comprehensive Testing Required
- **Issue**: Module lacks comprehensive test coverage and real-world validation
- **Files**: All files in this module
- **Solution**: Add integration tests with actual simulation runs, validate failure injection and recovery behavior

### Documentation and Usage Examples
- **Issue**: Limited documentation on how to use failure scenarios in experiments
- **Files**: `failure_manager.py`, `failure_types.py`
- **Solution**: Add usage examples and integration guide with the simulation engine

## Medium Priority

### Validate F3/F4 Failure Types
- **Issue**: SRLG (F3) and Geographic (F4) failure types need validation
- **Files**: `failure_types.py`
- **Solution**: Test with real network topologies that have SRLG/geographic data

### Recovery Metrics Integration
- **Issue**: Ensure recovery metrics are properly collected during failure scenarios
- **Files**: `failure_manager.py`
- **Solution**: Validate integration with `fusion.core.metrics` recovery tracking

## Low Priority

### Performance Optimization
- **Issue**: Failure injection performance not benchmarked for large networks
- **Files**: `failure_manager.py`
- **Solution**: Profile and optimize for large-scale topologies

## Notes

- This module supports F1 (link), F2 (node), F3 (SRLG), and F4 (geographic) failure types
- **FailureManager** (this module) simulates failures; **ProtectionPipeline** provisions backup paths - they are complementary but not yet fully integrated
- Legacy path (`use_orchestrator=False`) has full failure support; orchestrator path has a gap (see Critical section above)
- Coordinate with `fusion.core.metrics` for recovery time tracking
