# Interfaces TODOs

## High Priority

### Remove Legacy Abstract Base Classes
- **Issue**: Legacy ABC-based interfaces use outdated patterns incompatible with modern architecture
- **Files**:
  - `router.py`: `AbstractRoutingAlgorithm`
  - `spectrum.py`: `AbstractSpectrumAssigner`
  - `snr.py`: `AbstractSNRMeasurer`
  - `agent.py`: `AgentInterface`
- **Impact**: These classes use `engine_props: dict`, `SDNProps`, and mutable result storage instead of `NetworkState` and immutable result objects
- **Solution**: Migrate all implementations to Pipeline Protocols, then remove these files

### Remove Legacy Factory and SimulationPipeline
- **Issue**: `AlgorithmFactory` and `SimulationPipeline` depend on legacy ABCs
- **Files**: `factory.py`
- **Impact**: Factory creates instances of legacy ABCs and uses old property objects (`route_props`, `spectrum_props`)
- **Solution**: Replace with orchestrator pattern using Pipeline Protocols, then remove

## Medium Priority

### Update Module Exports After Legacy Removal
- **Issue**: `__init__.py` exports legacy classes that will be removed
- **Files**: `__init__.py`
- **Impact**: External code importing legacy classes will break
- **Solution**: After removing legacy code, update `__all__` to only export Pipeline Protocols and ControlPolicy

### Consolidate Test Files After Legacy Removal
- **Issue**: Test files for legacy ABCs will become obsolete
- **Files**:
  - `tests/test_router.py`
  - `tests/test_spectrum.py`
  - `tests/test_snr.py`
  - `tests/test_agent.py`
  - `tests/test_factory.py`
- **Impact**: Dead test code after legacy removal
- **Solution**: Remove test files for legacy code after migration complete

## Dependencies

### Migration Prerequisites
The following must be completed before legacy code can be removed:
1. All routing algorithms migrated to `RoutingPipeline` protocol
2. All spectrum algorithms migrated to `SpectrumPipeline` protocol
3. All SNR algorithms migrated to `SNRPipeline` protocol
4. RL module migrated to use `ControlPolicy` protocol
5. Orchestrator fully replaces `SimulationPipeline`

### External Dependencies
- `fusion/modules/routing/` - Must not depend on `AbstractRoutingAlgorithm`
- `fusion/modules/spectrum/` - Must not depend on `AbstractSpectrumAssigner`
- `fusion/modules/snr/` - Must not depend on `AbstractSNRMeasurer`
- `fusion/modules/rl/` - Must not depend on `AgentInterface`
- `fusion/core/` - Must not depend on `AlgorithmFactory` or `SimulationPipeline`
