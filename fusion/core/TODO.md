# Core Module TODOs

## High Priority

### Request Dictionary Key Mapping
- **Issue**: Manual key mapping ('mod_formats' → 'modulation_formats_dict', 'req_id' → 'request_id')
- **Files**: `simulation.py:95,115`
- **Solution**: Standardize request dictionary structure at source

### SimStats Checkpoint Support
- **Issue**: Cannot resume from previously saved simulation states
- **Files**: `metrics.py:25`
- **Solution**: Implement save/load checkpoint functionality

### Logging System Integration
- **Issue**: Replace print statements with proper logging
- **Files**: `metrics.py:469,483`
- **Solution**: Replace print with logger calls

## Medium Priority

### Break Up metrics.py
- **Issue**: File is 2048 lines with ~40 methods, well over 500-line guideline
- **Files**: `metrics.py`
- **Solution**: Extract into focused modules (e.g., `recovery_metrics.py`, `fragmentation_metrics.py`, `snapshot_metrics.py`)

### Reorganize simulation.py
- **Issue**: File is 2212 lines with mixed concerns (legacy path, orchestrator path, RL integration, failure handling)
- **Files**: `simulation.py`
- **Solution**: Extract into focused modules (e.g., `orchestrator_integration.py`, `rl_integration.py`, `failure_integration.py`)

### Reorganize snr_measurements.py
- **Issue**: File is 1668 lines, uses legacy Props pattern, currently wrapped by SNRAdapter
- **Files**: `snr_measurements.py`
- **Solution**: Break into focused modules (e.g., `snr_calculations.py`, `crosstalk.py`, `egn_model.py`) and migrate to pipeline pattern

### Multi-Core Configuration Support
- **Issue**: Cross-talk and SNR calculations limited to 7-core configurations
- **Files**: `spectrum_assignment.py:216`, `snr_measurements.py:12,213`
- **Solution**: Generalize methods to support all core configurations (4, 13, 19 cores)

### Configuration System Integration
- **Issue**: Hardcoded values should be configurable
- **Files**: `simulation.py:137` (band list), `metrics.py:399` (CI percent)
- **Solution**: Move hardcoded values to configuration files

### Multi-band Support Enhancement
- **Issue**: Limited multi-band support in spectrum assignment and SNR measurements
- **Files**: `spectrum_assignment.py:54,30`, `snr_measurements.py:78`
- **Solution**: Implement comprehensive multi-band support

### Replace Legacy Adapters
- **Issue**: Adapters are temporary migration layers wrapping legacy code
- **Files**: `adapters/routing_adapter.py`, `adapters/spectrum_adapter.py`, `adapters/snr_adapter.py`, `adapters/grooming_adapter.py`
- **Solution**: Replace adapters with clean pipeline implementations

## Low Priority

### Code Quality Improvements
- **Issue**: Large methods and type annotations need improvement
- **Files**: `metrics.py` (various methods), `spectrum_assignment.py:333,238`
- **Solution**: Refactor large methods, improve type hints

### Flexigrid Implementation
- **Issue**: Flexigrid support needs development
- **Files**: `spectrum_assignment.py:333`
- **Solution**: Research and implement flexigrid allocation algorithms
