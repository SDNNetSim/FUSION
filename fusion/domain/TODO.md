# Domain Module TODOs

## High Priority

### Hardcoded Spectrum Slots Default
- **Issue**: NetworkState silently defaults to 320 slots when band not found in config
- **Files**: `network_state.py:85`
- **Solution**: Raise explicit error or require band_slots config for all bands used

### Move Spectrum Algorithms Out of NetworkState
- **Issue**: `find_first_fit` and `find_last_fit` are allocation algorithms in a state container
- **Files**: `network_state.py:250-290`
- **Solution**: Move to spectrum assignment module; NetworkState should only manage state, not implement strategies

## Medium Priority (v6.1)

### Remove Legacy Adapter Methods
- **Issue**: Legacy conversion methods add complexity and maintenance burden
- **Files**:
  - `config.py:150-220` - `from_engine_props`, `to_engine_props`
  - `lightpath.py:344-507` - `from_legacy_dict`, `to_legacy_dict`, `to_legacy_key`
  - `network_state.py:400-480` - Legacy compatibility properties and methods
  - `request.py:180-250` - `from_request_info`, `to_request_info`
  - `results.py` - Various `from_*` and `to_*` methods
- **Solution**: Complete migration to domain objects, then remove legacy adapters

### Protection Features Beta Status
- **Issue**: 1+1 protection in Lightpath is not heavily tested
- **Files**: `lightpath.py:104-117,296-338`
- **Solution**: Add comprehensive test coverage for protection switching and backup path management

### Remove Legacy Compatibility Properties in NetworkState
- **Issue**: Properties like `network_spectrum_dict` exist only for backward compatibility
- **Files**: `network_state.py:350-380`
- **Solution**: Update all consumers to use new API, then remove compatibility layer

## Low Priority

### Consistent Error Messages
- **Issue**: Some validation errors could provide more context
- **Files**: Various `__post_init__` methods
- **Solution**: Include expected ranges, current values, and suggestions in error messages

### Time-Bandwidth Usage Tracking
- **Issue**: `time_bw_usage` dict in Lightpath may grow unbounded in long simulations
- **Files**: `lightpath.py:90`
- **Solution**: Add optional pruning or windowed tracking for long-running simulations

### Request Status State Machine
- **Issue**: Status transitions not formally validated
- **Files**: `request.py`
- **Solution**: Add state machine validation to prevent invalid transitions (e.g., BLOCKED -> ALLOCATED)
