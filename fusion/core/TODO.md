# Core Module TODOs

## High Priority

### Code Architecture Migration
- **Issue**: Legacy files need migration to new module structure
- **Files**: `spectrum_assignment.py`, `snr_measurements.py`
- **Next Steps**: Complete migration to modules/spectrum and modules/snr, then remove deprecated files

### Request Dictionary Key Mapping
- **Issue**: Manual key mapping ('mod_formats' → 'modulation_formats_dict', 'req_id' → 'request_id')
- **Files**: `simulation.py:95,115`
- **Next Steps**: Standardize request dictionary structure at source

### SimStats Checkpoint Support
- **Issue**: Cannot resume from previously saved simulation states
- **File**: `metrics.py:25`
- **Next Steps**: Implement save/load checkpoint functionality

### Logging System Integration
- **Issue**: Replace print statements with proper logging
- **Files**: `metrics.py:469,483`
- **Next Steps**: Replace print with logger calls

## Medium Priority

### Multi-Core Configuration Support
- **Issue**: Cross-talk and SNR calculations limited to 7-core configurations  
- **Files**: `spectrum_assignment.py:216`, `snr_measurements.py:12,213`
- **Next Steps**: Generalize methods to support all core configurations (4, 13, 19 cores)

### Configuration System Integration
- **Issue**: Hardcoded values should be configurable
- **Files**: `simulation.py:137` (band list), `metrics.py:399` (CI percent)
- **Next Steps**: Move hardcoded values to configuration files

### Variable Naming Consistency
- **Issue**: Inconsistent variable names across codebase
- **Files**: Multiple files (`req_dict`, `mod_format`, etc.)
- **Next Steps**: Audit and standardize variable naming conventions

### Multi-band Support Enhancement
- **Issue**: Limited multi-band support in spectrum assignment and SNR measurements
- **Files**: `spectrum_assignment.py:54,30`, `snr_measurements.py:78`
- **Next Steps**: Implement comprehensive multi-band support

## Low Priority

### Code Quality Improvements
- **Issue**: Large methods, debug prints, and type annotations need improvement
- **Files**: `metrics.py` (various methods), `spectrum_assignment.py:333,238`
- **Next Steps**: Refactor large methods, clean up debug prints, improve type hints

### Flexigrid Implementation
- **Issue**: Flexigrid support needs development
- **File**: `spectrum_assignment.py:333`
- **Next Steps**: Research and implement flexigrid allocation algorithms

### Dynamic Slicing Issues
- **Issue**: Unclear code paths and ignored parameters
- **File**: `light_path_slicing.py:89-91`
- **Next Steps**: Fix parameter usage and add proper testing