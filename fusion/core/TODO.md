# Core Module TODOs

This file tracks known issues and future improvements for the FUSION core modules.

## High Priority

### Code Architecture Migration
- **Issue**: Legacy files need migration to new module structure
- **Files**: `fusion/core/spectrum_assignment.py:11`, `fusion/core/snr_measurements.py:11`
- **Description**: Both spectrum assignment and SNR measurement files are marked for migration/deletion to modules/spectrum and modules/snr
- **Impact**: Code organization and maintainability
- **Next Steps**:
  1. Complete migration to modules/spectrum scripts
  2. Complete migration to modules/snr scripts
  3. Remove deprecated files after migration validation

### Request Dictionary Structure Issues
- **Issue**: Request dictionary key mapping requires manual conversion
- **Files**: `fusion/core/simulation.py:95`, `fusion/core/simulation.py:115`
- **Description**: Manual key mapping from 'mod_formats' to 'modulation_formats_dict' and 'req_id' to 'request_id'
- **Impact**: Code duplication and maintenance overhead
- **Next Steps**:
  1. Standardize request dictionary structure at source
  2. Update request generation to use proper keys
  3. Remove manual mapping code

### SimStats Class Enhancements (metrics.py)
- **Issue**: Implement ability to pick up from previously run simulations
- **File**: `fusion/core/metrics.py:25`
- **Description**: The SimStats class currently cannot resume from previously saved simulation states
- **Impact**: Forces complete restart of long-running simulations
- **Next Steps**: 
  1. Design serialization format for simulation state
  2. Implement save/load checkpoint functionality
  3. Add validation for checkpoint compatibility

### Logging System Integration
- **Issue**: Replace print statements with proper logging module
- **Files**: `fusion/core/metrics.py:469`, `fusion/core/metrics.py:483`
- **Description**: Direct print statements are used instead of the logging module
- **Impact**: Poor log management and debugging capabilities
- **Next Steps**:
  1. Import logging module
  2. Replace print statements with appropriate log levels
  3. Configure logging handlers for different output targets

## Medium Priority

### DRL Path Agents Configuration
- **Issue**: Band list configuration should be externalized
- **File**: `fusion/core/simulation.py:137`
- **Description**: Band list for DRL path agents is hardcoded and should be moved to configuration
- **Impact**: Reduces flexibility for different network configurations
- **Next Steps**:
  1. Move band list to configuration file or arguments script
  2. Update simulation engine to read from external configuration
  3. Add validation for band list configuration

### Variable Naming Consistency
- **Issue**: Band variable naming is inconsistent across codebase
- **File**: `fusion/core/simulation.py:152`
- **Description**: Variable names for bands change and are not consistent throughout the codebase
- **Impact**: Code readability and maintainability
- **Next Steps**:
  1. Audit all band-related variable names
  2. Establish consistent naming convention
  3. Update all references to use standard naming

### Spectrum Assignment Enhancements
- **Issue**: Multi-band support missing and methods need testing
- **Files**: `fusion/core/spectrum_assignment.py:54`, `fusion/core/spectrum_assignment.py:30`
- **Description**: Best-fit allocation lacks testing and multi-band support not implemented
- **Impact**: Limited spectrum allocation capabilities
- **Next Steps**:
  1. Implement multi-band support for spectrum assignment
  2. Add comprehensive testing for best-fit allocation
  3. Validate allocation algorithms across different scenarios

### Cross-talk Aware Allocation Limitation
- **Issue**: Cross-talk aware allocation only works for 7-core configurations
- **File**: `fusion/core/spectrum_assignment.py:216`
- **Description**: The handle_crosstalk_aware_allocation method has hardcoded limitations for seven-core fiber configurations
- **Impact**: Limited support for different core configurations (4, 13, 19 cores)
- **Next Steps**:
  1. Analyze code dependencies on seven-core assumption
  2. Generalize method to support all core configurations  
  3. Test with different core counts
  4. Update algorithm to dynamically determine core adjacency patterns

### SNR Measurement System Updates
- **Issue**: SNR measurement system needs variable renaming and feature additions
- **Files**: `fusion/core/snr_measurements.py:448`, `fusion/core/snr_measurements.py:451`, `fusion/core/snr_measurements.py:454`
- **Description**: Variables need renaming (to 'snr', 'xt') and external load check integration needed
- **Impact**: Code clarity and measurement accuracy
- **Next Steps**:
  1. Rename variables to follow naming conventions
  2. Integrate external load check with SNR calculations
  3. Update documentation for new variable names

### SNR Core Support Limitation
- **Issue**: SnrMeasurements class only works for seven cores
- **File**: `fusion/core/snr_measurements.py:12`
- **Description**: The class has a hardcoded limitation for seven-core fiber configurations
- **Impact**: Limited support for different core configurations (4, 13, 19 cores)
- **Next Steps**:
  1. Analyze code dependencies on seven-core assumption
  2. Generalize methods to support all core configurations
  3. Test with different core counts

### Adjacent Core Calculation Issue
- **Issue**: Number of adjacent cores hardcoded to negative 100
- **File**: `fusion/core/snr_measurements.py:213`
- **Description**: In _calculate_pxt method, num_adjacent is set to -100 which appears to be a placeholder value
- **Impact**: Incorrect cross-talk calculations when xt_noise is enabled
- **Next Steps**:
  1. Determine correct method to calculate number of adjacent cores
  2. Replace hardcoded value with proper calculation
  3. Validate cross-talk calculations with correct adjacent core count

### Simulation Engine Integration
- **Issue**: Integration with batch runner system incomplete
- **File**: `fusion/core/simulation.py:25`
- **Description**: Simulation engine needs integration with sim/batch_runner system
- **Impact**: Limited batch processing capabilities
- **Next Steps**:
  1. Design integration interface with batch runner
  2. Implement communication protocols
  3. Test batch processing functionality

### Configuration System Integration
- **Issue**: Add CI percent to configuration file
- **File**: `fusion/core/metrics.py:399`
- **Description**: Confidence interval percentage is hardcoded (5%) instead of being configurable
- **Impact**: Limited flexibility in simulation stopping criteria
- **Next Steps**:
  1. Add ci_percent parameter to engine configuration
  2. Update get_conf_inter method to use configurable value
  3. Add validation for ci_percent range (0-100)

### Code Quality Improvements
- **Issue**: Statistical calculation edge case handling
- **File**: `fusion/core/metrics.py:328`
- **Description**: Unclear comment about list length equal to one scenario
- **Impact**: Potential statistical calculation issues
- **Next Steps**:
  1. Research statistical validity for single-element lists
  2. Add proper handling or documentation for this edge case
  3. Consider alternative statistical approaches

## Low Priority

### Flexigrid Implementation
- **Issue**: Flexigrid functionality needs development
- **File**: `fusion/core/spectrum_assignment.py:333`
- **Description**: Flexigrid support needs to be developed for spectrum assignment
- **Impact**: Limited flexibility in spectrum allocation strategies
- **Next Steps**:
  1. Research flexigrid requirements and specifications
  2. Design flexigrid allocation algorithms
  3. Implement and test flexigrid functionality

### Spectrum Assignment Design Decisions
- **Issue**: Unclear handling strategy for certain allocation scenarios
- **File**: `fusion/core/spectrum_assignment.py:238`
- **Description**: Need to determine proper handling approach for specific allocation cases
- **Impact**: Potential allocation inefficiencies
- **Next Steps**:
  1. Analyze allocation scenarios requiring special handling
  2. Design consistent allocation strategy
  3. Document decision rationale

### SNR Measurement Band Support
- **Issue**: SNR measurements only work for C-band
- **File**: `fusion/core/snr_measurements.py:78`
- **Description**: Current SNR calculations are limited to C-band operations
- **Impact**: Reduced multi-band network support
- **Next Steps**:
  1. Extend SNR calculations to support all bands
  2. Update measurement algorithms for multi-band scenarios
  3. Test SNR accuracy across different bands

### Debug Statement Cleanup
- **Issue**: Debug print statements in production code
- **File**: `fusion/core/metrics.py:355-362`
- **Description**: Debug prints for normalization process should be removed or converted to debug logging
- **Impact**: Cluttered output in production runs
- **Next Steps**:
  1. Convert debug prints to logger.debug() calls
  2. Ensure debug logging can be controlled via configuration
  3. Clean up production output

### Method Size Reduction
- **Issue**: Several methods exceed 50-line guideline
- **Files**: 
  - `save_stats()`: 58 lines
  - `iter_update()`: 33 lines  
  - `_get_iter_means()`: 38 lines
- **Description**: Large methods reduce readability and maintainability
- **Impact**: Code is harder to understand and test
- **Next Steps**:
  1. Extract helper methods for complex logic
  2. Split large methods into focused functions
  3. Maintain single responsibility principle

### Type Annotation Improvements
- **Issue**: Generic object type annotations
- **File**: `fusion/core/metrics.py` (various methods)
- **Description**: Parameters like `sdn_data: object` should have specific type hints
- **Impact**: Reduced IDE support and type safety
- **Next Steps**:
  1. Define proper SDN data type or protocol
  2. Update type annotations throughout the class
  3. Add return type annotations where missing

## High Priority

### Variable Naming Consistency Check
- **Issue**: Review remaining variable names for consistency with coding standards
- **Files**: Multiple files across codebase  
- **Description**: Some variables like `req_dict`, `mod_format`, `route_matrix` are used across many files but don't follow naming conventions
- **Impact**: Inconsistent naming reduces code readability
- **Next Steps**:
  1. Audit usage across all 36+ files that use `mod_format`
  2. Create migration plan for high-usage variables
  3. Update tests and configuration files
  4. Consider backward compatibility for external APIs

### Dynamic Slicing Implementation Issues
- **Issue**: Dynamic slicing has unclear code paths and ignored parameters
- **File**: `fusion/modules/spectrum/light_path_slicing.py:89-91`
- **Description**: TODO comments indicate mod_format_list is ignored and path_len is unused
- **Impact**: Potential bugs in dynamic slicing allocation
- **Next Steps**:
  1. Research intended behavior for dynamic slicing
  2. Fix parameter usage or remove unused parameters
  3. Add proper testing for dynamic slicing scenarios

## Medium Priority

### Route Matrix Forced Modulation Inconsistency  
- **Issue**: Inconsistency when forcing modulation format with route matrix
- **File**: `fusion/core/sdn_controller.py:221`
- **Description**: Code comment indicates design issue with DRL path agents
- **Impact**: May affect ML model integration
- **Next Steps**:
  1. Review DRL path agent requirements
  2. Design consistent interface for forced parameters
  3. Update related documentation

---

## Contributing to TODOs

When adding new TODO items:

1. **Categorize by Priority**: High (blocking), Medium (enhances functionality), Low (nice to have)
2. **Provide Context**: Why is this needed? What's the impact?
3. **Define Next Steps**: Clear actionable items
4. **Reference Files**: List affected files with line numbers
5. **Estimate Effort**: Add complexity estimates when possible

When completing TODO items:
1. Move to "Completed Items" section with ✅ 
2. Add brief description of solution implemented
3. Update relevant documentation
4. Remove associated TODO comments from code