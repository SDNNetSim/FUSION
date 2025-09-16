# Core Module TODOs

This file tracks known issues and future improvements for the FUSION core modules.

## High Priority

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