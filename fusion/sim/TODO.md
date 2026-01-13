# Simulation Module TODOs

## High Priority

### Configuration Management for Hardcoded Paths
- **Issue**: Multiple files use hardcoded paths ('data', './evaluation_results')
- **Files**: `batch_runner.py:56`, `evaluate_pipeline.py:236`, `input_setup.py:94`
- **Impact**: Reduces portability and configurability
- **Solution**: Implement centralized configuration manager for all paths

### Type Annotation Consistency
- **Issue**: Some docstrings still use old type format (Dict vs dict)
- **Files**: `input_setup.py`, docstring inconsistencies across module
- **Impact**: Inconsistent documentation format
- **Solution**: Update all docstring type annotations to match function signatures

## Medium Priority

### Model Integration Placeholders
- **Issue**: Placeholder implementations for model loading and RL agents
- **Files**: `evaluate_pipeline.py:70`, `evaluate_pipeline.py:139`
- **Impact**: Limited evaluation functionality
- **Solution**: Implement actual model loading and RL agent integration

### Save Input Implementation
- **Issue**: Save input functionality not fully implemented
- **Files**: `batch_runner.py:71`
- **Impact**: Input files may not be saved when requested
- **Solution**: Implement proper save_input call mechanism

### Excel and Visualization Integration
- **Issue**: Excel export and visualization features are placeholders
- **Files**: `evaluate_pipeline.py:313`, `evaluate_pipeline.py:307`
- **Impact**: Limited reporting capabilities
- **Solution**: Integrate with Excel export and visualization modules

### RL Environment Integration
- **Issue**: RL evaluation uses placeholder implementation
- **Files**: `evaluate_pipeline.py:294`
- **Impact**: Cannot properly evaluate RL agents
- **Solution**: Integrate with actual RL environment

## Low Priority

### Multi-process Support
- **Issue**: Some utility functions only support single process
- **Files**: `utils.py:799`
- **Impact**: Limited scalability for certain operations
- **Solution**: Add multi-process support where applicable

### Algorithm Flexibility
- **Issue**: Hard-coded limitations (2 path levels, first fit core selection)
- **Files**: `utils.py:693`, `utils.py:658`
- **Impact**: Reduced algorithm flexibility
- **Solution**: Make algorithms more configurable

### Documentation References
- **Issue**: Missing academic references for formulations
- **Files**: `utils.py:632`
- **Impact**: Unclear mathematical basis
- **Solution**: Add proper academic references
