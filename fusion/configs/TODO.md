# Configuration System TODOs

This file tracks known issues and future improvements for the FUSION configuration system.

## High Priority

### DRL Path Agents Configuration
- **Issue**: `path_levels` parameter is currently hard-coded at the moment
- **Files**: `runtime_config.ini`, `xtar_example_config.ini`, `cross_platform.ini`
- **Description**: The path_levels parameter in RL settings is hard-coded and needs to be made configurable based on network topology or algorithm requirements
- **Impact**: Limits flexibility in deep reinforcement learning experiments
- **Next Steps**: 
  1. Analyze optimal path_levels values for different topologies
  2. Implement dynamic calculation based on network characteristics
  3. Add validation for path_levels range

### Unity Integration
- **Issue**: `save_step` parameter needs coordination with Unity visualization
- **Files**: `runtime_config.ini`, `cross_platform.ini`
- **Description**: The save_step parameter for simulation output needs to be synchronized with Unity visualization system
- **Impact**: Potential mismatch between simulation output frequency and visualization requirements
- **Next Steps**:
  1. Define Unity's expected data update frequency
  2. Ensure save_step aligns with Unity's requirements
  3. Add validation for compatible save_step values

## Medium Priority

### Template System Enhancements
- **Issue**: Add more specialized templates for specific research areas
- **Proposed Templates**:
  - `machine_learning.ini`: Optimized for ML training with larger datasets
  - `cross_talk_research.ini`: Enhanced cross-talk modeling parameters
  - `high_performance.ini`: Maximum performance configuration for HPC clusters
  - `debugging.ini`: Minimal setup with verbose logging for development
- **Next Steps**:
  1. Gather requirements from research teams
  2. Create templates based on common use cases
  3. Add comprehensive documentation for each template

### Parameter Validation Improvements
- **Issue**: Add more sophisticated parameter validation
- **Enhancements Needed**:
  - Cross-parameter validation (e.g., erlang_stop > erlang_start)
  - Topology-specific parameter constraints
  - Algorithm compatibility validation
  - Resource availability checks
- **Next Steps**:
  1. Define validation rules matrix
  2. Implement enhanced schema validation
  3. Add informative error messages

## Low Priority

### Template Description Injection
- **Issue**: Template export doesn't support description injection into INI files
- **Files**: `registry.py:240`
- **Description**: The export_config_template method accepts a description parameter but can't inject it into INI files due to save_config limitations
- **Impact**: Exported templates lack descriptive comments explaining their purpose
- **Next Steps**:
  1. Enhance save_config method to support comment injection
  2. Design comment formatting strategy for INI files
  3. Implement description insertion at template header
  4. Update export_config_template to use new functionality

### Configuration Migration Tools
- **Issue**: Need tools to migrate legacy configurations
- **Features**:
  - Automatic detection of outdated parameters
  - Migration suggestions for deprecated options
  - Bulk configuration updating utilities
- **Next Steps**:
  1. Catalog legacy configuration patterns
  2. Create migration mapping rules
  3. Implement automated migration tools

### Performance Optimization
- **Issue**: Configuration loading could be optimized for large-scale experiments
- **Opportunities**:
  - Configuration caching for repeated runs
  - Lazy loading of optional parameters
  - Parallel validation for multiple configurations
- **Next Steps**:
  1. Profile configuration loading performance
  2. Identify bottlenecks in validation process
  3. Implement caching mechanisms

## Completed Items

### ✅ Schema Validation System
- Implemented JSON schema-based validation
- Added comprehensive error reporting
- Created default configuration generation

### ✅ Template and Profile System
- Created template registry with multiple pre-configured setups
- Added configuration profiles for common use cases
- Implemented template export functionality

### ✅ CLI Integration
- Enhanced CLI to configuration mapping
- Added argument override capabilities
- Maintained backward compatibility

## Research Questions

### Algorithm-Specific Configuration
- **Question**: Should we create algorithm-specific configuration sections?
- **Context**: Different routing/allocation algorithms may need unique parameters
- **Research Needed**: Survey of algorithm-specific requirements across the codebase

### Multi-Objective Configuration
- **Question**: How to handle configurations for multi-objective optimization?
- **Context**: Some experiments require optimizing multiple metrics simultaneously
- **Research Needed**: Define multi-objective parameter structures

### Dynamic Configuration Updates
- **Question**: Should configurations be updatable during simulation runtime?
- **Context**: Long-running simulations might benefit from parameter adjustments
- **Research Needed**: Analyze safety and consistency requirements for runtime updates

---

## Contributing to TODOs

When adding new TODO items:

1. **Categorize by Priority**: High (blocking), Medium (enhances functionality), Low (nice to have)
2. **Provide Context**: Why is this needed? What's the impact?
3. **Define Next Steps**: Clear actionable items
4. **Reference Files**: List affected configuration files or code
5. **Estimate Effort**: Add complexity estimates when possible

When completing TODO items:
1. Move to "Completed Items" section with ✅ 
2. Add brief description of solution implemented
3. Update relevant documentation
4. Remove associated TODO comments from code