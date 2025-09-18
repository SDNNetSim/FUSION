# Reporting Module TODOs

This file tracks known issues and future improvements for the FUSION reporting system and logs organization.

## High Priority

### Logs Directory Organization
- **Issue**: StableBaselines3 output files are currently in the root logs directory
- **Description**: Need to organize RL training logs into a dedicated subdirectory structure
- **Impact**: Mixed log types make it difficult to find specific logs
- **Current State**: `logs/*` is gitignored, containing various RL training outputs
- **Next Steps**:
  1. Create `logs/stablebaselines3/` subdirectory
  2. Move existing RL training output to new location
  3. Update RL code to output to new subdirectory
  4. Consider further organization by model type or experiment
  5. Update `.gitignore` to preserve directory structure while ignoring contents

### Proposed Log Directory Structure
```
logs/
├── stablebaselines3/        # RL training logs
│   ├── PPO/                 # Organized by algorithm
│   ├── DQN/
│   └── A2C/
├── simulations/             # Runtime simulation logs
│   └── [sim_name]/          # Organized by simulation
└── system/                  # System and error logs
    ├── errors.log
    └── performance.log
```

## Medium Priority

### Reporter Class Enhancements
- **Issue**: Extend SimulationReporter for different output formats
- **Description**: Support multiple output formats (JSON, CSV, HTML reports)
- **Next Steps**:
  1. Create formatter classes for each output type
  2. Add configuration for output preferences
  3. Implement summary report generation

### Progress Reporting
- **Issue**: Real-time progress reporting for long-running simulations
- **Description**: Provide progress bars and ETA calculations
- **Next Steps**:
  1. Integrate with tqdm or similar progress library
  2. Add configurable update intervals
  3. Support both console and file-based progress tracking

### Visualization Integration
- **Issue**: Connect reporting module with visualization tools
- **Description**: Generate plots and charts as part of reports
- **Next Steps**:
  1. Define visualization requirements
  2. Create plotting utilities
  3. Integrate with existing visualization modules

## Low Priority

### Report Templates
- **Issue**: Create customizable report templates
- **Description**: Allow users to define custom report formats
- **Next Steps**:
  1. Design template system
  2. Create example templates
  3. Add template configuration

### Multi-format Export
- **Issue**: Support exporting reports in multiple formats simultaneously
- **Description**: Generate PDF, HTML, and Excel reports from same data
- **Next Steps**:
  1. Research multi-format libraries
  2. Design unified export interface
  3. Implement format converters

## Completed Items

### ✅ Reporting Module Creation
- Created `fusion/reporting/` module structure
- Separated reporting concerns from metrics collection
- Established clear module boundaries

---

## Contributing to TODOs

When adding new TODO items:

1. **Categorize by Priority**: High (blocking), Medium (enhances functionality), Low (nice to have)
2. **Provide Context**: Why is this needed? What's the impact?
3. **Define Next Steps**: Clear actionable items
4. **Reference Files**: List affected files or modules
5. **Consider Dependencies**: Note any prerequisites or related work

When completing TODO items:
1. Move to "Completed Items" section with ✅ 
2. Add brief description of solution implemented
3. Update relevant documentation
4. Remove associated TODO comments from code