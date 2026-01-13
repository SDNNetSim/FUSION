# CLI Module TODOs

## High Priority

### Custom Error Handling System
- **Issue**: CLI uses broad exception catching instead of specific error types (`run_sim.py:37-38`)
- **Impact**: Poor user experience with unclear error messages and difficult debugging
- **Solution**: Create custom CLI error module with specific exception types (SimulationError, ConfigurationError, ResourceError, ValidationError) and user-friendly error messages

### CLI Argument Validation Enhancement
- **Issue**: Need comprehensive CLI argument validation
- **Missing**: Cross-argument validation, file path checking, resource validation, algorithm compatibility
- **Solution**: Enhanced validation rules with informative error messages

### Multi-Processing Support
- **Issue**: Multi-processing functionality is no longer supported in the current implementation
- **Impact**: Users cannot run parallel simulations across multiple processes, limiting scalability for large-scale experiments
- **Solution**: Re-implement multi-processing support with proper process management, configuration handling, and result aggregation

## Medium Priority

### Single Entry Point CLI Architecture
- **Issue**: Current CLI design requires redundant subcommand pattern (`python -m fusion.cli.run_sim run_sim`)
- **Files**: `run_sim.py`, `run_train.py`, `main_parser.py`, `registry.py`
- **Description**: The current architecture uses separate module entry points (run_sim.py, run_train.py) each with their own subcommand parser, resulting in awkward invocation patterns like `python -m fusion.cli.run_sim run_sim` where "run_sim" appears twice
- **Impact**: Confusing user experience, inconsistent with modern CLI tools, harder to discover available commands
- **Proposed Solution**: Create single unified entry point with cleaner subcommand structure:
  - `fusion run_sim` instead of `python -m fusion.cli.run_sim run_sim`
  - `fusion train` instead of `python -m fusion.cli.run_train`
  - `fusion analyze` for future analysis commands
- **Next Steps**:
  1. Create main `fusion` entry point script
  2. Consolidate all subcommands under single parser
  3. Update setup.py/pyproject.toml with console_scripts entry point
  4. Migrate existing entry points to subcommand handlers
  5. Update all documentation with new invocation patterns
  6. Maintain backward compatibility during transition period
