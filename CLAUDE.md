# FUSION Project Context for AI Assistants

This document provides context for AI assistants (like Claude) working with the FUSION codebase.

## Project Overview

FUSION (Flexible Unified System for Intelligent Optical Networking) is an open-source simulation framework for Software Defined Elastic Optical Networks (SD-EONs) with extensions for AI/ML integration, particularly reinforcement learning for network optimization and survivability.

## Key Architectural Concepts

### Core Simulation Components
- **Discrete Event Simulation**: Event-driven network request processing
- **RSA (Routing and Spectrum Assignment)**: The fundamental resource allocation problem
- **Network Topology**: Graph-based representation using NetworkX
- **Spectrum Management**: Flexible grid optical spectrum allocation
- **Traffic Generation**: Poisson arrival process with configurable parameters

### Reinforcement Learning Integration
- **Offline RL**: Training from heuristic behavior logs (BC, IQL, CQL)
- **Online RL**: Integration with Stable-Baselines3 (PPO, A2C, DQN)
- **Action Masking**: Safety mechanism to prevent invalid actions
- **Policy Evaluation**: Baseline comparison and metrics collection

### Survivability Features
- **Failure Types**: Link (F1), Node (F2), SRLG (F3), Geographic (F4)
- **Protection Mechanisms**: 1+1 disjoint path protection
- **Recovery Metrics**: Blocking probability, recovery time, fragmentation

## Code Organization Philosophy

### Module Structure
- Each module contains its own `tests/` subdirectory
- Every module and test directory must have a `README.md`
- Use `__init__.py` to define public APIs with `__all__`
- Registry pattern for pluggable algorithms

### Naming Conventions
- **Standardized names**: `engine_props`, `sim_params`, `network_topology`
- **No type suffixes**: Use type hints instead of `_dict`, `_list` suffixes
- **Functions**: Verb phrases in `snake_case`
- **Classes**: `PascalCase`

### State Management
- Use `StateWrapper` for mutable configuration objects like `engine_props`
- Never hardcode paths - use `pathlib.Path` and configuration
- All RNG operations must be seeded for reproducibility

## Development Workflow

### Quality Tools
- **ruff**: Modern linting and formatting (replaces black, flake8, isort)
- **mypy**: Type checking with strict configuration
- **pytest**: Unit testing with coverage reporting
- **pre-commit**: Automated quality checks on commit

### Key Commands
```bash
make format      # Auto-format code
make lint-new    # Check for issues
make test-new    # Run tests with coverage
make check-all   # Full quality check
```

## Important Constraints and Guidelines

### What to Avoid
- No emojis in documentation or code (per user preference)
- No print statements - use logging
- No hardcoded paths - use configuration
- No broad exception catching
- No `fusion/gui` module changes (deprecated, requires revamp)

### What to Follow
- All functions require type annotations
- All code must pass ruff and mypy checks
- Functions should be under 50 lines
- Files should be under 500 lines
- Test coverage targets: 80-90% for most modules
- Use Sphinx-style docstrings

## Configuration System

FUSION uses INI-based configuration with:
- Template system in `fusion/configs/templates/`
- CLI argument integration via `fusion/cli/run_sim.py`
- Validation and type checking
- Environment-specific overrides

## Testing Standards

### Unit Testing
- Tests located in `tests/` subdirectory within each module
- One test file per module: `test_<module_name>.py`
- Mock all external dependencies
- Follow AAA pattern (Arrange, Act, Assert)
- Test naming: `test_<what>_<when>_<expected>`

### Integration Testing
- Located in top-level `tests/` directory
- Test component interactions
- May use real dependencies where appropriate

## Common Patterns

### Factory Pattern
See `fusion/interfaces/factory.py` for algorithm instantiation

### Interface/Abstract Base Classes
- `AbstractRoutingAlgorithm`
- `AbstractSpectrumAssigner`
- `AbstractSNRMeasurer`

### Registry Pattern
- Algorithm registration and discovery
- Plugin-style architecture
- Used in `fusion/modules/*/registry.py`

## Current Development Focus

As of the latest commits, the project is focused on:
1. Quality improvements: Resolving linting errors and test failures
2. Configuration system enhancements
3. Documentation improvements
4. Survivability experiment features (v1)
5. Offline RL policy integration

## Reference Documents

For detailed information, consult:
- `ARCHITECTURE.md`: System architecture and design
- `CODING_STANDARDS.md`: Code style and organization
- `TESTING_STANDARDS.md`: Testing requirements
- `DEVELOPMENT_WORKFLOW.md`: Development process and tools
- `CONTRIBUTING.md`: Contribution guidelines

## Working with This Codebase

When making changes:
1. Read relevant module READMEs first
2. Check `TODO.md` files for known issues
3. Follow the standardized naming conventions
4. Add tests for new functionality
5. Update documentation as needed
6. Run `make check-all` before committing
7. Use the TodoWrite tool to track multi-step tasks

## Key Files to Know

- `fusion/core/simulation.py`: Main simulation engine
- `fusion/interfaces/factory.py`: Algorithm factory and pipeline
- `fusion/cli/run_sim.py`: CLI entry point
- `fusion/configs/cli_to_config.py`: Configuration processing
- `fusion/modules/*/registry.py`: Algorithm registries

## Domain Knowledge

### Optical Networking Concepts
- **Spectrum slots**: Frequency spectrum divided into assignable units
- **Wavelength**: Individual optical carrier
- **Modulation formats**: BPSK, QPSK, 16-QAM, etc.
- **SNR (Signal-to-Noise Ratio)**: Quality metric for optical signals
- **Guard bands**: Spacing between spectrum assignments
- **Fragmentation**: Inefficient use of spectrum due to non-contiguous allocations

### Network Survivability
- **Protection**: Pre-provisioned backup resources (1+1, 1:1)
- **Restoration**: Dynamic re-routing after failure
- **SRLG (Shared Risk Link Group)**: Links that fail together
- **Disjoint paths**: Paths sharing no common links/nodes

This context should help AI assistants understand the project structure, conventions, and domain when assisting with development tasks.
