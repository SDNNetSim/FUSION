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

## Development Workflow

### Quality Tools
- **ruff**: Modern linting and formatting (replaces black, flake8, isort)
- **mypy**: Type checking with strict configuration
- **pytest**: Unit testing with coverage reporting
- **pre-commit**: Automated quality checks on commit (includes ruff, mypy, vulture, bandit)

### Key Commands
```bash
make install     # Install all dependencies
make install-dev # Install development tools only
make validate    # Run all pre-commit checks + tests (use before PRs)
make lint        # Run all pre-commit checks on all files
make test        # Run unit tests with pytest
make clean       # Clean up generated files
```

## Important Constraints and Guidelines

### What to Avoid
- No emojis in documentation or code (per user preference)
- No print statements - use logging
- No hardcoded paths - use configuration
- No broad exception catching
- The `fusion/gui` module has been removed (deprecated and deleted)

### What to Follow
- All functions require type annotations
- All code must pass ruff and mypy checks
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
1. Sphinx documentation: Comprehensive module documentation under `docs/developer/`
2. Quality improvements: Resolving linting errors and test failures
3. Survivability experiment features
4. Offline RL policy integration

## Documentation System

The project uses two documentation systems:
- **Module READMEs**: Each module has a `README.md` with overview and usage
- **Sphinx Documentation**: Comprehensive API and developer docs in `docs/`
  - Developer docs: `docs/developer/fusion/` with per-module documentation
  - Getting started guides: `docs/getting-started/`
  - API reference: `docs/api/`

## Reference Documents

For detailed information, consult:
- `CODING_STANDARDS.md`: Code style and organization
- `TESTING_STANDARDS.md`: Testing requirements
- `DEVELOPMENT_WORKFLOW.md`: Development process and tools
- `DEVELOPMENT_QUICKSTART.md`: Quick start guide for new developers
- `CONTRIBUTING.md`: Contribution guidelines

## Working with This Codebase

When making changes:
1. Read relevant module READMEs first
2. Check `TODO.md` files for known issues
3. Follow the standardized naming conventions
4. Add tests for new functionality
5. Update documentation as needed
6. Run `make validate` before committing
7. Use the TodoWrite tool to track multi-step tasks

## Key Files to Know

- `fusion/core/simulation.py`: Main simulation engine
- `fusion/interfaces/factory.py`: Algorithm factory and pipeline
- `fusion/cli/run_sim.py`: CLI entry point
- `fusion/configs/cli_to_config.py`: Configuration processing
- `fusion/modules/*/registry.py`: Algorithm registries

## RL Module Structure

The RL module (`fusion/modules/rl/`) has extensive sub-modules:
- `adapter/`: RL-simulation adapter layer
- `agents/`: RL agent implementations
- `algorithms/`: RL algorithm implementations
- `args/`: Argument parsing and configuration
- `environments/`: Custom RL environments
- `feat_extrs/`: Feature extractors for state representation
- `gymnasium_envs/`: Gymnasium-compatible environment wrappers
- `policies/`: Policy implementations (BC, IQL, CQL, etc.)
- `sb3/`: Stable-Baselines3 integration
- `utils/`: RL utility functions
- `visualization/`: Training visualization tools

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
