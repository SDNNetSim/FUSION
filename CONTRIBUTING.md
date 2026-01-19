# Contributing to FUSION

Thank you for your interest in contributing to FUSION! We value your contributions and want to make the process as smooth as possible for everyone involved. This document outlines how you can contribute, our coding guidelines, the pull request process, and our code of conduct.

## Getting Started

Before contributing, we recommend:

1. **Read the Documentation**: Our comprehensive documentation is available at [https://sdnnetsim.github.io/FUSION/](https://sdnnetsim.github.io/FUSION/)
2. **Understand the Codebase**: Review the [Developer Documentation](https://sdnnetsim.github.io/FUSION/developer/) to understand the project structure
3. **Set Up Your Environment**: Follow the [Installation Guide](https://sdnnetsim.github.io/FUSION/getting-started/installation.html) to get started

## How to Contribute

1. **Report Issues**: If you find a bug or have a suggestion for an improvement, please report it using the project's issue tracker. Be sure to search for existing issues before creating a new one.

2. **Submit Pull Requests**: If you'd like to contribute code or documentation, please submit a pull request (PR). Ensure your PR has a clear title and description, and follows our coding guidelines and PR process outlined below.

3. **Review Contributions**: You can also contribute by reviewing pull requests submitted by others. Providing feedback and suggestions can greatly improve the quality of contributions.

## Coding Guidelines

FUSION follows strict coding standards to ensure consistency and maintainability. Please review our comprehensive coding standards document before contributing:

**See [CODING_STANDARDS.md](CODING_STANDARDS.md) for complete guidelines.**

### Key Highlights

1. **Naming Conventions**:
   - Functions: `snake_case` verbs (e.g., `load_config`, `validate_data`)
   - Classes: `PascalCase`
   - Use type hints instead of type suffixes in variable names

2. **Code Organization**:
   - Each module must have a `README.md`
   - Tests in `tests/` subdirectory within each module

3. **Type Annotations**:
   - All function parameters and returns must have type annotations
   - Use modern Python type hints

4. **Documentation**:
   - Use Sphinx-style docstrings
   - Use `# TODO:` for planned enhancements
   - Use `# FIXME:` for areas needing fixes

5. **Quality Tools**:
   - Format and lint code with `ruff`
   - Type check with `mypy`
   - Test with `pytest`
   - Run `make validate` before submitting

## Pull Request Process

1. **Fork and Branch**: Fork the repository and create your branch from `main`.
2. **Code Quality**: Ensure your code passes all quality checks:
   ```bash
   make lint        # Run all pre-commit checks
   make test        # Run unit tests
   make validate    # Full validation (lint + tests)
   ```
3. **Documentation**: Update relevant documentation and add tests for new features.
4. **Commit Messages**: Follow [conventional commit](https://www.conventionalcommits.org/) format. See our [Commit Message Guide](.github/COMMIT_MESSAGE_GUIDE.md).
5. **PR Template**: Fill out the PR template completely when submitting.
6. **Review**: You need sign-off from two other developers before merging.

Check the [PR template](https://github.com/SDNNetSim/FUSION/blob/main/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md) for specific requirements.

## Issue Reporting Process

The GitHub issue tracker is our primary medium for raising and resolving issues.

### Bug Reports
When reporting bugs, please use the [bug report template](https://github.com/SDNNetSim/FUSION/blob/main/.github/ISSUE_TEMPLATE/01_bug_report.yml) and include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Feature Requests
When requesting features, please use the [feature request template](https://github.com/SDNNetSim/FUSION/blob/main/.github/ISSUE_TEMPLATE/02_feature_request.yml) and include:
- Clear description of the proposed feature
- Use case and motivation
- Potential implementation approach
- Any relevant examples or references

## Documentation

We use Sphinx for documentation. The full documentation is hosted at:

**[https://sdnnetsim.github.io/FUSION/](https://sdnnetsim.github.io/FUSION/)**

Key documentation sections:
- [Getting Started](https://sdnnetsim.github.io/FUSION/getting-started/) - Installation and setup
- [Developer Guide](https://sdnnetsim.github.io/FUSION/developer/) - Module documentation and architecture
- [API Reference](https://sdnnetsim.github.io/FUSION/api/) - Auto-generated API docs

When contributing documentation:
- Source files are in `docs/` directory
- Use reStructuredText (`.rst`) format
- Build locally with `make -C docs html`

## Code of Conduct

Our project adheres to a Code of Conduct that we expect all contributors to follow. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) document before participating in our community.

## Questions or Comments

If you have any questions or comments about contributing to the FUSION project, please feel free to reach out to us. We're more than happy to help you get started or clarify any points.

Thank you for contributing to the FUSION project!
