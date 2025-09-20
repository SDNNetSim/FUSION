# Contributing to the Simulator

Thank you for your interest in contributing to the simulator! We value your contributions and want to make the process as smooth as possible for everyone involved. This document outlines how you can contribute, our coding guidelines, the pull request process, and our code of conduct.

## Introduction

This simulator is an open-source initiative, and we welcome contributions from everyone. Whether you're fixing a bug, adding a new feature, or improving documentation, your help is appreciated. Before contributing, please take a moment to read through this document to understand our processes and guidelines.

## How to Contribute

1. **Report Issues**: If you find a bug or have a suggestion for an improvement, please report it using the project's issue tracker. Be sure to search for existing issues before creating a new one.

2. **Submit Pull Requests**: If you'd like to contribute code or documentation, please submit a pull request (PR). Ensure your PR has a clear title and description, and follows our coding guidelines and PR process outlined below.

3. **Review Contributions**: You can also contribute by reviewing pull requests submitted by others. Providing feedback and suggestions can greatly improve the quality of contributions.

## Coding Guidelines

### 1. Naming Conventions

1. **Helper Scripts**: Files in the `helper_scripts` directory should be named using the
   pattern `<script_name>_helpers.py`.
2. **Data Structures**: Name variables with their type, e.g., `<name>_list`, `<name>_dict`, `<name>_set`.
3. **Class Properties**: Include a dictionary named `<ClassName>_props` in class constructors for properties.
4. **Inner Classes**: Name classes within a constructor `<ClassName>_Obj` to indicate scope and relationship.

### 2. Directory and File Structure

1. **Argument Scripts**: Place external files with arguments in the `arg_scripts` directory,
   named `<file_name>_args.py`.
2. **Module Naming**: Directories with Python scripts should follow `<name>_scripts` naming convention.

### 3. Coding Practices

1. **Function Names**: Use assertive and descriptive names like `get`, `create`, `update`.
2. **Type Annotations**: Explicitly list variable types in all function parameters.
3. **Commenting and Documentation**:
    - Use `# FIXME:` for areas needing future fixes, with a brief explanation if necessary.
    - Use `# TODO:` for planned enhancements or tasks, with a concise description.
4. **Argument Labeling**: Label arguments explicitly when calling functions.

### 4. Testing and Quality Assurance

1. **Comprehensive Testing**: Test every function and its branches thoroughly.
2. **Development Tools**: We use a modern development stack for code quality:
   - **black**: Code formatter (automatic code style)
   - **ruff**: Fast linting and code analysis
   - **mypy**: Type checking for better code safety
   - **pytest**: Testing framework with coverage reporting
   - **pre-commit**: Automated quality checks before commits
3. **Development Workflow**: Follow our established development process:
   - Run `make format` to format code before commits
   - Run `make lint-new` for linting and type checking
   - Run `make test-new` for testing with coverage
   - Run `make check-all` for comprehensive quality checks
4. **Quality Standards**: All code must pass our automated quality checks including formatting, linting, type checking, and maintain test coverage above 80%.

### 5. Additional Considerations

1. **Class and File Naming Alignment**: Ensure class names match their `.py` file names.
2. **Argument Documentation**: Comment each argument in argument scripts to explain its purpose and expected values.

## Pull Request Process

### Before You Start
1. **Set up development environment**:
   ```bash
   pip install -r requirements-dev.txt
   make setup-hooks
   ```
2. **Fork and branch**: Fork the repository and create your branch from `main`.

### Development Process
1. **Code Quality**: All code must pass our quality checks:
   ```bash
   make check-all  # Runs format, lint, test, and analysis
   ```
2. **Pre-commit hooks**: Our hooks will automatically check your code on commit. If they fail, fix the issues and commit again.
3. **Testing**: Ensure all tests pass and maintain coverage above 80%:
   ```bash
   make test-new
   ```
4. **Documentation**: Update documentation if your changes affect user-facing functionality.

### Submitting Your PR
1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is semantic.
3. Your PR must pass all automated checks in our CI/CD pipeline.
4. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

*Check to ensure that your pull request meets the requirements as outlined by the [PR template](https://github.com/SDNNetSim/SDON_simulator/blob/v1_1/.github/pull_request_template.md) before merging.

## Issue Reporting Process
The github issue tracker is our primary medium for raising and resolving issues. Specific information related to reporting bugs and requesting features can be found below.

### Bug Reports
For bug reports, please be sure the structure of your report is consistent with the [bug report](https://github.com/SDNNetSim/SDON_simulator/blob/v1_1/.github/issue_template/bug_report.md) template.

### Feature Requests
For feature requests, please be sure the structure of your request is consistent with the [feature request](https://github.com/SDNNetSim/SDON_simulator/blob/v1_1/.github/issue_template/feature_request.md) template.


## Code of Conduct

Our project adheres to a Code of Conduct that we expect all contributors to follow. Please read the [code of conduct](CODE_OF_CONDUCT.md) document before participating in our community.

## Questions or Comments

If you have any questions or comments about contributing to the ACNL project, please feel free to reach out to us. We're more than happy to help you get started or clarify any points.

Thank you for contributing to the ACNL project!
