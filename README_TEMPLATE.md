# Module Name Template

> **Usage**: Copy this template when creating new modules. Replace placeholder text with actual content.

---

# [Module Name]

## Purpose
Brief description of what this module does and its role in the FUSION project.

**Example**: *This module handles network topology generation and management for optical network simulations.*

## Key Components

### Core Files
- `core.py`: [Description of main functionality]
- `utils.py` or `utils/`: [Description of utility functions]
- `constants.py`: [Module-specific constants]
- `errors.py`: [Custom exceptions for this module]

### Optional Files
- `registry.py`: [Component registry - only if needed for multi-component modules]
- `config.py`: [Module-specific configuration handling]

## Usage

### Basic Usage
```python
from fusion.[module_name] import [main_class_or_function]

# Example usage
example = MainClass(config_params)
result = example.primary_method()
```

### Advanced Usage
```python
# More complex examples
```

## Dependencies

### Internal Dependencies
- `fusion.core.config`: Configuration management
- `fusion.utils.logging`: Logging utilities
- [Other internal modules]

### External Dependencies
- `numpy`: [Why this is needed]
- `pathlib`: Path handling
- [Other external packages]

## Configuration

### Required Configuration
```python
# Example configuration structure
config = {
    "module_setting": "value",
    "another_setting": 42
}
```

### Environment Variables
- `FUSION_MODULE_DEBUG`: Enable debug mode for this module
- [Other environment variables]

## Testing

Unit tests are located in `tests/` directory:
```bash
# Run module tests
pytest fusion/[module_name]/tests/

# Run with coverage
pytest --cov=fusion.[module_name] fusion/[module_name]/tests/
```

## API Reference

### Main Classes
- `MainClass`: [Brief description]
- `HelperClass`: [Brief description]

### Key Functions
- `primary_function()`: [Brief description]
- `utility_function()`: [Brief description]

## Examples

### Example 1: Basic Setup
```python
# Code example showing typical usage
```

### Example 2: Advanced Configuration
```python
# Code example showing complex scenarios
```

## Notes

### Design Decisions
- [Important design choices and rationale]
- [Why certain patterns were chosen]

### Known Limitations
- [Current limitations or constraints]
- [Future improvement areas]

### Performance Considerations
- [Performance tips or warnings]
- [Memory usage considerations]

---

## Template Notes (Remove in actual README)

### Sections to customize:
1. **Replace [Module Name]** with actual module name
2. **Purpose section**: 2-3 sentences max, be specific
3. **Key Components**: List only files that actually exist
4. **Usage examples**: Real, working code examples
5. **Dependencies**: Only list actual dependencies with reasons
6. **Remove template notes** section when done

### Optional sections (add if relevant):
- **Architecture**: For complex modules with multiple components
- **Algorithms**: For modules implementing specific algorithms
- **Integration**: How this module integrates with others
- **Troubleshooting**: Common issues and solutions

### Writing guidelines:
- **Concise**: Each section should be scannable
- **Examples**: Always include working code examples
- **AI-friendly**: Clear structure for Claude to understand
- **Actionable**: Tell readers exactly what they can do