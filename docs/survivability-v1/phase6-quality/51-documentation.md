# Phase 6: Quality Assurance

## 51 - Documentation Requirements

**Section Reference**: Section 6 - Documentation Requirements

**Purpose**: Establish documentation standards for survivability v1 including module READMEs, docstrings, and configuration documentation to ensure maintainability and usability.

---

## 1. Module README.md Files (Required)

Each new module MUST have a `README.md` with:
- **Purpose**: Brief description of module functionality
- **Key Components**: List of main classes and files
- **Usage Examples**: Basic usage with code snippets
- **Dependencies**: Internal and external dependencies
- **Testing**: How to run tests for the module

### Example Module README

```markdown
# FUSION Failures Module

## Purpose
Provides failure injection and tracking for network survivability testing. Supports link (F1), SRLG (F3), and geographic (F4) failures.

## Key Components
- `FailureManager`: Main class for managing failure events
- `failure_types.py`: Implementations of failure types
- `registry.py`: Dynamic failure handler registration

## Usage Example
\`\`\`python
from fusion.modules.failures import FailureManager

manager = FailureManager(engine_props, topology)
event = manager.inject_failure(
    'geo',
    t_fail=100.0,
    t_repair=200.0,
    center_node=5,
    hop_radius=2
)
\`\`\`

## Dependencies
- Internal: `fusion.core.properties`, `fusion.core.simulation`
- External: `networkx`

## Testing
\`\`\`bash
pytest fusion/modules/failures/tests/ --cov=fusion.modules.failures
\`\`\`
```

---

## 2. Docstring Requirements (Required)

All public functions, classes, and methods MUST have Sphinx-style docstrings:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief description of function.

    Longer description with details about behavior,
    algorithms, or special cases.

    :param param1: Description of param1
    :type param1: int
    :param param2: Description of param2
    :type param2: str
    :return: Description of return value
    :rtype: bool
    :raises ValueError: When param1 is negative

    Example:
        >>> result = example_function(10, "test")
        >>> print(result)
        True
    """
    pass
```

### Required Elements
- Brief description (one line)
- Detailed description (if needed)
- All parameters documented with `:param:` and `:type:`
- Return value documented with `:return:` and `:rtype:`
- Exceptions documented with `:raises:`
- Usage example (when helpful)

---

## 3. Configuration Documentation (Required)

### Main README Update

Update `README.md` with survivability experiment instructions:

```markdown
## Survivability Experiments

FUSION supports survivability testing with failure injection and offline RL policies.

### Quick Start
\`\`\`bash
python -m fusion.cli.run_sim \
  --config_path fusion/configs/templates/survivability_experiment.ini \
  --failure_type geo \
  --geo_center_node 5 \
  --policy_type ksp_ff
\`\`\`

### Supported Failure Types
- **F1 (Link)**: Single link failure
- **F3 (SRLG)**: Shared Risk Link Group
- **F4 (Geographic)**: Hop-radius based regional failure

See `docs/survivability-v1/` for full documentation.
```

### Configuration Parameters Documentation

**Location**: `fusion/configs/README.md`

Document all new parameters:

```markdown
## Survivability Configuration

### Failure Settings
- `failure_type`: none, link, node, srlg, geo
- `t_fail_arrival_index`: Failure time (-1 = midpoint)
- `t_repair_after_arrivals`: Repair delay in arrivals
- `geo_center_node`: Center node for geographic failures
- `geo_hop_radius`: Radius in hops for geographic failures

### Protection Settings
- `protection_mode`: none, 1plus1
- `protection_switchover_ms`: Switchover latency (default: 50ms)

### RL Policy Settings
- `policy_type`: ksp_ff, one_plus_one, bc, iql
- `bc_model_path`: Path to BC model (.pt)
- `fallback_policy`: Fallback when all paths masked
```

---

## 4. Example Configurations

Provide example configurations for common scenarios:

### Example 1: Link Failure with KSP-FF
**Location**: `fusion/configs/examples/link_failure_ksp_ff.ini`

```ini
[failure_settings]
failure_type = link
failed_link_src = 3
failed_link_dst = 9
```

### Example 2: Geographic Failure with 1+1 Protection
**Location**: `fusion/configs/examples/geo_failure_protection.ini`

```ini
[failure_settings]
failure_type = geo
geo_center_node = 5
geo_hop_radius = 2

[protection_settings]
protection_mode = 1plus1
```

### Example 3: RL Policy Evaluation
**Location**: `fusion/configs/examples/rl_policy_eval.ini`

```ini
[offline_rl_settings]
policy_type = bc
bc_model_path = models/bc_model.pt
fallback_policy = ksp_ff
```

---

## 5. Acceptance Criteria

- [x] All modules have README.md files
- [x] All public APIs have Sphinx docstrings
- [x] Main README updated with survivability instructions
- [x] Configuration parameters documented
- [x] Example configurations provided
- [x] No missing or incomplete documentation

---

**Related Documents**:
- [50-testing.md](50-testing.md) (Testing standards)
- [12-configuration.md](../phase2-infrastructure/12-configuration.md) (Configuration system)
