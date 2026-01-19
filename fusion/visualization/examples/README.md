# Configuration Examples

This directory contains example configuration files for the FUSION visualization system.

## Files

### Basic Examples

- **`minimal_config_example.yml`** - Minimal working configuration
  - Demonstrates the bare minimum required fields
  - Uses all default settings
  - Great starting point for simple plots

- **`new_config_example.yml`** - Standard configuration
  - Shows typical usage with common options
  - Multiple plot types
  - Migrated from old format

### Advanced Examples

- **`advanced_config_example.yml`** - Comprehensive configuration
  - All available options demonstrated
  - Custom styling and statistical options
  - Performance tuning settings

### Legacy Examples

- **`old_config_example.yml`** - OLD (deprecated) format
  - **⚠️ DO NOT USE for new projects**
  - Included for reference only
  - Shows what to migrate FROM

## Usage

### Quick Start

```bash
# Copy an example
cp fusion/visualization/examples/minimal_config_example.yml my_config.yml

# Edit for your needs
nano my_config.yml

# Validate
fusion viz validate --config my_config.yml

# Generate plot
fusion viz plot --config my_config.yml
```

### Migration

If you have an old configuration file:

```bash
# Migrate from old to new format
fusion viz migrate --input old_config.yml --output new_config.yml --validate

# Review the new config
cat new_config.yml

# Test it
fusion viz plot --config new_config.yml
```

## Configuration Sections

### Required Fields

Every configuration must have:
```yaml
network: NSFNet  # Network topology name
dates: ["0606"]  # List of dates to process
plots:            # List of plots to generate
  - type: blocking
```

### Common Optional Fields

```yaml
defaults:
  format: png       # Output format: png, pdf, svg
  dpi: 300          # Resolution
  style: seaborn    # Plot style
  cache_enabled: true  # Enable caching for faster reloads

plots:
  - type: blocking
    algorithms: [ppo_obs_7, dqn_obs_7]  # Algorithms to compare
    traffic_volumes: [600, 700, 800]    # Traffic levels
    include_ci: true                     # Include confidence intervals
    save_path: ./figures/my_plot.png    # Custom output path
```

## Plot Types

### Available Plot Types

| Type | Description | Required Fields |
|------|-------------|----------------|
| `blocking` | Blocking probability vs traffic | algorithms, traffic_volumes |
| `rewards` | RL training rewards | algorithms |
| `memory` | Memory usage over time | algorithms |
| `computation` | Computation time | algorithms, traffic_volumes |
| `hops` | Average hop count | algorithms, traffic_volumes |
| `lengths` | Average path length | algorithms, traffic_volumes |

### Custom Plot Types

Create plugins for custom plot types:

```python
# fusion/modules/mymodule/visualization/my_plugin.py
from fusion.visualization.plugins import BasePlugin

class MyPlugin(BasePlugin):
    def register_plot_types(self):
        return {"my_custom_plot": PlotTypeRegistration(...)}
```

## Tips

### Performance

For faster plot generation:
```yaml
defaults:
  cache_enabled: true

performance:
  parallel: true
  max_workers: 4
```

### Styling

Match old visualization style:
```yaml
defaults:
  style: seaborn-whitegrid
  colors: ["#1f77b4", "#ff7f0e", "#2ca02c"]
  line_width: 2.0
  font_size: 12
```

### Batch Processing

Generate many plots at once:
```bash
# Put multiple configs in a directory
mkdir configs/
cp *_config.yml configs/

# Batch generate
fusion viz batch --config-dir configs/
```

## Validation

Always validate before running:
```bash
fusion viz validate --config my_config.yml
```

Common validation errors:
- Missing required fields (network, dates, plots)
- Invalid plot type
- Empty algorithm list
- Invalid file paths

## Getting Help

- **Migration Guide:** `MIGRATION_GUIDE.md`
- **API Reference:** `docs/visualization/api_reference.md`
- **Architecture Docs:** `docs/visualization/architecture.md`
- **CLI Help:** `fusion viz --help`

## Version Compatibility

These examples are for:
- FUSION version: 6.0+
- Visualization system: New architecture (DDD)
- Configuration format: v2

For legacy (pre-6.0) versions, see `old_config_example.yml` (deprecated).
