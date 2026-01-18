# RL Visualization Module

**Status: BETA**

This module provides RL-specific visualization capabilities as a **plugin extension** to the core FUSION visualization system.

## Architecture Overview

FUSION uses a **plugin-based visualization architecture**. The core visualization system (`fusion/visualization/`) provides:

- Domain-Driven Design (DDD) infrastructure
- Base metric definitions and plot types
- Data adapters, processors, and renderers
- Plugin discovery and registration

This RL visualization module (`fusion/modules/rl/visualization/`) extends that system by providing:

- RL-specific metrics (rewards, Q-values, policy entropy, etc.)
- RL-specific plot types (learning curves, convergence analysis, dashboards)
- RL-specific data processing strategies

```
fusion/visualization/              <-- Core visualization system
|-- plugins/
|   |-- base_plugin.py            <-- Plugin interface
|   `-- plugin_registry.py        <-- Plugin discovery/loading
|
fusion/modules/rl/visualization/   <-- This module (RL plugin)
|-- rl_plugin.py                  <-- Implements BasePlugin
|-- rl_metrics.py                 <-- RL metric definitions
|-- rl_plots.py                   <-- RL plot renderers
`-- rl_processors.py              <-- RL data processors
```

## How It Works

1. **Plugin Discovery**: The core visualization system discovers plugins via `get_plugin()` entry points
2. **Registration**: When loaded, `RLVisualizationPlugin` registers its metrics, plot types, and processors
3. **Usage**: Users can then use RL-specific plot types through the standard visualization API

## Components

### rl_plugin.py

The main plugin class implementing `BasePlugin`. Registers all RL-specific components:

- `register_metrics()` - Episode rewards, TD errors, Q-values, policy entropy, etc.
- `register_plot_types()` - Learning curves, Q-value heatmaps, convergence plots, dashboards
- `register_processors()` - Reward smoothing, Q-value processing, convergence detection

### rl_metrics.py

Defines RL training metrics using the core `MetricDefinition` class:

| Metric | Description |
|--------|-------------|
| `episode_reward` | Total reward per episode |
| `td_error` | Temporal difference errors |
| `q_values` | Action-value estimates |
| `policy_entropy` | Policy distribution entropy |
| `learning_rate` | Current learning rate |

### rl_plots.py

Custom renderers extending `BaseRenderer`:

| Renderer | Output |
|----------|--------|
| `RewardLearningCurveRenderer` | Episode rewards with confidence intervals |
| `QValueHeatmapRenderer` | Q-value heatmaps across states/actions |
| `ConvergencePlotRenderer` | Training convergence analysis |
| `MultiMetricDashboardRenderer` | Multi-panel RL training dashboard |

### rl_processors.py

Data processing strategies implementing `DataProcessorPort`:

| Processor | Purpose |
|-----------|---------|
| `RewardProcessingStrategy` | Smoothing, aggregation, confidence intervals |
| `QValueProcessingStrategy` | Q-value normalization and statistics |
| `ConvergenceDetectionStrategy` | Detect when training has converged |

## Usage

### Via Core Visualization API

```python
from fusion.visualization.plugins import get_global_registry

# Load RL plugin
registry = get_global_registry()
registry.discover_plugins()
registry.load_plugin("rl")

# Use RL plot types
from fusion.visualization.application.use_cases.generate_plot import generate_plot

result = generate_plot(
    config_path="rl_training_config.yml",
    plot_type="reward_learning_curve",  # RL-specific plot type
)
```

### Direct Import (Advanced)

```python
from fusion.modules.rl.visualization import RLVisualizationPlugin

plugin = RLVisualizationPlugin()
metrics = plugin.register_metrics()
plot_types = plugin.register_plot_types()
```

## Dependencies

- scipy (for smoothing filters)
- seaborn (for heatmaps)
- matplotlib (inherited from core visualization)

## Related Documentation

- Core visualization: `fusion/visualization/README.md`
- Sphinx docs: `docs/developer/fusion/modules/rl/visualization.rst`
- Plugin interface: `fusion/visualization/plugins/base_plugin.py`
