## Plugin Development Guide

**Version:** 1.0
**Last Updated:** 2025-10-02
**Audience:** FUSION Developers

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Plugin Architecture](#plugin-architecture)
4. [Creating a Plugin](#creating-a-plugin)
5. [Registering Metrics](#registering-metrics)
6. [Registering Plot Types](#registering-plot-types)
7. [Custom Processors](#custom-processors)
8. [Custom Renderers](#custom-renderers)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Best Practices](#best-practices)
12. [Examples](#examples)

---

## Introduction

The FUSION visualization plugin system allows you to extend the visualization capabilities with module-specific metrics, plot types, and processing strategies **without modifying core code**.

### Why Create a Plugin?

- Add custom metrics for your module
- Create specialized plot types
- Implement domain-specific processing strategies
- Share visualization tools across teams
- Maintain clean separation of concerns

### Plugin Capabilities

A plugin can:
- ✅ Register custom metrics
- ✅ Register custom plot types
- ✅ Provide processing strategies
- ✅ Provide custom renderers
- ✅ Define configuration schemas
- ✅ Validate configurations
- ✅ Declare dependencies on other plugins

---

## Quick Start

### 5-Minute Plugin

Create a basic plugin in 5 minutes:

```python
# fusion/modules/mymodule/visualization/mymodule_plugin.py

from fusion.visualization.plugins import BasePlugin, PlotTypeRegistration
from fusion.visualization.domain.entities.metric import (
    MetricDefinition, DataType, AggregationStrategy
)

class MyModulePlugin(BasePlugin):
    """Plugin for MyModule visualizations."""

    @property
    def name(self) -> str:
        return "mymodule"

    def register_metrics(self):
        """Register module-specific metrics."""
        return [
            MetricDefinition(
                name="my_metric",
                display_name="My Metric",
                data_type=DataType.FLOAT,
                source_path="$.mymodule.metric_value",
                aggregation=AggregationStrategy.MEAN,
                unit="units",
                description="Description of my metric"
            )
        ]
```

That's it! Your plugin will be auto-discovered and loaded.

---

## Plugin Architecture

### Plugin Lifecycle

```
1. Discovery      → Plugin files scanned in module directories
2. Registration   → Plugin class registered with PluginRegistry
3. Loading        → Dependencies resolved, plugin.on_load() called
4. Usage          → Metrics/plot types available to visualization system
5. Unloading      → plugin.on_unload() called (if needed)
```

### Plugin Structure

```
fusion/modules/mymodule/visualization/
├── __init__.py                    # Export plugin class
├── mymodule_plugin.py            # Main plugin file
├── mymodule_metrics.py           # Metric definitions (optional)
├── mymodule_processors.py        # Processing strategies (optional)
├── mymodule_plots.py             # Plot renderers (optional)
└── tests/                        # Plugin tests
    └── test_mymodule_plugin.py
```

---

## Creating a Plugin

### Step 1: Create Plugin Class

```python
from fusion.visualization.plugins import BasePlugin

class MyPlugin(BasePlugin):
    """My custom visualization plugin."""

    @property
    def name(self) -> str:
        """Unique plugin identifier."""
        return "myplugin"

    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Plugin description."""
        return "Visualization plugin for MyModule"

    @property
    def author(self) -> str:
        """Plugin author."""
        return "Your Name"
```

### Step 2: Implement Required Methods

```python
    def register_metrics(self) -> List[MetricDefinition]:
        """Register custom metrics."""
        return [
            # Your metrics here
        ]

    def register_plot_types(self) -> Dict[str, PlotTypeRegistration]:
        """Register custom plot types."""
        return {
            # Your plot types here
        }
```

### Step 3: Add Lifecycle Hooks (Optional)

```python
    def on_load(self) -> None:
        """Called when plugin is loaded."""
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {self.name} plugin v{self.version}")

    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        # Cleanup if needed
        pass
```

### Step 4: Check Dependencies (Optional)

```python
    def _check_dependencies(self) -> None:
        """Check plugin dependencies."""
        try:
            import required_package  # noqa: F401
        except ImportError as e:
            raise ImportError(f"Plugin requires required_package: {e}")
```

---

## Registering Metrics

### Basic Metric

```python
from fusion.visualization.domain.entities.metric import (
    MetricDefinition,
    DataType,
    AggregationStrategy
)

MetricDefinition(
    name="request_latency",
    display_name="Request Latency",
    data_type=DataType.FLOAT,
    source_path="$.performance.latency",
    aggregation=AggregationStrategy.MEAN,
    unit="ms",
    description="Average request latency in milliseconds"
)
```

### Data Types

```python
DataType.FLOAT      # Floating point number
DataType.INT        # Integer
DataType.ARRAY      # Numpy array or list
DataType.DICT       # Dictionary
DataType.STRING     # String value
```

### Aggregation Strategies

```python
AggregationStrategy.MEAN      # Average across runs
AggregationStrategy.MEDIAN    # Median value
AggregationStrategy.SUM       # Sum across runs
AggregationStrategy.MIN       # Minimum value
AggregationStrategy.MAX       # Maximum value
AggregationStrategy.LAST      # Last value
AggregationStrategy.FIRST     # First value
AggregationStrategy.STD       # Standard deviation
```

### Complete Example

```python
def register_metrics(self) -> List[MetricDefinition]:
    """Register all module metrics."""
    return [
        # Performance metrics
        MetricDefinition(
            name="throughput",
            display_name="Throughput",
            data_type=DataType.FLOAT,
            source_path="$.performance.throughput",
            aggregation=AggregationStrategy.MEAN,
            unit="Gbps",
            description="Network throughput in gigabits per second"
        ),

        # Resource metrics
        MetricDefinition(
            name="cpu_usage",
            display_name="CPU Usage",
            data_type=DataType.FLOAT,
            source_path="$.resources.cpu",
            aggregation=AggregationStrategy.MEAN,
            unit="percent",
            description="CPU utilization percentage"
        ),

        # Array metrics
        MetricDefinition(
            name="latency_samples",
            display_name="Latency Samples",
            data_type=DataType.ARRAY,
            source_path="$.performance.latency_samples",
            aggregation=AggregationStrategy.MEAN,
            unit="ms",
            description="Array of latency samples for distribution analysis"
        ),
    ]
```

---

## Registering Plot Types

### Basic Plot Type Registration

```python
from fusion.visualization.plugins import PlotTypeRegistration
from fusion.visualization.domain.strategies.processing_strategies import (
    GenericMetricProcessingStrategy
)

def register_plot_types(self) -> Dict[str, PlotTypeRegistration]:
    """Register custom plot types."""
    return {
        "my_plot": PlotTypeRegistration(
            processor=GenericMetricProcessingStrategy(),
            renderer=MyCustomRenderer(),
            description="My custom plot type",
            required_metrics=["my_metric"],
            default_config={
                "style": "default",
                "show_legend": True,
            }
        )
    }
```

### PlotTypeRegistration Parameters

```python
PlotTypeRegistration(
    processor=<ProcessingStrategy>,       # How to process data
    renderer=<Renderer>,                  # How to render plot
    description="Description of plot",    # User-friendly description
    required_metrics=["metric1", "metric2"],  # Required metrics
    default_config={...}                  # Default configuration
)
```

---

## Custom Processors

### Creating a Processing Strategy

```python
from fusion.visualization.domain.strategies.processing_strategies import (
    MetricProcessingStrategy,
    ProcessedMetric
)

class MyProcessor(MetricProcessingStrategy):
    """Custom processing strategy for my module."""

    def process(
        self,
        runs: List[Run],
        canonical_data: Dict[str, CanonicalData],
        metric_definition: MetricDefinition,
    ) -> ProcessedMetric:
        """Process metric data.

        Args:
            runs: List of simulation runs
            canonical_data: Canonical data for each run
            metric_definition: Metric definition

        Returns:
            ProcessedMetric with processed data
        """
        # Group by algorithm
        grouped = defaultdict(list)
        for run in runs:
            algo = run.algorithm
            data = canonical_data.get(run.id)
            if data:
                value = self._extract_value(data, metric_definition)
                grouped[algo].append(value)

        # Compute statistics
        results = {}
        for algo, values in grouped.items():
            results[algo] = {
                "mean": np.mean(values),
                "std": np.std(values, ddof=1),
                "ci": 1.96 * np.std(values) / np.sqrt(len(values)),
            }

        return ProcessedMetric(
            name=metric_definition.name,
            data=results,
            metadata={"aggregation": "mean_with_ci"}
        )

    def _extract_value(self, data, metric_def):
        """Extract metric value from data."""
        # Implement extraction logic
        pass
```

### Example: Smoothing Processor

```python
from scipy.ndimage import uniform_filter1d

class SmoothingProcessor(MetricProcessingStrategy):
    """Processor that applies smoothing to time series."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def process(self, runs, canonical_data, metric_definition):
        """Process with smoothing."""
        results = {}

        for run in runs:
            data = canonical_data.get(run.id)
            if data:
                # Extract time series
                time_series = self._extract_series(data, metric_definition)

                # Apply smoothing
                smoothed = uniform_filter1d(
                    time_series,
                    size=self.window_size,
                    mode='nearest'
                )

                results[run.algorithm] = {
                    "raw": time_series,
                    "smoothed": smoothed,
                }

        return ProcessedMetric(
            name=f"{metric_definition.name}_smoothed",
            data=results,
            metadata={"window_size": self.window_size}
        )
```

---

## Custom Renderers

### Creating a Renderer

```python
from fusion.visualization.infrastructure.renderers.base_renderer import (
    BaseRenderer,
    PlotResult
)
import matplotlib.pyplot as plt

class MyRenderer(BaseRenderer):
    """Custom plot renderer."""

    def render(self, spec: PlotSpecification) -> PlotResult:
        """Render the plot.

        Args:
            spec: Plot specification with data and configuration

        Returns:
            PlotResult with figure and metadata
        """
        # Create figure
        fig, ax = plt.subplots(figsize=spec.figure_size or (10, 6))

        # Plot data
        data = spec.data
        for algo, algo_data in data.items():
            ax.plot(
                algo_data["x"],
                algo_data["y"],
                label=algo,
                linewidth=2
            )

        # Styling
        ax.set_xlabel(spec.x_label or "X Axis", fontsize=12)
        ax.set_ylabel(spec.y_label or "Y Axis", fontsize=12)
        ax.set_title(spec.title or "Plot Title", fontsize=14, fontweight="bold")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save if path provided
        output_path = None
        if spec.save_path:
            output_path = Path(spec.save_path)
            fig.savefig(output_path, dpi=spec.dpi or 300, bbox_inches="tight")

        return PlotResult(
            success=True,
            figure=fig,
            output_path=output_path,
            metadata={"plot_type": "my_custom_plot"}
        )
```

### Example: Heatmap Renderer

```python
import seaborn as sns

class HeatmapRenderer(BaseRenderer):
    """Renderer for heatmap visualizations."""

    def render(self, spec: PlotSpecification) -> PlotResult:
        """Render heatmap."""
        fig, ax = plt.subplots(figsize=spec.figure_size or (10, 8))

        # Get matrix data
        matrix = spec.data.get("matrix", np.array([]))

        if matrix.size > 0:
            # Create heatmap
            sns.heatmap(
                matrix,
                ax=ax,
                cmap=spec.metadata.get("colormap", "viridis"),
                cbar=True,
                annot=spec.metadata.get("annotate", False),
            )

            ax.set_title(spec.title or "Heatmap", fontsize=14, fontweight="bold")

        plt.tight_layout()

        output_path = None
        if spec.save_path:
            output_path = Path(spec.save_path)
            fig.savefig(output_path, dpi=spec.dpi or 300, bbox_inches="tight")

        return PlotResult(
            success=True,
            figure=fig,
            output_path=output_path,
            metadata={"plot_type": "heatmap"}
        )
```

---

## Configuration

### Configuration Schema

```python
def get_config_schema(self) -> Dict:
    """Return JSON schema for plugin configuration."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "My Plugin Configuration",
        "type": "object",
        "properties": {
            "threshold": {
                "type": "number",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Threshold value for analysis"
            },
            "window_size": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Window size for smoothing"
            },
            "enable_feature": {
                "type": "boolean",
                "default": True,
                "description": "Enable advanced feature"
            }
        }
    }
```

### Configuration Validation

```python
def validate_config(self, config: Dict) -> bool:
    """Validate plugin configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    if "required_field" not in config:
        return False

    # Validate ranges
    if "threshold" in config:
        if not 0.0 <= config["threshold"] <= 1.0:
            return False

    # Validate types
    if "window_size" in config:
        if not isinstance(config["window_size"], int):
            return False

    return True
```

### Using Configuration

```yaml
# config.yml
plugins:
  myplugin:
    enabled: true
    threshold: 0.75
    window_size: 150
    enable_feature: false
```

---

## Testing

### Plugin Unit Tests

```python
# tests/test_myplugin.py

import pytest
from fusion.modules.mymodule.visualization import MyModulePlugin

class TestMyModulePlugin:
    """Tests for MyModule plugin."""

    def test_plugin_creation(self):
        """Test plugin can be created."""
        plugin = MyModulePlugin()
        assert plugin.name == "mymodule"
        assert plugin.version == "1.0.0"

    def test_plugin_registers_metrics(self):
        """Test plugin registers expected metrics."""
        plugin = MyModulePlugin()
        metrics = plugin.register_metrics()

        assert len(metrics) > 0
        metric_names = {m.name for m in metrics}
        assert "my_metric" in metric_names

    def test_plugin_registers_plot_types(self):
        """Test plugin registers plot types."""
        plugin = MyModulePlugin()
        plot_types = plugin.register_plot_types()

        assert "my_plot" in plot_types

    def test_config_validation(self):
        """Test configuration validation."""
        plugin = MyModulePlugin()

        # Valid config
        assert plugin.validate_config({"threshold": 0.5})

        # Invalid config
        assert not plugin.validate_config({"threshold": 1.5})
```

### Integration Tests

```python
from fusion.visualization.plugins import PluginRegistry

def test_plugin_loads_in_registry():
    """Test plugin loads successfully in registry."""
    registry = PluginRegistry()
    registry.register_plugin_class(MyModulePlugin)
    registry.load_plugin("mymodule")

    assert registry.is_loaded("mymodule")

    # Check metrics accessible
    metrics = registry.get_all_metrics()
    assert len(metrics) > 0
```

---

## Best Practices

### 1. Naming Conventions

```python
# Plugin name: lowercase, no spaces
@property
def name(self) -> str:
    return "mymodule"  # ✓ Good
    # return "My Module"  # ✗ Bad

# Metric names: snake_case
MetricDefinition(name="request_latency")  # ✓ Good
# MetricDefinition(name="RequestLatency")  # ✗ Bad
```

### 2. Documentation

```python
class MyPlugin(BasePlugin):
    """Short one-line description.

    Longer description explaining:
    - What the plugin does
    - What metrics it provides
    - What plot types it provides
    - Any special requirements

    Example:
        registry = PluginRegistry()
        registry.register_plugin_class(MyPlugin)
        registry.load_plugin("myplugin")
    """
```

### 3. Error Handling

```python
def register_metrics(self):
    """Register metrics with error handling."""
    try:
        metrics = self._create_metrics()
        self._validate_metrics(metrics)
        return metrics
    except Exception as e:
        logger.error(f"Failed to register metrics: {e}")
        return []  # Return empty list on failure
```

### 4. Performance

```python
# Cache expensive computations
@property
def metrics(self):
    if not hasattr(self, "_metrics_cache"):
        self._metrics_cache = self._compute_metrics()
    return self._metrics_cache
```

### 5. Dependencies

```python
# Declare dependencies explicitly
@property
def requires(self) -> List[str]:
    return ["base_plugin"]  # Required plugins

def _check_dependencies(self):
    """Check external package dependencies."""
    try:
        import required_package  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Plugin requires 'required_package'. "
            f"Install with: pip install required_package"
        )
```

---

## Examples

See the following for complete examples:
- **RL Plugin:** `fusion/modules/rl/visualization/rl_plugin.py`
- **Spectrum Plugin:** `fusion/modules/spectrum/visualization/spectrum_plugin.py`
- **SNR Plugin:** `fusion/modules/snr/visualization/snr_plugin.py`
- **Routing Plugin:** `fusion/modules/routing/visualization/routing_plugin.py`

---

## Support

- **Documentation:** `docs/visualization/`
- **Examples:** `fusion/visualization/examples/`
- **Issues:** GitHub Issues (label: `visualization`, `plugin`)
- **Questions:** GitHub Discussions

---

**Last updated:** 2025-10-02
**Document version:** 1.0
**FUSION version:** 6.0.0
