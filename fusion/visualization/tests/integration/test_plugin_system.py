"""Integration tests for the plugin system.

This module tests:
- Plugin discovery and loading
- Plugin registration
- Metric and plot type registration
- Plugin dependencies
- Plugin configuration validation
"""

from collections.abc import Generator
from pathlib import Path

import pytest

from fusion.visualization.application.ports.plot_renderer_port import RenderResult
from fusion.visualization.domain.entities.metric import (
    AggregationStrategy,
    DataType,
    MetricDefinition,
)
from fusion.visualization.domain.strategies.processing_strategies import (
    GenericMetricProcessingStrategy,
)
from fusion.visualization.domain.value_objects.plot_specification import (
    PlotSpecification,
)
from fusion.visualization.infrastructure.renderers.base_renderer import BaseRenderer
from fusion.visualization.plugins import (
    BasePlugin,
    PlotTypeRegistration,
    PluginRegistry,
    get_global_registry,
    reset_global_registry,
)


class TestPluginRegistry:
    """Tests for PluginRegistry functionality."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset global registry before each test."""
        reset_global_registry()
        yield
        reset_global_registry()

    def test_registry_singleton(self) -> None:
        """Test that get_global_registry returns singleton."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()
        assert registry1 is registry2

    def test_register_plugin(self) -> None:
        """Test registering a plugin instance."""
        registry = PluginRegistry()

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        plugin = TestPlugin()
        registry.register_plugin(plugin)

        assert registry.get_plugin("test") is plugin

    def test_duplicate_plugin_registration_raises(self) -> None:
        """Test that registering duplicate plugin raises error."""
        registry = PluginRegistry()

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        registry.register_plugin(TestPlugin())

        with pytest.raises(ValueError, match="already registered"):
            registry.register_plugin(TestPlugin())

    def test_register_plugin_class(self) -> None:
        """Test registering a plugin class."""
        registry = PluginRegistry()

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        registry.register_plugin_class(TestPlugin)

        plugin = registry.get_plugin("test")
        assert plugin is not None
        assert isinstance(plugin, TestPlugin)

    def test_load_plugin(self) -> None:  # type: ignore[misc]
        """Test loading a plugin."""
        registry = PluginRegistry()

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            def __init__(self) -> None:
                super().__init__()
                self.loaded = False

            def on_load(self) -> None:
                self.loaded = True

        registry.register_plugin_class(TestPlugin)
        registry.load_plugin("test")

        assert registry.is_loaded("test")
        plugin = registry.get_plugin("test")
        assert plugin is not None
        assert hasattr(plugin, "loaded")
        assert plugin.loaded  # type: ignore[attr-defined]

    def test_load_nonexistent_plugin_raises(self) -> None:
        """Test that loading nonexistent plugin raises KeyError."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.load_plugin("nonexistent")

    def test_unload_plugin(self) -> None:  # type: ignore[misc]
        """Test unloading a plugin."""
        registry = PluginRegistry()

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            def __init__(self) -> None:
                super().__init__()
                self.unloaded = False

            def on_unload(self) -> None:
                self.unloaded = True

        registry.register_plugin_class(TestPlugin)
        registry.load_plugin("test")
        registry.unload_plugin("test")

        assert not registry.is_loaded("test")
        plugin = registry.get_plugin("test")
        assert plugin is not None
        assert hasattr(plugin, "unloaded")
        assert plugin.unloaded  # type: ignore[attr-defined]

    def test_get_all_plugins(self) -> None:
        """Test getting all registered plugins."""
        registry = PluginRegistry()

        class TestPlugin1(BasePlugin):
            @property
            def name(self) -> str:
                return "test1"

        class TestPlugin2(BasePlugin):
            @property
            def name(self) -> str:
                return "test2"

        registry.register_plugin_class(TestPlugin1)
        registry.register_plugin_class(TestPlugin2)

        plugins = registry.get_all_plugins()
        assert len(plugins) == 2
        assert "test1" in plugins
        assert "test2" in plugins

    def test_get_loaded_plugins(self) -> None:
        """Test getting loaded plugins only."""
        registry = PluginRegistry()

        class TestPlugin1(BasePlugin):
            @property
            def name(self) -> str:
                return "test1"

        class TestPlugin2(BasePlugin):
            @property
            def name(self) -> str:
                return "test2"

        registry.register_plugin_class(TestPlugin1)
        registry.register_plugin_class(TestPlugin2)
        registry.load_plugin("test1")

        loaded = registry.get_loaded_plugins()
        assert len(loaded) == 1
        assert "test1" in loaded
        assert "test2" not in loaded


class TestPluginMetrics:
    """Tests for plugin metric registration."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset global registry before each test."""
        reset_global_registry()
        yield
        reset_global_registry()

    def test_plugin_registers_metrics(self) -> None:
        """Test that plugin can register metrics."""
        registry = PluginRegistry()

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            def register_metrics(self) -> list[MetricDefinition]:
                return [
                    MetricDefinition(
                        name="test_metric",
                        data_type=DataType.FLOAT,
                        source_path="$.test",
                        aggregation=AggregationStrategy.MEAN,
                    )
                ]

        registry.register_plugin_class(TestPlugin)
        registry.load_plugin("test")

        metrics = registry.get_all_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "test_metric"

    def test_get_specific_metric(self) -> None:
        """Test getting a specific metric by name."""
        registry = PluginRegistry()

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            def register_metrics(self) -> list[MetricDefinition]:
                return [
                    MetricDefinition(
                        name="metric1",
                        data_type=DataType.FLOAT,
                        source_path="$.m1",
                        aggregation=AggregationStrategy.MEAN,
                    ),
                    MetricDefinition(
                        name="metric2",
                        data_type=DataType.INT,
                        source_path="$.m2",
                        aggregation=AggregationStrategy.SUM,
                    ),
                ]

        registry.register_plugin_class(TestPlugin)
        registry.load_plugin("test")

        metric = registry.get_metric("metric2")
        assert metric is not None
        assert metric.name == "metric2"
        assert metric.data_type == DataType.INT

    def test_multiple_plugins_register_metrics(self) -> None:
        """Test that multiple plugins can register metrics."""
        registry = PluginRegistry()

        class Plugin1(BasePlugin):
            @property
            def name(self) -> str:
                return "plugin1"

            def register_metrics(self) -> list[MetricDefinition]:
                return [
                    MetricDefinition(
                        name="metric1",
                        data_type=DataType.FLOAT,
                        source_path="$.m1",
                        aggregation=AggregationStrategy.MEAN,
                    )
                ]

        class Plugin2(BasePlugin):
            @property
            def name(self) -> str:
                return "plugin2"

            def register_metrics(self) -> list[MetricDefinition]:
                return [
                    MetricDefinition(
                        name="metric2",
                        data_type=DataType.INT,
                        source_path="$.m2",
                        aggregation=AggregationStrategy.SUM,
                    )
                ]

        registry.register_plugin_class(Plugin1)
        registry.register_plugin_class(Plugin2)
        registry.load_all()

        metrics = registry.get_all_metrics()
        assert len(metrics) == 2
        metric_names = {m.name for m in metrics}
        assert metric_names == {"metric1", "metric2"}


class TestPluginPlotTypes:
    """Tests for plugin plot type registration."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset global registry before each test."""
        reset_global_registry()
        yield
        reset_global_registry()

    def test_plugin_registers_plot_types(self) -> None:
        """Test that plugin can register plot types."""
        registry = PluginRegistry()

        class DummyRenderer(BaseRenderer):
            def render(
                self,
                specification: PlotSpecification,
                output_path: Path,
                dpi: int = 300,
                format: str = "png",
            ) -> "RenderResult":
                from fusion.visualization.application.ports.plot_renderer_port import (
                    RenderResult,
                )

                return RenderResult(success=True, output_path=output_path)

            def supports_format(self, format: str) -> bool:
                return format in ["png", "pdf"]

            def get_supported_formats(self) -> list[str]:
                return ["png", "pdf"]

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            def register_plot_types(self) -> dict[str, PlotTypeRegistration]:
                return {
                    "test_plot": PlotTypeRegistration(
                        processor=GenericMetricProcessingStrategy(),
                        renderer=DummyRenderer(),
                        description="Test plot type",
                        required_metrics=["test_metric"],
                    )
                }

        registry.register_plugin_class(TestPlugin)
        registry.load_plugin("test")

        plot_types = registry.get_all_plot_types()
        assert len(plot_types) == 1
        assert "test_plot" in plot_types

    def test_get_specific_plot_type(self) -> None:
        """Test getting a specific plot type."""
        registry = PluginRegistry()

        class DummyRenderer(BaseRenderer):
            def render(
                self,
                specification: PlotSpecification,
                output_path: Path,
                dpi: int = 300,
                format: str = "png",
            ) -> "RenderResult":
                from fusion.visualization.application.ports.plot_renderer_port import (
                    RenderResult,
                )

                return RenderResult(success=True, output_path=output_path)

            def supports_format(self, format: str) -> bool:
                return format in ["png", "pdf"]

            def get_supported_formats(self) -> list[str]:
                return ["png", "pdf"]

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            def register_plot_types(self) -> dict[str, PlotTypeRegistration]:
                return {
                    "plot1": PlotTypeRegistration(
                        processor=GenericMetricProcessingStrategy(),
                        renderer=DummyRenderer(),
                        description="Plot 1",
                        required_metrics=["m1"],
                    ),
                    "plot2": PlotTypeRegistration(
                        processor=GenericMetricProcessingStrategy(),
                        renderer=DummyRenderer(),
                        description="Plot 2",
                        required_metrics=["m2"],
                    ),
                }

        registry.register_plugin_class(TestPlugin)
        registry.load_plugin("test")

        plot_type = registry.get_plot_type("plot2")
        assert plot_type is not None
        assert plot_type.description == "Plot 2"
        assert plot_type.required_metrics == ["m2"]


class TestPluginDependencies:
    """Tests for plugin dependency resolution."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset global registry before each test."""
        reset_global_registry()
        yield
        reset_global_registry()

    def test_plugin_with_satisfied_dependency(self) -> None:
        """Test loading plugin with satisfied dependency."""
        registry = PluginRegistry()

        class BasePlugin1(BasePlugin):
            @property
            def name(self) -> str:
                return "base"

        class DependentPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "dependent"

            @property
            def requires(self) -> list[str]:
                return ["base"]

        registry.register_plugin_class(BasePlugin1)
        registry.register_plugin_class(DependentPlugin)

        # Should load dependency automatically
        registry.load_plugin("dependent")

        assert registry.is_loaded("base")
        assert registry.is_loaded("dependent")

    def test_plugin_with_missing_dependency_fails(self) -> None:
        """Test that plugin with missing dependency fails to load."""
        from fusion.visualization.plugins.plugin_registry import PluginDependencyError

        registry = PluginRegistry()

        class DependentPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "dependent"

            @property
            def requires(self) -> list[str]:
                return ["missing"]

        registry.register_plugin_class(DependentPlugin)

        with pytest.raises(PluginDependencyError, match="not registered"):
            registry.load_plugin("dependent")


class TestRealPlugins:
    """Tests for actual FUSION plugins."""

    @pytest.fixture(autouse=True)
    def reset_registry(self) -> Generator[None, None, None]:
        """Reset global registry before each test."""
        reset_global_registry()
        yield
        reset_global_registry()

    def test_rl_plugin_loads(self) -> None:
        """Test that RL plugin loads successfully."""
        from fusion.modules.rl.visualization import RLVisualizationPlugin

        registry = PluginRegistry()
        registry.register_plugin_class(RLVisualizationPlugin)
        registry.load_plugin("rl")

        assert registry.is_loaded("rl")
        plugin = registry.get_plugin("rl")
        assert plugin is not None
        assert plugin.name == "rl"
        assert plugin.version == "1.0.0"

    def test_rl_plugin_metrics(self) -> None:
        """Test that RL plugin registers expected metrics."""
        from fusion.modules.rl.visualization import RLVisualizationPlugin

        registry = PluginRegistry()
        registry.register_plugin_class(RLVisualizationPlugin)
        registry.load_plugin("rl")

        metrics = registry.get_all_metrics()
        metric_names = {m.name for m in metrics}

        # Check for key RL metrics
        expected_metrics = {
            "episode_reward",
            "td_error",
            "q_values",
            "policy_entropy",
        }
        assert expected_metrics.issubset(metric_names)

    def test_rl_plugin_plot_types(self) -> None:
        """Test that RL plugin registers expected plot types."""
        from fusion.modules.rl.visualization import RLVisualizationPlugin

        registry = PluginRegistry()
        registry.register_plugin_class(RLVisualizationPlugin)
        registry.load_plugin("rl")

        plot_types = registry.get_all_plot_types()

        # Check for key RL plot types
        expected_plots = {
            "reward_learning_curve",
            "q_value_heatmap",
            "convergence_plot",
            "rl_dashboard",
        }
        assert expected_plots.issubset(set(plot_types.keys()))

    def test_all_plugins_load(self) -> None:
        """Test that all FUSION plugins load successfully."""
        from fusion.modules.rl.visualization import RLVisualizationPlugin
        from fusion.modules.routing.visualization import RoutingVisualizationPlugin
        from fusion.modules.snr.visualization import SNRVisualizationPlugin
        from fusion.modules.spectrum.visualization import SpectrumVisualizationPlugin

        registry = PluginRegistry()
        registry.register_plugin_class(RLVisualizationPlugin)
        registry.register_plugin_class(SpectrumVisualizationPlugin)
        registry.register_plugin_class(SNRVisualizationPlugin)
        registry.register_plugin_class(RoutingVisualizationPlugin)

        registry.load_all()

        # All should be loaded
        assert registry.is_loaded("rl")
        assert registry.is_loaded("spectrum")
        assert registry.is_loaded("snr")
        assert registry.is_loaded("routing")

    def test_all_plugins_unique_metrics(self) -> None:
        """Test that all plugins register unique metrics."""
        from fusion.modules.rl.visualization import RLVisualizationPlugin
        from fusion.modules.routing.visualization import RoutingVisualizationPlugin
        from fusion.modules.snr.visualization import SNRVisualizationPlugin
        from fusion.modules.spectrum.visualization import SpectrumVisualizationPlugin

        registry = PluginRegistry()
        registry.register_plugin_class(RLVisualizationPlugin)
        registry.register_plugin_class(SpectrumVisualizationPlugin)
        registry.register_plugin_class(SNRVisualizationPlugin)
        registry.register_plugin_class(RoutingVisualizationPlugin)
        registry.load_all()

        metrics = registry.get_all_metrics()
        metric_names = [m.name for m in metrics]

        # Check for uniqueness
        assert len(metric_names) == len(set(metric_names))

    def test_plugin_config_validation(self) -> None:
        """Test plugin configuration validation."""
        from fusion.modules.rl.visualization import RLVisualizationPlugin

        plugin = RLVisualizationPlugin()

        # Valid config
        valid_config = {"window_size": 100, "confidence_level": 0.95}
        assert plugin.validate_config(valid_config)

        # Invalid config (bad window size)
        invalid_config = {"window_size": -1}
        assert not plugin.validate_config(invalid_config)

        # Invalid config (bad confidence level)
        invalid_config2 = {"confidence_level": 1.5}
        assert not plugin.validate_config(invalid_config2)


class TestPluginConfigSchema:
    """Tests for plugin configuration schemas."""

    def test_rl_plugin_config_schema(self) -> None:
        """Test RL plugin configuration schema."""
        from fusion.modules.rl.visualization import RLVisualizationPlugin

        plugin = RLVisualizationPlugin()
        schema = plugin.get_config_schema()

        assert schema is not None
        assert "properties" in schema
        assert "window_size" in schema["properties"]
        assert "confidence_level" in schema["properties"]

    def test_spectrum_plugin_config_schema(self) -> None:
        """Test Spectrum plugin configuration schema."""
        from fusion.modules.spectrum.visualization import SpectrumVisualizationPlugin

        plugin = SpectrumVisualizationPlugin()
        schema = plugin.get_config_schema()

        assert schema is not None
        assert "properties" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
