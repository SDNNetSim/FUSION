"""Unit tests for BasePlugin class."""

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
from fusion.visualization.infrastructure.renderers.base_renderer import (
    BaseRenderer,
)
from fusion.visualization.plugins.base_plugin import BasePlugin, PlotTypeRegistration


class MockRenderer(BaseRenderer):
    """Mock renderer for testing."""

    def render(
        self,
        specification: PlotSpecification,
        output_path: Path,
        dpi: int = 300,
        format: str = "png",
    ) -> RenderResult:
        """Mock render method."""
        return RenderResult(success=True, output_path=output_path)

    def supports_format(self, format: str) -> bool:
        """Mock supports_format method."""
        return format in ["png", "pdf", "svg"]

    def get_supported_formats(self) -> list[str]:
        """Mock get_supported_formats method."""
        return ["png", "pdf", "svg"]


class TestPlotTypeRegistration:
    """Tests for PlotTypeRegistration."""

    def test_valid_registration(self) -> None:
        """Test creating valid plot type registration."""
        registration = PlotTypeRegistration(
            processor=GenericMetricProcessingStrategy(),
            renderer=MockRenderer(),
            description="Test plot",
            required_metrics=["metric1"],
        )

        assert registration.description == "Test plot"
        assert registration.required_metrics == ["metric1"]

    def test_registration_requires_description(self) -> None:
        """Test that registration requires description."""
        with pytest.raises(ValueError, match="Description is required"):
            PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=MockRenderer(),
                description="",
                required_metrics=["metric1"],
            )

    def test_registration_requires_metrics(self) -> None:
        """Test that registration requires at least one metric."""
        with pytest.raises(ValueError, match="At least one required metric"):
            PlotTypeRegistration(
                processor=GenericMetricProcessingStrategy(),
                renderer=MockRenderer(),
                description="Test",
                required_metrics=[],
            )


class TestBasePlugin:
    """Tests for BasePlugin abstract class."""

    def test_concrete_plugin_creation(self) -> None:
        """Test creating a concrete plugin."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test_plugin"

        plugin = TestPlugin()
        assert plugin.name == "test_plugin"
        assert plugin.version == "1.0.0"  # default
        assert plugin.author == "FUSION Team"  # default

    def test_plugin_name_required(self) -> None:
        """Test that plugin must implement name property."""

        class InvalidPlugin(BasePlugin):
            pass

        # Cannot instantiate abstract class without implementing name
        with pytest.raises(TypeError):
            InvalidPlugin()  # type: ignore[abstract]

    def test_plugin_custom_version(self) -> None:
        """Test plugin with custom version."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def version(self) -> str:
                return "2.0.0"

        plugin = TestPlugin()
        assert plugin.version == "2.0.0"

    def test_plugin_custom_description(self) -> None:
        """Test plugin with custom description."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "Custom description"

        plugin = TestPlugin()
        assert plugin.description == "Custom description"

    def test_plugin_is_available_default(self) -> None:
        """Test that plugin is available by default."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        plugin = TestPlugin()
        assert plugin.is_available()

    def test_plugin_with_dependency_check(self) -> None:
        """Test plugin with dependency checking."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            def _check_dependencies(self) -> None:
                import nonexistent_module  # noqa: F401

        plugin = TestPlugin()
        assert not plugin.is_available()

    def test_plugin_register_metrics_default(self) -> None:
        """Test that plugins return empty metrics by default."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        plugin = TestPlugin()
        assert plugin.register_metrics() == []

    def test_plugin_register_metrics_custom(self) -> None:
        """Test plugin registering custom metrics."""

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

        plugin = TestPlugin()
        metrics = plugin.register_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "test_metric"

    def test_plugin_register_plot_types_default(self) -> None:
        """Test that plugins return empty plot types by default."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        plugin = TestPlugin()
        assert plugin.register_plot_types() == {}

    def test_plugin_lifecycle_hooks(self) -> None:
        """Test plugin lifecycle hooks."""

        class TestPlugin(BasePlugin):
            def __init__(self) -> None:
                super().__init__()
                self.loaded = False
                self.unloaded = False

            @property
            def name(self) -> str:
                return "test"

            def on_load(self) -> None:
                self.loaded = True

            def on_unload(self) -> None:
                self.unloaded = True

        plugin = TestPlugin()
        assert not plugin.loaded
        assert not plugin.unloaded

        plugin.on_load()
        assert plugin.loaded

        plugin.on_unload()
        assert plugin.unloaded

    def test_plugin_get_config_schema_default(self) -> None:
        """Test default config schema is None."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        plugin = TestPlugin()
        assert plugin.get_config_schema() is None

    def test_plugin_validate_config_default(self) -> None:
        """Test default config validation returns True."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        plugin = TestPlugin()
        assert plugin.validate_config({"any": "config"})

    def test_plugin_custom_config_validation(self) -> None:
        """Test plugin with custom config validation."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            def validate_config(self, config: dict) -> bool:
                return "required_field" in config

        plugin = TestPlugin()
        assert not plugin.validate_config({})
        assert plugin.validate_config({"required_field": "value"})

    def test_plugin_repr(self) -> None:
        """Test plugin string representation."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def version(self) -> str:
                return "1.5.0"

        plugin = TestPlugin()
        repr_str = repr(plugin)
        assert "TestPlugin" in repr_str
        assert "test" in repr_str
        assert "1.5.0" in repr_str

    def test_plugin_requires_default(self) -> None:
        """Test plugin requires returns empty list by default."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

        plugin = TestPlugin()
        assert plugin.requires == []

    def test_plugin_with_requirements(self) -> None:
        """Test plugin with dependency requirements."""

        class TestPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "test"

            @property
            def requires(self) -> list[str]:
                return ["dep1", "dep2"]

        plugin = TestPlugin()
        assert plugin.requires == ["dep1", "dep2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
