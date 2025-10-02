"""Base plugin interface for visualization extensions.

This module provides the base class for creating visualization plugins that
extend the FUSION visualization system with module-specific metrics and plot types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

from fusion.visualization.domain.entities.metric import MetricDefinition
from fusion.visualization.domain.strategies.processing_strategies import MetricProcessingStrategy
from fusion.visualization.infrastructure.renderers.base_renderer import BaseRenderer


@dataclass
class PlotTypeRegistration:
    """Registration information for a custom plot type."""

    processor: MetricProcessingStrategy
    renderer: BaseRenderer
    description: str
    required_metrics: List[str]
    default_config: Optional[Dict] = None

    def __post_init__(self) -> None:
        """Validate registration."""
        if not self.description:
            raise ValueError("Description is required for plot type registration")
        if not self.required_metrics:
            raise ValueError("At least one required metric must be specified")


class BasePlugin(ABC):
    """Abstract base class for visualization plugins.

    Plugins extend the visualization system by providing:
    - Custom metric definitions
    - Custom plot types
    - Module-specific processing strategies

    Example:
        class MyPlugin(BasePlugin):
            @property
            def name(self) -> str:
                return "my_module"

            def register_metrics(self) -> List[MetricDefinition]:
                return [
                    MetricDefinition(name="my_metric", ...)
                ]

            def register_plot_types(self) -> Dict[str, PlotTypeRegistration]:
                return {
                    "my_plot": PlotTypeRegistration(...)
                }
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the plugin name (unique identifier)."""
        pass

    @property
    def version(self) -> str:
        """Return the plugin version."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Return a description of the plugin."""
        return f"Visualization plugin for {self.name}"

    @property
    def author(self) -> str:
        """Return the plugin author."""
        return "FUSION Team"

    @property
    def requires(self) -> List[str]:
        """Return list of required plugin dependencies."""
        return []

    def is_available(self) -> bool:
        """Check if plugin is available (dependencies satisfied)."""
        try:
            self._check_dependencies()
            return True
        except ImportError:
            return False

    def _check_dependencies(self) -> None:
        """Check plugin dependencies. Raise ImportError if not satisfied."""
        pass

    def register_metrics(self) -> List[MetricDefinition]:
        """Register custom metrics provided by this plugin.

        Returns:
            List of metric definitions
        """
        return []

    def register_plot_types(self) -> Dict[str, PlotTypeRegistration]:
        """Register custom plot types provided by this plugin.

        Returns:
            Dictionary mapping plot type name to registration info
        """
        return {}

    def register_processors(self) -> Dict[str, MetricProcessingStrategy]:
        """Register custom processing strategies.

        Returns:
            Dictionary mapping processor name to strategy instance
        """
        return {}

    def register_renderers(self) -> Dict[str, Type[BaseRenderer]]:
        """Register custom renderer types.

        Returns:
            Dictionary mapping renderer name to renderer class
        """
        return {}

    def on_load(self) -> None:
        """Called when plugin is loaded. Override for initialization."""
        pass

    def on_unload(self) -> None:
        """Called when plugin is unloaded. Override for cleanup."""
        pass

    def get_config_schema(self) -> Optional[Dict]:
        """Return JSON schema for plugin-specific configuration.

        Returns:
            JSON schema dict or None
        """
        return None

    def validate_config(self, config: Dict) -> bool:
        """Validate plugin-specific configuration.

        Args:
            config: Plugin configuration dictionary

        Returns:
            True if valid, False otherwise
        """
        return True

    def get_examples_dir(self) -> Optional[Path]:
        """Return path to plugin examples directory.

        Returns:
            Path to examples or None
        """
        return None

    def __repr__(self) -> str:
        """String representation of plugin."""
        return f"{self.__class__.__name__}(name={self.name}, version={self.version})"
