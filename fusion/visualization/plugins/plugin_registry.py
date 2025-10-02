"""Plugin registry for discovering and managing visualization plugins.

This module provides the PluginRegistry class that handles:
- Auto-discovery of plugins
- Plugin loading and initialization
- Plugin dependency resolution
- Access to registered metrics and plot types
"""

import importlib
import logging
from pathlib import Path

from fusion.visualization.domain.entities.metric import MetricDefinition
from fusion.visualization.plugins.base_plugin import BasePlugin, PlotTypeRegistration

logger = logging.getLogger(__name__)


class PluginLoadError(Exception):
    """Raised when a plugin fails to load."""

    pass


class PluginDependencyError(Exception):
    """Raised when plugin dependencies cannot be satisfied."""

    pass


class PluginRegistry:
    """Registry for visualization plugins.

    The registry provides:
    - Auto-discovery of plugins from specified directories
    - Lazy loading of plugins
    - Dependency resolution
    - Access to aggregated metrics and plot types

    Example:
        registry = PluginRegistry()
        registry.discover_plugins()
        registry.load_all()

        # Get all registered metrics
        metrics = registry.get_all_metrics()

        # Get specific plot type
        plot_reg = registry.get_plot_type("blocking")
    """

    def __init__(self) -> None:
        """Initialize the plugin registry."""
        self._plugins: dict[str, BasePlugin] = {}
        self._loaded: set[str] = set()
        self._failed: dict[str, str] = {}

    def register_plugin(self, plugin: BasePlugin) -> None:
        """Register a plugin instance.

        Args:
            plugin: Plugin instance to register

        Raises:
            ValueError: If plugin with same name already registered
        """
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin '{plugin.name}' already registered")

        logger.info(f"Registering plugin: {plugin.name} v{plugin.version}")
        self._plugins[plugin.name] = plugin

    def register_plugin_class(self, plugin_class: type[BasePlugin]) -> None:
        """Register a plugin class (will be instantiated).

        Args:
            plugin_class: Plugin class to register
        """
        plugin = plugin_class()
        self.register_plugin(plugin)

    def discover_plugins(self, search_paths: list[Path] | None = None) -> None:
        """Discover plugins from specified paths.

        Args:
            search_paths: List of directories to search for plugins.
                         If None, searches default locations.
        """
        if search_paths is None:
            search_paths = self._get_default_search_paths()

        logger.info(f"Discovering plugins in {len(search_paths)} paths")

        for path in search_paths:
            if not path.exists():
                logger.warning(f"Plugin search path does not exist: {path}")
                continue

            self._discover_in_path(path)

    def _get_default_search_paths(self) -> list[Path]:
        """Get default plugin search paths."""
        paths = []

        # Look for module-specific visualization directories
        fusion_root = Path(__file__).parent.parent.parent
        modules_dir = fusion_root / "modules"

        if modules_dir.exists():
            # Search for visualization subdirectories in each module
            for module_dir in modules_dir.iterdir():
                if module_dir.is_dir():
                    viz_dir = module_dir / "visualization"
                    if viz_dir.exists():
                        paths.append(viz_dir)

        return paths

    def _discover_in_path(self, path: Path) -> None:
        """Discover plugins in a specific path.

        Args:
            path: Directory to search for plugins
        """
        # Look for *_plugin.py files
        for plugin_file in path.glob("*_plugin.py"):
            try:
                self._load_plugin_file(plugin_file)
            except Exception as e:
                logger.error(f"Failed to load plugin from {plugin_file}: {e}")
                self._failed[plugin_file.stem] = str(e)

    def _load_plugin_file(self, plugin_file: Path) -> None:
        """Load a plugin from a Python file.

        Args:
            plugin_file: Path to plugin Python file
        """
        # Convert file path to module path
        module_name = self._file_to_module_name(plugin_file)

        try:
            module = importlib.import_module(module_name)

            # Look for plugin classes
            for name in dir(module):
                obj = getattr(module, name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, BasePlugin)
                    and obj is not BasePlugin
                ):
                    try:
                        self.register_plugin_class(obj)
                        logger.info(f"Loaded plugin class: {name} from {module_name}")
                    except Exception as e:
                        logger.error(f"Failed to register plugin class {name}: {e}")

        except ImportError as e:
            logger.error(f"Failed to import plugin module {module_name}: {e}")
            raise PluginLoadError(f"Import failed: {e}") from e

    def _file_to_module_name(self, plugin_file: Path) -> str:
        """Convert a file path to a module name.

        Args:
            plugin_file: Path to Python file

        Returns:
            Module name (e.g., "fusion.modules.rl.visualization.rl_plugin")
        """
        # Find the fusion package root
        parts = plugin_file.parts
        try:
            fusion_idx = parts.index("fusion")
            module_parts = parts[fusion_idx:-1] + (plugin_file.stem,)
            return ".".join(module_parts)
        except ValueError as exc:
            raise ValueError(
                f"Could not determine module name for {plugin_file}"
            ) from exc

    def load_plugin(self, name: str) -> None:
        """Load a specific plugin by name.

        Args:
            name: Plugin name

        Raises:
            KeyError: If plugin not found
            PluginLoadError: If plugin fails to load
        """
        if name in self._loaded:
            logger.debug(f"Plugin '{name}' already loaded")
            return

        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' not found in registry")

        plugin = self._plugins[name]

        # Check dependencies
        self._resolve_dependencies(plugin)

        # Check availability
        if not plugin.is_available():
            raise PluginLoadError(
                f"Plugin '{name}' is not available (dependencies not satisfied)"
            )

        # Load the plugin
        try:
            plugin.on_load()
            self._loaded.add(name)
            logger.info(f"Loaded plugin: {name}")
        except Exception as e:
            raise PluginLoadError(f"Failed to load plugin '{name}': {e}") from e

    def _resolve_dependencies(self, plugin: BasePlugin) -> None:
        """Resolve and load plugin dependencies.

        Args:
            plugin: Plugin to resolve dependencies for

        Raises:
            PluginDependencyError: If dependencies cannot be satisfied
        """
        for dep_name in plugin.requires:
            if dep_name not in self._plugins:
                raise PluginDependencyError(
                    f"Plugin '{plugin.name}' requires '{dep_name}' "
                    f"which is not registered"
                )

            # Load dependency if not already loaded
            if dep_name not in self._loaded:
                try:
                    self.load_plugin(dep_name)
                except PluginLoadError as e:
                    raise PluginDependencyError(
                        f"Failed to load dependency '{dep_name}' "
                        f"for '{plugin.name}': {e}"
                    ) from e

    def load_all(self) -> None:
        """Load all registered plugins."""
        for name in list(self._plugins.keys()):
            if name not in self._loaded:
                try:
                    self.load_plugin(name)
                except (PluginLoadError, PluginDependencyError) as e:
                    logger.error(f"Failed to load plugin '{name}': {e}")
                    self._failed[name] = str(e)

    def unload_plugin(self, name: str) -> None:
        """Unload a specific plugin.

        Args:
            name: Plugin name

        Raises:
            KeyError: If plugin not found
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' not found")

        if name in self._loaded:
            plugin = self._plugins[name]
            try:
                plugin.on_unload()
            except Exception as e:
                logger.error(f"Error unloading plugin '{name}': {e}")
            finally:
                self._loaded.remove(name)

    def get_plugin(self, name: str) -> BasePlugin | None:
        """Get a plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(name)

    def get_all_plugins(self) -> dict[str, BasePlugin]:
        """Get all registered plugins.

        Returns:
            Dictionary mapping plugin name to plugin instance
        """
        return self._plugins.copy()

    def get_loaded_plugins(self) -> list[str]:
        """Get list of loaded plugin names.

        Returns:
            List of loaded plugin names
        """
        return list(self._loaded)

    def get_all_metrics(self) -> list[MetricDefinition]:
        """Get all metrics from all loaded plugins.

        Returns:
            List of all metric definitions
        """
        metrics = []
        for name in self._loaded:
            plugin = self._plugins[name]
            metrics.extend(plugin.register_metrics())
        return metrics

    def get_metric(self, metric_name: str) -> MetricDefinition | None:
        """Get a specific metric by name.

        Args:
            metric_name: Name of the metric

        Returns:
            MetricDefinition or None if not found
        """
        for metric in self.get_all_metrics():
            if metric.name == metric_name:
                return metric
        return None

    def get_all_plot_types(self) -> dict[str, PlotTypeRegistration]:
        """Get all plot types from all loaded plugins.

        Returns:
            Dictionary mapping plot type name to registration
        """
        plot_types = {}
        for name in self._loaded:
            plugin = self._plugins[name]
            plot_types.update(plugin.register_plot_types())
        return plot_types

    def get_plot_type(self, plot_type_name: str) -> PlotTypeRegistration | None:
        """Get a specific plot type registration.

        Args:
            plot_type_name: Name of the plot type

        Returns:
            PlotTypeRegistration or None if not found
        """
        return self.get_all_plot_types().get(plot_type_name)

    def is_loaded(self, name: str) -> bool:
        """Check if a plugin is loaded.

        Args:
            name: Plugin name

        Returns:
            True if loaded, False otherwise
        """
        return name in self._loaded

    def get_failed_plugins(self) -> dict[str, str]:
        """Get plugins that failed to load.

        Returns:
            Dictionary mapping plugin name to error message
        """
        return self._failed.copy()


# Global plugin registry instance
_global_registry: PluginRegistry | None = None


def get_global_registry() -> PluginRegistry:
    """Get the global plugin registry instance.

    Returns:
        Global PluginRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_global_registry() -> None:
    """Reset the global plugin registry (mainly for testing)."""
    global _global_registry
    _global_registry = None
