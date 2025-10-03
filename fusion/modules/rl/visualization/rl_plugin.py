"""RL visualization plugin.

This plugin provides comprehensive visualization support for reinforcement
learning experiments including:
- Training metrics (rewards, losses, Q-values)
- Convergence analysis
- Policy/value visualization
- Multi-metric dashboards
"""

from fusion.modules.rl.visualization.rl_metrics import get_rl_metrics
from fusion.modules.rl.visualization.rl_plots import (
    ConvergencePlotRenderer,
    MultiMetricDashboardRenderer,
    QValueHeatmapRenderer,
    RewardLearningCurveRenderer,
)
from fusion.modules.rl.visualization.rl_processors import (
    ConvergenceDetectionStrategy,
    QValueProcessingStrategy,
    RewardProcessingStrategy,
)
from fusion.visualization.domain.entities.metric import MetricDefinition
from fusion.visualization.domain.strategies.processing_strategies import (
    MetricProcessingStrategy,
)
from fusion.visualization.infrastructure.renderers.base_renderer import BaseRenderer
from fusion.visualization.plugins.base_plugin import BasePlugin, PlotTypeRegistration


class RLVisualizationPlugin(BasePlugin):
    """Plugin for RL-specific visualizations.

    This plugin extends the FUSION visualization system with:
    - RL training metrics (rewards, TD errors, Q-values, policy entropy)
    - Specialized plot types (learning curves, convergence analysis, dashboards)
    - RL-specific processing strategies (reward smoothing, convergence detection)

    Example:
        # Plugin is auto-discovered and loaded
        from fusion.visualization.plugins import get_global_registry

        registry = get_global_registry()
        registry.discover_plugins()
        registry.load_plugin("rl")

        # Use RL plot types
        result = generate_plot(
            config_path="rl_config.yml",
            plot_type="reward_learning_curve"
        )
    """

    @property
    def name(self) -> str:
        """Return plugin name."""
        return "rl"

    @property
    def version(self) -> str:
        """Return plugin version."""
        return "1.0.0"

    @property
    def description(self) -> str:
        """Return plugin description."""
        return "Visualization plugin for Reinforcement Learning experiments"

    @property
    def author(self) -> str:
        """Return plugin author."""
        return "FUSION RL Team"

    def _check_dependencies(self) -> None:
        """Check plugin dependencies."""
        # Check for required packages
        try:
            import scipy  # noqa: F401
            import seaborn  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"RL visualization plugin requires scipy and seaborn: {e}"
            ) from e

    def register_metrics(self) -> list[MetricDefinition]:
        """Register RL-specific metrics.

        Returns:
            List of RL metric definitions
        """
        return get_rl_metrics()

    def register_plot_types(self) -> dict[str, PlotTypeRegistration]:
        """Register RL-specific plot types.

        Returns:
            Dictionary of plot type registrations
        """
        return {
            # New DDD-based plot types
            "reward_learning_curve": PlotTypeRegistration(
                processor=RewardProcessingStrategy(window_size=100),
                renderer=RewardLearningCurveRenderer(),
                description=(
                    "Learning curve showing episode rewards over training "
                    "with smoothing and confidence intervals"
                ),
                required_metrics=["episode_reward"],
                default_config={
                    "window_size": 100,
                    "confidence_level": 0.95,
                    "show_ci": True,
                },
            ),
            "q_value_heatmap": PlotTypeRegistration(
                processor=QValueProcessingStrategy(),
                renderer=QValueHeatmapRenderer(),
                description="Heatmap visualization of Q-values across states/actions",
                required_metrics=["q_values"],
                default_config={"colormap": "viridis", "annotate": False},
            ),
            "convergence_plot": PlotTypeRegistration(
                processor=ConvergenceDetectionStrategy(window_size=100, threshold=0.01),
                renderer=ConvergencePlotRenderer(),
                description=(
                    "Training convergence analysis showing when metrics stabilize"
                ),
                required_metrics=["episode_reward"],
                default_config={
                    "window_size": 100,
                    "threshold": 0.01,
                    "show_convergence_point": True,
                },
            ),
            "rl_dashboard": PlotTypeRegistration(
                processor=RewardProcessingStrategy(window_size=100),
                renderer=MultiMetricDashboardRenderer(),
                description=(
                    "Comprehensive dashboard showing multiple RL training metrics"
                ),
                required_metrics=[
                    "episode_reward",
                    "policy_loss",
                    "value_loss",
                    "policy_entropy",
                    "q_value_mean",
                ],
                default_config={"layout": "3x2", "window_size": 100},
            ),
        }

    def get_config_schema(self) -> dict:
        """Get configuration schema for this plugin.

        Returns:
            JSON schema for plugin configuration
        """
        return {
            "type": "object",
            "properties": {
                "window_size": {
                    "type": "integer",
                    "default": 100,
                    "description": "Smoothing window size for learning curves",
                },
                "confidence_level": {
                    "type": "number",
                    "default": 0.95,
                    "description": "Confidence level for intervals",
                },
                "colormap": {
                    "type": "string",
                    "default": "viridis",
                    "description": "Colormap for heatmaps",
                },
            },
        }

    def validate_config(self, config: dict) -> bool:
        """Validate plugin configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        # Validate window_size
        if "window_size" in config:
            window_size = config["window_size"]
            if not isinstance(window_size, int) or window_size <= 0:
                return False

        # Validate confidence_level
        if "confidence_level" in config:
            confidence_level = config["confidence_level"]
            if not isinstance(confidence_level, (int, float)) or not (
                0 < confidence_level < 1
            ):
                return False

        return True

    def register_processors(self) -> dict[str, MetricProcessingStrategy]:
        """Register RL-specific data processors.

        Returns:
            Dictionary of processor instances
        """
        return {
            "reward_processing": RewardProcessingStrategy(),
            "q_value_processing": QValueProcessingStrategy(),
            "convergence_detection": ConvergenceDetectionStrategy(),
        }

    def register_renderers(self) -> dict[str, type[BaseRenderer]]:
        """Register RL-specific renderers.

        Returns:
            Dictionary of renderer classes
        """
        return {
            "reward_learning_curve": RewardLearningCurveRenderer,
            "q_value_heatmap": QValueHeatmapRenderer,
            "convergence_plot": ConvergencePlotRenderer,
            "multi_metric_dashboard": MultiMetricDashboardRenderer,
        }


# Singleton instance for auto-discovery
_plugin_instance = RLVisualizationPlugin()


def get_plugin() -> RLVisualizationPlugin:
    """Return the singleton plugin instance for auto-discovery."""
    return _plugin_instance
