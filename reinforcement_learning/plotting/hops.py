from reinforcement_learning.plotting.transponders import _plot_metric
from reinforcement_learning.plotting.transponders import _plot_metric_minmax


def plot_hops(data, save_path=None, title="Hops"):
    """
    Plot hops.
    """
    _plot_metric(data, "Average # Hops", title, save_path)


def plot_hops_minmax(data, save_path=None, title="Hops – min vs max"):
    """
    Plot min max hops.
    """
    _plot_metric_minmax(data, "# Hops", title, save_path)
