# in lengths.py
from reinforcement_learning.plotting.transponders import _plot_metric
from reinforcement_learning.plotting.transponders import _plot_metric_minmax


def plot_lengths(data, save_path=None, title="Path Lengths (km)"):
    _plot_metric(data, "Average Length (km)", title, save_path)


def plot_lengths_minmax(data, save_path=None, title="Lengths – min vs max"):
    _plot_metric_minmax(data, "Length (km)", title, save_path)
