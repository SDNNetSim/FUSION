"""
FUSION Statistics Package.

This package provides statistics collection and aggregation
for simulation runs, consuming domain objects and producing
metrics compatible with run_comparison.py.
"""

from fusion.stats.collector import StatsCollector

__all__ = ["StatsCollector"]
