"""
Network analysis utilities for FUSION.

Provides functions for analyzing network topology, link usage,
and other network-related metrics.

TODO (legacy migration): This module currently expects the legacy dict format
for network_spectrum (dict mapping (src, dst) tuples to dicts with 'cores_matrix',
'usage_count', 'throughput', etc.). This format is provided by
NetworkState.network_spectrum_dict for backwards compatibility with v5.5.0.

This module should be updated to work directly with NetworkState and LinkSpectrum
objects (v6.1.0) instead of raw dicts. See fusion/domain/network_state.py for
the new data structures.
"""

from typing import Any

import numpy as np

from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


class NetworkAnalyzer:
    """
    Analyzes network topology and usage patterns.

    This class provides utilities for examining network characteristics,
    link utilization, and traffic patterns.
    """

    def __init__(self) -> None:
        """Initialize the network analyzer."""

    @staticmethod
    def get_link_usage_summary(network_spectrum: dict) -> dict[str, dict[str, Any]]:
        """
        Generate a summary of link usage across the network.

        Records usage for each directional link separately to maintain
        bidirectional link statistics.

        :param network_spectrum: Network spectrum database
        :return: Dictionary mapping link identifiers to usage statistics
        """
        usage_summary = {}

        for (src, dst), link_data in network_spectrum.items():
            # Create link key for this direction (src-dst format)
            link_key = f"{src}-{dst}"
            usage_summary[link_key] = {
                "usage_count": link_data.get("usage_count", 0),
                "throughput": link_data.get("throughput", 0),
                "link_num": link_data.get("link_num"),
            }

        logger.debug("Processed %d directional links", len(usage_summary))
        return usage_summary

    @staticmethod
    def analyze_network_congestion(
        network_spectrum: dict, specific_paths: list | None = None
    ) -> dict[str, Any]:
        """
        Analyze network congestion levels.

        :param network_spectrum: Network spectrum database
        :param specific_paths: Optional specific paths to analyze
        :return: Congestion analysis results
        """
        total_occupied_slots = 0
        total_guard_slots = 0
        active_requests = set()
        links_analyzed = 0

        # Skip by two because the link is bidirectional
        for link in list(network_spectrum.keys())[::2]:
            if specific_paths is not None and link not in specific_paths:
                continue

            link_data = network_spectrum[link]
            links_analyzed += 1

            for core in link_data["cores_matrix"]:
                requests = set(core[core > 0])
                active_requests.update(requests)

                total_occupied_slots += len(np.where(core != 0)[0])
                total_guard_slots += len(np.where(core < 0)[0])

        return {
            "total_occupied_slots": total_occupied_slots,
            "total_guard_slots": total_guard_slots,
            "active_requests": len(active_requests),
            "links_analyzed": links_analyzed,
            "avg_occupied_per_link": (
                total_occupied_slots / links_analyzed if links_analyzed > 0 else 0
            ),
            "avg_guard_per_link": (
                total_guard_slots / links_analyzed if links_analyzed > 0 else 0
            ),
        }

    @staticmethod
    def get_network_utilization_stats(network_spectrum: dict) -> dict[str, float]:
        """
        Calculate network-wide utilization statistics.

        :param network_spectrum: Network spectrum database
        :return: Dictionary of utilization statistics
        """
        total_slots = 0
        occupied_slots = 0
        link_utilization_list = []

        # Process each bidirectional link once
        processed_links = set()

        for (src, dst), link_data in network_spectrum.items():
            link_key = f"{min(src, dst)}-{max(src, dst)}"

            if link_key in processed_links:
                continue
            processed_links.add(link_key)

            cores_matrix = link_data.get("cores_matrix", [])

            for core in cores_matrix:
                core_total = len(core)
                core_occupied = len(np.where(core != 0)[0])

                total_slots += core_total
                occupied_slots += core_occupied

                if core_total > 0:
                    link_utilization_list.append(core_occupied / core_total)

        overall_utilization = occupied_slots / total_slots if total_slots > 0 else 0.0

        return {
            "overall_utilization": overall_utilization,
            "average_link_utilization": (
                float(np.mean(link_utilization_list)) if link_utilization_list else 0.0
            ),
            "max_link_utilization": (
                float(np.max(link_utilization_list)) if link_utilization_list else 0.0
            ),
            "min_link_utilization": (
                float(np.min(link_utilization_list)) if link_utilization_list else 0.0
            ),
            "total_slots": total_slots,
            "occupied_slots": occupied_slots,
            "links_processed": len(processed_links),
        }

    @staticmethod
    def identify_bottleneck_links(
        network_spectrum: dict, threshold: float = 0.8
    ) -> list:
        """
        Identify links that are above a utilization threshold.

        :param network_spectrum: Network spectrum database
        :param threshold: Utilization threshold (0.0 to 1.0)
        :return: List of bottleneck link identifiers
        """
        bottleneck_links = []
        processed_links = set()

        for (src, dst), link_data in network_spectrum.items():
            link_key = f"{min(src, dst)}-{max(src, dst)}"

            if link_key in processed_links:
                continue
            processed_links.add(link_key)

            cores_matrix = link_data.get("cores_matrix", [])
            link_utilization = 0.0

            for core in cores_matrix:
                if len(core) > 0:
                    core_utilization = len(np.where(core != 0)[0]) / len(core)
                    link_utilization = max(link_utilization, core_utilization)

            if link_utilization >= threshold:
                bottleneck_links.append(
                    {
                        "link_key": link_key,
                        "utilization": link_utilization,
                        "usage_count": link_data.get("usage_count", 0),
                        "throughput": link_data.get("throughput", 0),
                    }
                )

        # Sort by utilization (highest first)
        bottleneck_links.sort(key=lambda x: x["utilization"], reverse=True)

        logger.info(
            "Identified %d bottleneck links above %.1%% utilization",
            len(bottleneck_links),
            threshold * 100,
        )
        return bottleneck_links
