"""
Machine learning metrics collection for FUSION.

Handles collection and management of training data for ML models.
"""

import os
from typing import Any

import numpy as np
import pandas as pd

from fusion.sim.utils import find_core_cong, find_path_len
from fusion.utils.logging_config import get_logger
from fusion.utils.os import find_project_root

PROJECT_ROOT = find_project_root()
logger = get_logger(__name__)


class MLMetricsCollector:
    """
    Collects and manages machine learning training data.

    This class is responsible for gathering data that will be used
    to train machine learning models for network optimization.
    """

    def __init__(self, engine_props: dict[str, Any], sim_info: str):
        """
        Initialize the ML metrics collector.

        :param engine_props: Engine configuration properties
        :param sim_info: Simulation information string
        """
        self.engine_props = engine_props
        self.sim_info = sim_info
        self.train_data_list: list[dict[str, Any]] = []

    def update_train_data(
        self,
        old_request_info_dict: dict[str, Any],
        request_info_dict: dict[str, Any],
        network_spectrum_dict: dict[str, Any],
        current_transponders: int,
    ) -> None:
        """
        Update the training data list with current request information.

        :param old_request_info_dict: Request dictionary before any potential slicing
        :param request_info_dict: Request dictionary after potential slicing
        :param network_spectrum_dict: Network spectrum database
        :param current_transponders: Number of transponders used for this request
        """
        path_list = request_info_dict["path"]
        congestion_array = np.array([])

        # Calculate congestion for each core
        for core_num in range(self.engine_props["cores_per_link"]):
            current_congestion = find_core_cong(
                core_index=core_num,
                network_spectrum_dict=network_spectrum_dict,
                path_list=path_list,
            )
            congestion_array = np.append(congestion_array, current_congestion)

        # Calculate path length
        path_length = find_path_len(
            path_list=path_list, topology=self.engine_props["topology"]
        )

        # Prepare training data entry
        training_info_dict = {
            "old_bandwidth": old_request_info_dict["bandwidth"],
            "path_length": path_length,
            "longest_reach": np.max(
                old_request_info_dict["mod_formats"]["QPSK"]["max_length"]
            ),
            "average_congestion": float(np.mean(congestion_array)),
            "num_segments": current_transponders,
        }

        self.train_data_list.append(training_info_dict)

        logger.debug("Added training data entry: %s", training_info_dict)

    def save_train_data(
        self, iteration: int, max_iterations: int, base_file_path: str = "data"
    ) -> None:
        """
        Save training data to CSV file.

        Only saves on the last iteration to capture complete training data.

        :param iteration: Current iteration number
        :param max_iterations: Maximum number of iterations
        :param base_file_path: Base file path for output
        """
        if iteration == (max_iterations - 1):
            if not self.train_data_list:
                logger.warning("No training data to save")
                return

            save_df = pd.DataFrame(self.train_data_list)

            output_path = os.path.join(
                PROJECT_ROOT,
                base_file_path,
                "output",
                self.sim_info,
                f"{self.engine_props['erlang']}_train_data.csv",
            )

            save_df.to_csv(output_path, index=False)
            logger.info("Saved training data to: %s", output_path)

    def get_train_data(self) -> list[dict[str, Any]]:
        """
        Get the collected training data.

        :return: List of training data dictionaries
        """
        return self.train_data_list.copy()

    def clear_train_data(self) -> None:
        """
        Clear the training data list.

        Useful for resetting between iterations or experiments.
        """
        self.train_data_list.clear()
        logger.debug("Cleared training data list")

    def get_train_data_summary(self) -> dict[str, Any]:
        """
        Get a summary of the collected training data.

        :return: Dictionary containing summary statistics
        """
        if not self.train_data_list:
            return {
                "num_samples": 0,
                "avg_bandwidth": None,
                "avg_path_length": None,
                "avg_congestion": None,
                "avg_segments": None,
            }

        df = pd.DataFrame(self.train_data_list)

        return {
            "num_samples": len(self.train_data_list),
            "avg_bandwidth": df["old_bandwidth"].mean(),
            "avg_path_length": df["path_length"].mean(),
            "avg_congestion": df["average_congestion"].mean(),
            "avg_segments": df["num_segments"].mean(),
        }
