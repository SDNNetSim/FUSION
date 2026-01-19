"""
Remote cluster results fetching and synchronization module.

This module provides functionality for fetching simulation results from remote
cluster storage, including path manipulation, file synchronization, and manifest
parsing for the FUSION unity cluster management system.

The module handles:
- Converting between output and input directory paths
- Synchronizing directories and files via rsync
- Parsing simulation metadata and indices
- Managing cluster result downloads with proper organization
"""

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from time import sleep

import yaml

from fusion.unity.constants import (
    CLUSTER_RESULTS_DIR,
    CONFIG_FILE_PATH,
    OUTPUT_TO_INPUT_SEGMENTS,
    RSYNC_DEFAULT_OPTIONS,
    RUNS_INDEX_FILE,
    SIM_INPUT_PATTERN,
    SYNC_DELAY_SECONDS,
    TEMP_CONFIG_DIR,
)
from fusion.unity.errors import ConfigurationError, RemotePathError
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def convert_output_to_input_path(absolute_output_path: PurePosixPath) -> PurePosixPath:
    """
    Convert an output directory path to its corresponding input directory path.

    This function transforms paths by replacing 'output' with 'input' and removing
    the seed folder (e.g., s1, s2, etc.) from the end of the path.

    :param absolute_output_path: The absolute path to an output directory
    :type absolute_output_path: PurePosixPath
    :return: The corresponding input directory path with seed folder removed
    :rtype: PurePosixPath
    :raises RemotePathError: If 'output' is not found in the path

    Example:
        >>> output_path = PurePosixPath("/data/output/topology1/experiment/s1")
        >>> input_path = convert_output_to_input_path(output_path)
        >>> print(input_path)
        /data/input/topology1/experiment
    """
    path_parts_list = list(absolute_output_path.parts)

    if "output" not in path_parts_list:
        msg = f"Path does not contain 'output' directory: {absolute_output_path}"
        raise RemotePathError(msg)

    path_parts_list[path_parts_list.index("output")] = "input"
    return PurePosixPath(*path_parts_list[:-1])  # Remove seed folder


def get_last_path_segments(path: PurePosixPath, segment_count: int) -> PurePosixPath:
    """
    Extract the last n segments from a path.

    :param path: The path to extract segments from
    :type path: PurePosixPath
    :param segment_count: Number of segments to extract from the end
    :type segment_count: int
    :return: Path containing only the last n segments
    :rtype: PurePosixPath

    Example:
        >>> path = PurePosixPath("/data/output/topology1/experiment/s1")
        >>> result = get_last_path_segments(path, 3)
        >>> print(result)
        topology1/experiment/s1
    """
    if segment_count <= 0:
        return PurePosixPath()

    return PurePosixPath(*path.parts[-segment_count:])


def extract_topology_from_path(output_directory_path: PurePosixPath) -> str:
    """
    Extract the topology name from an output directory path.

    The topology is expected to be the directory immediately following 'output'
    in the path structure.

    :param output_directory_path: Path to the output directory
    :type output_directory_path: PurePosixPath
    :return: Name of the topology
    :rtype: str
    :raises RemotePathError: If 'output' is not found or no topology after 'output'

    Example:
        >>> output_path = PurePosixPath("/data/output/topology1/experiment/s1")
        >>> topology = extract_topology_from_path(output_path)
        >>> print(topology)
        topology1
    """
    path_parts_list = list(output_directory_path.parts)

    if "output" not in path_parts_list:
        msg = f"Path does not contain 'output' directory: {output_directory_path}"
        raise RemotePathError(msg)

    output_index = path_parts_list.index("output")
    if output_index + 1 >= len(path_parts_list):
        msg = f"No topology directory found after 'output' in path: {output_directory_path}"
        raise RemotePathError(msg)

    return path_parts_list[output_index + 1]


def _execute_command_with_delay(command_list: list[str], is_dry_run: bool) -> None:
    """
    Execute a shell command with a delay, optionally as a dry run.

    :param command_list: List of command parts to execute
    :type command_list: list[str]
    :param is_dry_run: If True, only log the command without executing
    :type is_dry_run: bool
    :raises subprocess.CalledProcessError: If command execution fails
    """
    sleep(SYNC_DELAY_SECONDS)
    if is_dry_run:
        logger.info("[dry‑run] %s", " ".join(command_list))
    else:
        logger.debug("Executing command: %s", " ".join(command_list))
        subprocess.run(command_list, check=True)


def synchronize_remote_directory(
    remote_root_path: str,
    absolute_directory_path: PurePosixPath,
    destination_root_path: Path,
    is_dry_run: bool,
) -> None:
    """
    Synchronize an entire remote directory to local filesystem using rsync.

    The directory is synchronized into dest_root/rel_path where rel_path
    consists of the last 4 segments of the absolute path.

    :param remote_root_path: Root path of the remote filesystem
    :type remote_root_path: str
    :param absolute_directory_path: Absolute path to the directory to sync
    :type absolute_directory_path: PurePosixPath
    :param destination_root_path: Local root directory for synchronized files
    :type destination_root_path: Path
    :param is_dry_run: If True, only log operations without executing
    :type is_dry_run: bool

    Example:
        >>> remote_root = "user@cluster:/work/"
        >>> remote_path = PurePosixPath("/work/data/output/topology1/exp1")
        >>> local_root = Path("/local/data")
        >>> synchronize_remote_directory(remote_root, remote_path, local_root, False)
    """
    relative_path = get_last_path_segments(absolute_directory_path, OUTPUT_TO_INPUT_SEGMENTS)
    local_target_directory = destination_root_path / relative_path
    local_target_directory.parent.mkdir(parents=True, exist_ok=True)

    rsync_command = [
        "rsync",
        *RSYNC_DEFAULT_OPTIONS,
        f"{remote_root_path}{absolute_directory_path}/",
        str(local_target_directory),
    ]

    try:
        _execute_command_with_delay(rsync_command, is_dry_run)
        logger.info("Successfully synchronized directory: %s", absolute_directory_path)
    except subprocess.CalledProcessError as error:
        logger.error("Failed to synchronize directory %s: %s", absolute_directory_path, error)


def synchronize_remote_file(
    remote_root_path: str,
    remote_file_path: PurePosixPath,
    local_file_path: Path,
    is_dry_run: bool,
) -> None:
    """
    Synchronize a single remote file to local filesystem using rsync.

    :param remote_root_path: Root path of the remote filesystem
    :type remote_root_path: str
    :param remote_file_path: Path to the remote file to synchronize
    :type remote_file_path: PurePosixPath
    :param local_file_path: Local path where the file should be saved
    :type local_file_path: Path
    :param is_dry_run: If True, only log operations without executing
    :type is_dry_run: bool
    """
    local_file_path.parent.mkdir(parents=True, exist_ok=True)

    rsync_command = [
        "rsync",
        *RSYNC_DEFAULT_OPTIONS,
        f"{remote_root_path}{remote_file_path}",
        str(local_file_path),
    ]

    try:
        _execute_command_with_delay(rsync_command, is_dry_run)
        logger.info("Successfully synchronized file: %s", remote_file_path)
    except subprocess.CalledProcessError as error:
        logger.error("Failed to synchronize file %s: %s", remote_file_path, error)


def synchronize_simulation_logs(
    remote_logs_root_path: str,
    path_algorithm_name: str,
    topology_name: str,
    date_timestamp_path: PurePosixPath,
    destination_root_path: Path,
    is_dry_run: bool,
) -> None:
    """
    Synchronize simulation log files from remote cluster storage.

    Downloads logs for a specific algorithm, topology, and timestamp combination.
    Logs are organized in the directory structure: path_algorithm/topology/timestamp/

    :param remote_logs_root_path: Root path for log files on remote system
    :type remote_logs_root_path: str
    :param path_algorithm_name: Name of the path algorithm used in simulation
    :type path_algorithm_name: str
    :param topology_name: Name of the network topology
    :type topology_name: str
    :param date_timestamp_path: Date/timestamp path segments for the simulation
    :type date_timestamp_path: PurePosixPath
    :param destination_root_path: Local root directory for log files
    :type destination_root_path: Path
    :param is_dry_run: If True, only log operations without executing
    :type is_dry_run: bool
    """
    remote_logs_path = PurePosixPath(path_algorithm_name) / topology_name / date_timestamp_path
    local_logs_path = destination_root_path / path_algorithm_name / topology_name / date_timestamp_path
    local_logs_path.mkdir(parents=True, exist_ok=True)

    rsync_command = [
        "rsync",
        *RSYNC_DEFAULT_OPTIONS,
        f"{remote_logs_root_path}{remote_logs_path}/",
        str(local_logs_path),
    ]

    try:
        _execute_command_with_delay(rsync_command, is_dry_run)
        logger.info(
            "Successfully synchronized logs for %s/%s",
            path_algorithm_name,
            topology_name,
        )
    except subprocess.CalledProcessError as error:
        logger.warning(
            "Logs not found for %s/%s at %s: %s",
            path_algorithm_name,
            topology_name,
            remote_logs_path,
            error,
        )


def extract_path_algorithm_from_input(input_directory_path: Path) -> str | None:
    """
    Extract the path algorithm name from simulation input files.

    Searches for simulation input JSON files (sim_input_s*.json) in the given
    directory and extracts the 'path_algorithm' field from the first valid file found.

    :param input_directory_path: Directory containing simulation input files
    :type input_directory_path: Path
    :return: Name of the path algorithm, or None if not found
    :rtype: str | None

    Example:
        >>> input_dir = Path("/data/input/topology1/experiment")
        >>> algorithm = extract_path_algorithm_from_input(input_dir)
        >>> print(algorithm)
        'shortest_path'
    """
    for simulation_input_file in input_directory_path.glob(SIM_INPUT_PATTERN):
        try:
            with simulation_input_file.open(encoding="utf-8") as file_handle:
                simulation_data = json.load(file_handle)
                path_algorithm = simulation_data.get("path_algorithm")
                if path_algorithm:
                    logger.debug(
                        "Found path algorithm '%s' in %s",
                        path_algorithm,
                        simulation_input_file,
                    )
                    return str(path_algorithm)
        except (json.JSONDecodeError, OSError) as error:
            logger.debug(
                "Failed to read path algorithm from %s: %s",
                simulation_input_file,
                error,
            )
            continue

    logger.warning("No valid path algorithm found in directory: %s", input_directory_path)
    return None


def iterate_runs_index_file(index_file_path: Path) -> Iterator[PurePosixPath]:
    """
    Iterate over entries in a runs index file.

    Each line in the index file should contain a JSON object with a 'path' field
    pointing to a simulation output directory.

    :param index_file_path: Path to the runs index file
    :type index_file_path: Path
    :return: Iterator yielding paths from the index file
    :rtype: Iterator[PurePosixPath]
    :raises json.JSONDecodeError: If a line contains invalid JSON
    :raises FileNotFoundError: If the index file doesn't exist

    Example:
        >>> index_file = Path("runs_index.json")
        >>> for output_path in iterate_runs_index_file(index_file):
        ...     print(f"Processing: {output_path}")
    """
    logger.debug("Reading runs index from: %s", index_file_path)

    with index_file_path.open(encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, 1):
            line = line.strip()
            if not line:
                continue

            try:
                index_entry = json.loads(line)
                output_path = PurePosixPath(index_entry["path"])
                yield output_path
            except (json.JSONDecodeError, KeyError) as error:
                logger.error(
                    "Invalid index entry at line %d: %s (Error: %s)",
                    line_number,
                    line,
                    error,
                )
                continue


def main() -> None:
    """
    Main entry point for the fetch results script.

    Controls the overall workflow of fetching simulation results from remote
    cluster storage based on configuration settings. Handles index processing,
    directory synchronization, and log file management.

    :raises ConfigurationError: If configuration file is not found or invalid
    :raises yaml.YAMLError: If configuration file format is invalid
    """
    logger.info("Starting results fetch process")

    try:
        configuration_file_path = Path(CONFIG_FILE_PATH)
        file_content = configuration_file_path.read_text(encoding="utf-8")
        configuration_data = yaml.safe_load(file_content)
    except FileNotFoundError as error:
        msg = f"Configuration file not found: {CONFIG_FILE_PATH}"
        raise ConfigurationError(msg) from error
    except yaml.YAMLError as error:
        msg = f"Invalid configuration file format: {error}"
        raise ConfigurationError(msg) from error

    metadata_root_path = configuration_data["metadata_root"]
    data_root_path = configuration_data["data_root"]
    logs_root_path = configuration_data["logs_root"]
    destination_directory = Path(configuration_data["dest"]).expanduser()

    # Normalize destination – ensure we have exactly one /data layer
    data_destination_directory = destination_directory if destination_directory.name == "data" else destination_directory / "data"

    experiment_relative_path = PurePosixPath(configuration_data["experiment"])
    is_dry_run = configuration_data.get("dry_run", False)

    # Set up temporary directory for metadata
    temporary_directory = Path(TEMP_CONFIG_DIR)
    temporary_directory.mkdir(exist_ok=True)

    # Download runs index file
    runs_index_remote_path = experiment_relative_path / RUNS_INDEX_FILE
    runs_index_local_path = temporary_directory / RUNS_INDEX_FILE

    logger.info("Downloading runs index file")
    synchronize_remote_file(metadata_root_path, runs_index_remote_path, runs_index_local_path, is_dry_run)

    processed_run_directories_set = set()

    logger.info("Processing runs from index file")
    for output_directory_path in iterate_runs_index_file(runs_index_local_path):
        # Skip failed jobs with empty directory names
        if output_directory_path.name == "":
            logger.debug("Skipping failed job with empty directory: %s", output_directory_path)
            continue

        parent_run_directory = output_directory_path.parent
        if parent_run_directory in processed_run_directories_set:
            logger.debug("Skipping already processed run directory: %s", parent_run_directory)
            continue
        processed_run_directories_set.add(parent_run_directory)

        # Synchronize output and input directories
        logger.info("Synchronizing run directory: %s", parent_run_directory)
        synchronize_remote_directory(data_root_path, parent_run_directory, data_destination_directory, is_dry_run)

        input_directory_path = convert_output_to_input_path(output_directory_path)
        synchronize_remote_directory(data_root_path, input_directory_path, data_destination_directory, is_dry_run)

        # Extract algorithm information and sync logs
        local_input_directory = Path(CLUSTER_RESULTS_DIR) / input_directory_path
        path_algorithm = extract_path_algorithm_from_input(local_input_directory)

        if not path_algorithm:
            logger.warning("No path algorithm found in input directory: %s", local_input_directory)
            continue

        topology_name = extract_topology_from_path(output_directory_path)
        date_timestamp_segments = get_last_path_segments(input_directory_path, 2)

        logger.info(
            "Synchronizing logs for algorithm '%s' and topology '%s'",
            path_algorithm,
            topology_name,
        )
        synchronize_simulation_logs(
            logs_root_path,
            path_algorithm,
            topology_name,
            date_timestamp_segments,
            destination_directory / "logs",
            is_dry_run,
        )

    logger.info("Results fetch process completed successfully")


if __name__ == "__main__":
    main()
