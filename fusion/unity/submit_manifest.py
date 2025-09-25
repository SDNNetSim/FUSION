"""
SLURM job submission module for Unity cluster management.

This module handles the submission of simulation manifests as SLURM array jobs
to the Unity cluster, including argument parsing, resource configuration, and
job queue management.
"""

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Any

from fusion.unity.constants import BASH_SCRIPTS_DIR, RESOURCE_KEYS, UNITY_BASE_DIR
from fusion.unity.errors import JobSubmissionError, ManifestNotFoundError
from fusion.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_cli() -> argparse.Namespace:
    """
    Parse the command line arguments and return an argparse.Namespace object.

    :return: Parsed command line arguments
    :rtype: argparse.Namespace
    """
    p = argparse.ArgumentParser()
    p.add_argument("exp", help="experiment directory under ./experiments")
    p.add_argument("script", help="bash script to run (e.g., run_rl_sim.sh)")
    p.add_argument(
        "--rows", type=int, help="number of jobs (defaults to line-count of manifest)"
    )
    return p.parse_args()


def read_first_row(manifest_path: Path) -> tuple[dict[str, Any], int]:
    """
    Read the first row of a manifest file and return a tuple (manifest, row).

    :param manifest_path: Path to the manifest CSV file
    :type manifest_path: Path
    :return: Tuple containing first row data and total row count
    :rtype: tuple[dict[str, Any], int]
    :raises ManifestNotFoundError: If manifest file cannot be read
    """
    try:
        with manifest_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                raise ManifestNotFoundError("Manifest is empty")
            return rows[0], len(rows)
    except OSError as error:
        msg = f"Cannot read manifest file: {manifest_path}"
        raise ManifestNotFoundError(msg) from error


def build_environment_variables(
    first_row: dict[str, Any],
    total_rows: int,
    job_directory: Path,
    experiment_name: str,
) -> dict[str, str]:
    """
    Build a dictionary with environment variables for SLURM job.

    :param first_row: First row data from manifest
    :type first_row: dict[str, Any]
    :param total_rows: Total number of rows in manifest
    :type total_rows: int
    :param job_directory: Path to job directory
    :type job_directory: Path
    :param experiment_name: Name of the experiment
    :type experiment_name: str
    :return: Environment variables for job submission
    :rtype: Dict[str, str]
    """
    # mandatory metadata
    env = {
        "MANIFEST": str(Path(UNITY_BASE_DIR) / job_directory / "manifest.csv"),
        "N_JOBS": str(total_rows - 1),  # Slurm arrays are 0-indexed
        "JOB_DIR": str(job_directory),
        "NETWORK": first_row.get("network", ""),
        "DATE": experiment_name.split("_")[0],
        "JOB_NAME": (
            f"{first_row['path_algorithm']}_{first_row['erlang_start']}_"
            f"{experiment_name.replace('/', '_')}"
        ),
    }

    # propagate resources ⇢ upper-case so bash can ${PARTITION}
    for key in RESOURCE_KEYS:
        if key in first_row and first_row[key]:
            env[key.upper()] = str(first_row[key])

    return env


def main() -> None:
    """
    Main entry point for the submit manifest script.

    Parses command line arguments, validates inputs, and submits the manifest
    as a SLURM array job to the cluster.

    :raises JobSubmissionError: If job submission fails
    :raises ManifestNotFoundError: If required files are missing
    """
    args = parse_cli()
    job_directory = Path(args.exp)
    if not job_directory.exists():
        raise ManifestNotFoundError(f"Experiment directory not found: {job_directory}")

    manifest_path = job_directory / "manifest.csv"
    if not manifest_path.exists():
        raise ManifestNotFoundError(f"Missing manifest file: {manifest_path}")

    first_row, total_rows = read_first_row(manifest_path)
    job_count = args.rows if args.rows is not None else total_rows

    environment_variables = build_environment_variables(
        first_row, job_count, job_directory, args.exp
    )

    jobs_directory = job_directory / "jobs"
    jobs_directory.mkdir(parents=True, exist_ok=True)

    slurm_output = jobs_directory / "slurm_%A_%a.out"

    script_path = Path(BASH_SCRIPTS_DIR) / args.script
    if not script_path.exists():
        raise ManifestNotFoundError(f"Bash script not found: {script_path}")

    cmd = [
        "sbatch",
        f"--partition={environment_variables['PARTITION']}",
        f"--gpus={environment_variables['GPUS']}",
        f"--cpus-per-task={environment_variables['CPUS']}",
        f"--mem={environment_variables['MEM']}",
        f"--time={environment_variables['TIME']}",
        f"--array=0-{environment_variables['N_JOBS']}",
        f"--output={slurm_output}",
        f"--job-name={environment_variables['JOB_NAME']}",
    ]

    if environment_variables["PARTITION"] in ("gpu", "cpu"):
        cmd.append("--qos=long")

    cmd.append(str(script_path))

    logger.info("Submitting SLURM command: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            env={**os.environ, **environment_variables},
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Job submitted successfully: %s", result.stdout.strip())
    except subprocess.CalledProcessError as error:
        logger.error("SLURM submission failed: %s", error.stderr)
        raise JobSubmissionError(f"Job submission failed: {error.stderr}") from error


if __name__ == "__main__":
    main()
