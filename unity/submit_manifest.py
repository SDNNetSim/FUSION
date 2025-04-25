import argparse
import pathlib
import subprocess
import sys
import csv
import os

SBATCH = [
    "--job-name=sim",
    "--output=slurm_out/%x_%A_%a.out",
    "--error=slurm_out/%x_%A_%a.err",
    "--partition=compute",
    "--cpus-per-task=1",
    "--mem=4G",
    "--time=1:00:00",
]


def parse_cli():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", help="experiment directory under ./experiments")
    parser.add_argument("script", help="bash script to run (e.g., launch.sh)")
    parser.add_argument("--rows", type=int, required=True, help="number of jobs")
    return parser.parse_args()


def get_network_from_manifest(manifest_path: pathlib.Path) -> str:
    """
    Get network from manifest file.
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader)
        if "network" not in first_row:
            sys.exit("Manifest file missing 'network' field.")
        return first_row["network"]


def main():
    """
    Controls the script.
    """
    args = parse_cli()

    job_dir = pathlib.Path("experiments") / args.exp
    if not job_dir.exists():
        sys.exit(f"Experiment directory not found: {job_dir}")

    manifest = job_dir / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"Missing manifest file: {manifest}")

    network = get_network_from_manifest(manifest)

    env = {
        "MANIFEST": str(manifest),
        "N_JOBS": str(args.rows - 1),
        "JOB_DIR": str(job_dir),
        "NETWORK": network,
        "DATE": args.exp.split("_")[0],
    }

    cmd = ["sbatch"] + SBATCH + [args.script]
    result = subprocess.run(cmd, env={**env, **dict(**os.environ)}, check=False)
    if result.returncode != 0:
        sys.exit("Job submission failed.")


if __name__ == "__main__":
    main()
