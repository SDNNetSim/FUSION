import argparse
import csv
import os
import pathlib
import subprocess
import sys

# Keep this list in-sync with _RESOURCE_KEYS in make_manifest.py
RESOURCE_KEYS = {
    "partition", "time", "mem", "cpus", "gpus", "nodes"
}


def parse_cli() -> argparse.Namespace:
    """
    Parse the command line arguments and return an argparse.Namespace object.
    """
    p = argparse.ArgumentParser()
    p.add_argument("exp", help="experiment directory under ./experiments")
    p.add_argument("script", help="bash script to run (e.g., run_rl_sim.sh)")
    p.add_argument("--rows", type=int,
                   help="number of jobs (defaults to line-count of manifest)")
    return p.parse_args()


def read_first_row(manifest: pathlib.Path) -> tuple[dict, int]:
    """
    Read the first row of a manifest file and return a tuple (manifest, row).
    """
    with manifest.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            sys.exit("Manifest is empty.")
        return rows[0], len(rows)


def build_env(first: dict, n_rows: int, job_dir: pathlib.Path, exp: str) -> dict:
    """
    Build a dictionary with environment variables.
    """
    # mandatory metadata
    env = {
        "MANIFEST": str(pathlib.Path("unity") / job_dir / "manifest.csv"),
        "N_JOBS": str(n_rows - 1),  # Slurm arrays are 0-indexed
        "JOB_DIR": str(job_dir),
        "NETWORK": first.get("network", ""),
        "DATE": exp.split("_")[0],
        "JOB_NAME": f"{first['path_algorithm']}_{first['erlang_start']}_{exp.replace('/', '_')}",
    }

    # propagate resources ⇢ upper-case so bash can ${PARTITION}
    for key in RESOURCE_KEYS:
        if key in first and first[key]:
            env[key.upper()] = str(first[key])

    return env


def main() -> None:
    """
    Controls the script.
    """
    args = parse_cli()
    job_dir = pathlib.Path(args.exp)
    if not job_dir.exists():
        sys.exit(f"Experiment directory not found: {job_dir}")

    manifest = job_dir / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"Missing manifest file: {manifest}")

    first_row, total_rows = read_first_row(manifest)
    n_rows = args.rows if args.rows is not None else total_rows

    env = build_env(first_row, n_rows, job_dir, args.exp)

    jobs_dir = job_dir / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    slurm_output = jobs_dir / "slurm_%A_%a.out"

    script_path = pathlib.Path("bash_scripts") / args.script
    if not script_path.exists():
        sys.exit(f"Bash script not found: {script_path}")

    cmd = [
        "sbatch",
        f"--partition={env['PARTITION']}",
        f"--gpus={env['GPUS']}",
        f"--cpus-per-task={env['CPUS']}",
        f"--mem={env['MEM']}",
        f"--time={env['TIME']}",
        f"--array=0-{env['N_JOBS']}",
        f"--output={slurm_output}",
        f"--job-name={env['JOB_NAME']}",
        str(script_path),
    ]

    print(f'[DEBUG] Command submitted: {cmd}')

    result = subprocess.run(cmd, env={**os.environ, **env}, check=False)
    if result.returncode != 0:
        sys.exit("Job submission failed.")


if __name__ == "__main__":
    main()
