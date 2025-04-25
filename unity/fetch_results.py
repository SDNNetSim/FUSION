#!/usr/bin/env python3
"""
fetch_results.py  –  Copy result folders listed in runs_index.json
                     to a destination using rsync.

Example (run on your laptop):

    ./fetch_results.py \
        --experiment    experiments/0425/NSFNet/161152 \
        --cluster-root  user@login.cluster.edu:/work/project \
        --dest          ./cluster_results

Arguments
---------
--experiment   Experiment directory that contains runs_index.json
--cluster-root Prefix before the absolute paths stored in runs_index.json
               (include user@host: for remote rsync).
--dest         Local directory to copy into.
--dry-run      Show rsync commands without executing them.
"""
import argparse
import json
import pathlib
import subprocess
import sys


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--cluster-root", required=True)
    p.add_argument("--dest", required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def iter_paths(index_file: pathlib.Path):
    with index_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield pathlib.PurePosixPath(json.loads(line)["path"])


def rsync_one(src_prefix: str, abs_path: pathlib.PurePosixPath,
              dest_root: pathlib.Path, dry: bool):
    dest_root.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avP", "--compress",
           f"{src_prefix}{abs_path}", str(dest_root / abs_path.name)]
    if dry:
        print("[dry-run]", " ".join(cmd))
    else:
        print("→", " ".join(cmd))
        subprocess.run(cmd, check=True)


def main() -> None:
    args = cli()
    index_file = pathlib.Path(args.experiment) / "runs_index.json"
    if not index_file.exists():
        sys.exit(f"runs_index.json not found: {index_file}")

    for folder in iter_paths(index_file):
        rsync_one(args.cluster_root, folder, pathlib.Path(args.dest),
                  args.dry_run)


if __name__ == "__main__":
    main()
