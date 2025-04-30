#!/usr/bin/env python3
"""
fetch_results.py – Copy simulation folders from cluster to local machine
using rsync, based on a configuration YAML file.

• Fetches `runs_index.json` and `manifest.csv` from a metadata root.
• Pulls input/output folders from a data root.
• Pulls logs from a logs root, stored as:
      logs/<path_algorithm>/<topology>/<date>/<timestamp>/
"""

import json
import logging
import pathlib
import subprocess
from typing import Iterator, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def twin_input_path(abs_path: pathlib.PurePosixPath) -> pathlib.PurePosixPath:
    """Convert an output path to the corresponding input path (drop seed)."""
    parts = list(abs_path.parts)
    idx = parts.index("output")
    parts[idx] = "input"
    return pathlib.PurePosixPath(*parts[:-1])  # drop seed folder


def last_n_parts(p: pathlib.PurePosixPath, n: int) -> pathlib.PurePosixPath:
    """Return the last n parts of a path."""
    return pathlib.PurePosixPath(*p.parts[-n:])


def topology_from_output(out_path: pathlib.PurePosixPath) -> str:
    """Extract the topology name from an output path."""
    parts = list(out_path.parts)
    topo = parts[parts.index("output") + 1]
    return topo


def _run(cmd: list[str], dry: bool) -> None:
    """Execute a shell command, respecting dry-run mode."""
    if dry:
        logging.info("[dry‑run] %s", " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def rsync_dir(remote_root: str, abs_path: pathlib.PurePosixPath,
              dest_root: pathlib.Path, dry: bool) -> None:
    """Rsync a directory from remote to local."""
    rel = last_n_parts(abs_path, 4)
    local_dir = dest_root / rel
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avP", "--compress",
           f"{remote_root}{abs_path}/", str(local_dir)]
    _run(cmd, dry)


def rsync_file(remote_root: str, remote_path: pathlib.PurePosixPath,
               local_path: pathlib.Path, dry: bool) -> None:
    """Rsync a file from remote to local."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-avP", "--compress",
           f"{remote_root}{remote_path}", str(local_path)]
    _run(cmd, dry)


def rsync_logs(remote_logs_root: str, path_alg: str, topology: str,
               date_ts: pathlib.PurePosixPath, dest_root: pathlib.Path,
               dry: bool) -> None:
    """Attempt to pull logs if present, using rsync.

    Ensures the *local* destination tree exists before invoking rsync
    to avoid mkdir errors on the receiver.
    """
    remote = pathlib.PurePosixPath(path_alg) / topology / date_ts
    local = dest_root / path_alg / topology / date_ts
    local.mkdir(parents=True, exist_ok=True)

    cmd = ["rsync", "-avP", "--compress",
           f"{remote_logs_root}{remote}/", str(local)]
    try:
        _run(cmd, dry)
    except subprocess.CalledProcessError as err:
        logging.warning("Logs not found: %s (%s)", remote, err)


def read_path_algorithm(input_dir: pathlib.Path) -> Optional[str]:
    """Read the path_algorithm value from one of the sim_input JSON files."""
    for f in input_dir.glob("sim_input_s*.json"):
        try:
            with f.open(encoding="utf-8") as fh:
                return json.load(fh).get("path_algorithm")
        except (json.JSONDecodeError, OSError):
            continue
    return None


def iter_index(index_file: pathlib.Path) -> Iterator[pathlib.PurePosixPath]:
    """Yield paths listed in the JSON index file."""
    with index_file.open(encoding="utf-8") as fh:
        for line in fh:
            if line := line.strip():
                yield pathlib.PurePosixPath(json.loads(line)["path"])


def main() -> None:
    """Main entry point – orchestrates syncing of simulation results."""
    cfg = yaml.safe_load(pathlib.Path("configs/config.yml").read_text(encoding="utf-8"))
    meta_root, data_root, logs_root = cfg["metadata_root"], cfg["data_root"], cfg["logs_root"]
    dest = pathlib.Path(cfg["dest"])
    exp_rel = pathlib.PurePosixPath(cfg["experiment"])
    dry = cfg.get("dry_run", False)

    tmp = pathlib.Path(".tmp_config")
    tmp.mkdir(exist_ok=True)
    rsync_file(meta_root, exp_rel / "runs_index.json", tmp / "runs_index.json", dry)

    index = tmp / "runs_index.json"
    for out_p in iter_index(index):
        rsync_dir(data_root, out_p, dest / "output", dry)
        in_p = twin_input_path(out_p)
        rsync_dir(data_root, in_p, dest, dry)

        rel_full, rel_trim = last_n_parts(in_p, 4), last_n_parts(in_p, 3)
        local_in = dest / rel_full if (dest / rel_full).exists() else dest / rel_trim
        path_alg = read_path_algorithm(local_in)
        if not path_alg:
            logging.warning("No path_algorithm in %s", local_in)
            continue

        topo = topology_from_output(out_p)
        date_ts = last_n_parts(in_p, 2)
        rsync_logs(logs_root, path_alg, topo, date_ts, dest / "logs", dry)


if __name__ == "__main__":
    main()
