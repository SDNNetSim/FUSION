import json
import logging
import pathlib
import subprocess
from typing import Iterator, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def twin_input_path(abs_path: pathlib.PurePosixPath) -> pathlib.PurePosixPath:
    """Convert an *output* path to its corresponding *input* path (strip seed)."""
    parts = list(abs_path.parts)
    parts[parts.index("output")] = "input"
    return pathlib.PurePosixPath(*parts[:-1])  # drop seed folder (s1, s2, ...)


def last_n_parts(p: pathlib.PurePosixPath, n: int) -> pathlib.PurePosixPath:
    """
    Return the last n parts of the path.
    """
    return pathlib.PurePosixPath(*p.parts[-n:])


def topology_from_output(out_path: pathlib.PurePosixPath) -> str:
    """
    Return the topology from an output path.
    """
    parts = list(out_path.parts)
    return parts[parts.index("output") + 1]


def _run(cmd: list[str], dry: bool) -> None:
    if dry:
        logging.info("[dry‑run] %s", " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


def rsync_dir(remote_root: str, abs_path: pathlib.PurePosixPath,
              dest_root: pathlib.Path, dry: bool) -> None:
    """Sync an entire directory (abs_path) into dest_root/rel_path."""
    rel = last_n_parts(abs_path, 4)  # keep last 4 segments
    local_dir = dest_root / rel
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["rsync", "-avP", "--compress", f"{remote_root}{abs_path}/", str(local_dir)], dry)


def rsync_file(remote_root: str, remote_path: pathlib.PurePosixPath,
               local_path: pathlib.Path, dry: bool) -> None:
    """
    Sync an entire file (remote_path) into local_path/rel_path.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _run(["rsync", "-avP", "--compress", f"{remote_root}{remote_path}", str(local_path)], dry)


def rsync_logs(remote_logs_root: str, path_alg: str, topology: str,
               date_ts: pathlib.PurePosixPath, dest_root: pathlib.Path,
               dry: bool) -> None:
    """
    Rsync logs download.
    """
    remote = pathlib.PurePosixPath(path_alg) / topology / date_ts
    local = dest_root / path_alg / topology / date_ts
    local.mkdir(parents=True, exist_ok=True)
    try:
        _run(["rsync", "-avP", "--compress", f"{remote_logs_root}{remote}/", str(local)], dry)
    except subprocess.CalledProcessError as err:
        logging.warning("Logs not found: %s (%s)", remote, err)


def read_path_algorithm(input_dir: pathlib.Path) -> Optional[str]:
    """
    Read a path algorithm from the input directory.
    """
    for f in input_dir.glob("sim_input_s*.json"):
        try:
            with f.open(encoding="utf-8") as fh:
                return json.load(fh).get("path_algorithm")
        except (json.JSONDecodeError, OSError):
            continue
    return None


def iter_index(index_file: pathlib.Path) -> Iterator[pathlib.PurePosixPath]:
    """
    Iter index function.
    """
    with index_file.open(encoding="utf-8") as fh:
        for line in fh:
            if line := line.strip():
                yield pathlib.PurePosixPath(json.loads(line)["path"])


def main() -> None:
    """
    Controls the script.
    """
    cfg = yaml.safe_load(pathlib.Path("configs/config.yml").read_text(encoding="utf-8"))
    meta_root, data_root, logs_root = cfg["metadata_root"], cfg["data_root"], cfg["logs_root"]
    dest = pathlib.Path(cfg["dest"]).expanduser()

    # Normalise destination – ensure we have exactly one /data layer
    data_dest = dest if dest.name == "data" else dest / "data"

    exp_rel = pathlib.PurePosixPath(cfg["experiment"])
    dry = cfg.get("dry_run", False)

    tmp = pathlib.Path(".tmp_config")
    tmp.mkdir(exist_ok=True)
    rsync_file(meta_root, exp_rel / "runs_index.json", tmp / "runs_index.json", dry)

    index = tmp / "runs_index.json"
    fetched = set()

    for out_p in iter_index(index):
        # Failed job, no output directory
        if out_p.name == '':
            print(f'[DEBUG] Skipping {out_p} due to empty directory (failed job).')
            continue

        parent_run_dir = out_p.parent
        if parent_run_dir in fetched:
            continue
        fetched.add(parent_run_dir)

        rsync_dir(data_root, parent_run_dir, data_dest, dry)
        in_p = twin_input_path(out_p)
        rsync_dir(data_root, in_p, data_dest, dry)

        local_in = pathlib.Path('cluster_results') / in_p
        path_alg = read_path_algorithm(local_in)
        if not path_alg:
            logging.warning("No path_algorithm in %s", local_in)
            continue

        topo = topology_from_output(out_p)
        date_ts = last_n_parts(in_p, 2)
        rsync_logs(logs_root, path_alg, topo, date_ts, dest / "logs", dry)


if __name__ == "__main__":
    main()
