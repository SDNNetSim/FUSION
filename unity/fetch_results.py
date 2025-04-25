#!/usr/bin/env python3
"""
fetch_results.py – copy simulation folders from cluster to local machine
using either rsync (default) or Globus CLI.

Examples
--------
# 1. rsync (default)
python fetch_results.py \
   --experiment experiments/0425/NSFNet/182105 \
   --cluster-root user@login.cluster.edu:/work/project/data/ \
   --dest ./cluster_results

# 2. Globus
python fetch_results.py \
   --experiment experiments/0425/NSFNet/182105 \
   --mode globus \
   --src-ep 01234567-89ab-cdef-0123-456789abcdef \
   --dst-ep 89abcdef-0123-4567-89ab-cdef01234567 \
   --dest ./cluster_results
"""
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile


# ──────────────────────────────────────────────────────────────────────────────
# path helpers
# ──────────────────────────────────────────────────────────────────────────────
def twin_input_path(abs_path: pathlib.PurePosixPath) -> pathlib.PurePosixPath:
    """Turn /…/output/…/<run>/s1 → /…/input/…/<run> (drop trailing s1)."""
    parts = list(abs_path.parts)
    try:
        idx = parts.index("output")
        parts[idx] = "input"
    except ValueError:  # pragma: no cover
        raise RuntimeError(f"'output' not found in path: {abs_path}") from None
    return pathlib.PurePosixPath(*parts[:-1])  # drop 's1'


def last_n_parts(p: pathlib.PurePosixPath, n: int) -> pathlib.PurePosixPath:
    return pathlib.PurePosixPath(*p.parts[-n:])


# ──────────────────────────────────────────────────────────────────────────────
# CLI + utils
# ──────────────────────────────────────────────────────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True,
                   help="experiment directory containing runs_index.json")
    p.add_argument("--cluster-root", required=True,
                   help="user@host: or user@host:/base/path/")
    p.add_argument("--dest", required=True, help="local destination root")
    p.add_argument("--mode", choices=["rsync", "globus"], default="rsync")
    p.add_argument("--src-ep", help="Globus SOURCE endpoint UUID")
    p.add_argument("--dst-ep", help="Globus DEST   endpoint UUID")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def iter_paths(index_file: pathlib.Path):
    with index_file.open(encoding="utf-8") as fh:
        for line in fh:
            if line := line.strip():
                yield pathlib.PurePosixPath(json.loads(line)["path"])


# ──────────────────────────────────────────────────────────────────────────────
# rsync implementation
# ──────────────────────────────────────────────────────────────────────────────
def rsync_one(src_prefix: str, abs_path: pathlib.PurePosixPath,
              dest_root: pathlib.Path, dry: bool):
    """Copy folder contents with rsync, creating parents as needed."""
    rel = last_n_parts(abs_path, 4)  # NSFNet/0425/<ts>/s1
    local_dir = dest_root.joinpath(rel)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    src = f"{src_prefix}{abs_path}/"  # trailing '/' = contents
    cmd = ["rsync", "-avP", "--compress", src, str(local_dir)]

    if dry:
        print("[dry-run]", " ".join(cmd))
    else:
        subprocess.run(cmd, check=True)


# ──────────────────────────────────────────────────────────────────────────────
# Globus implementation
# ──────────────────────────────────────────────────────────────────────────────
def globus_batch(src_ep: str, dst_ep: str, pairs: list[tuple[str, str]],
                 dry: bool):
    """Create batch file + launch single Globus transfer."""
    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        batch_path = pathlib.Path(tmp.name)
        for remote, local in pairs:
            tmp.write(f"{remote} {local}\n")

    cmd = [
        "globus", "transfer",
        "--label", "sim_results_fetch",
        "--batch", str(batch_path),
        src_ep, dst_ep
    ]
    if dry:
        print("[dry-run]", " ".join(cmd))
        print(batch_path.read_text())  # show batch content
    else:
        subprocess.run(cmd, check=True)
        batch_path.unlink()  # clean up


# ──────────────────────────────────────────────────────────────────────────────
# main orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = cli()
    index_file = pathlib.Path(args.experiment) / "runs_index.json"
    if not index_file.exists():
        sys.exit(f"runs_index.json not found: {index_file}")

    dest_root = pathlib.Path(args.dest)
    mode = args.mode

    if mode == "rsync":
        for out_path in iter_paths(index_file):
            rsync_one(args.cluster_root, out_path,
                      dest_root / "output", args.dry_run)

            in_path = twin_input_path(out_path)
            rsync_one(args.cluster_root, in_path,
                      dest_root / "input", args.dry_run)

    else:  # mode == "globus"
        if not (args.src_ep and args.dst_ep):
            sys.exit("--src-ep and --dst-ep are required for --mode globus")

        transfers: list[tuple[str, str]] = []
        for out_path in iter_paths(index_file):
            # output
            rel_out = last_n_parts(out_path, 4)
            local_out = dest_root / "output" / rel_out
            transfers.append((str(out_path) + "/", str(local_out)))

            # input
            in_path = twin_input_path(out_path)
            rel_in = last_n_parts(in_path, 3)  # NSFNet/0425/<ts>
            local_in = dest_root / "input" / rel_in
            transfers.append((str(in_path) + "/", str(local_in)))

        globus_batch(args.src_ep, args.dst_ep, transfers, args.dry_run)


if __name__ == "__main__":
    main()
