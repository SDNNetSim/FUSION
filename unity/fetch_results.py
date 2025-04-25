import argparse
import json
import pathlib
import subprocess
import sys


def twin_input_path(abs_path: pathlib.PurePosixPath) -> pathlib.PurePosixPath:
    """
    Convert an output path to the matching input path by:
    - replacing 'output' with 'input'
    - dropping the last component (like 's1')
    """
    parts = list(abs_path.parts)
    try:
        output_index = parts.index("output")
        parts[output_index] = "input"
    except ValueError:
        raise RuntimeError(f"'output' not found in path: {abs_path}")

    return pathlib.PurePosixPath(*parts[:-1])  # remove 's1' at the end


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
              dest_root: pathlib.Path, dry: bool, allow_missing: bool = False):
    """
    Copies everything inside abs_path to dest_root preserving folder structure.
    If allow_missing=True, skip paths that don't exist on the remote.
    """
    relative_parts = abs_path.parts[-4:]
    local_subdir = dest_root.joinpath(*relative_parts)

    local_subdir.parent.mkdir(parents=True, exist_ok=True)

    src_path = f"{src_prefix}{abs_path}/"
    dest_path = str(local_subdir)

    cmd = ["rsync", "-avP", "--compress", src_path, dest_path]

    if dry:
        print("[dry-run]", " ".join(cmd))
    else:
        print("→", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            if allow_missing and e.returncode == 23:
                # Code 23 = some files not transferred (ok for missing folders)
                print(f"⚠️  Skipping missing input folder: {src_path}")
            else:
                raise


def main() -> None:
    args = cli()
    index_file = pathlib.Path(args.experiment) / "runs_index.json"
    if not index_file.exists():
        sys.exit(f"runs_index.json not found: {index_file}")

    for output_path in iter_paths(index_file):
        # Copy output
        rsync_one(args.cluster_root, output_path,
                  pathlib.Path(args.dest) / "output", args.dry_run)

        # Copy input (allow missing folders)
        input_path = twin_input_path(output_path)
        rsync_one(args.cluster_root, input_path,
                  pathlib.Path(args.dest) / "input", args.dry_run,
                  allow_missing=True)


if __name__ == "__main__":
    main()
