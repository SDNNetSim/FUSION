from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys


def _sync_rsync(remote: str, exp: str, local_path: pathlib.Path) -> None:
    """Synchronise using rsync."""
    # TODO: Ensure this file path is correct and copies to the correct place.
    remote_path = f"{remote}:$HOME/experiments/{exp}/results/"
    subprocess.run(
        ["rsync", "-avzP", remote_path, str(local_path)],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _sync_globus(
        src_ep: str,
        dst_ep: str,
        remote_path: str,
        local_path: pathlib.Path,
) -> None:
    """Synchronise using Globus CLI."""
    subprocess.run(
        [
            "globus",
            "transfer",
            src_ep + ":" + remote_path,
            dst_ep + ":" + str(local_path),
            "--recursive",
        ],
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def _parse_args() -> argparse.Namespace:
    """CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, help="20250424_nsfnet")
    parser.add_argument("--method", choices=("rsync", "globus"), default="rsync")
    parser.add_argument("--remote", default="unity", help="SSH alias for rsync mode")
    parser.add_argument("--src-endpoint", help="Globus source endpoint UUID")
    parser.add_argument("--dst-endpoint", help="Globus destination endpoint UUID")
    return parser.parse_args()


def main() -> None:
    """Entrypoint."""
    args = _parse_args()
    # TODO: Ensure that this path is correct
    local_path = pathlib.Path("experiments") / args.exp / "results"
    local_path.mkdir(parents=True, exist_ok=True)

    if args.method == "rsync":
        _sync_rsync(args.remote, args.exp, local_path)
    else:
        src_ep = args.src_endpoint or os.getenv("GLOBUS_SRC_EP")
        dst_ep = args.dst_endpoint or os.getenv("GLOBUS_DST_EP")
        if not src_ep or not dst_ep:
            sys.exit(
                "Globus mode requires --src-endpoint and --dst-endpoint "
                "or env vars GLOBUS_SRC_EP / GLOBUS_DST_EP."
            )
        remote_rel = f"/~/experiments/{args.exp}/results/"
        _sync_globus(src_ep, dst_ep, remote_rel, local_path)


if __name__ == "__main__":
    main()
