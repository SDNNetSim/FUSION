# TODO: (version 6) This is probably not scalable, find a more creative way or combine them somehow for a comprehensive
#   test
# TODO: (version 5.5) Ensure the starting directory is consistent

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone regression harness for the optical-network simulator.

  • Discovers every case under tests/fixtures/expected_results/<CASE_NAME>/
  • Overrides the simulator's ini, sets mod_assumption_path, runs the sim
  • Harvests artefacts produced *after* the run and diffs them against the
    fixtures (exact match, no tolerance)
  • Prints pylint-style diagnostics and exits non-zero on failure
  • Deletes artefacts for passing cases unless --no-cleanup is supplied

This script is **not** collected by pytest; run it directly:
    python -m tests.run_comparison
"""
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from config_scripts.parse_args import parse_args
from config_scripts.setup_config import read_config
from run_sim import run as run_simulation

LOGGER = logging.getLogger(__name__)


# ───────────────────────────── CLI / ENTRY-POINT ────────────────────────────


def _build_cli() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_comparison",
        description="Execute every expected-results fixture and compare outputs.",
    )
    parser.add_argument(
        "--fixtures-root",
        type=Path,
        default=(
                Path(__file__).resolve().parent / "fixtures" / "expected_results"
        ),
        help="Root directory that contains per-case expected results.",
    )
    parser.add_argument(
        "--cleanup",
        dest="cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete generated artefacts when the case passes (default: true).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity level.",
    )
    return parser.parse_args()


# ─────────────────────── Discovery / Validation Helpers ─────────────────────


def _discover_cases(fixtures_root: Path) -> List[Path]:
    """Return all case directories under *fixtures_root*, sorted by name."""
    cases = sorted([p for p in fixtures_root.iterdir() if p.is_dir()])
    if not cases:
        LOGGER.error("No cases found under %s", fixtures_root)
        sys.exit(2)
    return cases


def _locate_single(pattern: str, directory: Path) -> Path:
    """Return exactly one file matching *pattern* or exit with error."""
    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        LOGGER.error(
            "Expected exactly one '%s' in %s, found %d", pattern, directory, len(matches)
        )
        sys.exit(2)
    return matches[0]


def _override_ini(base_args: Dict, new_config: Path) -> Dict:
    """Return a copy of *base_args* with config_path swapped out."""
    result = dict(base_args)
    result["config_path"] = str(new_config)
    return result


# ───────────────────────── File-level Diff Functions ────────────────────────


def _compare_json(expected: Path, actual: Path, rel: str) -> List[str]:
    with expected.open(encoding="utf-8") as f_exp, actual.open(encoding="utf-8") as f_act:
        exp_data = json.load(f_exp)
        act_data = json.load(f_act)

    failures: List[str] = []

    def _walk(old: Dict, new: Dict, path: str = "") -> None:
        for key in old:
            cur = f"{path}.{key}" if path else key
            if key not in new:
                failures.append(f"{rel}:{cur} missing in actual")
            elif isinstance(old[key], dict) and isinstance(new[key], dict):
                _walk(old[key], new[key], cur)
            elif old[key] != new[key]:
                failures.append(f"{rel}:{cur} expected {old[key]!r} got {new[key]!r}")
        for key in new:
            if key not in old:
                cur = f"{path}.{key}" if path else key
                failures.append(f"{rel}:{cur} extra in actual")

    _walk(exp_data, act_data)
    return failures


def _compare_text(expected: Path, actual: Path, rel: str) -> List[str]:
    with expected.open(encoding="utf-8") as f_exp, actual.open(encoding="utf-8") as f_act:
        exp_lines = f_exp.readlines()
        act_lines = f_act.readlines()

    if exp_lines == act_lines:
        return []

    diff = difflib.unified_diff(
        exp_lines, act_lines, fromfile=f"expected/{rel}", tofile=f"actual/{rel}", lineterm=""
    )
    return list(diff)


def _compare_binary(expected: Path, actual: Path, rel: str) -> List[str]:
    if expected.read_bytes() == actual.read_bytes():
        return []
    return [f"{rel}: binary files differ"]


def _diff_files(expected: Path, actual: Path, rel: str) -> List[str]:
    suffix = expected.suffix.lower()
    if suffix == ".json":
        return _compare_json(expected, actual, rel)
    if suffix in {".txt", ".csv", ".ini"}:
        return _compare_text(expected, actual, rel)
    return _compare_binary(expected, actual, rel)


# ──────────────────────────── Per-case Execution ────────────────────────────


def _run_single_case(case_dir: Path, base_args: Dict, cleanup: bool) -> bool:
    """Execute one regression case – return True if it passes."""
    LOGGER.info("▶  running case: %s", case_dir.name)

    # Mandatory inputs
    config_path = _locate_single("*_config.ini", case_dir)
    mod_assumption_path = _locate_single("mod_formats.json", case_dir)

    start_ts = time.time()  # timestamp before simulation

    # Build args for this case and inject the custom mod_formats path
    args_for_case = _override_ini(base_args, config_path)
    sim_dict = read_config(args_dict=args_for_case, config_path=str(config_path))

    for thread_params in sim_dict.values():
        thread_params["mod_assumption_path"] = str(mod_assumption_path)

    # Kick off the simulation (inherits cwd set to repo root)
    run_simulation(sims_dict=sim_dict)
    LOGGER.debug("simulation completed for %s", case_dir.name)

    # for start_ts in ['10_37_46_978428', '10_38_16_959304']:
    # Collect artefacts produced since *start_ts*
    artefact_root = Path("data") / "output"
    produced: Dict[str, Path] = {}
    for file_path in artefact_root.rglob("*"):
        if file_path.is_file() and file_path.stat().st_mtime >= start_ts:
            produced.setdefault(file_path.name, file_path)

    if not produced:
        LOGGER.error("%s: no artefacts created under %s", case_dir.name, artefact_root)
        return False

    # Compare against expected files
    failures: List[str] = []
    for expected in case_dir.iterdir():
        if expected.name in {config_path.name, "mod_formats.json"}:
            continue  # skip inputs

        actual = produced.get(expected.name)
        rel_name = f"{case_dir.name}/{expected.name}"

        if actual is None:
            failures.append(f"{rel_name}: expected file not produced")
            continue

        diff_lines = _diff_files(expected, actual, rel_name)
        failures.extend(diff_lines)

        if cleanup and not diff_lines:
            actual.unlink(missing_ok=True)

    # Outcome
    if failures:
        for msg in failures:
            LOGGER.error("FAIL: %s", msg)
        LOGGER.error("✖  %s FAILED (%d mismatch%s)", case_dir.name, len(failures),
                     "" if len(failures) == 1 else "es")
        return False

    LOGGER.info("✔  %s PASSED", case_dir.name)
    return True


# ────────────────────────────────── MAIN ────────────────────────────────────


def main() -> None:  # noqa: D401
    """Entry-point."""
    cli = _build_cli()

    # Absolute path to fixtures (must survive cwd change)
    fixtures_root = cli.fixtures_root.resolve()

    # Switch cwd to repo root so 'data/raw' etc. are reachable
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    logging.basicConfig(
        level=getattr(logging, cli.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    cases = _discover_cases(fixtures_root)
    base_args = parse_args()  # returns args_dict

    all_ok = True
    for case in cases:
        all_ok &= _run_single_case(case, base_args, cleanup=cli.cleanup)

    if all_ok:
        LOGGER.info("All %d cases passed ✓", len(cases))
        sys.exit(0)

    LOGGER.error("One or more cases FAILED")
    sys.exit(1)


if __name__ == "__main__":
    main()
