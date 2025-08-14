# pylint: disable=import-error

# TODO: (version 6) This is probably not scalable, find a more creative way or combine them somehow for a comprehensive
#   test
# TODO: (version 5.5) Ensure the starting directory is consistent
# TODO: (version 5.5) Ensure the text fixtures have README files
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List
import multiprocessing

from stable_baselines3.common.callbacks import CallbackList
from fusion.cli.config_setup import load_config
from fusion.sim.utils import get_start_time
from fusion.sim.network_simulator import run as run_simulation
from fusion.modules.rl import workflow_runner
from fusion.modules.rl.gymnasium_envs.general_sim_env import SimEnv
from fusion.modules.rl.utils.callbacks import EpisodicRewardCallback, LearnRateEntCallback
from fusion.cli.args.simulation_args import add_run_sim_args
from fusion.cli.args.registry import add_routing_args, add_spectrum_args, add_snr_args, add_sdn_args, add_stats_args

# TODO: (version 5.5) Use mock YML instead of using the originals from sb3

LOGGER = logging.getLogger(__name__)
IGNORE_KEYS = {'route_times_max', 'route_times_mean', 'route_times_min', 'sim_end_time'}


def run_rl_simulation(input_dict: dict, config_path: str):
    """
    Wrapper function to run RL simulations in the style expected by the comparison test.
    """

    # Load config directly without CLI args dependency
    base_args = {"run_id": "rl_test", "config_path": config_path}
    sim_dict = load_config(config_path, base_args)

    # RL only supports s1 - remove any other simulation threads from input_dict
    filtered_input_dict = {k: v for k, v in input_dict.items() if k == 's1' or not k.startswith('s')}

    # Create callbacks
    ep_call_obj = EpisodicRewardCallback(verbose=1)
    param_call_obj = LearnRateEntCallback(verbose=1)
    callback_list = CallbackList([ep_call_obj, param_call_obj])

    # RL simulations only run 's1' - ignore other simulation threads
    # Extract only s1 data from both sim_dict and input_dict
    if 's1' in sim_dict:
        thread_dict = sim_dict['s1'].copy()

        # Only apply s1 overrides from filtered_input_dict
        if 's1' in filtered_input_dict:
            thread_dict.update(filtered_input_dict['s1'])

        # Force single Erlang run for RL (prevent multiple erlang values from creating s2, s3, etc.)
        if 'erlang_start' in thread_dict and 'erlang_stop' in thread_dict:
            # Set erlang_stop = erlang_start to run only one Erlang value
            thread_dict['erlang_stop'] = thread_dict['erlang_start']

        # Force single thread/simulation for RL
        thread_dict['thread_erlangs'] = False
        if 'num_threads' in thread_dict:
            thread_dict['num_threads'] = 1

        # Debug: print configuration to see what's being passed
        print(f"DEBUG: Original input_dict keys: {list(input_dict.keys())}")
        print(f"DEBUG: Filtered input_dict keys: {list(filtered_input_dict.keys())}")
        print(f"DEBUG: RL sim_dict keys: {list(sim_dict.keys())}")
        print(f"DEBUG: thread_dict erlang_start: {thread_dict.get('erlang_start')}")
        print(f"DEBUG: thread_dict erlang_stop: {thread_dict.get('erlang_stop')}")
        print(f"DEBUG: thread_dict erlang_step: {thread_dict.get('erlang_step')}")

        thread_dict["callback"] = callback_list

        # Create single-thread sim_dict for RL (only s1)
        rl_sim_dict = {'s1': thread_dict}

        # Create environment for s1 only
        env = SimEnv(render_mode=None, custom_callback=ep_call_obj, sim_dict=rl_sim_dict)

        # Run the workflow for s1 only - workflow expects flat dict, environment expects nested dict
        workflow_runner.run(env=env, sim_dict=thread_dict, callback_list=callback_list)

        # Save stats for s1
        if hasattr(env, 'engine_obj') and hasattr(env.engine_obj, 'stats_obj'):
            stats_obj = env.engine_obj.stats_obj
            stats_obj.end_iter_update()
            stats_obj.save_stats(base_fp='data')


def _build_cli() -> argparse.Namespace:
    """Return parsed command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run_comparison",
        description="Execute every expected-results fixture and compare outputs.\n"
                    "This script compares simulation results against expected outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        default=False,
        help="Delete generated artefacts when the case passes (default: true).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity level.",
    )
    return parser.parse_args()


def _discover_cases(fixtures_root: Path) -> List[Path]:
    """Return all case directories under *fixtures_root*, sorted by name."""

    looks_like_case = (
            list(fixtures_root.glob("*_config.ini"))
            and (fixtures_root / "mod_formats.json").exists()
    )

    if looks_like_case:
        return [fixtures_root]  # ← always a list✅

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


def _compare_json(expected: Path, actual: Path, rel: str) -> List[str]:
    with expected.open(encoding="utf-8") as f_exp, actual.open(encoding="utf-8") as f_act:
        exp_data = json.load(f_exp)
        act_data = json.load(f_act)

    failures: List[str] = []

    def _walk(old: Dict, new: Dict, path: str = "") -> None:
        for key in old:
            cur = f"{path}.{key}" if path else key
            if key in IGNORE_KEYS or cur in IGNORE_KEYS:
                continue
            if key not in new:
                failures.append(f"{rel}:{cur} missing in actual")
            elif isinstance(old[key], dict) and isinstance(new[key], dict):
                _walk(old[key], new[key], cur)
            elif old[key] != new[key]:
                failures.append(f"{rel}:{cur} expected {old[key]!r} got {new[key]!r}")
        for key in new:
            cur = f"{path}.{key}" if path else key
            if key in IGNORE_KEYS or cur in IGNORE_KEYS:
                continue
            if key not in old:
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


def _find_output_directory(sim_dict: Dict, artefact_root: Path) -> Path:
    """Find the output directory for the simulation results."""
    output_dir = None

    for topo_dir in artefact_root.iterdir():
        if not topo_dir.is_dir():
            continue
        for date_dir in topo_dir.iterdir():
            if topo_dir.name != sim_dict['s1']['network'] or date_dir.name != sim_dict['s1']['date']:
                continue
            if not date_dir.is_dir():
                continue

            # Find the most recent time directory that contains an s1 subdirectory
            time_dirs = [d for d in date_dir.iterdir() if d.is_dir()]
            time_dirs.sort(key=lambda x: x.name, reverse=True)  # Most recent first

            for time_dir in time_dirs:
                s1_path = time_dir / "s1"
                if s1_path.exists() and s1_path.is_dir():
                    output_dir = s1_path
                    print(f"DEBUG: Found output directory: {output_dir}")
                    break

    return output_dir


def _compare_files(case_dir: Path, output_dir: Path, cleanup: bool) -> List[str]:
    """Compare expected files with actual output files."""
    produced = {p.name: p for p in output_dir.iterdir() if p.is_file()}
    config_path = _locate_single("*_config.ini", case_dir)
    failures: List[str] = []

    for expected in case_dir.iterdir():
        if expected.name in {config_path.name, "mod_formats.json"}:
            continue  # skip inputs

        expected_name = "_".join(expected.name.split("_")[-2:])
        actual = produced.get(expected_name)
        rel_name = f"{case_dir.name}/{expected.name}"

        if actual is None:
            failures.append(f"{rel_name}: expected file not produced")
            continue

        diff_lines = _diff_files(expected, actual, rel_name)
        failures.extend(diff_lines)

        if cleanup and not diff_lines:
            actual.unlink(missing_ok=True)

    return failures


def _run_single_case(case_dir: Path, base_args: Dict, cleanup: bool) -> bool:
    """Execute one regression case – return True if it passes."""
    LOGGER.info("▶  running case: %s", case_dir.name)

    # Mandatory inputs
    config_path = _locate_single("*_config.ini", case_dir)
    mod_assumption_path = _locate_single("mod_formats.json", case_dir)

    # Build args for this case and inject the custom mod_formats path
    args_for_case = _override_ini(base_args, config_path)

    sim_dict = load_config(str(config_path), args_for_case)
    sim_dict = get_start_time(sim_dict)
    sim_start = sim_dict['s1']['sim_start']

    # Inject the custom mod_formats path
    for thread_params in sim_dict.values():
        thread_params["mod_assumption_path"] = str(mod_assumption_path)
        thread_params['sim_start'] = sim_start

    # Kick off the simulation (inherits cwd set to repo root)
    if case_dir.name in ('epsilon_greedy_bandit', 'ucb_bandit', 'ppo'):
        LOGGER.info("▶ Running a REINFORCEMENT LEARNING simulation.")
        run_rl_simulation(input_dict=sim_dict, config_path=config_path)
    else:
        LOGGER.info("▶ Running a VANILLA simulation.")
        run_simulation(sims_dict=sim_dict, stop_flag=multiprocessing.Event())
    LOGGER.debug("simulation completed for %s", case_dir.name)

    # Locate the 's1' directory written by this run
    artefact_root = Path("data") / "output"
    output_dir = _find_output_directory(sim_dict, artefact_root)

    if output_dir is None:
        LOGGER.error("No output directory found after simulation under %s", artefact_root)
        return False

    # Compare against expected files
    failures = _compare_files(case_dir, output_dir, cleanup)

    # Outcome
    if failures:
        for msg in failures:
            LOGGER.error("FAIL: %s", msg)
        LOGGER.error("✖  %s FAILED (%d mismatch%s)", case_dir.name, len(failures),
                     "" if len(failures) == 1 else "es")
        return False

    LOGGER.info("✔  %s PASSED", case_dir.name)
    return True


def main() -> None:
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

    # Create a simple parser for comparison tests (no subcommands needed)

    simple_parser = argparse.ArgumentParser(description="Comparison Test Parser")
    add_run_sim_args(simple_parser)
    add_routing_args(simple_parser)
    add_spectrum_args(simple_parser)
    add_snr_args(simple_parser)
    add_sdn_args(simple_parser)
    add_stats_args(simple_parser)

    # Override required args with defaults for comparison tests
    # Only provide essential args, let the test-specific config files control other parameters
    default_args = [
        "--config_path", "ini/run_ini/config.ini",
        "--run_id", "comparison_test"
    ]
    base_args = vars(simple_parser.parse_args(default_args))

    # Remove non-essential args that could override test-specific config values
    # Keep only the essential args needed for the test framework
    essential_args = {'config_path', 'run_id'}
    base_args = {k: v for k, v in base_args.items() if k in essential_args or v is None}

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
