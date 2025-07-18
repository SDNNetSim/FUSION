from __future__ import annotations

import os
import argparse
import json
import sys
import traceback
from pathlib import Path

from reinforcement_learning.workflow_runner import (  # type: ignore
    run_optuna_study,
    run,
)
from reinforcement_learning.utils.gym_envs import create_environment  # type: ignore

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# TODO: (drl_path_agents) Put 'utils' file ending (imports) in the standards
#       and guidelines
# TODO: (drl_path_agents) No support for core or spectrum assignment
# TODO: (drl_path_agents) Does not support multi-band
# TODO: (drl_path_agents) Q-Learning does not save iteratively
# TODO: (drl_path_agents) Create run mods directory and raise an error if it
#       doesn't exist
# TODO: (base_agent) Fix load_model function to prevent file_prefix error
#       outside of training
# TODO: Add this to a helper or utils script
# TODO: Run multiple erlangs support
def json_friendly(d: dict) -> dict:
    """
    Recursively strip unserializable values from a dict.
    Keeps only JSON-serializable keys and values.
    """

    def safe(v):
        try:
            json.dumps(v)
            return v
        except (TypeError, OverflowError):
            return str(v)

    return {k: safe(v) for k, v in d.items()}


def _extract_bookkeeping_flags() -> tuple[argparse.Namespace, list[str]]:
    """
    Pull ``--run_id`` and ``--output_dir`` out of ``sys.argv`` without touching
    the rest of the command-line parameters.  Returns the parsed bookkeeping
    args as well as the *remaining* argv list so we can forward it verbatim to
    the environment builder.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run_id", default=None)
    pre_parser.add_argument("--output_dir", default=None)

    parsed_args, remaining_argv = pre_parser.parse_known_args()
    return parsed_args, remaining_argv


def _write_bookkeeping_files(
        sim_dict: dict,
        run_id: str | None,
        output_dir: Path | None,
        had_error: bool = False,
        is_testing: bool = False,
) -> None:
    """
    Write ``meta.json`` into *output_dir* and print the path so the bash
    wrapper can capture it.  If *output_dir* is ``None`` do nothing.
    Exit with 0 only if no error occurred.
    """
    if output_dir is None:
        sys.exit(1 if had_error else 0)

    output_dir.mkdir(parents=True, exist_ok=True)
    sim_dict["run_id"] = run_id

    with (output_dir / "meta.json").open("w", encoding="utf-8") as mfile:
        json.dump(json_friendly(sim_dict), mfile, indent=2)

    print(f"OUTPUT_DIR={output_dir}")

    if not is_testing:
        sys.exit(1 if had_error else 0)


def run_rl_sim(input_dict: dict = None, is_testing: bool = False, config_path: Path = None) -> None:
    """
    Main function orchestrating a single simulation run.
    """
    bk_args, remaining_argv = _extract_bookkeeping_flags()
    sys.argv = [sys.argv[0], *remaining_argv]

    env, sim_dict, callback_list = create_environment(config_path=config_path)

    # TODO: (version 6.0) Three or four different logged times!? Please fix!
    if input_dict is not None:
        env.sim_dict.update(input_dict['s1'])
        parts = env.engine_obj.sim_info.rsplit("/", 1)
        parts[-1] = input_dict['s1']['sim_start']

        env.engine_obj.sim_info = "/".join(parts)
        env.engine_obj.stats_obj.sim_info = "/".join(parts)
        sim_dict.update(input_dict['s1'])

    out_path = Path('data') / 'output' / sim_dict['network'] / sim_dict['date'] / sim_dict['sim_start'] / sim_dict[
        'thread_num']

    had_error = False

    try:
        if not sim_dict["optimize"] and not sim_dict["optimize_hyperparameters"]:
            run(env=env, sim_dict=sim_dict, callback_list=callback_list)
        else:
            run_optuna_study(sim_dict=sim_dict, callback_list=callback_list)
    # TODO: (version 5.5-6) Address this
    except Exception:  # pylint: disable=broad-exception-caught
        traceback.print_exc()
        had_error = True
    finally:
        # Always try to write bookkeeping files
        _write_bookkeeping_files(sim_dict, bk_args.run_id, out_path, had_error, is_testing=is_testing)


if __name__ == "__main__":
    run_rl_sim()
