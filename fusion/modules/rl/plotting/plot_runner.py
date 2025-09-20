# pylint: disable=unsupported-binary-operation

# Standard library imports
import argparse
from inspect import signature
from pathlib import Path

# Standard library imports
from typing import Dict, List, Optional, Any

# Third-party imports
import yaml

# Local imports
from fusion.modules.rl.plotting.errors import InvalidConfigurationError, PlottingFileNotFoundError
from fusion.modules.rl.plotting.loaders import load_metric_for_runs, discover_all_run_ids
from fusion.modules.rl.plotting import processors
from fusion.modules.rl.plotting.registry import PLOTS


def call_processor(processor_function, raw_runs: Dict[str, Any], runid_to_algo: Dict[str, str], **context) -> Any:
    """
    Call processor_function with (raw_runs, runid_to_algo) and pass **context
    only if the function signature accepts it.
    
    :param processor_function: Function to process the data
    :param raw_runs: Raw run data
    :type raw_runs: Dict[str, Any]
    :param runid_to_algo: Mapping from run ID to algorithm name
    :type runid_to_algo: Dict[str, str]
    :param context: Additional context parameters
    :return: Processed data
    :rtype: Any
    """
    parameters = signature(processor_function).parameters
    if len(parameters) >= 3:  # processor wants a 3rd arg
        return processor_function(raw_runs, runid_to_algo, context)
    return processor_function(raw_runs, runid_to_algo)  # legacy 2-arg processors


def _collect_run_ids(config_algorithm: str, config_variants: List[Dict[str, Any]], discovered: Dict[str, List[str]]) -> List[str]:
    """Collect run IDs for a specific algorithm from configuration or discovered runs."""
    if config_variants:
        return [variant["run_id"] for variant in config_variants]
    # include partial match
    return [
        run_id
        for key, run_ids in discovered.items()
        if key.startswith(config_algorithm)
        for run_id in run_ids
    ]


def _load_and_validate_cfg(config_path: str) -> dict:
    """Load and validate plotting configuration from YAML file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise PlottingFileNotFoundError(f"Failed to load configuration file {config_path}: {exc}") from exc

    if not config.get("network") or not config.get("dates"):
        raise InvalidConfigurationError("YAML configuration must contain 'network' and 'dates' fields.")

    return config


def _get_algorithms(config: Dict[str, Any], network: str, dates: List[str], observation_spaces: Optional[List[str]]) -> List[str]:
    """Get list of algorithms to process from configuration or discovery."""
    variants = config["runs"].get("variants", {})
    if variants:
        return list(variants.keys())

    if config.get("algorithms"):
        return config["algorithms"]

    all_drl = discover_all_run_ids(network, dates, drl=True, obs_filter=observation_spaces)
    all_non_drl = discover_all_run_ids(network, dates, drl=False, obs_filter=None)
    return sorted(set(all_drl) | set(all_non_drl))


def _process_plot(config: Dict[str, Any], plot_name: str, network: str, dates: List[str], algorithms: List[str],
                  observation_spaces: Optional[List[str]]):
    plot_metadata = PLOTS[plot_name]
    plot_function = plot_metadata["plot"]
    processor_function = getattr(processors, plot_metadata["process"])

    variants_block = config["runs"].get("variants", {})
    combined_raw, combined_runid_to_algo, combined_start_stamps = {}, {}, {}

    for run_type in ("drl", "non_drl"):
        if not config["runs"].get(run_type, False):
            continue

        drl_flag = run_type == "drl"
        discovered = discover_all_run_ids(network, dates, drl=drl_flag, obs_filter=observation_spaces)

        for algorithm in algorithms:
            run_ids = _collect_run_ids(algorithm, variants_block.get(algorithm, []), discovered)
            print(f"[DEBUG] Using run_ids for {algorithm} ({'DRL' if drl_flag else 'non-DRL'}): {run_ids}")
            if not run_ids:
                continue

            raw_metric, runid_to_algo, start_stamps = load_metric_for_runs(
                run_ids=set(run_ids),
                metric=plot_name,
                drl=drl_flag,
                network=network,
                dates=dates
            )
            combined_raw.update(raw_metric)
            combined_runid_to_algo.update(runid_to_algo)
            combined_start_stamps.update(start_stamps)

    if not combined_raw:
        return

    context = {"start_stamps": combined_start_stamps} if plot_name == "sim_times" else {}

    processed = call_processor(processor_function, combined_raw, combined_runid_to_algo, **context)

    save_directory = config.get("save_dir")
    title = f"{plot_name.capitalize()} – {network}"

    if save_directory:
        filename = f"{plot_name}_{network}.png"
        plot_path = Path(save_directory) / filename
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        plot_function(processed, save_path=plot_path, title=title)
    else:
        plot_function(processed, save_path=None, title=title)


def main(config_path: str):
    """
    Entrypoint to control the plotting script.
    """
    config = _load_and_validate_cfg(config_path)
    network = config["network"]
    dates = config["dates"]
    observation_spaces = config.get("observation_spaces", None)

    algorithms = _get_algorithms(config, network, dates, observation_spaces)

    for plot_name in config["plots"]:
        _process_plot(config, plot_name, network, dates, algorithms, observation_spaces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="plot_config.yml", help="Path to plot YAML")
    args = parser.parse_args()
    main(args.config)
