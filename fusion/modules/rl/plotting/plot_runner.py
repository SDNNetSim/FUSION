# pylint: disable=unsupported-binary-operation

from inspect import signature
from pathlib import Path
import argparse
import yaml
from fusion.modules.rl.plotting.loaders import load_metric_for_runs, discover_all_run_ids
from fusion.modules.rl.plotting import processors
from fusion.modules.rl.plotting.registry import PLOTS


def call_processor(proc_fn, raw_runs, runid_to_algo, **context):
    """
    Call *proc_fn* with (raw_runs, runid_to_algo) and pass **context
    only if the function signature accepts it.
    """
    params = signature(proc_fn).parameters
    if len(params) >= 3:  # proc wants a 3rd arg
        return proc_fn(raw_runs, runid_to_algo, context)
    return proc_fn(raw_runs, runid_to_algo)  # legacy 2-arg processors


def _collect_run_ids(cfg_algo: str, cfg_variants: list[dict], discovered: dict[str, list[str]]) -> list[str]:
    if cfg_variants:
        return [v["run_id"] for v in cfg_variants]
    # include partial match
    return [
        run_id
        for key, run_ids in discovered.items()
        if key.startswith(cfg_algo)
        for run_id in run_ids
    ]


def _load_and_validate_cfg(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not cfg.get("network") or not cfg.get("dates"):
        raise ValueError("YAML must contain 'network' and 'dates' fields.")

    return cfg


def _get_algorithms(cfg: dict, network: str, dates: list[str], obs_spaces: list[str] | None) -> list[str]:
    variants = cfg["runs"].get("variants", {})
    if variants:
        return list(variants.keys())

    if cfg.get("algorithms"):
        return cfg["algorithms"]

    all_drl = discover_all_run_ids(network, dates, drl=True, obs_filter=obs_spaces)
    all_non_drl = discover_all_run_ids(network, dates, drl=False, obs_filter=None)
    return sorted(set(all_drl) | set(all_non_drl))


def _process_plot(cfg: dict, plot_name: str, network: str, dates: list[str], algos: list[str],
                  obs_spaces: list[str] | None):
    plot_meta = PLOTS[plot_name]
    plot_fn = plot_meta["plot"]
    proc_fn = getattr(processors, plot_meta["process"])

    variants_block = cfg["runs"].get("variants", {})
    combined_raw, combined_runid_to_algo, combined_start_stamps = {}, {}, {}

    for run_type in ("drl", "non_drl"):
        if not cfg["runs"].get(run_type, False):
            continue

        drl_flag = run_type == "drl"
        discovered = discover_all_run_ids(network, dates, drl=drl_flag, obs_filter=obs_spaces)

        for algo in algos:
            run_ids = _collect_run_ids(algo, variants_block.get(algo, []), discovered)
            print(f"[DEBUG] Using run_ids for {algo} ({'DRL' if drl_flag else 'non-DRL'}): {run_ids}")
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

    processed = call_processor(proc_fn, combined_raw, combined_runid_to_algo, **context)

    save_dir = cfg.get("save_dir")
    title = f"{plot_name.capitalize()} – {network}"

    if save_dir:
        filename = f"{plot_name}_{network}.png"
        plot_path = Path(save_dir) / filename
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plot_fn(processed, save_path=plot_path, title=title)
    else:
        plot_fn(processed, save_path=None, title=title)


def main(cfg_path: str):
    """
    Entrypoint to control the plotting script.
    """
    cfg = _load_and_validate_cfg(cfg_path)
    network = cfg["network"]
    dates = cfg["dates"]
    obs_spaces = cfg.get("observation_spaces", None)

    algos = _get_algorithms(cfg, network, dates, obs_spaces)

    for plot_name in cfg["plots"]:
        _process_plot(cfg, plot_name, network, dates, algos, obs_spaces)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="plot_config.yml", help="Path to plot YAML")
    args = parser.parse_args()
    main(args.config)
