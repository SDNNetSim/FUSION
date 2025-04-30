# ✅ plot_runner.py (updated to pass runid→algorithm to processors)
import argparse
import yaml
from pathlib import Path
from reinforcement_learning.plotting.loaders import load_metric_for_runs, discover_all_run_ids
from reinforcement_learning.plotting import processors
from reinforcement_learning.plotting.registry import PLOTS


def _collect_run_ids(cfg_algo: str, cfg_variants: list[dict], discovered: dict[str, list[str]]) -> list[str]:
    if cfg_variants:
        return [v["run_id"] for v in cfg_variants]
    return discovered.get(cfg_algo, [])


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    network = cfg.get("network")
    dates = cfg.get("dates")
    if not network or not dates:
        raise ValueError("YAML must contain 'network' and 'dates' fields.")

    for plot_name in cfg["plots"]:
        plot_meta = PLOTS[plot_name]
        plot_fn = plot_meta["plot"]
        proc_fn = getattr(processors, plot_meta["process"])

        variants_block = cfg["runs"].get("variants", {})
        explicit_algos = cfg.get("algorithms", [])

        if variants_block:
            algos = list(variants_block.keys())
        elif explicit_algos:
            algos = explicit_algos
        else:
            all_drl = discover_all_run_ids(network, dates, drl=True)
            all_non_drl = discover_all_run_ids(network, dates, drl=False)
            algos = sorted(set(all_drl.keys()) | set(all_non_drl.keys()))

        combined_raw = {}
        combined_runid_to_algo = {}

        for run_type in ("drl", "non_drl"):
            if not cfg["runs"].get(run_type, False):
                continue

            drl_flag = run_type == "drl"
            discovered = discover_all_run_ids(network, dates, drl=drl_flag)

            for algo in algos:
                variants = variants_block.get(algo, [])
                run_ids = _collect_run_ids(algo, variants, discovered)

                raw_metric, runid_to_algo = load_metric_for_runs(
                    run_ids=run_ids,
                    metric=plot_name,
                    drl=drl_flag,
                    network=network,
                    dates=dates
                )

                combined_raw.update(raw_metric)
                combined_runid_to_algo.update(runid_to_algo)

        if not combined_raw:
            continue

        processed = proc_fn(combined_raw, combined_runid_to_algo)

        save_dir = cfg.get("save_dir")
        title = f"{plot_name.capitalize()} – {network}"

        if save_dir:
            filename = f"{plot_name}_{network}.png"
            plot_path = Path(save_dir) / filename
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            plot_fn(processed, save_path=plot_path, title=title)
        else:
            plot_fn(processed, save_path=None, title=title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="plot_config.yml", help="Path to plot YAML")
    args = parser.parse_args()
    main(args.config)
