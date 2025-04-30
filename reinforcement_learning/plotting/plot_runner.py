"""
Command-line entry-point:

    python plot_runner.py --config plot_config.yml

It walks the YAML, loads raw JSON via loaders.py, processes it,
and hands tidy data to the chosen plotting function.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import yaml
from reinforcement_learning.plotting.loaders import load_metric_for_runs
from reinforcement_learning.plotting import processors
from reinforcement_learning.plotting.registry import PLOTS


def _collect_run_ids(cfg_algo: str, cfg_variants: list[dict], drl: bool) -> list[str]:
    """
    Helper: build the list of run-IDs we need to pull from disk.
    """
    if not cfg_variants:
        # fallback – expected naming scheme
        variant_tag = "drl" if drl else "baseline"
        return [f"{cfg_algo}_{variant_tag}"]

    return [variant["run_id"] for variant in cfg_variants]


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for plot_name in cfg["plots"]:
        plot_meta = PLOTS[plot_name]
        plot_fn = plot_meta["plot"]
        proc_fn = getattr(processors, plot_meta["process"])

        for algo, variants in cfg["runs"]["variants"].items():
            for run_type in ("drl", "non_drl"):
                if not cfg["runs"].get(run_type, False):
                    continue

                drl_flag = run_type == "drl"
                run_ids = _collect_run_ids(algo, variants, drl_flag)
                raw_metric = load_metric_for_runs(run_ids, plot_name, drl=drl_flag)

                if not raw_metric:
                    continue  # nothing on disk

                processed = proc_fn(raw_metric)

                # optional save directory can be specified in YAML
                save_dir = cfg.get("save_dir")
                if save_dir:
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    plot_fn(processed, save_path=None, title=None)  # adjust if your plotter signature allows save_path
                else:
                    plot_fn(processed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="plot_config.yml", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
