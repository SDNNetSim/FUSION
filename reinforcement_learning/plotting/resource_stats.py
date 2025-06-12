from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap, BoundaryNorm

def get_resource_usage_colormap_and_norm():
    """
    Return a colormap and normalization for resource usage heatmaps
    using the same bucket thresholds as _bucket() in the resource table.
    """
    # Use colors in order of buckets from best to worst
    colors = [
        "mediumseagreen",      # Large Benefit (≤ -20)
        "palegreen",           # Moderate Benefit (≤ -10)
        "honeydew",            # Small Benefit (≤ -5)
        "lightyellow",         # Negligible (< 5)
        "moccasin",            # Small Disadvantage (< 10)
        "lightsalmon",         # Moderate Disadvantage (< 20)
        "lightcoral",          # Large Disadvantage (≥ 20)
    ]

    # Edges matching _bucket logic exactly
    bounds = [-21, -20, -10, -5, 5, 10, 20, 21]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, len(colors))

    return cmap, norm


def plot_resource_percent_delta_heatmaps(
        processed_dict: dict,
        save_path: Path | None = None,  # pylint: disable=unsupported-binary-operation
        title_prefix: str = "Resource Usage %Δ vs Baseline",

):
    """
    Generate three separate heatmaps for percentage deltas in:
    - Path Length
    - Hop Count
    - Transponders
    Compared to both congestion-aware and k=4 baselines.

    Parameters:
        processed_dict (dict): Must contain keys 'lengths', 'hops', 'trp'.
        save_path (Path): Base path for saving figures (if desired).
        title_prefix (str): Prefix for each plot title.
    """
    lengths = processed_dict["lengths"]
    hops = processed_dict["hops"]
    trps = processed_dict["trp"]

    baselines = {
        "cong_aware": "Congestion Aware",
        "k_shortest_path_4": "K-Shortest Path (k=4)",
    }
    metrics = {
        "Length (km)": lengths,
        "Hop Count": hops,
        "Transponders": trps,
    }

    def get_algos(d):
        return sorted([a for a in d.keys() if a not in {"cong_aware", "k_shortest_path_4", "k_shortest_path_1"}])

    def get_erlangs(d):
        return sorted({float(e) for a in d for e in d[a]})

    algos = get_algos(lengths)
    erlangs = [str(e) for e in get_erlangs(lengths)]

    for metric_name, metric_dict in metrics.items():
        for bl_key, bl_label in baselines.items():
            data = pd.DataFrame(index=algos, columns=erlangs, dtype=float)

            for algo in algos:
                for erlang in erlangs:
                    try:
                        algo_val = metric_dict[algo][erlang]["mean"]
                        base_val = metric_dict[bl_key][erlang]["mean"]
                        pct_delta = 100.0 * (algo_val - base_val) / base_val if base_val else np.nan
                        data.at[algo, erlang] = pct_delta
                    except KeyError:
                        data.at[algo, erlang] = np.nan

            _, ax = plt.subplots(figsize=(len(erlangs) * 0.9 + 3, len(algos) * 0.45 + 1.5), dpi=300)
            cmap, norm = get_resource_usage_colormap_and_norm()
            sns.heatmap(
                data,
                cmap=cmap,
                norm=norm,
                annot=True,
                fmt=".1f",
                linewidths=0.5,
                linecolor="gray",
                cbar_kws={"label": "Avg % Δ vs BL"},
                ax=ax
            )

            ax.set_title(f"{title_prefix} – {metric_name} vs {bl_label}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Erlang Traffic", fontsize=12, fontweight="bold")
            ax.set_ylabel("Algorithm", fontsize=12, fontweight="bold")
            ax.tick_params(axis='x', labelrotation=45)
            ax.tick_params(labelsize=10)

            plt.tight_layout()
            if save_path:
                fname = f"{metric_name.lower().replace(' ', '_')}_vs_{bl_key}.png"
                out_path = save_path.parent / fname if isinstance(save_path, Path) else Path(save_path) / fname
                plt.savefig(out_path, bbox_inches="tight")
                print(f"[plot_resource] ✅ Saved {out_path}")
                plt.close()
            else:
                plt.show()
