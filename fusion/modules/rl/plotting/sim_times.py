# TODO: (version 5.5-6) Consider having a function like this in a shared plotting module

import matplotlib.pyplot as plt
import numpy as np


def plot_sim_times(
    sim_times_data: dict,
    title: str = "Simulation Time by Traffic Volume",
    save_path: str | None = None,  # pylint: disable=unsupported-binary-operation
    log_y: bool = False,
):
    """
    Draws a grouped bar-chart of mean simulation time per traffic volume.

    Parameters
    ----------
    sim_times_data : dict
        Processed mapping: {algorithm: {traffic_volume: mean_time_seconds}}
    title : str
        Figure title.
    save_path : str or None
        If given, the figure is saved to this path.
    log_y : bool
        If True, use a log scale on the y-axis.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not sim_times_data:
        print("No simulation-time data to plot.")
        return None

    available_styles = plt.style.available
    if "seaborn-whitegrid" in available_styles:
        plt.style.use("seaborn-whitegrid")
    elif "seaborn-white" in available_styles:
        plt.style.use("seaborn-white")
    else:
        plt.style.use("default")

    traffic_labels = sorted(
        {
            float(tv)
            for algo_data in sim_times_data.values()
            for tv in algo_data.keys()
            if int(float(tv)) % 100 == 0
        }
    )
    algos = sorted(sim_times_data.keys())

    means = np.array(
        [
            [sim_times_data[algo].get(str(tv), 0.0) for algo in algos]
            for tv in traffic_labels
        ]
    )  # shape: (#traffic_volumes, #algorithms)

    x = np.arange(len(traffic_labels))
    bar_w = 0.8 / len(algos)

    fig = plt.figure(figsize=(14, 6), dpi=300)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    algo_colors = {algo: colors[i % len(colors)] for i, algo in enumerate(algos)}

    for i, algo in enumerate(algos):
        hatch_style = "//" if "k_shortest_path" in algo else None
        plt.bar(
            x + i * bar_w,
            means[:, i],
            bar_w,
            label=algo,
            color=algo_colors[algo],
            alpha=0.9,
            hatch=hatch_style,
            edgecolor="black",
            linewidth=0.7,
        )

    plt.xticks(
        x + bar_w * (len(algos) / 2 - 0.5),
        [str(int(tv)) for tv in traffic_labels],
        rotation=45,
        ha="right",
        fontsize=12,
    )
    plt.xlabel("Traffic Volume (Erlang)", fontsize=14, fontweight="bold")
    plt.ylabel("Mean Simulation Time (s)", fontsize=14, fontweight="bold")
    plt.yscale("log", base=10)
    plt.title(title, fontsize=16, fontweight="bold")

    if log_y:
        plt.yscale("log")

    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(
        title="Algorithm",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=12,
        title_fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[plot_sim_times] ✅ Saved: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.clf()

    return fig
