import numpy as np
import matplotlib.pyplot as plt


def plot_memory_usage(
        memory_usage_data: dict,
        title: str = "Memory Usage by Traffic Volume",
        save_path: str | None = None,  # pylint: disable=unsupported-binary-operation
        log_y: bool = False,
):
    """
    Grouped bar-chart of average peak memory (MB).

    Parameters
    ----------
    memory_usage_data : dict
        {algo: {traffic_volume: {'overall': mean_MB}}}
    title : str
        Plot title.
    save_path : str | None
        Path to save image, if given.
    log_y : bool
        True → log-scale y-axis.
    """
    if not memory_usage_data:
        print("[plot_memory_usage] ⚠️ No data provided.")
        return None

    avail = plt.style.available
    if "seaborn-whitegrid" in avail:
        plt.style.use("seaborn-whitegrid")
    elif "seaborn-white" in avail:
        plt.style.use("seaborn-white")
    else:
        plt.style.use("default")

    raw_labels = {
        tv for algo_data in memory_usage_data.values() for tv in algo_data.keys()
    }
    only_overall = raw_labels == {"overall"}

    if only_overall:
        traffic_labels = ["overall"]
    else:
        traffic_labels = sorted(float(lbl) for lbl in raw_labels if lbl != "overall")

    algos = sorted(memory_usage_data.keys())

    means_list = []
    for tv in traffic_labels:
        row_list = []
        for algo in algos:
            val_dict = memory_usage_data[algo].get(str(tv), {})
            row_list.append(float(val_dict.get("overall", 0.0)))
        means_list.append(row_list)
    means_arr = np.array(means_list)  # shape: (#traffic, #algorithms)

    x = np.arange(len(traffic_labels))
    bar_w = 0.8 / len(algos)

    plt.figure(figsize=(14, 6), dpi=300)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    algo_colors = {a: palette[i % len(palette)] for i, a in enumerate(algos)}

    for i, algo in enumerate(algos):
        hatch = "//" if "k_shortest_path" in algo else None
        plt.bar(
            x + i * bar_w,
            means_arr[:, i],
            bar_w,
            label=algo,
            color=algo_colors[algo],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            hatch=hatch,
        )

    tick_labels = (
        ["overall"]
        if only_overall
        else [str(int(tv)) for tv in traffic_labels]
    )
    plt.xticks(
        x + bar_w * (len(algos) / 2 - 0.5),
        tick_labels,
        rotation=45,
        ha="right",
        fontsize=12,
    )
    plt.xlabel(
        "Traffic Volume (Erlang)" if not only_overall else "Scenario",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Mean Peak Memory (MB)", fontsize=14, fontweight="bold")
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
        print(f"[plot_memory_usage] ✅ Saved: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.clf()

    return plt.gcf()
