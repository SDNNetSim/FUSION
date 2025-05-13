# transponders.py  – copy‐paste & change y‑label for hops / lengths
import matplotlib.pyplot as plt


def _plot_metric(
        data,
        ylabel,
        title,
        save_path=None,
        show_band=False,  # <‑‑ add
):
    plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(10, 6), dpi=300)

    for algo, tv_dict in data.items():
        erlangs, means, mins, maxs = [], [], [], []
        for tv, stats in tv_dict.items():
            erlangs.append(float(tv))
            means.append(stats["mean"])
            mins.append(stats["min"])
            maxs.append(stats["max"])

        # sort by erlang
        erlangs, means, mins, maxs = zip(*sorted(zip(erlangs, means, mins, maxs)))
        plt.plot(erlangs, means, marker='o', linewidth=2, label=algo)

        if show_band:  # <‑‑ gate the shading
            plt.fill_between(
                erlangs,
                mins,
                maxs,
                alpha=0.15,  # lighter
                zorder=1,  # behind line
            )

    plt.xlabel("Erlang Values", fontsize=14, fontweight='bold')
    plt.ylabel(ylabel, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        plt.savefig(save_path)
        print(f"[plot_{ylabel}] ✅ Saved {save_path}")
        plt.close()
    else:
        plt.show()
        plt.clf()


def plot_transponders(data, save_path=None, title="Transponders"):
    _plot_metric(data, "Average # Transponders", title, save_path)


# ---------------- Min‑vs‑Max companion (cleaner style) -----------------
def _plot_metric_minmax(data, ylabel, title, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.style.use('seaborn-whitegrid' if 'seaborn-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # will collect one handle per algorithm for the 1st legend
    algo_handles = []

    # consistent style for the two bands
    style_min = dict(linestyle='--', linewidth=1.4, marker='v', markersize=4)
    style_max = dict(linestyle=':', linewidth=1.4, marker='^', markersize=4)

    for algo, tv_dict in data.items():
        erlangs, mins, maxs = [], [], []
        for tv, stats in tv_dict.items():
            erlangs.append(float(tv))
            mins.append(stats["min"])
            maxs.append(stats["max"])

        erlangs, mins, maxs = zip(*sorted(zip(erlangs, mins, maxs)))
        colour = ax._get_lines.get_next_color()

        # MIN
        ax.plot(
            erlangs, mins, color=colour, label=algo, **style_min
        )
        # store a handle for legend‑1 ONLY ONCE (min curve)
        algo_handles.append(ax.lines[-1])

        # MAX   (no label so it won't appear twice)
        ax.plot(
            erlangs, maxs, color=colour, label='_nolegend_', **style_max
        )

    ax.set_xlabel("Erlang Values", fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', linewidth=0.5)

    # ── LEGENDS ─────────────────────────────────────────────────────────
    # 1) algorithms   (handles collected above)
    first_leg = ax.legend(
        handles=algo_handles,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=11,
        title="Algorithms",
    )
    ax.add_artist(first_leg)  # keep it when we add a second legend

    # 2) line‑style key
    style_handles = [
        Line2D([], [], color='k', **style_min, label='min'),
        Line2D([], [], color='k', **style_max, label='max'),
    ]
    ax.legend(
        handles=style_handles,
        bbox_to_anchor=(1.02, 0.45),
        loc='upper left',
        fontsize=11,
        title="Statistic",
    )

    plt.tight_layout(rect=[0, 0, 0.8, 1])

    if save_path:
        plt.savefig(save_path)
        print(f"[plot_{ylabel}_minmax] ✅ Saved {save_path}")
        plt.close(fig)
    else:
        plt.show()
        plt.clf()


# Public wrappers (mirror for hops / lengths)
def plot_transponders_minmax(data, save_path=None, title="Transponders – min vs max"):
    _plot_metric_minmax(data, "# Transponders", title, save_path)
