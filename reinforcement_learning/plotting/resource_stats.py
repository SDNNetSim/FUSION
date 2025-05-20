"""resource_stats.py

Generate a comparison table that summarises mean / min / max **path length (km),
hop count, and transponder count** for every algorithm and Erlang traffic level,
relative to two baselines (congestion‑aware and k‑shortest‑path, k = 4).

Lower numbers are **better** – the last column shows the average percentage
change versus the chosen baseline (negative ⇒ benefit, green; positive ⇒
disadvantage, red).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

__all__ = [
    "plot_resource_stats_table",  # main worker
    "plot_resource_stats_table_entry",  # registry‑friendly wrapper
]

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

# Baseline algorithm keys present in the processed dicts
_BASELINES = {
    "cong_aware",            # congestion‑aware shortest path
    "k_shortest_path_4",    # k‑SP with k = 4 (principal baseline)
}
# Additional keys that appear in some datasets but are still baselines and
# must be *omitted* from the *Algorithm* column
_EXTRA_BASELINES = {"k_shortest_path_1"}
_ALL_BASELINES = _BASELINES | _EXTRA_BASELINES


def _pct_delta(a: float, b: float) -> float:
    """Percentage change from *b* (baseline) to *a* (algo)."""
    return (a - b) / b * 100.0 if b else 0.0


def _aggregate_percent_delta(
    len_mean: float,
    hop_mean: float,
    trp_mean: float,
    len_bl: float,
    hop_bl: float,
    trp_bl: float,
) -> float:
    """Average of the three percentage deltas (length, hops, trp)."""

    deltas = (
        _pct_delta(len_mean, len_bl),
        _pct_delta(hop_mean, hop_bl),
        _pct_delta(trp_mean, trp_bl),
    )
    return float(np.mean(deltas))


# ---------------------------------------------------------------------------
# Plotting core
# ---------------------------------------------------------------------------

def plot_resource_stats_table(
    lengths: Mapping[str, Mapping[str, Mapping[str, float]]],
    hops: Mapping[str, Mapping[str, Mapping[str, float]]],
    transponders: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    save_path: Path | None = None,
    title: str = "Resource_stats – metro_net",
):
    """Create the resource‑use comparison table figure + optional CSV.

    Parameters
    ----------
    lengths, hops, transponders
        Output of the respective `process_*` functions – nested dicts
        ``{algo: {erlang: {"mean": x, "min": y, "max": z}}}``.
    save_path
        If supplied, ``<save_path>.png`` and ``<save_path>.csv`` are written.
    title
        Figure title.
    """

    # ------------------------------------------------------------------
    # 1. Gather rows (skip baseline algos in the *Algorithm* column)
    # ------------------------------------------------------------------
    rows: list[list[Any]] = []

    erlangs_all = sorted({float(e) for alg in lengths.values() for e in alg})
    erlangs_keep = {
        str(erlangs_all[0]),
        str(erlangs_all[len(erlangs_all) // 2]),
        str(erlangs_all[-1]),
    }

    for algo_key in lengths:
        if algo_key in _ALL_BASELINES:  # omit baseline algorithms from col‑0
            continue

        for erlang in erlangs_keep:
            for bl_key, bl_label in (
                ("cong_aware", "cong_aware"),
                ("k_shortest_path_4", "k = 4"),
            ):
                # metric means ------------------------------------------------
                lm = lengths[algo_key][erlang]["mean"]
                hm = hops[algo_key][erlang]["mean"]
                tm = transponders[algo_key][erlang]["mean"]

                # baseline means ---------------------------------------------
                lb = lengths[bl_key][erlang]["mean"]
                hb = hops[bl_key][erlang]["mean"]
                tb = transponders[bl_key][erlang]["mean"]

                pct = _aggregate_percent_delta(lm, hm, tm, lb, hb, tb)

                # build table row --------------------------------------------
                rows.append(
                    [
                        algo_key.replace("_", " ").title(),
                        bl_label,
                        float(erlang),
                        f"{lm:.0f}, {hm:.2f}, {tm:.2f}",
                        f"{lengths[algo_key][erlang]['min']:.0f}, "
                        f"{hops[algo_key][erlang]['min']:.2f}, "
                        f"{transponders[algo_key][erlang]['min']:.2f}",
                        f"{lengths[algo_key][erlang]['max']:.0f}, "
                        f"{hops[algo_key][erlang]['max']:.2f}, "
                        f"{transponders[algo_key][erlang]['max']:.2f}",
                        f"{pct:+.1f} %",
                    ]
                )

    df = (
        pd.DataFrame(
            rows,
            columns=[
                "Algorithm",
                "Baseline",
                "Erlang",
                "Mean (km, hops, trp)",
                "Min",
                "Max",
                "Avg % Δ vs BL (↓ better)",
            ],
        )
        .sort_values(["Erlang", "Baseline", "Algorithm"], ignore_index=True)
    )

    # ------------------------------------------------------------------
    # 2. Draw table
    # ------------------------------------------------------------------
    # – make fonts larger than the blocking‑stats table for readability
    base_font = 12
    plt.rcParams.update({"font.size": base_font})

    fig_h = 0.7 + 0.40 * len(df)  # row‑dependent height
    fig, ax = plt.subplots(figsize=(15, fig_h), dpi=300)
    ax.axis("off")
    fig.suptitle(title, fontsize=base_font + 2, fontweight="bold", y=0.97)

    col_w = [0.16, 0.10, 0.07, 0.21, 0.12, 0.12, 0.22]  # total = 1.00
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        colWidths=col_w,
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(base_font)
    tbl.scale(1.35, 1.55)  # wider & taller cells

    # ------------------------------------------------------------------
    # 3. Colour‑code last column (green benefit → red disadvantage)
    # ------------------------------------------------------------------
    palette = {
        "Large Benefit": to_rgba("mediumseagreen", 0.60),
        "Moderate Benefit": to_rgba("palegreen", 0.60),
        "Small Benefit": to_rgba("honeydew", 0.85),
        "Negligible": to_rgba("lightyellow", 0.85),
        "Small Disadvantage": to_rgba("moccasin", 0.80),
        "Moderate Disadvantage": to_rgba("lightsalmon", 0.75),
        "Large Disadvantage": to_rgba("lightcoral", 0.55),
    }

    def _bucket(val_str: str) -> str:
        v = float(val_str.rstrip(" %"))
        if v <= -20:
            return "Large Benefit"
        if v <= -10:
            return "Moderate Benefit"
        if v <= -5:
            return "Small Benefit"
        if v < 5:
            return "Negligible"
        if v < 10:
            return "Small Disadvantage"
        if v < 20:
            return "Moderate Disadvantage"
        return "Large Disadvantage"

    ncols = len(df.columns)
    boundary = df["Erlang"].ne(df["Erlang"].shift(-1))

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
        if col == 0 and row > 0:
            cell._loc = "left"
        if row > 0:
            if col == ncols - 1:
                cell.set_facecolor(palette[_bucket(cell.get_text().get_text())])
            # grey separator after each Erlang block (except last)
            if boundary.iloc[row - 1] and col != ncols - 1:
                cell.set_facecolor("#e6e6e6")
                cell.set_text_props(weight="bold")

    # ------------------------------------------------------------------
    # 4. Save / show
    # ------------------------------------------------------------------
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path.with_suffix(".png"), bbox_inches="tight")
        df.to_csv(save_path.with_suffix(".csv"), index=False)
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Registry‑friendly wrapper
# ---------------------------------------------------------------------------

def plot_resource_stats_table_entry(
    processed_dict: Dict[str, Any],
    *,
    save_path: Path | None = None,
    title: str | None = None,
):
    """Thin wrapper so `registry.py` can call us with one dict."""

    return plot_resource_stats_table(
        processed_dict["lengths"],
        processed_dict["hops"],
        processed_dict["trp"],
        save_path=save_path,
        title=title or "Resource_stats – metro_net",
    )
