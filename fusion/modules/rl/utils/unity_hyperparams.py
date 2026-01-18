"""
Unity hyperparameter optimization utilities.

This module provides functionality for parsing SLURM output files,
collecting hyperparameter tuning results, and finding robust optimal
hyperparameters across different network loads.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fusion.utils.logging_config import get_logger

# Default paths for Unity experiments - can be overridden
DEFAULT_IN_ROOT = Path("../experiments/input/0502")
DEFAULT_OUT_ROOT = Path("../experiments/output/0502/")
DEFAULT_GLOB_PATTERN = "**/*.out"

CSV_ROW_RE = re.compile(r"CSV Row \d+:\s*(.*)")
TRIAL_RE = re.compile(
    r"Trial\s+(?P<trial>\d+)\s+finished\s+with\s+value:\s+"
    r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+and\s+parameters:\s+(?P<params>\{.*?\})"
)
DATE_DIR_RE = re.compile(r"experiments[\\/](\d{4})[\\/]", re.IGNORECASE)


def _parse_csv_row(row_str: str, header_str: str) -> dict[str, str]:
    """Return dict mapping header fields -> row values."""
    headers = [h.strip() for h in header_str.split(",")]
    values = [v.strip() for v in row_str.split(",")]
    return dict(zip(headers, values, strict=False))


def _parse_one_out(path: Path) -> tuple[dict[str, str | None], pd.DataFrame]:
    """Parse a single SLURM .out file and return (meta, trials_df)."""
    meta_dict: dict[str, str | None] = {}
    trials_list: list[dict[str, Any]] = []
    row_str = None
    header_str = None

    with path.open("r", errors="ignore") as fh:
        for line in fh:
            # Capture *any* CSV Row X line
            m_row = CSV_ROW_RE.search(line)
            if m_row:
                row_str = m_row.group(1).strip()
                continue

            # Header appears on the same line ("Header: run_id,...")
            if line.startswith("Header:"):
                header_str = line.split("Header:", 1)[1].strip()
                if row_str:
                    meta_dict.update(_parse_csv_row(row_str, header_str))
                    row_str = header_str = None  # reset for safety
                continue

            # Trials
            m_trial = TRIAL_RE.search(line)
            if m_trial:
                trials_list.append(
                    {
                        "trial": int(m_trial.group("trial")),
                        "objective_value": float(m_trial.group("value")),
                        **ast.literal_eval(m_trial.group("params")),
                    }
                )

    if not trials_list:
        raise ValueError(f"No trial lines detected in {path}")

    keep = ["run_id", "path_algorithm", "network", "erlang_start"]
    meta_small_dict: dict[str, str | None] = {k: meta_dict.get(k) for k in keep}

    trials_df = pd.DataFrame(trials_list).sort_values("trial").reset_index(drop=True)

    return meta_small_dict, trials_df


def _destination(
    meta: dict[str, str | None], out_root: Path, orig_path: Path
) -> tuple[Path, str]:
    """Build destination directory & filename for a given (meta, source_path)."""
    alg = meta["path_algorithm"] or "unknown_algorithm"
    net = meta["network"] or "unknown_network"
    run_id = meta["run_id"] or "unknown_run"
    erlang = meta["erlang_start"] or "0"

    match = DATE_DIR_RE.search(str(orig_path))
    date_chunk = match.group(1) if match else datetime.today().strftime("%m%d")

    dest_dir = out_root / alg / net / date_chunk / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir, f"{erlang}_results.csv"


def collect(in_root: Path, out_root: Path, glob_pattern: str = "**/*.out") -> None:
    """Parse every .out file under `in_root` and write CSV/JSON."""
    logger = get_logger(__name__)
    files = sorted(in_root.glob(glob_pattern))
    logger.info("[collect] Found %d log file(s) under %s", len(files), in_root)

    for fp in files:
        try:
            meta, df = _parse_one_out(fp)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logger.warning("[skip] %s: Failed to parse file - %s", fp.name, e)
            continue
        except Exception as e:
            logger.warning("[skip] %s: Unexpected error - %s", fp.name, e)
            continue

        dest_dir, csv_name = _destination(meta, out_root, fp)
        df.to_csv(dest_dir / csv_name, index=False)
        (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(
            "Processed %s -> %s/%s",
            fp.relative_to(in_root),
            dest_dir.relative_to(out_root),
            csv_name,
        )


def _encode_param_matrix(
    df: pd.DataFrame,
    ignore: tuple[str, ...] = ("trial", "objective_value", "erlang_start"),
) -> tuple[np.ndarray, ColumnTransformer, list[str]]:
    """Return (X, enc, param_cols) where X is the encoded feature matrix."""
    param_cols = [c for c in df.columns if c not in ignore]
    num_cols = [c for c in param_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in param_cols if c not in num_cols]

    logger = get_logger(__name__)
    logger.debug("[encode] #numeric=%d  #categorical=%d", len(num_cols), len(cat_cols))

    # Fill missing numerics (e.g. unused layers) with a sentinel
    num_tf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical → one-hot
    cat_tf = Pipeline(
        [("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    enc = ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", cat_tf, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    curr_x = enc.fit_transform(df[param_cols])

    logger.debug("[encode] Encoded feature matrix shape: %s", curr_x.shape)
    if np.isnan(curr_x).any():
        raise ValueError(
            "[encode] NaNs remain after preprocessing. Investigate source."
        )
    return curr_x, enc, param_cols


def _knn_predict_matrix(
    df: pd.DataFrame, curr_x: np.ndarray, k: int = 5
) -> tuple[np.ndarray, list[float]]:
    """
    Build one k‑NN model per Erlang load and return a (n_samples, n_loads) matrix
    where entry (i, j) is the *predicted* objective of config i at load j.
    """
    logger = get_logger(__name__)
    loads = sorted(df["erlang_start"].unique())
    n, _ = curr_x.shape
    preds = np.empty((n, len(loads)), dtype=float)

    for j, load in enumerate(loads):
        mask_load = df["erlang_start"] == load
        x_load = curr_x[mask_load]
        y_load = df.loc[mask_load, "objective_value"].to_numpy()

        # Choose k adaptively per load
        n_i = len(x_load)
        local_k = max(3, min(7, n_i // 2))  # floor=3  ceiling=7
        nbrs = NearestNeighbors(n_neighbors=local_k, metric="euclidean").fit(x_load)

        # Query whole set
        dists, idxs = nbrs.kneighbors(curr_x, return_distance=True)
        # Weight by inverse distance (add ε to avoid /0)
        weights = 1.0 / (dists + 1e-9)
        weights /= weights.sum(axis=1, keepdims=True)
        pred_load = (weights * y_load[idxs]).sum(axis=1)
        preds[:, j] = pred_load

        logger.debug("[knn]  load=%s   rows_in_load=%d   k=%d", load, len(x_load), k)

    return preds, loads


def _knn_robust_aggregate(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Return DataFrame with one row per hyper‑parameter vector and robustness stats
    computed from k‑NN‑predicted returns across *all* Erlang loads.
    """
    logger = get_logger(__name__)
    logger.info(
        "[knn_agg] Incoming rows: %d   unique loads: %d",
        len(df),
        df["erlang_start"].nunique(),
    )
    curr_x, _, param_cols = _encode_param_matrix(df)

    pred_mat, _ = _knn_predict_matrix(df, curr_x, k=k)
    logger.debug("[knn_agg] Prediction matrix shape: %s", pred_mat.shape)

    keys, _ = zip(*df.apply(_hash_params, axis=1), strict=False)
    df["_key"] = keys

    summary_rows = []
    for key, idxs in df.groupby("_key").groups.items():
        pmat = pred_mat[list(idxs)]
        obs = pmat.mean(axis=0)
        summary_rows.append(
            {
                **df.loc[idxs[0], param_cols].to_dict(),
                "worst_pred_return": obs.min(),
                "mean_pred_return": obs.mean(),
                "std_pred_return": obs.std(ddof=0),
                "samples": len(idxs),
                "_key": key,
            }
        )

    out = pd.DataFrame(summary_rows)
    out = out.sort_values(
        ["worst_pred_return", "mean_pred_return"], ascending=[False, False]
    )  # higher is better
    logger.info("[knn_agg] Aggregation complete")
    return out


def _hash_params(
    row: pd.Series,
    ignore: tuple[str, ...] = ("trial", "objective_value", "erlang_start"),
) -> tuple[str, dict[str, Any]]:
    """Return (md5‑hash, params_dict) for a trial row, ignoring bookkeeping cols."""
    items = sorted((k, row[k]) for k in row.index if k not in ignore)
    # MD5 used only for creating unique identifiers, not for security
    hash_value = hashlib.md5(str(items).encode(), usedforsecurity=False).hexdigest()
    return hash_value, dict(items)


def _gather_csvs(topo_dir: Path) -> list[Path]:
    csvs = list(topo_dir.glob("**/*_results.csv"))
    if csvs:
        logger = get_logger(__name__)
        rel = [c.relative_to(DEFAULT_OUT_ROOT) for c in csvs]
        logger.info(
            "[gather_csvs] Found %d CSVs under %s:",
            len(csvs),
            topo_dir.relative_to(DEFAULT_OUT_ROOT),
        )
        for c in rel:
            logger.info("              • %s", c)
    return csvs


def find_best_params_for_topology(topo_dir: Path, out_root: Path | None = None) -> None:
    """
    Finds the best parameters for a topology.
    """
    logger = get_logger(__name__)
    out_root = out_root or DEFAULT_OUT_ROOT
    csv_files = _gather_csvs(topo_dir)
    if not csv_files:
        logger.warning(
            "[Phase 2] No CSVs under %s; skipping.", topo_dir.relative_to(out_root)
        )
        return

    frames = []
    for f in csv_files:
        erlang = int(f.stem.split("_", 1)[0])  # "200_results.csv" → 200
        df = pd.read_csv(f)
        df["erlang_start"] = erlang
        frames.append(df)
        logger.info("[Phase 2] Loaded %s → rows=%d, erlang=%d", f.name, len(df), erlang)

    df_all = pd.concat(frames, ignore_index=True)
    logger.info("[Phase 2] Total concatenated rows: %d", len(df_all))
    leaderboard = _knn_robust_aggregate(df_all, k=5)
    logger.info(
        "Top 5 results:\n%s",
        leaderboard.head(5)[
            [
                "worst_pred_return",
                "mean_pred_return",
                "std_pred_return",
                "samples",
            ]
        ],
    )
    best = leaderboard.iloc[0]

    best_params_path = topo_dir / "best_params.json"
    best_dict = {
        str(k): (
            v.item() if hasattr(v, "item") else v
        )  # converts np.int64, np.float64, np.bool_ to int/float/bool
        for k, v in best.items()
        if k != "_key"
    }
    best_params_path.write_text(json.dumps(best_dict, indent=2))

    logger.info(
        "[Phase 2] Best params saved: %s -> best_params.json (worst=%.2f, mean=%.2f)",
        topo_dir.relative_to(out_root),
        best["worst_pred_return"],
        best["mean_pred_return"],
    )
    # Optional: log top‑3 summary
    logger.debug(
        "Top 3 summary:\n%s",
        leaderboard.head(3)[
            [
                "worst_pred_return",
                "mean_pred_return",
                "std_pred_return",
            ]
        ],
    )


def sweep_all_topologies(out_root: Path) -> None:
    """
    Sweeps all topologies.
    """
    logger = get_logger(__name__)
    logger.info("[sweep_all_topologies] Scanning %s ...", out_root)
    for alg_dir in out_root.iterdir():
        if not alg_dir.is_dir():
            continue
        logger.info("[sweep_all_topologies] Algorithm: %s", alg_dir.name)
        for net_dir in alg_dir.iterdir():
            if net_dir.is_dir():
                logger.info("[sweep_all_topologies]   Topology: %s", net_dir.name)
                find_best_params_for_topology(net_dir, out_root)


if __name__ == "__main__":
    collect(DEFAULT_IN_ROOT, DEFAULT_OUT_ROOT, DEFAULT_GLOB_PATTERN)
    sweep_all_topologies(DEFAULT_OUT_ROOT)
