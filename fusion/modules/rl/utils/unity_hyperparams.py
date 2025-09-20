from __future__ import annotations

import ast
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

IN_ROOT = Path("../experiments/input/0502")
OUT_ROOT = Path("../experiments/output/0502/")
GLOB_PATTERN = "**/*.out"

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


def _parse_one_out(path: Path) -> tuple[dict[str, str], pd.DataFrame]:
    """Parse a single SLURM .out file and return (meta, trials_df)."""
    meta: dict[str, str] = {}
    trials: list[dict] = []
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
                    meta.update(_parse_csv_row(row_str, header_str))
                    row_str = header_str = None  # reset for safety
                continue

            # Trials
            m_trial = TRIAL_RE.search(line)
            if m_trial:
                trials.append(
                    {
                        "trial": int(m_trial.group("trial")),
                        "objective_value": float(m_trial.group("value")),
                        **ast.literal_eval(m_trial.group("params")),
                    }
                )

    if not trials:
        raise ValueError(f"No trial lines detected in {path}")

    keep = ["run_id", "path_algorithm", "network", "erlang_start"]
    meta_small = {k: meta.get(k) for k in keep}

    trials_df = pd.DataFrame(trials).sort_values("trial").reset_index(drop=True)

    return meta_small, trials_df


def _destination(
    meta: dict[str, str], out_root: Path, orig_path: Path
) -> tuple[Path, str]:
    """Build destination directory & filename for a given (meta, source_path)."""
    alg = meta["path_algorithm"]
    net = meta["network"]
    run_id = meta["run_id"]
    erlang = meta["erlang_start"]

    match = DATE_DIR_RE.search(str(orig_path))
    date_chunk = match.group(1) if match else datetime.today().strftime("%m%d")

    dest_dir = out_root / alg / net / date_chunk / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir, f"{erlang}_results.csv"


def collect(in_root: Path, out_root: Path, glob_pattern: str = "**/*.out") -> None:
    "Parse every .out file under `in_root` and write CSV/JSON."
    files = sorted(in_root.glob(glob_pattern))
    print(f"[collect] Found {len(files)} log file(s) under {in_root}")

    for fp in files:
        try:
            meta, df = _parse_one_out(fp)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"   [skip] {fp.name}: Failed to parse file - {e}")
            continue
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"   [skip] {fp.name}: Unexpected error - {e}")
            continue

        dest_dir, csv_name = _destination(meta, out_root, fp)
        df.to_csv(dest_dir / csv_name, index=False)
        (dest_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(
            f"   ✓ {fp.relative_to(in_root)} → {dest_dir.relative_to(out_root)}/{csv_name}"
        )


def _encode_param_matrix(
    df: pd.DataFrame, ignore=("trial", "objective_value", "erlang_start")
):
    """Return (X, enc, param_cols) where X is the encoded feature matrix."""
    param_cols = [c for c in df.columns if c not in ignore]
    num_cols = [c for c in param_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in param_cols if c not in num_cols]

    print(f"[encode] #numeric={len(num_cols)}  #categorical={len(cat_cols)}")

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

    print(f"[encode] Encoded feature matrix shape: {curr_x.shape}")
    if np.isnan(curr_x).any():
        raise ValueError(
            "[encode] ❌ NaNs remain after preprocessing! Investigate source."
        )
    return curr_x, enc, param_cols


def _knn_predict_matrix(df: pd.DataFrame, curr_x: np.ndarray, k: int = 5):
    """
    Build one k‑NN model per Erlang load and return a (n_samples, n_loads) matrix
    where entry (i, j) is the *predicted* objective of config i at load j.
    """
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

        print(f"[knn]  load={load:<4}   rows_in_load={len(x_load):>3}   k={k}")

    return preds, loads


def _knn_robust_aggregate(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Return DataFrame with one row per hyper‑parameter vector and robustness stats
    computed from k‑NN‑predicted returns across *all* Erlang loads.
    """
    print(
        f"[knn_agg] Incoming rows: {len(df)}   unique loads: {df['erlang_start'].nunique()}"
    )
    curr_x, _, param_cols = _encode_param_matrix(df)

    pred_mat, _ = _knn_predict_matrix(df, curr_x, k=k)
    print(f"[knn_agg] Prediction matrix shape: {pred_mat.shape}")

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
    print("[knn_agg] Aggregation complete")
    return out


def _hash_params(
    row: pd.Series, ignore=("trial", "objective_value", "erlang_start")
) -> tuple[str, dict]:
    """Return (md5‑hash, params_dict) for a trial row, ignoring bookkeeping cols."""
    items = sorted((k, row[k]) for k in row.index if k not in ignore)
    return hashlib.md5(str(items).encode()).hexdigest(), dict(items)


def _gather_csvs(topo_dir: Path) -> list[Path]:
    csvs = list(topo_dir.glob("**/*_results.csv"))
    if csvs:
        rel = [c.relative_to(OUT_ROOT) for c in csvs]
        print(
            f"[gather_csvs] Found {len(csvs)} CSVs under {topo_dir.relative_to(OUT_ROOT)}:"
        )
        for c in rel:
            print(f"              • {c}")
    return csvs


def find_best_params_for_topology(topo_dir: Path) -> None:
    """
    Finds the best parameters for a topology.
    """
    csv_files = _gather_csvs(topo_dir)
    if not csv_files:
        print(f"[Phase 2] No CSVs under {topo_dir.relative_to(OUT_ROOT)}; skipping.")
        return

    frames = []
    for f in csv_files:
        erlang = int(f.stem.split("_", 1)[0])  # "200_results.csv" → 200
        df = pd.read_csv(f)
        df["erlang_start"] = erlang
        frames.append(df)
        print(f"[Phase 2] Loaded {f.name:<20} → rows={len(df):>4}, erlang={erlang}")

    df_all = pd.concat(frames, ignore_index=True)
    print(f"[Phase 2] Total concatenated rows: {len(df_all)}")
    leaderboard = _knn_robust_aggregate(df_all, k=5)
    print(
        leaderboard.head(5)[
            ["worst_pred_return", "mean_pred_return", "std_pred_return", "samples"]
        ]
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

    print(
        f"[Phase 2] 🏆  {topo_dir.relative_to(OUT_ROOT)} → best_params.json saved "
        f"(worst={best['worst_pred_return']:.2f}, mean={best['mean_pred_return']:.2f})"
    )
    # Optional: print top‑3 summary
    print(
        leaderboard.head(3)[
            ["worst_pred_return", "mean_pred_return", "std_pred_return"]
        ]
    )


def sweep_all_topologies(out_root: Path) -> None:
    """
    Sweeps all topologies.
    """
    print(f"[sweep_all_topologies] Scanning {out_root} ...")
    for alg_dir in out_root.iterdir():
        if not alg_dir.is_dir():
            continue
        print(f"[sweep_all_topologies] ▶ Algorithm: {alg_dir.name}")
        for net_dir in alg_dir.iterdir():
            if net_dir.is_dir():
                print(f"[sweep_all_topologies]   └─ Topology: {net_dir.name}")
                find_best_params_for_topology(net_dir)


if __name__ == "__main__":
    collect(IN_ROOT, OUT_ROOT, GLOB_PATTERN)
    sweep_all_topologies(OUT_ROOT)
