"""
Call from any notebook or script:

    from scripts.plot_experiment import plot_experiment
    plot_experiment('20250424_nsfnet',
                    include_alg=['ppo','qr_dqn'],
                    metric='rewards',
                    variant='per_seed')
"""
import pathlib, pandas as pd, json, itertools, typing as T
from reinforcement_learning.plot import (
    blocking as bl, rewards as rw, state_values as sv)

ROOT = pathlib.Path('data/output')


def _collect(exp: str) -> pd.DataFrame:
    date, net = exp.split('_')
    rows = []
    for daydir in (ROOT / net / date).iterdir():
        for er_dir in daydir.iterdir():
            run_dir = er_dir
            meta_f = run_dir / 'meta.json'
            if not meta_f.exists(): continue
            meta = json.loads(meta_f.read_text())
            rows.append({**meta,
                         'res_dir': str(run_dir),
                         'erlang': float(er_dir.name)})
    return pd.DataFrame(rows)


def plot_experiment(exp: str,
                    include_alg: T.List[str] = None,
                    metric: str = 'rewards',
                    variant: str = 'per_seed'):
    df = _collect(exp)
    if include_alg: df = df[df.algorithm.isin(include_alg)]
    # ------------------------------------------------------------------
    if metric == 'blocking':
        final = bl.prepare_blocking(df)  # write helper if needed
        bl.plot_blocking_probabilities(final)
    elif metric == 'rewards':
        data = rw.load_all_rewards_files(df)  # your existing util
        if variant == 'per_seed':
            rw.plot_rewards_per_seed(data)
        else:
            rw.plot_rewards_averaged_with_variance(data)
    # add other metrics here
