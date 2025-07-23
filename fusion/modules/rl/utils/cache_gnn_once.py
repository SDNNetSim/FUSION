import os
from pathlib import Path
import torch
from reinforcement_learning.feat_extrs.constants import CACHE_DIR
from reinforcement_learning.feat_extrs.path_gnn_cached import PathGNNEncoder

# TODO: (version 5.5-6) Does not save in the correct path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
from reinforcement_learning.utils.gym_envs import create_environment  # pylint: disable=wrong-import-position


def main():
    """
    Controls the function.
    """
    config_path = 'ini/run_ini/config.ini'
    root = Path(__file__).resolve().parents[2]
    env, sim_dict, _ = create_environment(config_path=str(root / config_path))
    obs, _ = env.reset()

    cache_fp = CACHE_DIR / f"{sim_dict['network']}.pt"
    if cache_fp.exists():
        print("Cache already exists:", cache_fp)
        return

    enc = PathGNNEncoder(
        env.observation_space,
        emb_dim=env.engine_obj.engine_props["emb_dim"],
        gnn_type=env.engine_obj.engine_props["gnn_type"],
        layers=env.engine_obj.engine_props["layers"],
    ).to(sim_dict.get("device", "cpu")).eval()

    device = torch.device(sim_dict.get("device", "cpu"))

    def to_tensor(arr, *, dtype=None):
        """Return a torch.Tensor on the correct device."""
        if isinstance(arr, torch.Tensor):
            return arr.to(device)
        return torch.as_tensor(arr, dtype=dtype, device=device)

    x = to_tensor(obs["x"])
    edge_index = to_tensor(obs["edge_index"], dtype=torch.long)
    path_masks = to_tensor(obs["path_masks"])

    with torch.inference_mode():
        emb = enc(x, edge_index, path_masks).cpu()

    torch.save(emb, cache_fp)
    print("✅  Saved cache to", cache_fp)


if __name__ == "__main__":
    main()
