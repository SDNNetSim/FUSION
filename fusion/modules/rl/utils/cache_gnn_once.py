import os
from pathlib import Path

import torch

from fusion.modules.rl.feat_extrs.constants import CACHE_DIR
from fusion.modules.rl.feat_extrs.path_gnn_cached import PathGNNEncoder
from fusion.modules.rl.utils.errors import CacheError

# NOTE: Cache save path may need adjustment depending on project structure
# Consider making cache directory configurable through environment variables
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Only change directory if this script is run directly, not when imported
if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)


def main():
    """
    Creates and caches GNN embeddings for the specified network configuration.
    
    :raises CacheError: If caching operation fails
    """
    # Import here to avoid issues when this module is imported as a library
    from fusion.modules.rl.utils.gym_envs import create_environment  # pylint: disable=import-outside-toplevel
    from fusion.utils.logging_config import get_logger  # pylint: disable=import-outside-toplevel

    logger = get_logger(__name__)

    config_path = 'ini/run_ini/config.ini'
    root = Path(__file__).resolve().parents[2]
    try:
        env, sim_dict, _ = create_environment(config_path=str(root / config_path))
        obs, _ = env.reset()
    except Exception as e:
        raise CacheError(
            f"Failed to create environment from config '{config_path}': {e}. "
            "Please verify the configuration file exists and is valid."
        ) from e

    cache_fp = CACHE_DIR / f"{sim_dict['network']}.pt"
    if cache_fp.exists():
        print("Cache already exists:", cache_fp)
        return

    try:
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
        logger.info("Saved cache to %s", cache_fp)
    except Exception as e:
        raise CacheError(
            f"Failed to generate or save GNN cache for network '{sim_dict.get('network')}': {e}. "
            f"Cache path: {cache_fp}"
        ) from e


if __name__ == "__main__":
    main()
