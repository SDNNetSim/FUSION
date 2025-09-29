"""
GNN caching utilities for path-based feature extraction.

This module provides functionality to create and cache GNN embeddings
for network configurations, enabling faster training by avoiding
repetitive embedding calculations.
"""

# Standard library imports
import os
from pathlib import Path

# Third-party imports
import torch

# Local imports
from fusion.modules.rl.feat_extrs.constants import CACHE_DIR
from fusion.modules.rl.feat_extrs.path_gnn_cached import PathGNNEncoder
from fusion.modules.rl.utils.errors import CacheError

__all__ = [
    "main",
]

# NOTE: Cache save path may need adjustment depending on project structure
# Consider making cache directory configurable through environment variables
PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]

# Only change directory if this script is run directly, not when imported
if __name__ == "__main__":
    os.chdir(PROJECT_ROOT_PATH)


def main() -> None:
    """
    Creates and caches GNN embeddings for the specified network configuration.

    :raises CacheError: If caching operation fails
    """
    # Import here to avoid issues when this module is imported as a library
    from fusion.modules.rl.utils.gym_envs import create_environment
    from fusion.utils.logging_config import get_logger

    logger = get_logger(__name__)

    config_path = "ini/run_ini/config.ini"
    root = Path(__file__).resolve().parents[2]
    try:
        env, sim_dict, _ = create_environment(config_path=str(root / config_path))
        obs, _ = env.reset()
    except Exception as e:
        raise CacheError(
            f"Failed to create environment from config '{config_path}': {e}. "
            "Please verify the configuration file exists and is valid."
        ) from e

    cache_path = CACHE_DIR / f"{sim_dict['network']}.pt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        logger.info("Cache already exists: %s", cache_path)
        return

    try:
        enc = PathGNNEncoder(
            env.observation_space,
            emb_dim=env.engine_obj.engine_props["emb_dim"],
            gnn_type=env.engine_obj.engine_props["gnn_type"],
            layers=env.engine_obj.engine_props["layers"],
        ).to(sim_dict.get("device", "cpu")).eval()

        device = torch.device(sim_dict.get("device", "cpu"))

        def to_tensor(
            arr: torch.Tensor | list | tuple, *, dtype: torch.dtype | None = None
        ) -> torch.Tensor:
            """
            Return a torch.Tensor on the correct device.

            :param arr: Input array to convert to tensor
            :type arr: torch.Tensor | list | tuple
            :param dtype: Optional data type for the tensor
            :type dtype: torch.dtype | None
            :return: Tensor on the correct device
            :rtype: torch.Tensor
            """
            if isinstance(arr, torch.Tensor):
                return arr.to(device)
            return torch.as_tensor(arr, dtype=dtype, device=device)

        x = to_tensor(obs["x"])
        edge_index = to_tensor(obs["edge_index"], dtype=torch.long)
        path_masks = to_tensor(obs["path_masks"])

        with torch.inference_mode():
            emb = enc(x, edge_index, path_masks).cpu()

        torch.save(emb, cache_path)
        logger.info("Saved cache to %s", cache_path)
    except Exception as e:
        raise CacheError(
            f"Failed to generate or save GNN cache for network "
            f"'{sim_dict.get('network')}': {e}. "
            f"Cache path: {cache_path}"
        ) from e


if __name__ == "__main__":
    main()
