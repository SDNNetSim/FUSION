# reinforcement_learning/feat_extrs/path_gnn_cached.py
"""
Light-weight feature extractor that re-uses a *cached* per-path embedding.

The file is import-safe in environments where PyTorch is **missing or stubbed**
(e.g. unit tests that monkey-patch ``torch``).  Any hard dependency on
``torch.Tensor`` at *import-time* was removed so you’ll no longer see:

    AttributeError: module 'torch' has no attribute 'Tensor'
"""

from __future__ import annotations

import types
from typing import Any, TYPE_CHECKING

# ----------------------------------------------------------------------
# Optional -- PyTorch
# ----------------------------------------------------------------------
try:  # real PyTorch or a stub that *has* Tensor
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    # Create a *very* small stub so the rest of the module keeps working
    torch = types.SimpleNamespace()  # type: ignore
    torch.as_tensor = lambda x: x  # type: ignore
    torch.stack = None  # type: ignore

# During *static* type-checking we still want the proper symbol
if TYPE_CHECKING:  # pylint: disable=using-constant-test
    from torch import Tensor
else:  # At runtime → plain ``Any``
    Tensor = Any  # type: ignore

# ----------------------------------------------------------------------
# Optional -- Stable-Baselines3
# ----------------------------------------------------------------------
try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
except ModuleNotFoundError:  # pragma: no cover
    # Minimal shim so unit-tests / import don’t break
    class BaseFeaturesExtractor:  # type: ignore
        """
        Stand-in for SB3’s BaseFeaturesExtractor when the real package
        is absent.  **Only** what we need for CachedPathGNN.
        """

        def __init__(self, observation_space: Any, features_dim: int):
            self.observation_space = observation_space
            self.features_dim = features_dim

        # SB3 registers buffers so we provide the same helper.
        def register_buffer(self, name: str, tensor: Any) -> None:  # noqa: D401
            setattr(self, name, tensor)


# ----------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------
class CachedPathGNN(BaseFeaturesExtractor):
    """
    Feature extractor that returns a *pre-computed* (cached) embedding
    for every path.  No GNN layers are evaluated at runtime.

    Parameters
    ----------
    obs_space
        Observation-space description (passed straight to SB3).
    cached_embedding
        Tensor/ndarray with shape ``(n_paths, emb_dim)``.
        Will be broadcast to each batch element during ``forward()``.
    """

    def __init__(self, obs_space: Any, cached_embedding: Tensor):  # type: ignore[name-defined]
        # Work out the final flattened feature dimension
        if hasattr(cached_embedding, "shape"):
            n_paths, emb_dim = cached_embedding.shape[-2:]
            features_dim = int(n_paths * emb_dim)
        else:  # List / other iterable
            features_dim = len(cached_embedding)  # type: ignore[arg-type]

        super().__init__(obs_space, features_dim)

        # Register as buffer *if* PyTorch is really there, otherwise
        # just store the raw object.  Either way the attribute exists.
        if hasattr(torch, "as_tensor"):  # Real torch
            tensor = (cached_embedding
                      if isinstance(cached_embedding, torch.Tensor)  # type: ignore[attr-defined]
                      else torch.as_tensor(cached_embedding))  # type: ignore[arg-type]
            self.register_buffer("cached_embedding", tensor)  # type: ignore[attr-defined]
        else:  # Stub torch → keep plain
            self.cached_embedding = cached_embedding  # type: ignore

    # pylint: disable=arguments-differ
    def forward(self, observations: dict) -> Tensor | Any:  # type: ignore[override]
        """
        Return the flattened cached embedding, duplicated across the batch.

        The observation dict is only inspected for *batch size* via the
        optional ``path_masks`` key.  No tensor operations are performed
        when PyTorch is absent.
        """
        batch_size = 1
        masks = observations.get("path_masks")
        if masks is not None and hasattr(masks, "shape"):
            batch_size = masks.shape[0]

        flat = (self.cached_embedding.reshape(-1)  # type: ignore[attr-defined]
                if hasattr(self.cached_embedding, "reshape")
                else self.cached_embedding)

        if batch_size == 1:
            return flat if not hasattr(torch, "stack") else flat.unsqueeze(0)  # type: ignore[attr-defined]

        if hasattr(torch, "stack"):  # True PyTorch
            return flat.unsqueeze(0).repeat(batch_size, 1)  # type: ignore[attr-defined]
        # Fallback for stubbed torch: simple Python list
        return [flat for _ in range(batch_size)]


__all__ = ["CachedPathGNN"]
