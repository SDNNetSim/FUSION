# TODO: (version 5.5-6) No longer supported.

import math

import torch

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PointerHead(torch.nn.Module):
    """
    Pointer head used in the pointer policy
    """

    def __init__(self, dim):
        super().__init__()
        self.qkv = torch.nn.Linear(dim, dim * 3)

    def forward(self, path_feats):  # shape (batch, 3, dim)
        """
        Forward pass of the pointer head.
        """
        qkv = self.qkv(path_feats)  # (batch,3,3*dim)
        q, k, v = qkv.chunk(3, dim=-1)
        # attention scores
        scores = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(q.size(-1))
        attn = torch.softmax(scores, dim=-1)  # (batch,3,3)
        # weighted sum
        out = torch.einsum("bij,bjd->bid", attn, v)  # (batch,3,dim)
        # return logits per path
        logits = out.sum(dim=-1)  # (batch,3)
        return logits


class PointerPolicy(ActorCriticPolicy):
    """
    The pointer policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp_extractor = None

    def _build_mlp_extractor(self):
        # override to use PointerHead
        self.mlp_extractor = BaseFeaturesExtractor(self.features_extractor.observation_space,
                                                   features_dim=self.features_extractor.features_dim)
        self.mlp_extractor.policy_net = PointerHead(self.features_extractor.features_dim)
        self.mlp_extractor.value_net = torch.nn.Linear(self.features_extractor.features_dim, 1)
