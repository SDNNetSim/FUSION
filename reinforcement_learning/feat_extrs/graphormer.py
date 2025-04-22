import torch
from torch_geometric.nn import TransformerConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GraphTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, emb_dim, heads, layers):
        super().__init__(obs_space, features_dim=emb_dim * 3)
        in_dim = obs_space["x"].shape[1]
        self.convs = torch.nn.ModuleList([
            TransformerConv(
                in_dim if i == 0 else emb_dim,
                emb_dim // heads,
                heads=heads,
                concat=True
            )
            for i in range(layers)
        ])
        self.readout = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, obs):
        x = obs["x"]  # (N, node_feat)
        ei = obs["edge_index"]  # (2, E)
        for conv in self.convs:
            x = conv(x, ei).relu()  # (N, emb_dim)
        # build edge embeddings
        src, dst = ei
        edge_emb = (x[src] + x[dst]) * 0.5  # (E, emb_dim)
        # path masks summation
        masks = obs["path_masks"]  # (3, E)
        path_emb = torch.einsum("re,pe->pr", edge_emb, masks)
        # project and flatten
        path_vec = self.readout(path_emb)  # (3, emb_dim)
        return path_vec.flatten()  # (3*emb_dim,)
