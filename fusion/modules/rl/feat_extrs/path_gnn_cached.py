import torch
from torch_geometric.nn import GATv2Conv, SAGEConv, GraphConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PathGNNEncoder(torch.nn.Module):
    """
    Cache the GNN encoder to a file.
    """
    def __init__(self, obs_space, emb_dim=64, gnn_type="gat", layers=2):
        super().__init__()
        conv_map = {"gat": GATv2Conv, "sage": SAGEConv, "graphconv": GraphConv}
        conv = conv_map.get(gnn_type, GATv2Conv)
        in_dim = obs_space["x"].shape[1]
        self.convs = torch.nn.ModuleList(
            [conv(in_dim if i == 0 else emb_dim, emb_dim) for i in range(layers)]
        )
        self.readout = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x, edge_index, path_masks):
        """
        Forward propagation.
        """
        y = x
        for c in self.convs:
            y = c(y, edge_index).relu()
        src, dst = edge_index
        edge_emb = (y[src] + y[dst]) * 0.5
        path_emb = path_masks @ edge_emb
        return self.readout(path_emb).flatten()


class CachedPathGNN(BaseFeaturesExtractor):
    """
    Cache the GNN encoder to a file.
    """
    def __init__(self, obs_space, cached_embedding: torch.Tensor):
        super().__init__(obs_space, features_dim=cached_embedding.numel())
        self.register_buffer("cached", cached_embedding)

    def forward(self, obs):
        """
        Forward propagation.
        """
        batch = obs["x"].shape[0] if obs["x"].dim() == 3 else 1
        return self.cached.unsqueeze(0).repeat(batch, 1)
