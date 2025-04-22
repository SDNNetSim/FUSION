import torch
from torch_geometric.nn import GATv2Conv, SAGEConv, GraphConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class PathGNN(BaseFeaturesExtractor):
    def __init__(self, obs_space, emb_dim, gnn_type, layers):
        super().__init__(obs_space, features_dim=emb_dim * obs_space["path_masks"].shape[0])
        conv_map = {"gat": GATv2Conv, "sage": SAGEConv, "graphconv": GraphConv}
        Conv = conv_map.get(gnn_type, GATv2Conv)
        in_dim = obs_space["x"].shape[1]
        self.convs = torch.nn.ModuleList([
            Conv(in_dim if i == 0 else emb_dim, emb_dim)
            for i in range(layers)
        ])
        self.readout = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, obs):
        x = obs["x"]
        ei = obs["edge_index"].long()
        masks = obs["path_masks"]

        if x.dim() == 2:
            x = x.unsqueeze(0)
            ei = ei.unsqueeze(0)
            masks = masks.unsqueeze(0)

        batch_size = x.size(0)
        outputs = []
        for b in range(batch_size):
            xb = x[b]
            eib = ei[b]
            mb = masks[b]

            y = xb
            for conv in self.convs:
                y = conv(y, eib).relu()

            src_idx, dst_idx = eib
            edge_emb = (y[src_idx] + y[dst_idx]) * 0.5
            path_emb = mb @ edge_emb
            pv = self.readout(path_emb).flatten()
            outputs.append(pv)

        return torch.stack(outputs, dim=0)
