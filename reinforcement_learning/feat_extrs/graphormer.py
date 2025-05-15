import torch
from torch_geometric.nn import TransformerConv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# TODO: Add params to optuna

class GraphTransformerExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Graph Transformer with SB3.
    """

    def __init__(self, obs_space, emb_dim, heads, layers):
        num_paths = obs_space["path_masks"].shape[0]
        in_dim = obs_space["x"].shape[1]
        out_per_head = emb_dim // heads
        conv_out_dim = heads * out_per_head
        super().__init__(obs_space, features_dim=emb_dim * num_paths)

        self.convs = torch.nn.ModuleList([
            TransformerConv(
                in_channels=(in_dim if i == 0 else conv_out_dim),
                out_channels=out_per_head,
                heads=heads,
                concat=True
            )
            for i in range(layers)
        ])
        self.readout = torch.nn.Linear(conv_out_dim, emb_dim)

    def forward(self, obs):
        """
        Convert observation to feature vector.
        """
        x = obs["x"]  # [B, N, F] or [N, F]
        ei = obs["edge_index"].long()  # [B, 2, E] or [2, E]
        masks = obs["path_masks"]  # [B, k, E] or [k, E]

        # Handle batch dimension
        if x.dim() == 3:
            batch_size = x.size(0)
            if batch_size > 1:
                outputs = []
                for b in range(batch_size):
                    xb = x[b]
                    eib = ei[b] if ei.dim() == 3 else ei
                    mb = masks[b] if masks.dim() == 3 else masks
                    yb = xb
                    for conv in self.convs:
                        yb = conv(yb, eib).relu()
                    src_idx, dst_idx = eib
                    edge_emb_b = (yb[src_idx] + yb[dst_idx]) * 0.5
                    path_emb_b = mb @ edge_emb_b
                    pv_b = self.readout(path_emb_b).flatten()
                    outputs.append(pv_b)
                return torch.stack(outputs, dim=0)

            x = x.squeeze(0)
            ei = ei.squeeze(0) if ei.dim() == 3 else ei
            masks = masks.squeeze(0) if masks.dim() == 3 else masks

        # Single sample (no batch) or after squeezing batch=1
        y = x
        for conv in self.convs:
            y = conv(y, ei).relu()

        src_idx, dst_idx = ei
        edge_emb = (y[src_idx] + y[dst_idx]) * 0.5
        path_emb = masks @ edge_emb
        path_vec = self.readout(path_emb)
        flat = path_vec.flatten()
        return flat.unsqueeze(0)
